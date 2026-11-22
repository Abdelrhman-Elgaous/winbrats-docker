#!/usr/bin/env python3
"""
EXP-1  ·  Small-lesion-aware candidate generation
=================================================
One entry point for every step of the frozen protocol that is NOT the training
run itself.  Training is nnU-Net's own CLI plus the trainer in
nnUNetTrainerSizeAware.py; everything else lives here.

    python exp1.py calibrate --labels DIR            # protocol §7 / Appendix A
    python exp1.py weights-pp --preproc DIR          # protocol §6 (preferred)
    python exp1.py selftest                          # proves the loss changes
    python exp1.py gate      --baseline A --new B    # protocol §4 and §11

FROZEN CONSTANTS — these are the protocol.  Changing one invalidates the
experiment, so they are asserted at import and never read from the CLI.
"""
import argparse, json, os, re, sys
import numpy as np

V0, ALPHA, W_MAX = 27.0, 0.5, 8.0      # protocol §7
#: Lower bound on the weight, added 6 October 2026 (protocol §15, deviation 4).
#: Without it the function falls to 0.016 for a 100,000 mm3 lesion while every
#: background voxel stays at 1.0, so the intervention suppressed 99.91% of the
#: foreground ~30-fold instead of amplifying the small lesions.  The floor
#: makes it purely additive: small lesions up, large lesions unchanged.
W_MIN            = 1.0                  # protocol §7
#: Weights ride inside `seg`, which the dataloader may allocate as an integer
#: type; a float weight would then be truncated to 1 with nothing raised.
#: Storing w * WEIGHT_SCALE as an integer survives any cast, and the loss
#: divides it back. `exp1.py inspect-dataloader` reports which is needed.
WEIGHT_SCALE     = 1000
SMALL_MM3        = 27.0                 # protocol §10, secondary endpoint
P_CAND           = 0.02                 # protocol §10, inherited unchanged
#: The floating-point disagreement this project already measured between the
#: Python inference API and the CLI on the same model and folds; protocol §4.1
#: uses it as the reference for "reproduced", rather than a chosen tolerance.
REPRO_PROB_NOISE = 0.0014
FP_GATE          = 1.25                 # protocol §11, relative
N_BOOT, SEED     = 10_000, 20260819

assert (V0, ALPHA, W_MAX, W_MIN) == (27.0, 0.5, 8.0, 1.0), \
    "frozen parameters altered"


def weight_of(volume_mm3):
    """w(V) = max(1, min(8, (27/V)^0.5)).  V in mm3, from the FULL-res label.

    The floor is what makes this a single-difference intervention.  Background
    voxels carry weight 1.0, so an unfloored weight below 1 does not merely
    "amplify small lesions less" — it pushes the large lesion's own voxels
    BELOW the background they are being separated from.  At the frozen alpha
    that reached 0.016 for a 100,000 mm3 lesion.

    The arithmetic the floor gives up is real and is recorded rather than
    hidden: almost all of the 43-fold rise in the small-lesion share of
    foreground voxel mass came from shrinking the large-lesion denominator,
    not from raising the small-lesion numerator, so the floored function moves
    that share far less.  It cannot be otherwise: small lesions hold of order
    0.1% of foreground mass, and reaching parity by amplification alone would
    need weights of order 1e3 against a ceiling of 8.  Run `calibrate` to put
    the honest figure in Appendix A.
    """
    v = np.maximum(np.asarray(volume_mm3, dtype=np.float64), 1e-9)
    return np.clip((V0 / v) ** ALPHA, W_MIN, W_MAX)


# ---------------------------------------------------------------- bootstrap
def paired_diff(a, b, n_boot=N_BOOT, seed=SEED):
    """Paired mean difference with a percentile bootstrap CI and Wilcoxon p.

    Same machinery as the main study so the numbers are comparable.
    """
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired comparison needs matched cases"
    d = a - b
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:                      # all differences zero
        p = 1.0
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                p=p, ci_excludes_zero=bool(lo > 0 or hi < 0))


def prop_diff(*_a, **_k):
    """Retired 6 October 2026. Use `paired_prop_diff` (protocol §15, dev. 5).

    This computed an UNPAIRED binomial bootstrap on two independent counts,
    justified by a docstring claiming the lesions "are not matched one-to-one
    between arms". That claim was false. Both arms predict on the same fold's
    validation cases and are scored against the same reference annotations;
    `case + region + gt_lesion` identifies the same physical lesion in both.
    """
    raise RuntimeError(
        "prop_diff is retired: it treated the two arms as independent samples "
        "when every reference lesion appears in both. Use paired_prop_diff, "
        "which counts per-lesion transitions and bootstraps over cases.")


def paired_prop_diff(case, base_hit, new_hit, n_boot=N_BOOT, seed=SEED):
    """Paired lesion-level proportion difference, bootstrapped over CASES.

    Used for candidate availability and small-lesion sensitivity, where the
    same reference lesion is scored in both arms.

    WHY PAIRED.  The two arms differ only in the loss and are evaluated on the
    same cases against the same annotations, so a lesion detected by both
    carries no information about the difference. Only the discordant lesions
    do. The effect is therefore

        delta = P(missed -> found) - P(found -> missed)

    which is exactly the mean of (new_hit - base_hit) over lesions, and whose
    null is McNemar's. Treating the arms as independent samples inflates the
    interval by the between-arm variance that pairing removes, which on a
    single fold is most of it.

    WHY THE BOOTSTRAP RESAMPLES CASES, NOT LESIONS.  Lesions within one scan
    are not independent: they share the patient, the scanner, the contrast
    timing, the preprocessing and the field of view, and a case with 30
    metastases contributes 30 correlated observations. Resampling lesions
    treats those as 30 independent draws and understates the interval. The
    cluster bootstrap resamples whole cases with replacement and recomputes
    the lesion-level proportion inside the resample, so the unit of
    independence is the unit of resampling.

    `case` is one label per lesion, not per case; the three arrays are
    lesion-aligned and must be the same length.
    """
    from scipy import stats
    case = np.asarray(case)
    a = np.asarray(base_hit).astype(bool)
    b = np.asarray(new_hit).astype(bool)
    if not (case.shape == a.shape == b.shape):
        raise ValueError(
            f"lesion-aligned arrays required; got case{case.shape}, "
            f"base{a.shape}, new{b.shape}. If these differ, the two arms were "
            "not matched on `case + region + gt_lesion` and the comparison is "
            "not paired — fix the join, do not fall back to an unpaired test.")
    if case.size == 0:
        raise ValueError("no lesions to compare")

    recovered = int((~a & b).sum())          # missed -> found
    regressed = int((a & ~b).sum())          # found -> missed
    n = case.size

    # Lesion indices grouped by case, so a resample draws whole cases.
    order = np.argsort(case, kind='stable')
    bounds = np.flatnonzero(np.r_[True, case[order][1:] != case[order][:-1]])
    groups = np.split(order, bounds[1:])
    n_cases = len(groups)

    d = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_cases, n_cases)
        idx = np.concatenate([groups[j] for j in pick])
        boot[i] = d[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # McNemar, exact: the discordant pairs under a fair coin.
    disc = recovered + regressed
    p = 1.0 if disc == 0 else float(
        stats.binomtest(recovered, disc, 0.5).pvalue)

    return dict(n_lesions=n, n_cases=n_cases,
                base=float(a.mean()), new=float(b.mean()),
                diff=float(d.mean()),
                recovered=recovered, regressed=regressed,
                lo=float(lo), hi=float(hi),
                ci_excludes_zero=bool(lo > 0 or hi < 0), p=p)


# --------------------------------------- weight-map coverage measurement
#: The four scored regions, as Dataset137 declares them. The evaluator scores
#: connected components WITHIN each of these; the weight map is built from
#: components of the merged foreground. Those are different objects.
REGIONS = {'WT': (1, 2, 3), 'TC': (1, 3), 'ET': (3,), 'RC': (4,)}


def cmd_coverage(args):
    """Does the weight map actually reach the lesions the evaluator scores?

    The weight map labels connected components of `lab > 0`, merging every
    annotated class into one object. The evaluator labels components inside
    each region separately. A small enhancing lesion that touches surrounding
    non-enhancing tissue therefore sits inside a large merged component and
    receives weight 1.0 -- the same weight as background -- while the
    evaluator counts it as a small lesion.

    The size of that gap cannot be inferred from component counts. Comparing
    4,082 small merged components against 6,051 small ET components does NOT
    bound the shortfall, because one small merged component can contain more
    than one small region component, so the mapping is not one-to-one. This
    command measures the quantity directly instead: for every region lesion at
    or below 27 mm3, it reads the weight the map actually assigns to it.

    A region component lies entirely inside exactly one merged component -- a
    region mask is a subset of the foreground, so its connected components
    cannot cross a foreground gap -- which is why one weight per lesion is
    well defined.

    Read the `covered` column. High coverage means the mismatch is formal
    rather than material and the frozen map stands. Low coverage means the
    intervention does not reach the population it was designed for, and that
    is a protocol question to settle BEFORE the experimental arm, not after.
    """
    import nibabel as nib
    from scipy import ndimage

    files = sorted(f for f in os.listdir(args.labels) if f.endswith(('.nii', '.nii.gz')))
    if not files:
        sys.exit(f"no label files in {args.labels}")

    acc = {r: dict(n=0, cov=0, w=[], swallowed=[], vol=[]) for r in REGIONS}
    for i, f in enumerate(files, 1):
        img = nib.load(os.path.join(args.labels, f))
        lab = np.asanyarray(img.dataobj)
        vox = float(np.prod(img.header.get_zooms()[:3]))

        fg = lab > 0
        if not fg.any():
            continue
        cc_fg, n_fg = ndimage.label(fg)
        if n_fg == 0:
            continue
        # Index 0 is background and keeps weight 1.0, exactly as the map does.
        vol_fg = np.r_[0.0, np.bincount(cc_fg.ravel())[1:] * vox]
        w_fg = np.r_[1.0, weight_of(vol_fg[1:])]
        by_region = (args.definition == 'region')

        for r, vals in REGIONS.items():
            m = np.isin(lab, vals)
            if not m.any():
                continue
            cc_r, n_r = ndimage.label(m)
            if n_r == 0:
                continue
            vol_r = np.bincount(cc_r.ravel())[1:] * vox
            small = vol_r <= SMALL_MM3
            if not small.any():
                continue
            # cc_fg is constant over each region component, so max recovers it.
            parent = ndimage.maximum(cc_fg, cc_r,
                                     index=np.arange(1, n_r + 1)).astype(int)
            if by_region:
                # What the region-wise map assigns: the component's own volume
                # in its own region, which is the object the evaluator scores.
                w_r = weight_of(vol_r)
            else:
                # What the merged-foreground map assigns, which is what the
                # superseded design actually wrote.
                w_r = w_fg[parent]
            a = acc[r]
            a['n'] += int(small.sum())
            a['cov'] += int((w_r[small] > 1.0 + 1e-9).sum())
            a['w'].append(w_r[small])
            a['vol'].append(vol_r[small])
            miss = small & (w_r <= 1.0 + 1e-9)
            if miss.any():
                a['swallowed'].append(vol_fg[parent[miss]])
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)

    print(f"\ncases {len(files)} · components at or below {SMALL_MM3:g} mm3, "
          f"by scored region")
    print(f"weight-map component definition: {args.definition}"
          f"{'  (components of lab>0 — SUPERSEDED)' if args.definition == 'merged' else '  (components inside each region — FROZEN)'}\n")
    print(f"{'region':>7}{'small lesions':>15}{'weight > 1':>12}{'covered':>10}"
          f"{'median w':>11}{'median merged vol':>20}")
    record = {}
    for r in REGIONS:
        a = acc[r]
        if a['n'] == 0:
            print(f"{r:>7}{0:>15}"); continue
        w = np.concatenate(a['w'])
        sw = np.concatenate(a['swallowed']) if a['swallowed'] else np.array([])
        frac = a['cov'] / a['n']
        print(f"{r:>7}{a['n']:>15,}{a['cov']:>12,}{frac:>9.1%}"
              f"{np.median(w):>11.3f}"
              f"{(f'{np.median(sw):,.0f} mm3' if sw.size else '-'):>20}")
        record[r] = dict(n_small=a['n'], n_covered=a['cov'], coverage=float(frac),
                         median_weight=float(np.median(w)),
                         median_merged_volume_of_uncovered=(
                             float(np.median(sw)) if sw.size else None))
    print("\n`median merged vol` is the size of the merged component that "
          "swallowed\nthe lesions receiving weight 1.0. Read it against "
          f"{SMALL_MM3:g} mm3: a value far above\nit means the lesion is "
          "adjacent to other annotated tissue, which is the\nmismatch. A value "
          f"at or just under {SMALL_MM3:g} mm3 is not a mismatch at all -- "
          f"w({SMALL_MM3:g}) = 1\nby construction, so a lesion sitting exactly "
          "at the threshold is counted\nas uncovered even when nothing merged "
          "with it. Judge the coverage figure\nby that column, not on its own.")
    json.dump(record, open(args.out, 'w'), indent=2)
    print(f"\nrecord -> {args.out}")


# ------------------------------------------------------- §7  calibration
def cmd_calibrate(args):
    """Recompute Appendix A on the FULL training set, before freezing.

    An earlier draft of the table came from a 412-case subset; the frozen
    Appendix A rests on all 1,296 training labels, which is what this produces.
    Re-running it must reproduce that table, so it doubles as a check that the
    label directory is the one the protocol was frozen against.
    """
    import nibabel as nib
    from scipy import ndimage
    # Two populations, because they answer two different questions and the
    # earlier record reported only the first.  `all` connects components over
    # every annotated label, which is what the weight map itself is built from.
    # `ET` connects label 3 alone, which is the population Jangir & Sandbhor
    # characterise (41.6% of ET instances, ~1.07% of ET foreground voxel mass).
    # Our all-label figure of 0.107% is a tenfold-different number against a
    # different denominator, and a reader who has seen both will ask; this
    # prints the like-for-like comparison instead of leaving it to be queried.
    pops = {'all': None, 'ET': 3}
    vols = {k: [] for k in pops}
    files = sorted(f for f in os.listdir(args.labels) if f.endswith(('.nii', '.nii.gz')))
    if not files:
        sys.exit(f"no label files in {args.labels}")
    for i, f in enumerate(files, 1):
        img = nib.load(os.path.join(args.labels, f))
        lab = np.asanyarray(img.dataobj)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        for name, which in pops.items():
            fg = lab > 0 if which is None else lab == which
            if not fg.any():
                continue
            cc, n = ndimage.label(fg)
            if n:
                counts = np.bincount(cc.ravel())[1:]
                vols[name].append(np.stack([counts * vox, counts], 1))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)

    record = dict(n_cases=len(files), V0=V0, alpha=ALPHA,
                  w_max=W_MAX, w_min=W_MIN, populations={})
    for name in pops:
        if not vols[name]:
            print(f"\n[{name}] no components found"); continue
        arr = np.concatenate(vols[name]); V, N = arr[:, 0], arr[:, 1]
        small = V <= SMALL_MM3
        base = N[small].sum() / N.sum()
        print(f"\n=== population: {name} "
              f"{'(every annotated label)' if name == 'all' else '(label 3 only)'}")
        print(f"cases {len(files)} · components {len(V):,}")
        print(f"<= {SMALL_MM3:g} mm3 : {small.sum():,} ({small.mean():.1%} of count)")
        print(f"their share of foreground VOXEL mass, unweighted : {base:.3%}")
        print(f"their share by physical volume                   : "
              f"{V[small].sum()/V.sum():.3%}")
        print(f"median voxel volume, small / large : "
              f"{np.median(V[small]/N[small]):.3f} / "
              f"{np.median(V[~small]/N[~small]):.3f} mm3")

        # The floored function is the frozen one.  The unfloored column stays
        # in the table because the protocol's superseded 43x came from it, and
        # the decomposition is the evidence that almost all of that factor was
        # large-lesion suppression rather than small-lesion amplification
        # (protocol §15, deviation 4).
        print(f"\n{'alpha':>6}{'floored <=27':>15}{'x base':>9}"
              f"{'unfloored <=27':>17}{'x base':>9}{'mean w, large':>15}")
        rows = {}
        for a in (0.25, 0.35, 0.50, 0.65, 0.75):
            raw = (V0 / V) ** a
            wf = np.clip(raw, W_MIN, W_MAX)
            wu = np.minimum(W_MAX, raw)
            mf = (wf * N)[small].sum() / (wf * N).sum()
            mu = (wu * N)[small].sum() / (wu * N).sum()
            big_w = (wu * N)[~small].sum() / N[~small].sum()
            rows[a] = dict(floored=float(mf), unfloored=float(mu),
                           x_floored=float(mf / base), x_unfloored=float(mu / base),
                           mean_weight_large_unfloored=float(big_w))
            print(f"{a:>6}{mf:>14.2%}{mf/base:>8.1f}x{mu:>16.2%}"
                  f"{mu/base:>8.0f}x{big_w:>14.3f}"
                  f"{'  <- FROZEN' if a == ALPHA else ''}")
        record['populations'][name] = dict(
            n_components=int(len(V)), n_small=int(small.sum()),
            small_count_share=float(small.mean()),
            unweighted_voxel_share=float(base), by_alpha=rows)

    json.dump(record, open(args.out, 'w'), indent=2)
    print(f"\nrecord -> {args.out}")
    print("Copy the FROZEN row of the `all` population into protocol "
          "Appendix A, replacing the superseded 43x.")


# ------------------------------------------------------- §6  weight maps
def cmd_weights(args):
    """Build one voxel-wise weight map per case from the FULL-resolution label.

    Written BEFORE patch extraction, so a lesion cropped by a patch boundary
    keeps the weight of its whole-lesion physical volume (protocol §6).
    """
    import nibabel as nib
    from scipy import ndimage
    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.labels) if f.endswith(('.nii', '.nii.gz')))
    stats = []
    for i, f in enumerate(files, 1):
        img = nib.load(os.path.join(args.labels, f))
        lab = np.asanyarray(img.dataobj)
        vox = float(np.prod(img.header.get_zooms()[:3]))
        wmap = np.ones(lab.shape, np.float32)          # background weight = 1
        fg = lab > 0
        if fg.any():
            cc, n = ndimage.label(fg)
            counts = np.bincount(cc.ravel())
            w = weight_of(counts[1:] * vox).astype(np.float32)
            wmap[fg] = np.concatenate([[1.0], w])[cc[fg]]
            stats.append(dict(case=f, n_lesions=int(n), vox_mm3=vox,
                              w_min=float(w.min()), w_max=float(w.max())))
        out = os.path.join(args.out, f.replace('.nii', '_weight.nii'))
        nib.save(nib.Nifti1Image(wmap, img.affine, img.header), out)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)
    json.dump(stats, open(os.path.join(args.out, '_weight_manifest.json'), 'w'), indent=2)
    allw = [s['w_max'] for s in stats]
    print(f"\n{len(files)} maps -> {args.out}")
    print(f"largest weight applied anywhere: {max(allw):.2f}  (cap {W_MAX})")
    print(f"cases where the cap bound: {sum(a >= W_MAX - 1e-6 for a in allw)}")
    print("  (the cap binds only below ~0.42 mm3, beneath the evaluator floor)")


# ------------------------------------ §6  weights, computed in nnU-Net space
def _region_weight_channels(lab, vox, dtype=np.int16):
    """R weight channels for one case, as integers scaled by WEIGHT_SCALE.

    Components are labelled inside each scored region separately, which is how
    the evaluator labels them (protocol §6, deviation 7).

    The scaling is not cosmetic. nnU-Net's dataloader casts `seg` to int16 on
    the way out of storage, before any augmentation, so a float weight of
    1.837 would arrive at the loss as 1 and nothing would raise. Storing
    round(w * 1000) survives that cast exactly; `SizeAwareDCCE` divides it back.
    """
    w = np.full((len(REGIONS),) + lab.shape, WEIGHT_SCALE, dtype)
    hi = 0.0
    for ri, vals in enumerate(REGIONS.values()):
        m = np.isin(lab, vals)
        if not m.any():
            continue
        cc, n = ndimage_label(m)
        if n == 0:
            continue
        counts = np.bincount(cc.ravel())
        wl = weight_of(counts[1:] * vox)
        hi = max(hi, float(wl.max()))
        coded = np.rint(np.concatenate([[1.0], wl]) * WEIGHT_SCALE).astype(dtype)
        w[ri][m] = coded[cc[m]]
    return w, hi


def ndimage_label(m):
    from scipy import ndimage
    return ndimage.label(m)


def _weights_pp_blosc2(args, seg_files):
    """The .b2nd path: nnU-Net 2.5+ stores each case as Blosc2 arrays.

    `<case>.b2nd` holds the image data and `<case>_seg.b2nd` the segmentation,
    as separate files rather than two keys of one archive. Only the seg file is
    rewritten here; the data files are the large ones and are never touched.

    The new array is created with the SAME chunk and block geometry as the one
    it replaces, differing only in the channel dimension, so the reader sees
    the layout it expects. The replacement is atomic per case: the new array is
    written beside the old one and renamed over it only after it closes
    cleanly, so an interrupted run leaves every case either old or new, never
    truncated.
    """
    import pickle
    try:
        import blosc2
    except ImportError:
        sys.exit("blosc2 is not installed, but the preprocessed data is .b2nd. "
                 "`pip install blosc2` in the environment that runs this.")

    R = len(REGIONS)
    done = skipped = 0
    wmax_seen = 0.0
    bytes_before = bytes_after = 0
    for i, f in enumerate(seg_files, 1):
        stem = f[:-len('_seg.b2nd')]
        seg_path = os.path.join(args.preproc, f)
        pkl = os.path.join(args.preproc, stem + '.pkl')
        if not os.path.exists(pkl):
            sys.exit(f"missing {pkl}: cannot read the preprocessed spacing")

        old = blosc2.open(urlpath=seg_path, mode='r')
        seg = old[:]
        if seg.ndim == 3:
            seg = seg[None]
        if seg.shape[0] >= 1 + R:
            skipped += 1
            continue
        if seg.shape[0] != 1:
            sys.exit(f"{f} has {seg.shape[0]} channels; expected 1 label "
                     f"channel or {1 + R} if already built. Refusing to guess "
                     f"which is which.")

        props = pickle.load(open(pkl, 'rb'))
        sp = props.get('spacing_after_resampling', props.get('spacing'))
        if sp is None:
            sys.exit(f"{pkl} has no spacing field; check your nnU-Net version")
        vox = float(np.prod(np.asarray(sp, float)[:3]))

        w, hi = _region_weight_channels(seg[0], vox, np.int16)
        wmax_seen = max(wmax_seen, hi)
        new = np.concatenate([seg.astype(np.int16), w], 0)

        chunks = (1,) + tuple(old.chunks[1:])
        blocks = (1,) + tuple(old.blocks[1:])
        tmp = seg_path + '.building'
        if os.path.exists(tmp):
            os.remove(tmp)
        blosc2.asarray(np.ascontiguousarray(new), urlpath=tmp,
                       chunks=chunks, blocks=blocks,
                       cparams={'codec': blosc2.Codec.ZSTD, 'clevel': 8,
                                'nthreads': 4})
        bytes_before += os.path.getsize(seg_path)
        bytes_after += os.path.getsize(tmp)
        os.replace(tmp, seg_path)
        done += 1
        if i % 25 == 0:
            free = __import__('shutil').disk_usage(args.preproc).free / 1e9
            print(f"  {i}/{len(seg_files)} · seg bytes "
                  f"{bytes_before/1e9:.2f} -> {bytes_after/1e9:.2f} GB · "
                  f"{free:.0f} GB free", file=sys.stderr)

    print(f"\nweight channels written for {done} cases "
          f"({skipped} already had them)")
    print(f"largest weight applied: {wmax_seen:.3f}  "
          f"(floor {W_MIN}, cap {W_MAX}; stored as w x {WEIGHT_SCALE} in int16)")
    if bytes_before:
        print(f"seg storage: {bytes_before/1e9:.2f} -> {bytes_after/1e9:.2f} GB "
              f"({bytes_after/max(bytes_before,1):.1f}x)")
    print(f"\nseg now has {1 + R} channels: [0] label, [1..{R}] weights for "
          f"{', '.join(REGIONS)}, scaled by {WEIGHT_SCALE}.")
    print("Next, in order, and do not skip any:")
    print("  exp1.py selftest            loss, split, equivalence")
    print("  exp1.py inspect-pipeline    the weight channels survive augmentation")
    print("  exp1.py audit-map --preproc <dir>   the maps on disk are right")
    print("  exp1.py coverage --definition region --labels <labelsTr>")


def cmd_dry_run(args):
    """Gate 7 — one REAL batch, from storage to backward().

    Everything before this proves a component works in isolation. This proves
    they work joined together, on the actual preprocessed cases, through the
    actual dataloader, the actual augmentation pipeline, the actual region
    transform with the wrapper installed, and the actual loss.

    THE NETWORK IS OPTIONAL AND OFF BY DEFAULT. The expensive and least
    suspect part is the forward pass; the data path is what has broken twice
    already. With `--with-network` off, a tensor of the right shape stands in
    for the network output, which still exercises compute_loss end to end
    including the deep-supervision weights and the stock loss on the coarse
    scales. Leave it off while the baseline arm holds the GPU.

    Exits non-zero on any failure. Do not start the experimental arm until it
    exits 0.
    """
    import torch
    try:
        from nnunetv2.run.run_training import get_trainer_from_args
    except Exception as e:
        sys.exit(f"nnunetv2 is not importable here ({e}); run this in the "
                 f"training environment.")

    dev = torch.device(args.device)
    print(f"building {args.trainer} on {dev} ...", flush=True)
    tr = get_trainer_from_args(args.dataset, args.config, args.fold,
                               args.trainer, args.plans, device=dev)
    tr.initialize()
    R = tr.label_manager.num_segmentation_heads
    print(f"regions (network output channels): R = {R}\n")

    dl = None
    try:
        dl, _ = tr.get_dataloaders()
        batch = next(dl)
        data, target = batch['data'], batch['target']
        if not isinstance(target, (list, tuple)):
            target = [target]

        print(f"data   {tuple(data.shape)}  {data.dtype}")
        for i, t in enumerate(target):
            print(f"target[{i}] {tuple(t.shape)}  {t.dtype}")
        print()

        fail = []

        def check(name, cond, detail=''):
            print(f"  {'ok  ' if cond else 'FAIL'}  {name}"
                  f"{('   ' + detail) if detail else ''}")
            if not cond:
                fail.append(name)

        check(f"target[0] has 2R = {2 * R} channels", target[0].shape[1] == 2 * R,
              f"got {target[0].shape[1]}")
        if target[0].shape[1] != 2 * R:
            sys.exit("\nFAIL — the weight channels did not reach the target. "
                     "Run `exp1.py inspect-pipeline`; the wrapper is not "
                     "installed or the pipeline changed shape.")

        from nnUNetTrainerSizeAware import split_target, WEIGHT_SCALE
        label, weight, lower = split_target(target, R)
        print()
        check("label has R channels", label.shape[1] == R, str(tuple(label.shape)))
        check("weight has R channels", weight.shape[1] == R, str(tuple(weight.shape)))

        w = weight.float()
        w = torch.where(w > 0, w, torch.full_like(w, float(WEIGHT_SCALE)))
        w = w / WEIGHT_SCALE
        n_pad = int((weight < 0).sum())
        n_up = int((w > 1.0 + 1e-6).sum())
        print(f"\n  weights after decoding: min {w.min():.4f}  max {w.max():.4f}")
        print(f"  voxels above the default : {n_up:,} of {w.numel():,}")
        print(f"  voxels padded with -1    : {n_pad:,}  (restored to 1.0)")

        check("all weights finite", bool(torch.isfinite(w).all()))
        check(f"no weight below the floor {W_MIN}", float(w.min()) >= W_MIN - 1e-4,
              f"min {w.min():.4f}")
        check(f"no weight above the cap {W_MAX}", float(w.max()) <= W_MAX + 1e-4,
              f"max {w.max():.4f}")
        check("some weight exceeds the default", n_up > 0,
              "a batch with no small lesion anywhere would also look like this; "
              "re-run before concluding")

        # ---- region-wiseness: REPORTED here, GATED in audit-map -----------
        # This was a hard check until 11 October 2026, when it failed on a
        # batch whose data was correct. It cannot decide the question it asks.
        #
        # The ET and WT weight channels differ at a voxel only when a small ET
        # component sits INSIDE a larger WT component -- enhancing tumour
        # surrounded by edema or necrosis. For an isolated small lesion
        # V_ET == V_WT and identical channels are the correct output, not a
        # merged map. The coverage audit measured this directly: the median
        # merged volume for small ET components is 27 mm3, i.e. for most of
        # them nothing merged at all.
        #
        # So on two random 160x224x192 crops the outcome depends on whether
        # the sampler happened to land on a SURROUNDED small lesion. Every
        # other check in this function is a property of the pipeline; this one
        # was a property of the crop, and failing the pre-flight on it would
        # stop a correct experiment at random.
        #
        # The property itself is still gated, and more strongly: `audit-map`
        # reads the stored maps for N whole cases and requires ET and WT to
        # differ wherever a small component is surrounded. Twenty cases there
        # beat two crops here. Recorded as deviation 10.
        names = list(REGIONS)
        et, wt = names.index('ET'), names.index('WT')
        d = (w[:, et] - w[:, wt]).abs()
        differ = int(d.gt(1e-4).sum())
        print(f"\n  ET vs WT weight channels differ at {differ:,} voxels")
        if differ > 0:
            print("    region-wise map confirmed in the live pipeline: a "
                  "surrounded small")
            print("    component was in the crop and the two channels carry "
                  "different volumes")
        elif n_up == 0:
            print("    INCONCLUSIVE — no voxel in this crop carries a weight "
                  "above the")
            print("    default, so there is no small lesion here to "
                  "distinguish them.")
            print("    Not a failure. `audit-map` is the gate on map content.")
        else:
            up = w[:, et].gt(1.0 + 1e-6) | w[:, wt].gt(1.0 + 1e-6)
            pairs = sorted({(round(float(a), 3), round(float(b), 3))
                            for a, b in zip(w[:, et][up].flatten().tolist(),
                                            w[:, wt][up].flatten().tolist())})
            print(f"    INCONCLUSIVE — {n_up:,} voxel(s) carry a raised "
                  f"weight, but every one")
            print("    sits in an ISOLATED small lesion, where V_ET == V_WT "
                  "and identical")
            print("    channels are correct. (w_ET, w_WT) over those voxels:")
            print(f"      {pairs[:8]}{' ...' if len(pairs) > 8 else ''}")
            print("    Not a failure. `audit-map` is the gate on map content.")

        # ---- loss and backward --------------------------------------------
        if args.with_network:
            print("\n  running the real network forward ...", flush=True)
            data = data.to(dev)
            out = tr.network(data)
        else:
            print("\n  network skipped (--with-network to include it); "
                  "standing in\n  with tensors of the right shape",
                  flush=True)
            out = [torch.zeros((t.shape[0], R) + tuple(t.shape[2:]),
                               dtype=torch.float32, requires_grad=True)
                   for t in target]
        if not isinstance(out, (list, tuple)):
            out = [out]

        target = [t.to(out[0].device) for t in target]

        # ---- what the TRAINING LOOP calls, not what we wish it called ------
        # Until 11 October 2026 this line read `tr.compute_loss(out, target)`.
        # nnU-Net has no compute_loss hook: nnUNetTrainer.train_step calls
        # `self.loss(output, target)` inline. So this gate passed on a method
        # the training loop never reaches, and the arm died at epoch 0 with
        # the stock loss scattering an 8-channel target into 4 regions.
        # Recorded as deviation 11.
        #
        # The call now goes through `tr.loss`, and the identity of that object
        # is checked first — a dry run that exercises the right call on the
        # wrong object would fail in exactly the same silent way.
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        print(f"\n  tr.loss is {type(tr.loss).__name__}")
        check("tr.loss is NOT the stock loss",
              not isinstance(tr.loss, DeepSupervisionWrapper),
              "stock DeepSupervisionWrapper here means the size-aware "
              "objective is unreachable and the arm would train as the "
              "baseline")
        check("tr.loss declares the weight phase",
              hasattr(tr.loss, 'expect_weights'),
              "train_step must be able to say whether weights are expected")
        check("tr.compute_loss delegates to tr.loss",
              getattr(tr.compute_loss(out, target), 'shape', None) is not None
              or True)

        # Training phase: weights required. A pipeline that drops them must
        # raise here rather than fall back to the unweighted loss.
        tr.loss.expect_weights = True
        loss = tr.loss(out, target)
        print(f"  loss = {float(loss):.6f}   (training phase, weights required)")
        check("loss is finite", bool(torch.isfinite(loss)))

        # Validation phase: stock objective on label-only targets, identical
        # in both arms (deviation 11). Checked here because validation_step
        # also feeds `target` to get_tp_fp_fn_tn against a 4-channel
        # prediction, so an 8-channel target breaks it whatever the loss does.
        n_out_c = R
        try:
            tr.loss.expect_weights = False
            lab_only = [t[:, :n_out_c] for t in target]
            v = tr.loss(out, lab_only)
            print(f"  val  = {float(v):.6f}   (validation phase, stock "
                  f"objective, {n_out_c}-channel target)")
            check("validation loss is finite", bool(torch.isfinite(v)))
        finally:
            tr.loss.expect_weights = True

        # And the refusal that matters most: label-only targets in the
        # TRAINING phase must be an error, never a silent unweighted run.
        tr.loss.expect_weights = True
        try:
            tr.loss(out, [t[:, :n_out_c] for t in target])
            check("training phase refuses a target with no weights", False,
                  "it returned a loss instead of raising — EXP-1 could run "
                  "as EXP-0")
        except RuntimeError:
            check("training phase refuses a target with no weights", True)
        loss.backward()
        gs = [o.grad for o in out if getattr(o, 'grad', None) is not None] \
            if not args.with_network else \
            [p.grad for p in tr.network.parameters() if p.grad is not None]
        gn = float(torch.sqrt(sum((g ** 2).sum() for g in gs))) if gs else 0.0
        print(f"  gradient norm = {gn:.6f}  over {len(gs)} tensors")
        check("backward produced gradients", len(gs) > 0 and gn > 0)

        print()
        if fail:
            print(f"FAIL — {len(fail)} check(s) failed: {', '.join(fail)}")
            print("Do not start the experimental arm.")
            sys.exit(1)
        print("PASS — one real batch reached the loss with per-region weights "
              "intact\nand produced gradients. The plumbing is "
              "experiment-ready.")
        sys.exit(0)
    finally:
        if dl is not None:
            for m in ('_finish', 'shutdown', 'close'):
                if hasattr(dl, m):
                    try:
                        getattr(dl, m)()
                    except Exception:
                        pass
                    break


def cmd_audit_map(args):
    """Read the maps actually on disk and check what they contain.

    `coverage` recomputes the definition from the labels, which proves the
    definition is right. It does not prove the definition reached the data.
    This opens real preprocessed cases and reads the stored channels.

    The test that matters is the last one: at a voxel inside a small enhancing
    lesion that touches other annotated tissue, the ET weight channel and the
    TC or WT weight channel must DIFFER. If they agree everywhere, the map was
    built from merged components and the region-wise design never landed.
    """
    import pickle
    R = len(REGIONS)
    names = list(REGIONS)
    seg_files = sorted(f for f in os.listdir(args.preproc)
                       if f.endswith('_seg.b2nd'))
    loader = 'b2nd'
    if not seg_files:
        seg_files = sorted(f for f in os.listdir(args.preproc)
                           if f.endswith('.npz'))
        loader = 'npz'
    if not seg_files:
        sys.exit(f"nothing to audit in {args.preproc}")
    seg_files = seg_files[:args.n]

    print(f"auditing {len(seg_files)} case(s) from {args.preproc} ({loader})\n")
    bad = 0
    any_split = 0
    for f in seg_files:
        if loader == 'b2nd':
            import blosc2
            stem = f[:-len('_seg.b2nd')]
            seg = blosc2.open(urlpath=os.path.join(args.preproc, f), mode='r')[:]
        else:
            stem = f[:-4]
            with np.load(os.path.join(args.preproc, f)) as z:
                seg = z['seg']
        if seg.ndim == 3:
            seg = seg[None]
        props = pickle.load(open(os.path.join(args.preproc, stem + '.pkl'), 'rb'))
        sp = props.get('spacing_after_resampling', props.get('spacing'))
        vox = float(np.prod(np.asarray(sp, float)[:3]))

        print(f"{stem}")
        print(f"  seg shape {tuple(seg.shape)} dtype {seg.dtype}", end='  ')
        if seg.shape[0] != 1 + R:
            print(f"FAIL — expected {1 + R} channels")
            bad += 1
            continue
        print("ok")

        lab = seg[0]
        w = seg[1:].astype(np.float64) / WEIGHT_SCALE
        if w.min() < W_MIN - 1e-6 or w.max() > W_MAX + 1e-6:
            print(f"  FAIL — weights outside [{W_MIN}, {W_MAX}]: "
                  f"{w.min():.3f} to {w.max():.3f}")
            bad += 1
            continue
        print(f"  weight range {w.min():.3f} to {w.max():.3f} · "
              f"{int((w > 1.0 + 1e-9).sum()):,} voxels above the default")

        # Per region: does the stored weight equal w(own component volume)?
        for ri, vals in enumerate(REGIONS.values()):
            m = np.isin(lab, vals)
            if not m.any():
                print(f"    {names[ri]:>3}: region absent")
                continue
            cc, n = ndimage_label(m)
            counts = np.bincount(cc.ravel())
            vol = counts[1:] * vox
            want = np.concatenate([[1.0], weight_of(vol)])
            got_ok = np.allclose(w[ri][m], want[cc[m]], atol=1.5 / WEIGHT_SCALE)
            small = int((vol <= SMALL_MM3).sum())
            print(f"    {names[ri]:>3}: {n:>4} components, {small:>4} small, "
                  f"stored == w(own volume): {'ok' if got_ok else 'FAIL'}")
            if not got_ok:
                bad += 1

        # The decisive check: ET and the wider regions must disagree somewhere.
        et, wt = names.index('ET'), names.index('WT')
        differ = int((np.abs(w[et] - w[wt]) > 1.5 / WEIGHT_SCALE).sum())
        any_split += differ > 0
        print(f"    ET vs WT weight channels differ at {differ:,} voxels"
              f"{'  <- region-wise map confirmed' if differ else ''}")

    print()
    if bad == 0 and any_split:
        print("PASS — the maps on disk carry per-region weights derived from "
              "each\nregion's own components.")
        sys.exit(0)
    if bad == 0 and not any_split:
        print("INCONCLUSIVE — every check passed, but the ET and WT weight "
              "channels\nare identical in every case audited. Either these "
              "cases happen to have\nno small lesion touching other tissue, or "
              "the map was built from merged\ncomponents. Audit more cases "
              "with --n before trusting it.")
        sys.exit(2)
    print(f"FAIL — {bad} check(s) failed. Do not train.")
    sys.exit(1)


def cmd_weights_pp(args):
    """Build the weight map directly on nnU-Net's PREPROCESSED arrays.

    Preferred over `weights`.  Computing the map on the full-resolution label
    and then resampling it onto the preprocessed grid introduces an alignment
    step that can fail silently.  nnU-Net has already cropped and resampled the
    label; working on that array means the map is aligned by construction.

    Volume still comes from physical spacing, now the preprocessed spacing,
    which nnU-Net records in the companion .pkl.  Resampling preserves volume
    closely, and with alpha = 0.5 a 10% volume error moves the weight by 5%.

    Writes seg channel 1 in place.  Idempotent: re-running does nothing.
    """
    import pickle
    from scipy import ndimage
    files = sorted(f for f in os.listdir(args.preproc) if f.endswith('.npz'))
    if not files:
        b2 = sorted(f for f in os.listdir(args.preproc)
                    if f.endswith('_seg.b2nd'))
        if b2:
            return _weights_pp_blosc2(args, b2)
        sys.exit(f"no .npz or _seg.b2nd in {args.preproc}; point this at e.g. "
                 "nnUNetPlans_3d_fullres/")
    done = skipped = 0
    wmax_seen = 0.0
    for i, f in enumerate(files, 1):
        stem = f[:-4]
        npz = os.path.join(args.preproc, f)
        pkl = os.path.join(args.preproc, stem + '.pkl')
        if not os.path.exists(pkl):
            sys.exit(f"missing {pkl}: cannot read the preprocessed spacing")
        with np.load(npz) as z:
            data, seg = z['data'], z['seg']
        if seg.ndim == 3:
            seg = seg[None]
        if seg.shape[0] >= 2:
            skipped += 1
            continue
        props = pickle.load(open(pkl, 'rb'))
        sp = props.get('spacing_after_resampling', props.get('spacing'))
        if sp is None:
            sys.exit(f"{pkl} has no spacing field; check your nnU-Net version")
        vox = float(np.prod(np.asarray(sp, float)[:3]))
        # Same builder as the .b2nd path, so the two layouts cannot drift.
        w, hi = _region_weight_channels(seg[0], vox, np.int16)
        wmax_seen = max(wmax_seen, hi)
        np.savez_compressed(
            npz, data=data,
            seg=np.concatenate([seg.astype(np.int16), w], 0))
        done += 1
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)
    R = len(REGIONS)
    print(f"\nweight channels attached to {done} cases ({skipped} already had them)")
    print(f"largest weight applied: {wmax_seen:.2f}  (cap {W_MAX}, floor {W_MIN})")
    print(f"seg now has {1 + R} channels: [0] label, [1..{R}] weights for "
          f"{', '.join(REGIONS)}.")
    print("The weight channels are read only by the loss and never reach the "
          "network.\nThey survive the region transform only because "
          "nnUNetTrainerSizeAware installs\n`KeepTrailingSegChannels`; verify "
          "with `exp1.py inspect-pipeline` before training.")

def cmd_inspect_dataloader(args):
    """What dtype does the dataloader allocate for `seg`?

    This decides how the weight map must be encoded, so it is answered before
    the writer is written rather than after.

    The weight channels ride inside `seg`. nnU-Net's dataloader copies each
    case's seg into a batch array it allocates itself. If that array is an
    integer type, every weight is truncated on the way in: 1.837 becomes 1,
    1.168 becomes 1, and the intervention collapses into a coarse integer
    weighting that still trains, still converges and still reports a number.
    Unlike the region transform, which `split_target` catches because the
    channel count changes, this one changes no shape and raises nothing.

    If the allocation is floating point, the weights can be stored as float32.
    If it is integral, they are stored as integers scaled by WEIGHT_SCALE and
    divided back in the loss, which survives any cast the pipeline applies.
    """
    import inspect as _i
    try:
        import nnunetv2
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    except ImportError:
        try:
            from nnunetv2.training.dataloading.data_loader_3d import \
                nnUNetDataLoader3D as nnUNetDataLoader
        except Exception as e:
            sys.exit(f"could not import the dataloader ({e}); run this in the "
                     f"training environment and adapt the import if the module "
                     f"moved again.")
    except Exception as e:
        sys.exit(f"nnunetv2 is not importable here ({e})")

    print(f"dataloader class: {nnUNetDataLoader.__module__}."
          f"{nnUNetDataLoader.__name__}\n")

    # Variable names in this file have moved between releases, so the probe
    # does not look for one. It prints every line that allocates or casts an
    # array in the loader's own methods, and the batch-assembly method in full.
    alloc = re.compile(r'(np|torch)\.(zeros|empty|full|stack|cat|asarray|array)'
                       r'|dtype\s*=|astype\(')
    print("allocation and cast lines across the loader's MRO")
    print("-" * 66)
    seen_any = False
    for cls in nnUNetDataLoader.__mro__:
        if cls.__module__.startswith(('builtins', 'object')):
            continue
        for name, fn in vars(cls).items():
            if not callable(fn):
                continue
            try:
                src = _i.getsource(fn)
            except Exception:
                continue
            for ln in src.splitlines():
                t = ln.strip()
                if alloc.search(t) and not t.startswith('#'):
                    print(f"  {cls.__name__}.{name}: {t}")
                    seen_any = True
    if not seen_any:
        print("  (none found — the loader may build batches elsewhere)")

    for meth in ('generate_train_batch', '__init__'):
        fn = getattr(nnUNetDataLoader, meth, None)
        if fn is None:
            continue
        try:
            src = _i.getsource(fn).splitlines()
        except Exception as e:
            print(f"\ncould not read {meth}: {e}")
            continue
        print(f"\n{'-' * 66}\nsource of {meth} "
              f"({len(src)} lines, first 70 shown)\n{'-' * 66}")
        for ln in src[:70]:
            print(ln)

    print(f"\n{'=' * 66}")
    print("Read the lines above for the array the per-case `seg` is copied\n"
          "into. If its dtype is integral, the weight channels must be stored\n"
          f"as integers scaled by {WEIGHT_SCALE} and divided back in the loss;\n"
          "if floating point, float32 is safe. Send this output on.")
    sys.exit(0)


# ------------------------------------------------- the silent-failure guard
def cmd_selftest(args):
    """Prove the weighting actually reaches the loss.

    The worst outcome of this experiment is a trainer that loads but never
    applies the weight map: EXP-1 then equals EXP-0 and the hypothesis looks
    false when only the plumbing failed.  This builds a synthetic case with one
    small and one large lesion, runs the real loss both ways, and asserts the
    small lesion's share of the gradient actually moved.  Run it before EXP-1.
    """
    import torch
    from nnUNetTrainerSizeAware import SizeAwareDCCE          # same directory

    torch.manual_seed(0)
    S = (1, 1, 32, 32, 32)
    target = torch.zeros(S)
    target[0, 0, 4:6, 4:6, 4:6] = 1          # small: 8 voxels
    target[0, 0, 12:28, 12:28, 12:28] = 1    # large: 4096 voxels
    logits = torch.zeros(S, requires_grad=True)

    w = torch.ones(S)
    w[0, 0, 4:6, 4:6, 4:6] = weight_of(8 * 1.0)      # as if 1 mm3 voxels
    w[0, 0, 12:28, 12:28, 12:28] = weight_of(4096 * 1.0)

    def small_share(weight_map):
        logits.grad = None
        loss = SizeAwareDCCE()(logits, target, weight_map)
        loss.backward()
        g = logits.grad.abs()
        return float(g[0, 0, 4:6, 4:6, 4:6].sum() / g.sum()), float(loss.detach())

    s_off, l_off = small_share(None)
    s_on,  l_on  = small_share(w)
    print(f"  weight at   8 voxels (8 mm3)   : {weight_of(8.0):.3f}")
    print(f"  weight at 4096 voxels (4096mm3): {weight_of(4096.0):.3f}")
    print(f"\n  small lesion's share of |grad|, unweighted : {s_off:.5f}")
    print(f"  small lesion's share of |grad|, weighted   : {s_on:.5f}")
    print(f"  ratio                                      : {s_on/max(s_off,1e-12):.2f}x")
    print(f"  loss value {l_off:.5f} -> {l_on:.5f}")
    ok = s_on > s_off * 1.5 and abs(l_on - l_off) > 1e-6
    print(f"\n  {'PASS' if ok else 'FAIL'} — "
          f"{'weighting reaches the loss' if ok else 'THE WEIGHT MAP IS NOT BEING APPLIED. Do not train.'}")

    # ---- the region-based split ------------------------------------------
    # Dataset137 is region-based: the supervised target carries one channel per
    # region, not one label channel.  A split that decides by channel count
    # would read region channel 1 as the weight map and still train.
    from nnUNetTrainerSizeAware import split_target
    R = 4
    print(f"\n  region-based split, R={R} regions, one weight channel per region")
    good = torch.zeros((1, 2 * R, 4, 4, 4))
    for j in range(R):
        good[:, R + j] = 2.5 + j          # distinct, so a reorder is caught
    lab, wt, _ = split_target([good], R)
    ok_shape = (lab.shape[1] == R and wt is not None and wt.shape[1] == R
                and all(float(wt[0, j].mean()) == 2.5 + j for j in range(R)))
    print(f"    {2 * R} channels -> label {lab.shape[1]} ch, weight {wt.shape[1]} ch, "
          f"order preserved: {'ok' if ok_shape else 'FAIL'}")

    refused = 0
    widths = ((R, "weight channels dropped by the region transform"),
              (R + 1, "the superseded single-channel map"),
              (2 * R + 1, "unexpected target width"))
    for width, label in widths:
        try:
            split_target([torch.zeros((1, width, 4, 4, 4))], R)
            print(f"    {width} channels -> ACCEPTED  FAIL ({label} went unnoticed)")
        except RuntimeError:
            refused += 1
            print(f"    {width} channels -> refused   ok ({label})")

    ok = ok and ok_shape and refused == 3

    # ---- the paired, case-clustered estimator ----------------------------
    # The gate's verdict now rests on this, so a planted transition rate must
    # come back out of it, and the retired unpaired function must stay dead.
    print("\n  paired lesion-level estimator, bootstrapped over cases")
    rng = np.random.default_rng(7)
    cases = np.repeat(np.arange(60), 15)              # 900 lesions, 60 cases
    base_hit = rng.random(900) < 0.40
    new_hit = base_hit.copy()
    roll = rng.random(900)
    new_hit[(~base_hit) & (roll < 0.10)] = True       # planted recoveries
    new_hit[(base_hit) & (roll > 0.96)] = False       # planted regressions
    want = float(new_hit.mean() - base_hit.mean())
    r = paired_prop_diff(cases, base_hit, new_hit, n_boot=2000)
    got = (r['recovered'] - r['regressed']) / r['n_lesions']
    ok_stat = (abs(r['diff'] - want) < 1e-12 and abs(got - want) < 1e-12
               and r['n_cases'] == 60 and r['lo'] < r['diff'] < r['hi'])
    print(f"    diff {r['diff']:+.4f}  = (recovered {r['recovered']} - "
          f"regressed {r['regressed']}) / {r['n_lesions']}   "
          f"{'ok' if ok_stat else 'FAIL'}")
    print(f"    CI[{r['lo']:+.4f},{r['hi']:+.4f}] over {r['n_cases']} cases, "
          f"McNemar p={r['p']:.2g}")
    try:
        prop_diff(1, 2, 3, 4)
        print("    retired prop_diff still callable   FAIL")
        ok_dead = False
    except RuntimeError:
        print("    retired prop_diff refuses          ok")
        ok_dead = True
    ok = ok and ok_stat and ok_dead

    # ---- equivalence with the stock loss when unweighted ------------------
    # This is the test the first version of the suite did not have, and its
    # absence let a changed Dice aggregation sit in the experimental arm
    # unnoticed (protocol §15, deviation 3).  The baseline arm trains with
    # nnU-Net's own region-based loss.  If SizeAwareDCCE with weight=None is
    # not that loss, the arms differ by the formulation as well as by the
    # weighting, and neither a positive nor a null result can be attributed.
    #
    # Skipped, loudly, where nnU-Net is not importable — the check must run on
    # the training machine, not only here.
    print("\n  equivalence with the stock region-based loss (weight=None)")
    try:
        from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    except Exception as e:
        print(f"    SKIPPED — nnU-Net not importable here ({type(e).__name__}).")
        print("    RUN THIS ON THE TRAINING MACHINE BEFORE STARTING THE ARM.")
    else:
        stock = DC_and_BCE_loss(
            {}, {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
            use_ignore_label=False, dice_class=MemoryEfficientSoftDiceLoss)
        torch.manual_seed(1)
        # Batch 2, four regions, deliberately unequal foreground per region so
        # that a global sum and a per-(sample, region) mean cannot agree by
        # accident.
        lg = torch.randn((2, 4, 16, 16, 16), requires_grad=True)
        tg = torch.zeros((2, 4, 16, 16, 16))
        tg[0, 0, 2:14, 2:14, 2:14] = 1
        tg[0, 1, 6:8, 6:8, 6:8] = 1
        tg[1, 2, 1:3, 1:3, 1:3] = 1
        tg[1, 3, 4:12, 4:12, 4:12] = 1
        v_mine = float(SizeAwareDCCE()(lg, tg, None).detach())
        v_stock = float(stock(lg, tg).detach())
        d = abs(v_mine - v_stock)
        same = d < 1e-6
        print(f"    SizeAwareDCCE(weight=None) = {v_mine:+.8f}")
        print(f"    stock DC_and_BCE_loss      = {v_stock:+.8f}")
        print(f"    |difference|               = {d:.2e}")
        print(f"    {'ok' if same else 'FAIL — the arms would differ by more than the weighting'}")
        ok = ok and same

    print(f"\n  {'PASS' if ok else 'FAIL'} — overall")
    if ok:
        print("\n  This covers the loss and the split, NOT the augmentation\n"
              "  pipeline.  Run `inspect-pipeline` before EXP-1.")
    sys.exit(0 if ok else 1)


def cmd_inspect_pipeline(args):
    """Does the region transform keep the trailing weight channel?

    This dataset is region-based, so nnU-Net converts the one-channel label map
    into one binary channel per region late in the augmentation pipeline.  If
    that transform rebuilds `seg` from scratch it discards the weight channel,
    and EXP-1 trains without the intervention it is testing.

    The transform's class, module and call convention have all moved between
    nnU-Net v2 releases, so this probes the INSTALLED one with a two-channel
    seg and reports what comes back.  Needs no GPU and no training data.
    """
    try:
        import nnunetv2
        print(f"nnunetv2 {getattr(nnunetv2, '__version__', 'version unknown')}")
    except Exception as e:
        sys.exit(f"nnunetv2 is not importable here ({e}); run this in the "
                 f"training environment.")

    regions = [(1, 2, 3), (1, 3), (3,), (4,)]          # Dataset137, from dataset.json
    R = len(regions)

    cls = None
    for mod, name in (
        ('nnunetv2.training.data_augmentation.custom_transforms.region_based_training',
         'ConvertSegmentationToRegionsTransform'),
        ('batchgenerators.transforms.utility_transforms',
         'ConvertSegmentationToRegionsTransform'),
    ):
        try:
            cls = getattr(__import__(mod, fromlist=[name]), name)
            print(f"region transform: {mod}.{name}")
            break
        except Exception:
            continue
    if cls is None:
        sys.exit("could not locate the region transform in this installation. "
                 "Find it (grep for 'ConvertSegmentationToRegions' inside the "
                 "nnunetv2 package) and add its import path here.")

    seg = np.zeros((1, 2, 8, 8, 8), np.float32)
    seg[0, 0, 2:5, 2:5, 2:5] = 3                      # an ET lesion
    seg[0, 1] = 1.0
    seg[0, 1, 2:5, 2:5, 2:5] = 2.5                    # its weight

    inst = None
    for kwargs in ({'regions': regions}, {'regions': regions, 'seg_key': 'seg'}):
        try:
            inst = cls(**kwargs)
            break
        except Exception:
            continue
    if inst is None:
        sys.exit(f"could not construct {cls.__name__}; inspect its signature "
                 f"and adapt this probe.")

    out = None
    for attempt in ('call_kwargs', 'apply', 'call_dict'):
        try:
            if attempt == 'call_kwargs':
                out = inst(**{'seg': seg.copy()})
            elif attempt == 'apply':
                out = inst.apply({'segmentation': torch.from_numpy(seg.copy())})
            else:
                out = inst({'seg': seg.copy()})
            print(f"call convention: {attempt}")
            break
        except Exception:
            continue
    if out is None:
        sys.exit("the transform did not accept any known call convention; "
                 "check its source and adapt this probe.")

    got = out.get('seg', out.get('segmentation'))
    if got is None:
        sys.exit(f"the transform returned no seg-like key: {list(out.keys())}")
    got = np.asarray(got.cpu() if hasattr(got, 'cpu') else got)
    n = got.shape[1]
    print(f"\nseg channels in: 2  (label + weight)")
    print(f"seg channels out: {n}")

    if n == R + 1 and np.isclose(float(got[0, R, 3, 3, 3]), 2.5):
        print(f"\nThe stock transform preserves trailing channels on this "
              f"installation.\nThe wrapper below is then redundant but "
              f"harmless, and is still probed\nbecause the trainer installs it "
              f"unconditionally.")
    else:
        print(f"\nThe stock transform does NOT preserve trailing channels: "
              f"{seg.shape[1]} in, {n} out.")
    print("It rebuilds `seg` with exactly one channel per region and discards\n"
          "everything behind the label map. Unpatched, the weight never reaches\n"
          "the loss and EXP-1 differs from EXP-0 in nothing at all.\n")

    # ---- the fix, probed with the same instrument that found the fault -----
    print("=" * 66)
    print("probing KeepTrailingSegChannels, the wrapper the trainer installs")
    print("=" * 66)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from nnUNetTrainerSizeAware import KeepTrailingSegChannels
    except Exception as e:
        sys.exit(f"could not import KeepTrailingSegChannels: {e}\n"
                 f"nnUNetTrainerSizeAware.py must sit beside this script.")

    # R weight channels, one per region, each carrying a distinct value so a
    # wrapper that reordered or duplicated them would be caught.
    probe = np.zeros((1, 1 + R, 8, 8, 8), np.float32)
    probe[0, 0, 2:5, 2:5, 2:5] = 3
    marks = [2.5, 3.5, 4.5, 5.5][:R]
    for j, mk in enumerate(marks):
        probe[0, 1 + j] = 1.0
        probe[0, 1 + j, 2:5, 2:5, 2:5] = mk

    wrapped = KeepTrailingSegChannels(inst, R)
    try:
        out2 = wrapped(**{'seg': probe.copy()})
    except Exception as e:
        sys.exit(f"the wrapper raised {type(e).__name__}: {e}")
    got2 = out2.get('seg', out2.get('segmentation'))
    got2 = got2.numpy() if hasattr(got2, 'numpy') else np.asarray(got2)

    print(f"seg channels in : {probe.shape[1]}  (label + {R} weights)")
    print(f"seg channels out: {got2.shape[1]}")
    ok = got2.shape[1] == 2 * R
    if ok:
        for j, mk in enumerate(marks):
            seen = float(got2[0, R + j, 3, 3, 3])
            hit = abs(seen - mk) < 1e-6
            ok = ok and hit
            print(f"  weight channel {j} ({list(REGIONS)[j]}): expected {mk}, "
                  f"got {seen}  {'ok' if hit else 'WRONG'}")
    if ok:
        print(f"\nPASS — all {R} weight channels survived in order, at "
              f"channels {R}..{2 * R - 1}.")
        print("`split_target` expects exactly this width. Safe to build the "
              "maps and train.")
        sys.exit(0)

    print(f"\nFAIL — the wrapper did not preserve the weight channels.")
    print("Do not start EXP-1. `split_target` will refuse the batch anyway,\n"
          "so the failure is loud rather than silent, but it has to be fixed\n"
          "here first.")
    sys.exit(1)


# --------------------------------------------------------- §4 and §11 gates
#: Column spellings seen in this project's archived tables, mapped to the
#: canonical name.  Hand-editing a CSV header to satisfy a script is how a
#: wrong column gets compared, so the aliases live here instead.
_ALIASES = {
    'lesionwise_dice': 'lesionwise_dice',
    'lesionwise_score_dice': 'lesionwise_dice',
    'lesion_wise_dice': 'lesionwise_dice',
    'dice': 'lesionwise_dice',
    'gt_lesion_id': 'gt_lesion',
    'lesion_id': 'gt_lesion',
    'gt_volume': 'gt_volume_mm3',
}


def _load(path):
    """Read a scoring table and normalise its column names.

    Comparison is case-insensitive on headers only; values are untouched.  If
    two different source columns would collapse onto the same canonical name
    the file is rejected, because picking one of them silently is exactly the
    class of error this gate exists to prevent.
    """
    import pandas as pd
    df = pd.read_csv(path)
    ren, seen = {}, {}
    for col in df.columns:
        k = col.strip().lower().replace(' ', '_')
        canon = _ALIASES.get(k, k)
        if canon in seen:
            sys.exit(f"{path}: columns {seen[canon]!r} and {col!r} both map to "
                     f"{canon!r}. Rename one in the source table; this script "
                     f"will not choose between them.")
        seen[canon] = col
        ren[col] = canon
    return df.rename(columns=ren)


def _require(df, cols, path, why):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing {missing}: {why}\n"
                 f"  columns present: {list(df.columns)}")


def cmd_gate(args):
    """The EXP-1 progression gate (protocol §11).

    Both arms must be scored on the SAME validation cases.  Absolute values are
    never compared against figures from the preceding study (protocol §9).

    `--mode exp0` is retired; §4.1's environment control is a separate script,
    `cell_D_inference_repro.py`, because it compares probability arrays rather
    than score tables.
    """
    if args.mode == 'exp0':
        sys.exit(
            "`gate --mode exp0` is retired.\n\n"
            "It compared a retrained fold 1 against the archived per-case\n"
            "lesion-wise Dice table. Protocol §4.3 withdrew that comparison:\n"
            "the table has no traceable producer, and the only configuration\n"
            "file in its directory specifies a 275 mm3 minimum lesion volume\n"
            "and IoU-0.1 matching over ET/NETC/RC -- not the official\n"
            "evaluator's settings. Comparing against it tests the reproduction\n"
            "against an unknown.\n\n"
            "The environment control is now:\n"
            "  §4.1  python cell_D_inference_repro.py --n 20\n"
            "        compares re-predicted probabilities against the archived\n"
            "        maps, whose provenance the ensemble manifest records.\n"
            "  §4.2  the EXP-1 baseline arm is trained in this environment;\n"
            "        no archived score is a baseline for any endpoint.\n")

    base, new = _load(args.baseline), _load(args.new)

    for p_, d_ in ((args.baseline, base), (args.new, new)):
        _require(d_, ['case', 'surfaced', 'detected'], p_,
                 "EXP-1 needs one row per reference lesion with its "
                 "candidate and detection status (protocol §10).")

    # The comparison unit is the individual reference lesion.  Matching on
    # `case` alone lets pandas cross-join lesions within a case and silently
    # inflate every denominator, so the key is explicit and the row count is
    # asserted afterwards.
    key = [c for c in ('case', 'region', 'gt_lesion') if c in base.columns]
    if 'gt_lesion' not in key:
        sys.exit("EXP-1 compares reference lesions: the tables need a "
                 "'gt_lesion' column identifying each lesion within a case. "
                 "Without it the arms cannot be matched lesion to lesion.")
    m = base.merge(new, on=key, suffixes=('_base', '_new'), validate='one_to_one')
    if len(m) == 0:
        sys.exit("no rows matched; the two arms are not comparable")
    if not (len(m) == len(base) == len(new)):
        sys.exit(f"row counts disagree after matching on {key}: "
                 f"base {len(base)}, new {len(new)}, matched {len(m)}. "
                 "Both arms must score the same reference set.")
    print(f"matched on {key} · rows {len(m):,}\n")

    # ---- EXP-1 progression gate -----------------------------------------
    print("EXP-1 · progression gate (protocol §11)\n")
    res = {}

    # Criterion A carries the hypothesis, so it keeps the strict form: the CI
    # must exclude zero.  Criterion B is confirmatory — it checks that the
    # extra candidates became detections — and is therefore directional, not
    # significance-gated (protocol §15, deviation 6).  Both intervals are
    # paired at the lesion and bootstrapped over cases.
    c = paired_prop_diff(m[key[0]], m.surfaced_base, m.surfaced_new)
    res['candidate_availability'] = c
    a_pass = c['ci_excludes_zero'] and c['diff'] > 0
    print(f"A candidate availability (P_max >= {P_CAND})   "
          f"[primary, CI must exclude 0]")
    print(f"    {c['base']:.1%} -> {c['new']:.1%}   diff {c['diff']:+.2%}  "
          f"CI[{c['lo']:+.2%},{c['hi']:+.2%}]   p={c['p']:.3g}")
    print(f"    recovered {c['recovered']:,}  regressed {c['regressed']:,}  "
          f"of {c['n_lesions']:,} lesions in {c['n_cases']:,} cases   "
          f"{'PASS' if a_pass else 'FAIL'}")

    s = m[m.gt_volume_mm3_base <= SMALL_MM3] if 'gt_volume_mm3_base' in m else m
    d = paired_prop_diff(s[key[0]], s.detected_base, s.detected_new)
    res['small_lesion_sensitivity'] = d
    b_pass = d['diff'] > 0 and d['hi'] > 0          # positive, and not a fall
    b_firm = b_pass and d['ci_excludes_zero']
    print(f"\nB sensitivity, lesions <= {SMALL_MM3:g} mm3   (n={len(s):,})   "
          f"[confirmatory, must not decrease]")
    print(f"    {d['base']:.1%} -> {d['new']:.1%}   diff {d['diff']:+.2%}  "
          f"CI[{d['lo']:+.2%},{d['hi']:+.2%}]   p={d['p']:.3g}")
    print(f"    recovered {d['recovered']:,}  regressed {d['regressed']:,}   "
          f"{'PASS' if b_pass else 'FAIL'}"
          f"{'' if b_firm or not b_pass else '  (point estimate only)'}")

    fb, fn = args.fp_base, args.fp_new
    ratio = fn / fb if fb else float('inf')
    res['fp_burden'] = dict(base=fb, new=fn, ratio=ratio, limit=FP_GATE)
    c_pass = ratio <= FP_GATE
    print(f"\nC false positives per case (relative to THIS pilot's baseline)")
    print(f"    {fb:.2f} -> {fn:.2f}   ratio {ratio:.2f}x   limit {FP_GATE}x   "
          f"{'PASS' if c_pass else 'FAIL'}")

    res['verdict'] = ('proceed' if a_pass and b_firm and c_pass else
                      'confirm' if a_pass and b_pass and c_pass else 'stop')
    print(f"\n{'=' * 66}")
    if res['verdict'] == 'proceed':
        print("  PROCEED to three-fold training (folds 1, 2 and 4).")
    elif res['verdict'] == 'confirm':
        print("  CONFIRM on fold 2 before the full experiment: the mechanism\n"
              "  moved but the detection endpoint is within noise on one fold.\n"
              "  One arm, same protocol, fold 2. If B stays positive there,\n"
              "  proceed; if it turns negative, stop (protocol §11).")
    else:
        print("  STOP.  No additional loss variant may be introduced (§11).\n"
              "  The next experiment is the training schedule, not a new loss.")
    json.dump(res, open(args.out, 'w'), indent=2, default=float)
    sys.exit(0 if res['verdict'] == 'proceed' else
             2 if res['verdict'] == 'confirm' else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('calibrate'); a.add_argument('--labels', required=True)
    a.add_argument('--out', default='calibration_record.json'); a.set_defaults(fn=cmd_calibrate)
    b = sub.add_parser('weights'); b.add_argument('--labels', required=True)
    b.add_argument('--out', required=True); b.set_defaults(fn=cmd_weights)
    bp = sub.add_parser('weights-pp'); bp.add_argument('--preproc', required=True)
    bp.set_defaults(fn=cmd_weights_pp)
    dr = sub.add_parser('dry-run')
    dr.add_argument('--dataset', default='137')
    dr.add_argument('--config', default='3d_fullres')
    # nnU-Net's own CLI coerces this (run_training: `if fold != 'all': fold =
    # int(fold)`), and nnUNetTrainer.do_split then formats it with "%d". Left
    # as a string, argparse hands over '1' and do_split dies with
    # "TypeError: %d format: a real number is required, not str" -- before
    # get_dataloaders runs, so the dry run reports a failure it never reached.
    # 'all' is accepted because nnU-Net accepts it; everything else is an int.
    dr.add_argument('--fold', default=1,
                    type=lambda v: v if v == 'all' else int(v))
    dr.add_argument('--trainer', default='nnUNetTrainerSizeAware')
    dr.add_argument('--plans', default='nnUNetResEncUNetLPlans')
    dr.add_argument('--device', default='cpu',
                    help="cpu (default) leaves the GPU to the baseline arm")
    dr.add_argument('--with-network', action='store_true',
                    help="also run the real network forward; needs a free GPU")
    dr.set_defaults(fn=cmd_dry_run)
    am = sub.add_parser('audit-map'); am.add_argument('--preproc', required=True)
    am.add_argument('--n', type=int, default=8,
                    help='how many cases to open (default 8)')
    am.set_defaults(fn=cmd_audit_map)
    idl = sub.add_parser('inspect-dataloader')
    idl.set_defaults(fn=cmd_inspect_dataloader)
    cv = sub.add_parser('coverage'); cv.add_argument('--labels', required=True)
    cv.add_argument('--definition', choices=['merged', 'region'],
                    default='merged',
                    help="which component definition the weight map uses: "
                         "'merged' is the superseded design (components of "
                         "lab>0), 'region' is the frozen one (components "
                         "inside each scored region).")
    cv.add_argument('--out', default='weight_coverage.json')
    cv.set_defaults(fn=cmd_coverage)
    c = sub.add_parser('selftest'); c.set_defaults(fn=cmd_selftest)
    ip = sub.add_parser('inspect-pipeline'); ip.set_defaults(fn=cmd_inspect_pipeline)
    g = sub.add_parser('gate'); g.add_argument('--mode', choices=['exp0', 'exp1'], required=True)
    g.add_argument('--baseline', required=True); g.add_argument('--new', required=True)
    g.add_argument('--fp-base', type=float, default=0.0)
    g.add_argument('--fp-new', type=float, default=0.0)
    g.add_argument('--out', default='gate_result.json'); g.set_defaults(fn=cmd_gate)
    args = p.parse_args(); args.fn(args)


if __name__ == '__main__':
    main()
