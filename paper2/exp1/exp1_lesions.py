# -*- coding: utf-8 -*-
"""
===========================================================================
 EXP-1 · lesion table builder  —  one row per reference lesion, per arm
===========================================================================
WHY THIS FILE EXISTS

Protocol section 10 used to say the primary endpoint is "read from the
`cand@0.02` column of the c0_lesions.csv table that c0_fn_recoverability.py
produces". That is wrong for EXP-1 and would have produced a Phase B number
under an EXP-1 label. c0_fn_recoverability.py:

  * calls ns.load_mean_prob  -> the MEAN of folds 1, 2 and 4;
    EXP-1 is a single fold.
  * calls ns.build_nested_label with the Phase B tuned thresholds
    (WT .40  TC .35  ET .45  RC .60); EXP-1 uses nnU-Net's own region
    output at 0.5.
  * runs on leak_class_567.csv (patient_only + seen, clean stratum locked
    out); EXP-1 runs on the 259 fold-1 validation cases.

Three different populations and two different prediction sources. Section 10
now names THIS file instead.

WHAT IS CARRIED OVER, AND WHY IT IS THE PART WORTH CARRYING

The matching logic below is copied from c0_fn_recoverability.py without
modification: merge_by_dilation, prediction_components, reference_lesions and
the detection loop. That logic is the only part of C0 that was ever validated
against an external oracle -- its gate reproduced the official evaluator's
false-negative counts AND reference-lesion counts exactly, in every case and
every region tested. Reusing the functions rather than reimplementing them is
the point: it removes a class of divergence instead of guarding against it.

What is NOT carried over is the input. That is the whole change.

THE AXIS-ORDER TRAP, AND HOW IT IS AVOIDED RATHER THAN HANDLED

nnU-Net writes <case>.npz in its own internal axis order and <case>.nii.gz in
the file's orientation; the reader/writer transposes between them. Mixing the
two sources is the "axis order between the npz and the reference" failure that
C0's gate lists as a known biter.

This file does not transpose anything. Everything comes from one order:

    probabilities   <- the npz                        (internal order)
    baseline label  <- convert_probabilities_to_segmentation(those same
                       probabilities)                 (internal order)
    reference       <- the plans' own reader_writer .read_seg()
                                                      (internal order)

The derived baseline label must then be voxel-identical to the .nii.gz
nnU-Net wrote. That is checked, not assumed: --gate asserts it per region on
a probe of cases, against nnU-Net's own artefact. If it holds, the axis order
and the region composition are proven.

  --gate       layer 1: derived segmentation == nnU-Net's written .nii.gz
  --selftest   structural checks on synthetic volumes, no data needed
  --run        build the table

--run refuses to start until --gate has passed for that arm.
===========================================================================
"""

import os
import sys
import json
import glob
import time
import argparse

import numpy as np
import pandas as pd
from scipy import ndimage

# ---- frozen, and identical to c0_fn_recoverability.py ---------------------
DIL_FACTOR = 1      # evaluator's, metrics_MET:679
VOL_THRESH = 2      # evaluator's, metrics_MET:681
P_CAND = 0.02       # protocol section 10, pre-specified, not optimised
PROPOSAL_THR = [0.30, 0.20, 0.10, 0.05, 0.02]
RADII_MM = [10.0, 20.0, 40.0]
GATE_PROBE = 5


# ===========================================================================
#  carried over from c0_fn_recoverability.py, unmodified
# ===========================================================================
def merge_by_dilation(mask):
    """
    Components merged across a one-voxel dilation, at their original extent
    (metrics_MET 271-279 for the reference, 307-315 for the prediction). No
    size filtering: the two paths filter differently and do it themselves.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32), np.array([]), np.array([])
    struct = ndimage.generate_binary_structure(3, 2)
    dil = ndimage.binary_dilation(mask, structure=struct, iterations=DIL_FACTOR)
    lab, _ = ndimage.label(dil, structure=np.ones((3, 3, 3)))
    lab = (lab * mask).astype(np.int32)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    return lab, ids, counts


def prediction_components(mask):
    """
    The prediction path. Components at or below TWO VOXELS are removed here,
    before anything else sees them (metrics_MET 317-323). The ids they vacate
    are not reused, so the numbering has gaps -- which is what the evaluator
    works with.
    """
    lab, ids, counts = merge_by_dilation(mask)
    if not len(ids):
        return lab, ids, counts
    small = counts <= VOL_THRESH
    if small.any():
        lab[np.isin(lab, ids[small])] = 0
    return lab, ids[~small], counts[~small]


def reference_lesions(gt, labels, vox_mm3):
    """
    The reference path. NO filtering at construction: the scoring loop
    (metrics_MET 584) runs over every component regardless of size, and the
    volume criterion is applied afterwards by subtracting from the counts
    (743-749). Filtering here instead would delete lesions the evaluator
    counts, which is exactly what made N_gt disagree on four cases.

    Note the floor is 2 mm3 on this path and 2 voxels on the prediction path.
    Same constant, different units, different stage.
    """
    mask = np.isin(gt, labels)
    lab, ids, counts = merge_by_dilation(mask)
    return lab, ids, counts.astype(float) * vox_mm3


# ===========================================================================
#  EXP-1 specific: one arm's validation folder
# ===========================================================================
def _nnunet(preprocessed, plans_name):
    """The plans' own label manager and reader/writer. Nothing is reimplemented."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    dj = json.load(open(os.path.join(preprocessed, 'dataset.json')))
    pm = PlansManager(os.path.join(preprocessed, f'{plans_name}.json'))
    lm = pm.get_label_manager(dj)
    if not lm.has_regions:
        sys.exit("this dataset is not region-based; EXP-1's endpoints are "
                 "defined per region (WT/TC/ET/RC). STOP.")
    return dj, pm, lm


def _region_names(lm):
    """
    Channel order comes from the label manager, never hardcoded. For
    Dataset137 this yields WT [1,2,3] / TC [1,3] / ET [3] / RC [4] in
    channels 0..3, matching c0's CH map -- but it is read, so a plans change
    cannot silently reorder the columns.
    """
    guess = {(1, 2, 3): 'WT', (1, 3): 'TC', (3,): 'ET', (4,): 'RC'}
    out = []
    for i, r in enumerate(lm.foreground_regions):
        t = tuple(r) if isinstance(r, (list, tuple)) else (r,)
        out.append((i, guess.get(t, f'ch{i}'), t))
    return out


def _retry_read(fn, what, tries=4, pause=3.0):
    """
    Retry a read that failed for no reason the filesystem will admit to.

    Added 20 October 2026. The gate died on BraTS-MET-00030-000.nii.gz with
    SimpleITK's "does not exist" on a file that was present the whole time:
    259/259 .nii.gz, .npz and .pkl, verified immediately afterwards. Colab's
    Drive FUSE mount does that — an existing file momentarily fails to open.

    Over a 7-hour pass across 259 cases this matters more than it looks. The
    `--run` loop catches per-case exceptions and keeps going, so one transient
    failure would quietly leave a case out of the table, and `exp1.py gate`
    asserts that both arms carry the same lesion set. A single dropped read in
    one arm therefore breaks the comparison after seven hours of work.

    Retries only. Nothing here changes a value; a path that is genuinely wrong
    still raises, with the original error.
    """
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i + 1 < tries:
                print(f"      read failed on {what} "
                      f"({type(e).__name__}), retry {i+1}/{tries-1}",
                      flush=True)
                time.sleep(pause)
    raise last


def load_case(val_dir, gt_dir, case, lm, rw):
    """
    Probabilities, the baseline label derived from them, and the reference --
    all three in nnU-Net's internal axis order. See the module docstring on
    why nothing here is transposed.
    """
    npz = os.path.join(val_dir, f'{case}.npz')
    if not os.path.isfile(npz):
        raise FileNotFoundError(
            f"{case}.npz missing. The arm was run without --npz, so no "
            f"probability maps exist and the primary endpoint (protocol "
            f"section 10) cannot be computed. Re-export with:\n"
            f"  nnUNetv2_train 137 3d_fullres 1 -tr <TRAINER> "
            f"-p nnUNetResEncUNetLPlans --val --npz")

    def _np():
        with np.load(npz) as z:
            key = 'probabilities' if 'probabilities' in z else z.files[0]
            return np.asarray(z[key], dtype=np.float32)
    prob = _retry_read(_np, f'{case}.npz')

    seg = np.asarray(lm.convert_probabilities_to_segmentation(prob))

    gt_path = os.path.join(gt_dir, f'{case}.nii.gz')
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"no reference for {case} in {gt_dir}")
    gt_arr, gt_props = _retry_read(lambda: rw.read_seg(gt_path),
                                   f'{case} reference')
    gt = np.asarray(gt_arr)[0]

    if gt.shape != seg.shape:
        raise ValueError(f"{case}: reference {gt.shape} vs prediction "
                         f"{seg.shape} -- these are not the same case or the "
                         f"orders disagree. STOP.")

    sp = [float(v) for v in gt_props['spacing']]
    vox_mm3 = float(np.prod(sp))
    return prob, seg, gt, vox_mm3, sp


def analyse_case(case, val_dir, gt_dir, lm, rw, regions, want_proposals=True):
    """
    Per reference lesion: was it detected by this arm, and does any
    lower-threshold component touch it. The evaluator's loop, in its order
    (metrics_MET 584-650): for every reference component, dilate it, intersect
    with the predicted components, call it a true positive if anything
    intersects. No component is skipped for being small. Walked rather than
    summarised, so any divergence in what "detected" means stays visible.
    """
    prob, seg, gt, vox_mm3, sp = load_case(val_dir, gt_dir, case, lm, rw)
    struct = ndimage.generate_binary_structure(3, 2)
    spacing = np.array(sp)

    rows, counts = [], {}
    for ch, reg, labels in regions:
        gt_lab, gt_ids, gt_vols = reference_lesions(gt, labels, vox_mm3)
        base_lab, base_ids, _ = prediction_components(np.isin(seg, labels))

        detected, tp_ids_all, tp_centroids = {}, [], []
        for gid in gt_ids:
            sel = gt_lab == gid
            sel_dil = ndimage.binary_dilation(sel, structure=struct,
                                              iterations=DIL_FACTOR)
            hit = np.unique(base_lab[sel_dil])
            hit = hit[hit > 0]
            detected[int(gid)] = bool(hit.size)
            tp_ids_all.extend(int(h) for h in hit)
            if hit.size:
                tp_centroids.append(ndimage.center_of_mass(sel))

        # Only now the volume criterion, subtracted from the counts as the
        # evaluator does (743-749) rather than removing lesions beforehand.
        small = {int(g): (v <= VOL_THRESH) for g, v in zip(gt_ids, gt_vols)}
        raw_fn = [g for g in detected if not detected[g]]
        raw_tp = [g for g in detected if detected[g]]
        n_fn = len(raw_fn) - sum(1 for g in raw_fn if small[g])
        n_tp = len(raw_tp) - sum(1 for g in raw_tp if small[g])
        n_fp = int(len(set(map(int, base_ids)) - set(tp_ids_all)))

        counts[reg] = {'n_gt': n_tp + n_fn, 'TP': n_tp, 'FN': n_fn, 'FP': n_fp,
                       'n_base_components': int(len(base_ids)),
                       'n_gt_raw': int(len(gt_ids))}

        if not want_proposals:
            continue

        # Proposals stay on the channel: the question is whether probability
        # evidence exists inside the lesion at a lower threshold, and the
        # nested composition would let a competing region claim those voxels
        # first.
        prop = {}
        for t in PROPOSAL_THR:
            pl, pids, _ = prediction_components(prob[ch] >= t)
            prop[t] = pl
            counts[reg][f'n_cand@{t}'] = int(len(pids))

        tp_arr = np.array(tp_centroids) if tp_centroids else np.empty((0, 3))

        for gid, vol in zip(gt_ids, gt_vols):
            sel = gt_lab == gid
            rec = {'case': case, 'region': reg, 'gt_lesion': int(gid),
                   'gt_volume_mm3': round(float(vol), 4),
                   'gt_voxels': int(sel.sum()),
                   # named for the gate directly: exp1.py gate wants
                   # 'detected' and 'surfaced', and renaming downstream is
                   # one more place to map the wrong column.
                   'detected': int(detected[int(gid)]),
                   'max_prob_in_lesion': round(float(prob[ch][sel].max()), 4),
                   'mean_prob_in_lesion': round(float(prob[ch][sel].mean()), 4)}

            first = None
            for t in PROPOSAL_THR:
                hit = np.unique(prop[t][sel])        # DIRECT overlap, no
                hit = hit[hit > 0]                   # dilation tolerance
                rec[f'cand@{t}'] = int(hit.size > 0)
                if hit.size:
                    rec[f'cand_vox@{t}'] = int(max((prop[t] == h).sum()
                                                   for h in hit))
                    if first is None:
                        first = t
                else:
                    rec[f'cand_vox@{t}'] = 0
            rec['first_threshold'] = first if first is not None else np.nan
            rec['surfaced'] = rec[f'cand@{P_CAND}']

            if len(tp_arr):
                c = np.array(ndimage.center_of_mass(sel))
                d = np.sqrt((((tp_arr - c) * spacing) ** 2).sum(axis=1))
                rec['dist_nearest_detected_mm'] = round(float(d.min()), 2)
                for r in RADII_MM:
                    rec[f'n_detected_within_{int(r)}mm'] = int((d <= r).sum())
            else:
                rec['dist_nearest_detected_mm'] = np.nan
                for r in RADII_MM:
                    rec[f'n_detected_within_{int(r)}mm'] = 0
            rows.append(rec)

    del prob, seg, gt
    return rows, counts


# ===========================================================================
#  gate -- layer 1, against nnU-Net's own artefact
# ===========================================================================
def gate(a):
    print("=" * 74)
    print(" EXP-1 lesion gate | layer 1: derived segmentation vs nnU-Net's "
          ".nii.gz")
    print("=" * 74)

    dj, pm, lm = _nnunet(a.preprocessed, a.plans)
    rw = pm.image_reader_writer_class()
    regions = _region_names(lm)
    print(f"  channels: " + "  ".join(f"{i}={n}{list(t)}" for i, n, t in regions))

    cases = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(f'{a.val_dir}/*.npz'))
    if not cases:
        sys.exit(f"no .npz under {a.val_dir}.\n"
                 f"This arm was run without --npz, so the probability maps do "
                 f"not exist and the primary endpoint cannot be computed.\n"
                 f"Re-export off the final checkpoint (no retraining):\n"
                 f"  nnUNetv2_train 137 3d_fullres 1 -tr <TRAINER> "
                 f"-p {a.plans} --val --npz")
    print(f"  cases with probabilities: {len(cases)}")
    print(f"  probe: {min(GATE_PROBE, len(cases))} cases x "
          f"{len(regions)} regions\n")

    rows = []
    for c in cases[:GATE_PROBE]:
        prob, seg, gt, vox, sp = load_case(a.val_dir, a.gt_dir, c, lm, rw)
        # nnU-Net's written label, read back through the SAME reader, so the
        # comparison is in one axis order.
        wp = os.path.join(a.val_dir, f'{c}.nii.gz')
        if not os.path.isfile(wp):
            raise FileNotFoundError(
                f"{c}.nii.gz is not in {a.val_dir}, but {c}.npz is. nnU-Net "
                f"writes the npz BEFORE the label in the same call, so an "
                f"export killed midway leaves exactly this. Count both sets "
                f"before re-exporting: if they are equal, the read was "
                f"transient (Drive FUSE) and re-running this gate is enough.")
        w, _ = _retry_read(lambda: rw.read_seg(wp), f'{c} written label')
        written = np.asarray(w)[0]
        for ch, reg, labels in regions:
            mine = np.isin(seg, labels)
            theirs = np.isin(written, labels)
            rows.append({'case': c, 'region': reg,
                         'mine': int(mine.sum()), 'nnunet': int(theirs.sum()),
                         'identical': bool(np.array_equal(mine, theirs))})

    d = pd.DataFrame(rows)
    n_ok = int(d.identical.sum())
    print(f"  {n_ok}/{len(d)} region masks voxel-identical")
    if n_ok < len(d):
        print("  " + d[~d.identical].to_string(index=False).replace("\n", "\n  "))
        print("""
  GATE FAILED.

  DIAGNOSTIC STOP. The segmentation derived from the probability maps is not
  the segmentation nnU-Net wrote, so the probabilities and the reference are
  not in the same frame and every lesion match built on them would be
  measuring a misalignment. Do not proceed and do not transpose until it
  passes -- find out which of the two is being read wrongly.

  Where to look, in the order these have bitten before:
    - axis order between the npz and the reference
    - regions_class_order in the plans vs foreground_regions order
    - the npz holding logits rather than sigmoid probabilities""")
        return False

    flag = os.path.join(os.path.dirname(a.out) or '.',
                        f'.exp1_lesions_gate_{a.arm}')
    os.makedirs(os.path.dirname(flag) or '.', exist_ok=True)
    with open(flag, 'w') as f:
        json.dump({'passed': True, 'arm': a.arm, 'cases': len(cases),
                   'comparisons': len(d),
                   'when': time.strftime('%Y-%m-%d %H:%M')}, f)
    print(f"""
  GATE PASSED. The probability maps and the reference are in the same frame,
  and the region composition derived from the probabilities reproduces
  nnU-Net's own output exactly. The matching functions below are the ones
  C0's gate validated against the official evaluator counts.

  Run --run.""")
    return True


# ===========================================================================
#  selftest -- synthetic, no data
# ===========================================================================
def selftest(a):
    print("=" * 74)
    print(" EXP-1 lesion builder | structural selftest")
    print("=" * 74)
    fails = []

    def check(name, cond, detail=''):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"   {detail}" if detail else ''))
        if not cond:
            fails.append(name)

    # two-voxel floor on the prediction path, 2 mm3 on the reference path
    m = np.zeros((20, 20, 20), bool)
    m[2, 2, 2:4] = True            # 2 voxels -> dropped
    m[10, 10, 10:13] = True        # 3 voxels -> kept
    _, ids, cnt = prediction_components(m)
    check("prediction floor drops <=2 voxels, keeps 3",
          len(ids) == 1 and cnt[0] == 3, f"ids={len(ids)} counts={list(cnt)}")

    _, rids, rvol = reference_lesions(np.where(m, 1, 0), (1,), 1.0)
    check("reference path does NOT filter at construction",
          len(rids) == 2, f"ids={len(rids)} (both components kept)")

    # MERGING. Both sides of a gap are dilated by one, so the gap that closes
    # is TWO voxels wide, not one. Asserted at the boundary in both
    # directions, because an off-by-one here silently changes what counts as
    # one lesion.
    def gap(n):
        m_ = np.zeros((24, 24, 24), bool)
        m_[5, 5, 5] = True
        m_[5, 5, 5 + n + 1] = True
        return len(merge_by_dilation(m_)[1])

    check("merge closes a one-voxel gap", gap(1) == 1, f"-> {gap(1)} component")
    check("merge closes a two-voxel gap", gap(2) == 1, f"-> {gap(2)} component")
    check("merge does NOT close a three-voxel gap", gap(3) == 2,
          f"-> {gap(3)} components; the tolerance is exactly 2")

    # 26-connectivity, not 6: a pure diagonal is one component
    dg = np.zeros((20, 20, 20), bool)
    dg[5, 5, 5] = dg[6, 6, 6] = True
    check("26-connectivity joins a diagonal pair",
          len(merge_by_dilation(dg)[1]) == 1)

    # DETECTION. Only the REFERENCE is dilated, once. So a prediction must
    # reach a voxel adjacent to the reference; one empty voxel between them is
    # already a miss. This is a tighter tolerance than the merge above, and
    # the asymmetry is the evaluator's, not an oversight.
    struct = ndimage.generate_binary_structure(3, 2)
    ref = np.zeros((24, 24, 24), bool)
    ref[10, 10, 10:14] = True                 # occupies 10..13
    rd = ndimage.binary_dilation(ref, structure=struct, iterations=DIL_FACTOR)

    def detects(start):
        p = np.zeros((24, 24, 24), bool)
        p[10, 10, start:start + 3] = True
        lab, _, _ = prediction_components(p)
        hit = np.unique(lab[rd])
        return bool(hit[hit > 0].size)

    check("prediction touching the reference is detected", detects(14) is True,
          "starts at 14, reference ends at 13")
    check("prediction one empty voxel away is NOT detected",
          detects(15) is False, "starts at 15, voxel 14 empty")
    check("distant prediction is not detected", detects(18) is False)

    # SURFACING is direct overlap with no dilation at all (protocol
    # section 10, as corrected 7 October). So the component at 14 -- which
    # detection above DOES find -- must not surface. If this test ever passes
    # for both, the two criteria have collapsed into one.
    p14 = np.zeros((24, 24, 24), bool)
    p14[10, 10, 14:17] = True
    lab, _, _ = prediction_components(p14)
    hit = np.unique(lab[ref])
    check("a touching component surfaces only on DIRECT overlap",
          hit[hit > 0].size == 0,
          "detected=1 but surfaced=0; the criteria are distinct")

    p_in = np.zeros((24, 24, 24), bool)
    p_in[10, 10, 12:15] = True                # overlaps 12,13
    lab, _, _ = prediction_components(p_in)
    hit = np.unique(lab[ref])
    check("an overlapping component does surface", hit[hit > 0].size == 1)

    print("\n" + ("  all structural checks passed" if not fails
                  else f"  {len(fails)} FAILED: {fails}"))
    return not fails


# ===========================================================================
def run(a):
    flag = os.path.join(os.path.dirname(a.out) or '.',
                        f'.exp1_lesions_gate_{a.arm}')
    if not os.path.exists(flag) and not a.force:
        sys.exit("  The gate has not passed for this arm. Run --gate first.\n"
                 "  (--force overrides, and records nothing; do not use it "
                 "for a result.)")

    dj, pm, lm = _nnunet(a.preprocessed, a.plans)
    rw = pm.image_reader_writer_class()
    regions = _region_names(lm)

    cases = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(f'{a.val_dir}/*.npz'))
    if a.n:
        cases = cases[:a.n]
        print(f"  DRY RUN on {len(cases)} cases -- checks the code, not the "
              f"science.\n")

    print("=" * 74)
    print(f" EXP-1 lesion table | arm = {a.arm}")
    print("=" * 74)
    print(f"  cases           : {len(cases)}")
    print(f"  proposal thr    : {PROPOSAL_THR}  (pre-specified)")
    print(f"  surfaced        : connected component at p >= {P_CAND}, "
          f"direct overlap")
    print(f"  prediction floor: <= {VOL_THRESH} voxels dropped")
    print(f"  reference floor : <= {VOL_THRESH} mm3 subtracted from counts\n")

    les, cas, errs = [], [], []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        try:
            rows, cnt = analyse_case(c, a.val_dir, a.gt_dir, lm, rw, regions)
            les += rows
            rec = {'case': c, 'arm': a.arm}
            for reg in cnt:
                for k, v in cnt[reg].items():
                    rec[f'{reg}_{k}'] = v
            cas.append(rec)
        except Exception as e:
            errs.append((c, f"{type(e).__name__}: {e}"))
        if i % 20 == 0 or i == len(cases):
            pd.DataFrame(les).to_csv(a.out, index=False)
            pd.DataFrame(cas).to_csv(a.out.replace('.csv', '_cases.csv'),
                                     index=False)
            el = time.time() - t0
            print(f"    {i}/{len(cases)}  {len(les):,} lesions  "
                  f"{el/60:.0f}m  eta {el/i*(len(cases)-i)/60:.0f}m")

    d = pd.DataFrame(les)
    d['arm'] = a.arm
    d.to_csv(a.out, index=False)
    cs = pd.DataFrame(cas)
    cs.to_csv(a.out.replace('.csv', '_cases.csv'), index=False)

    if errs:
        print(f"\n  ERRORS on {len(errs)} cases:")
        for c, e in errs[:5]:
            print(f"    {c}: {e}")
        print("  A partial table is not comparable between arms. Resolve "
              "these before using it.")

    print("\n" + "=" * 74)
    print(f" {a.arm}  ·  {len(d):,} reference lesions in "
          f"{d.case.nunique()} cases")
    print("=" * 74)
    print(f"  {'region':<8}{'lesions':>9}{'detected':>10}{'surfaced':>10}"
          f"{'<=27mm3':>10}{'det|small':>11}")
    for _, reg, _ in regions:
        s = d[d.region == reg]
        if not len(s):
            continue
        sm = s[s.gt_volume_mm3 <= 27.0]
        print(f"  {reg:<8}{len(s):>9,}{s.detected.mean():>9.1%}"
              f"{s.surfaced.mean():>10.1%}{len(sm):>10,}"
              f"{(sm.detected.mean() if len(sm) else float('nan')):>10.1%}")
    fp = sum(cs[f'{r}_FP'].sum() for _, r, _ in regions
             if f'{r}_FP' in cs.columns)
    print(f"\n  false positives per case: {fp/max(len(cs),1):.2f}   "
          f"(criterion C input, --fp-{a.arm})")
    print(f"\n  written: {a.out}")
    print(f"           {a.out.replace('.csv', '_cases.csv')}")
    print("\n  Feed both arms to the gate:")
    print(f"    python exp1.py gate --baseline <baseline>.csv "
          f"--new <sizeaware>.csv \\\n           --fp-base <x> --fp-new <y>")
    if a.n:
        print("\n  DRY RUN -- draw no conclusions. Re-run without --n.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not argparse-required: --selftest is synthetic and must run anywhere,
    # including a machine with no data on it. The commands that do need paths
    # check for them themselves, by name, so the error says which one.
    ap.add_argument('--val-dir', help="an arm's fold_1/validation folder "
                                      "(needs .npz)")
    ap.add_argument('--gt-dir',
                    help='nnUNet_preprocessed/<dataset>/gt_segmentations')
    ap.add_argument('--preprocessed',
                    help='nnUNet_preprocessed/<dataset> (dataset.json, plans)')
    ap.add_argument('--plans', default='nnUNetResEncUNetLPlans')
    ap.add_argument('--arm', choices=['baseline', 'sizeaware'])
    ap.add_argument('--out', default='exp1_lesions.csv')
    ap.add_argument('--n', type=int, default=0, help='dry run on N cases')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='skip the gate check; not for a result')
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest(a) else 1)

    if a.gate or a.run:
        need = [k for k in ('val_dir', 'gt_dir', 'preprocessed', 'arm')
                if not getattr(a, k)]
        if need:
            ap.error("missing for --" + ("gate" if a.gate else "run") + ": "
                     + ", ".join("--" + k.replace("_", "-") for k in need))

    if a.gate:
        sys.exit(0 if gate(a) else 1)
    if a.run:
        run(a)
    else:
        print(__doc__)
        ap.print_help()


if __name__ == '__main__':
    main()
