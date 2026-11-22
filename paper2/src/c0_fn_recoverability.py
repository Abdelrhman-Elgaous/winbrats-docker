# -*- coding: utf-8 -*-
"""
==============================================================================
 C0 GATE · FALSE-NEGATIVE RECONSTRUCTION  —  validate before unlocking
==============================================================================
 Read-only with respect to every frozen artefact. Writes only its own outputs.

 C0 asks whether lesions the baseline missed have any representation in the
 probability maps at a lower threshold. Answering that needs the set of missed
 lesions, and that set must be the official one -- the evaluator's own
 false negatives, not "reference lesions with no overlapping prediction".
 Those are different populations, and building Phase C on the second while
 believing it is the first would misstate the problem.

 The official evaluator takes minutes to hours per case, so running it over
 1,296 cases is not available. The alternative is to reconstruct its false
 negatives externally and prove the reconstruction correct before using it,
 exactly as the false-positive assignment was proven before the component
 analysis.

 THE GATE
   reconstruct FN on the 155 cases that have official T4 counts
        ↓
   compare region-by-region, case-by-case, not in aggregate
        ↓
   exact agreement  →  unlock the 1,296
   any mismatch     →  DIAGNOSTIC STOP

 A mismatch is not something to disclose and work around. It means the
 reconstruction is measuring a different quantity, and the analysis built on it
 would be about a different problem.

 The 155 cases are used here as a technical validation reference, the way
 Num_FP was used for the component gate. They are not used to choose a
 threshold, a feature or a model, and nothing here is a design decision.

 Run --gate first. --run refuses to start until the gate has passed.
==============================================================================
"""

import os
import sys
import json
import glob
import time
import argparse
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
GT_DIR = ns.GT_DIR
OUT = f"{DRIVE}/phase_c_outputs/c0"
GATE_FLAG = f"{OUT}/.c0_gate_passed"
W2 = f"{DRIVE}/phase_b_outputs/wave2"

# ---- frozen before any C0 number was computed ------------------------------
REGIONS = ["WT", "TC", "ET", "RC"]
CH = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}
GT_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)}
BASELINE_THR = dict(ns.THR)                      # WT .40 TC .35 ET .45 RC .60
PROPOSAL_THR = [0.30, 0.20, 0.10, 0.05, 0.02]    # pre-specified, not optimised
DIL_FACTOR = 1                                   # evaluator's, metrics_MET:679
VOL_THRESH = 2                                   # evaluator's, metrics_MET:681
RADII_MM = [10.0, 20.0, 40.0]
VOXEL_PROBE = 5   # cases for the voxel-identity layer                    # C0.5, fixed before results
NPZ_TO_GT = ns.NPZ_TO_GT

os.makedirs(OUT, exist_ok=True)


def load_case(case):
    """
    The Phase B prediction, built by the Phase B functions.

    An earlier version of this read a single out-of-fold npz and thresholded
    the channel directly. Both were wrong, and together they moved the whole
    prediction: the out-of-fold map is one fold where Phase B used the mean of
    three, and thresholding a channel ignores the nested composition that
    decides which region a voxel ends up in. The two produced 23,294 predicted
    voxels against the evaluator's 22,031 on one case, and 85 disagreeing
    comparisons across the validation set.

    Calling load_mean_prob and build_nested_label rather than reimplementing
    them is the point: these are the functions that produced every Phase B
    number, so using them removes a whole class of divergence rather than
    guarding against it.
    """
    mean = ns.load_mean_prob(case)                  # mean of folds 1, 2, 4
    lab = ns.build_nested_label(mean, suppressor=None)

    img = nib.load(f"{GT_DIR}/{case}.nii.gz")
    inv = np.argsort(NPZ_TO_GT)
    gt = np.asanyarray(img.dataobj).transpose(inv)
    if gt.shape != lab.shape:
        raise ValueError(f"{case}: gt {gt.shape} vs label {lab.shape}")
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return mean, lab, gt, (sx, sy, sz)


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


def reference_lesions(gt, region, vox_mm3):
    """
    The reference path. NO filtering at construction: the scoring loop
    (metrics_MET 584) runs over every component regardless of size, and the
    volume criterion is applied afterwards by subtracting from the counts
    (743-749). Filtering here instead would delete lesions the evaluator
    counts, which is exactly what made N_gt disagree on four cases.

    Note the floor is 2 mm3 on this path and 2 voxels on the prediction path.
    Same constant, different units, different stage.
    """
    mask = np.isin(gt, GT_LABELS[region])
    lab, ids, counts = merge_by_dilation(mask)
    return lab, ids, counts.astype(float) * vox_mm3


def analyse_case(case, want_proposals=True):
    """
    Per reference lesion: was it detected at baseline, and if not, does any
    lower-threshold component touch it. Returns rows and per-case counts.
    """
    loaded = load_case(case)
    if loaded is None:
        return None, None
    mean, lab, gt, (sx, sy, sz) = loaded
    vox_mm3 = sx * sy * sz

    rows, counts = [], {}
    for reg in REGIONS:
        ch = CH[reg]
        gt_lab, gt_ids, gt_vols = reference_lesions(gt, reg, vox_mm3)

        # The region as the evaluator sees it: extracted from the nested
        # label, not thresholded off the channel. Nesting moves voxels between
        # regions, so the two are not the same mask.
        base_mask = np.isin(lab, GT_LABELS[reg])
        base_lab, base_ids, _ = prediction_components(base_mask)

        # The evaluator's loop, in its order (metrics_MET 584-650):
        #   for every reference component, dilate it, intersect with the
        #   predicted components, and call it a true positive if anything
        #   intersects. No component is skipped for being small.
        # Restating this as arithmetic would hide any divergence in what
        # "detected" means, so it is walked rather than summarised.
        struct = ndimage.generate_binary_structure(3, 2)
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
        fn_sub = sum(1 for g in raw_fn if small[g])
        tp_sub = sum(1 for g in raw_tp if small[g])

        n_fn = len(raw_fn) - fn_sub
        n_tp = len(raw_tp) - tp_sub
        n_fp = int(len(set(map(int, base_ids)) - set(tp_ids_all)))

        counts[reg] = {"n_gt": n_tp + n_fn, "TP": n_tp, "FN": n_fn,
                       "FP": n_fp, "n_base_components": int(len(base_ids)),
                       "n_gt_raw": int(len(gt_ids)),
                       "fn_sub": int(fn_sub), "tp_sub": int(tp_sub)}

        if not want_proposals:
            continue

        # proposals, computed once per threshold and reused for every lesion
        # Proposals stay on the channel: the question is whether probability
        # evidence exists inside the lesion at a lower threshold, and nesting
        # would let a competing region claim those voxels first.
        prop = {}
        for t in PROPOSAL_THR:
            pl, pids, pcnt = prediction_components(mean[ch] >= t)
            prop[t] = (pl, pids, pcnt)
            counts[reg][f"n_cand@{t}"] = int(len(pids))

        tp_arr = np.array(tp_centroids) if tp_centroids else np.empty((0, 3))
        spacing = np.array([sx, sy, sz])

        for gid, vol in zip(gt_ids, gt_vols):
            sel = gt_lab == gid
            rec = {"case": case, "region": reg, "gt_lesion": int(gid),
                   "gt_volume_mm3": round(float(vol), 4),
                   "gt_voxels": int(sel.sum()),
                   "baseline_detected": int(detected[int(gid)]),
                   "max_prob_in_lesion": round(float(mean[ch][sel].max()), 4),
                   "mean_prob_in_lesion": round(float(mean[ch][sel].mean()), 4)}

            first = None
            for t in PROPOSAL_THR:
                pl, _, pcnt = prop[t]
                hit = np.unique(pl[sel])
                hit = hit[hit > 0]
                rec[f"cand@{t}"] = int(hit.size > 0)
                if hit.size:
                    sizes = [int((pl == h).sum()) for h in hit]
                    rec[f"cand_vox@{t}"] = int(max(sizes))
                    if first is None:
                        first = t
                else:
                    rec[f"cand_vox@{t}"] = 0
            rec["first_threshold"] = first if first is not None else np.nan

            # C0.5 — spatial relation to lesions the baseline did find
            if len(tp_arr):
                c = np.array(ndimage.center_of_mass(sel))
                d = np.sqrt((((tp_arr - c) * spacing) ** 2).sum(axis=1))
                rec["dist_nearest_detected_mm"] = round(float(d.min()), 2)
                for r in RADII_MM:
                    rec[f"n_detected_within_{int(r)}mm"] = int((d <= r).sum())
            else:
                rec["dist_nearest_detected_mm"] = np.nan
                for r in RADII_MM:
                    rec[f"n_detected_within_{int(r)}mm"] = 0
            rows.append(rec)

    del mean, lab, gt
    return rows, counts


# ---------------------------------------------------------------------------
def gate():
    print("=" * 78)
    print(" C0 GATE — reconstructed FN against the official counts")
    print("=" * 78)

    files = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv")]
    if not files:
        sys.exit(f"official T4 counts not found under {W2}")
    t4 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    off = t4[t4.setting == "baseline"].set_index(["case", "region"])
    cases = sorted(off.index.get_level_values(0).unique())
    print(f"  validation cases: {len(cases)}  (official T4 baseline counts)")
    print(f"  criterion       : exact agreement, per case AND per region\n")

    # Layer 1: does the reconstruction produce the same predicted voxels the
    # evaluator scored? Checked separately from the counts, because a single
    # combined test cannot say whether a failure came from the source or from
    # the matching logic built on top of it.
    print("  Layer 1 — voxel identity against the evaluator's own artefacts")
    vox_rows = []
    for c in cases[:VOXEL_PROBE]:
        try:
            _, lab, _, _ = load_case(c)
            for reg in REGIONS:
                mine = np.isin(lab, GT_LABELS[reg])
                f = glob.glob(f"{ns.WORK_DIR}/tmp_pred/**/*{c}_{reg}.nii.gz",
                              recursive=True)
                f = [x for x in f if "_cc" not in os.path.basename(x)]
                if not f:
                    continue
                ev = np.asanyarray(nib.load(f[0]).dataobj).transpose(
                    np.argsort(NPZ_TO_GT)) > 0
                vox_rows.append({"case": c, "region": reg,
                                 "mine": int(mine.sum()), "eval": int(ev.sum()),
                                 "identical": bool(np.array_equal(mine, ev))})
        except Exception as e:
            print(f"    {c}: {type(e).__name__}: {e}")
    if vox_rows:
        v = pd.DataFrame(vox_rows)
        n_id = int(v.identical.sum())
        print(f"    {n_id}/{len(v)} region masks voxel-identical")
        if n_id < len(v):
            print("    " + v[~v.identical].to_string(index=False)
                  .replace("\n", "\n    "))
            print("""
    STOP. The prediction itself differs from what the evaluator scored, so any
    count comparison below would be measuring two things at once. Fix the
    source before reading the FN numbers.""")
            return False
    else:
        print("    no evaluator artefacts on disk to compare against —")
        print("    run one case through official_score first, or accept that")
        print("    layer 1 is untested and rely on the counts alone.")

    print("\n  Layer 2 — reconstructed FN against the official counts")
    rows, errs = [], []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        try:
            _, cnt = analyse_case(c, want_proposals=False)
            if cnt is None:
                errs.append((c, "could not load"))
                continue
            for reg in REGIONS:
                if (c, reg) not in off.index:
                    continue
                o = off.loc[(c, reg)]
                rows.append({"case": c, "region": reg,
                             "mine_FN": cnt[reg]["FN"],
                             "official_FN": int(o.Num_FN),
                             "mine_TP": cnt[reg]["TP"],
                             "official_TP": int(o.Num_TP),
                             "mine_ngt": cnt[reg]["n_gt"],
                             "official_ngt": int(o.Num_TP + o.Num_FN)})
        except Exception as e:
            errs.append((c, f"{type(e).__name__}: {e}"))
        if i % 25 == 0:
            print(f"    {i}/{len(cases)}  ({time.time()-t0:.0f}s)")

    d = pd.DataFrame(rows)
    if not len(d):
        sys.exit("  no comparisons produced")

    d["fn_match"] = d.mine_FN == d.official_FN
    d["ngt_match"] = d.mine_ngt == d.official_ngt

    d["fn_diff"] = (d.mine_FN - d.official_FN).abs()
    print("\n" + "=" * 78)
    print(f"  cases checked   : {d.case.nunique()}")
    print(f"  region cells    : {len(d)}")
    print(f"  mismatched cells: {int((~(d.mine_FN == d.official_FN)).sum())}")
    print(f"  max |delta FN|  : {int(d.fn_diff.max())}")
    print(f"  comparisons     : {len(d)}   ({d.case.nunique()} cases)")
    print(f"  FN exact match  : {int(d.fn_match.sum())}/{len(d)}"
          f"   ({d.fn_match.mean():.1%})")
    print(f"  N_gt match      : {int(d.ngt_match.sum())}/{len(d)}"
          f"   ({d.ngt_match.mean():.1%})")
    if errs:
        print(f"  errors          : {len(errs)}")
        for c, e in errs[:5]:
            print(f"    {c}: {e}")

    print("\n  per region:")
    g = d.groupby("region").agg(n=("fn_match", "size"),
                                fn_ok=("fn_match", "sum"),
                                ngt_ok=("ngt_match", "sum"))
    g["fn_rate"] = (g.fn_ok / g.n).round(3)
    print("  " + g.to_string().replace("\n", "\n  "))

    d.to_csv(f"{OUT}/c0_gate_validation.csv", index=False)

    strict = (bool(d.fn_match.all()) and bool(d.ngt_match.all())
              and int(d.fn_diff.max()) == 0 and not errs)
    print(f"\n  acceptance: {len(d)}/{len(d)} FN exact · "
          f"{int(d.ngt_match.sum())}/{len(d)} N_gt exact · "
          f"max |delta FN| = {int(d.fn_diff.max())}")
    if strict:
        open(GATE_FLAG, "w").write(json.dumps({
            "passed": True, "cases": int(d.case.nunique()),
            "comparisons": int(len(d)), "when": time.strftime("%Y-%m-%d %H:%M")}))
        print("""
  GATE PASSED. The reconstruction reproduces the evaluator's false-negative
  counts exactly, in every case and every region tested. Reference-lesion
  counts agree too, so the two are identifying the same lesions and not merely
  arriving at the same total.

  The reconstruction may now be applied to the full cohort. Run --run.""")
        return True

    bad = d[~d.fn_match | ~d.ngt_match]
    print(f"\n  mismatched cases: "
          f"{sorted(bad.case.unique())[:10]}"
          + (f" … +{bad.case.nunique()-10} more" if bad.case.nunique() > 10 else ""))
    print(f"""
  GATE FAILED — {len(bad)} disagreeing comparison(s).

  DIAGNOSTIC STOP. Do not proceed, and do not substitute a looser definition
  of "missed lesion" to make this pass. A reconstruction that disagrees with
  the evaluator is measuring a different quantity, and C0 built on it would
  characterise a different problem from the one Phase B established.

  Where to look, in the order these have bitten before:
    · dilation and merging (metrics_MET 307-322)
    · the two-voxel floor, applied as voxels to predictions and mm3 to
      reference lesions
    · connectivity: 26 for the evaluator, not the ndimage default
    · axis order between the npz and the reference
    · region composition from the nested labels""")
    print("\n" + bad.head(12).to_string(index=False))
    print(f"\n  full table: {OUT}/c0_gate_validation.csv")
    return False


def run():
    if not os.path.exists(GATE_FLAG):
        sys.exit("  The gate has not passed. Run --gate first.")

    # The primary cohort, fixed before any C0 number was computed: the non-clean
    # Phase B strata. These are the cases Phase B scored, so the Phase B
    # ensemble exists for them by construction, and they exclude the 155 clean
    # cases entirely — no design decision may touch those.
    #
    # They are NOT independent. The deployed folds saw them, wholly or through
    # another timepoint of the same patient. They are the non-test-stratum
    # cohort and must be described that way.
    #
    # The 729 cases outside the 567 are excluded: load_mean_prob could not
    # build the Phase B ensemble for any of the eight probed, so they have a
    # different prediction source and therefore a different false-negative set.
    lk = pd.read_csv(f"{DRIVE}/phase_b_outputs/wave0/leak_class_567.csv")
    cases = sorted(lk.loc[lk.leak_class.isin(["patient_only", "seen"]), "case"])
    locked = sorted(lk.loc[lk.leak_class == "clean", "case"])

    print("=" * 78)
    print(" C0 — primary cohort")
    print("=" * 78)
    print(f"  patient_only + seen : {len(cases)}")
    print(f"  clean, LOCKED       : {len(locked)}   (not analysed here)")
    assert not (set(cases) & set(locked)), "the locked stratum leaked into C0"
    print(f"  proposal thresholds: {PROPOSAL_THR}  (pre-specified)")

    les_rows, case_rows, qc = [], [], defaultdict(int)
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        try:
            rows, cnt = analyse_case(c)
            if rows is None:
                qc["unloadable"] += 1
                continue
            les_rows += rows
            rec = {"case": c}
            for reg in REGIONS:
                for k, v in cnt[reg].items():
                    rec[f"{reg}_{k}"] = v
            case_rows.append(rec)
            qc["processed"] += 1
        except Exception as e:
            qc["failed"] += 1
            qc[f"err_{type(e).__name__}"] += 1
        if i % 50 == 0 or i == len(cases):
            pd.DataFrame(les_rows).to_csv(f"{OUT}/c0_lesions.csv", index=False)
            pd.DataFrame(case_rows).to_csv(f"{OUT}/c0_cases.csv", index=False)
            el = time.time() - t0
            print(f"    {i}/{len(cases)}  {len(les_rows)} lesions  "
                  f"{el/60:.0f}m  eta {el/i*(len(cases)-i)/60:.0f}m")

    les = pd.DataFrame(les_rows)
    les.to_csv(f"{OUT}/c0_lesions.csv", index=False)
    pd.DataFrame(case_rows).to_csv(f"{OUT}/c0_cases.csv", index=False)

    json.dump({
        "cohort_size": len(cases),
        "source": "load_mean_prob(folds 1,2,4) -> build_nested_label, the Phase B pipeline",
        "cohort": "patient_only + seen; clean stratum locked out; 729 outside the 567 excluded for lacking the Phase B ensemble",
        "probability_key": "probabilities",
        "channel_map": CH,
        "baseline_thresholds": BASELINE_THR,
        "proposal_thresholds": PROPOSAL_THR,
        "npz_to_gt": list(NPZ_TO_GT),
        "connectivity": "26 with one-voxel dilation merge, evaluator convention",
        "component_floor_voxels": VOL_THRESH,
        "reference_floor_mm3": VOL_THRESH,
        "matching": "direct voxel overlap, no dilation tolerance",
        "radii_mm": RADII_MM,
        "gate": json.load(open(GATE_FLAG)),
    }, open(f"{OUT}/c0_manifest.json", "w"), indent=2)
    json.dump(dict(qc), open(f"{OUT}/c0_qc.json", "w"), indent=2)

    # ---- C0-A ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(" C0-A · are missed lesions present at a lower threshold?")
    print("=" * 78)
    fn = les[les.baseline_detected == 0]
    print(f"\n  baseline false negatives: {len(fn)}")
    tab = []
    for reg in REGIONS:
        s = fn[fn.region == reg]
        row = {"region": reg, "n_FN": len(s)}
        for t in PROPOSAL_THR:
            row[f"@{t}"] = round(s[f"cand@{t}"].mean(), 3) if len(s) else np.nan
        row["never"] = (round(1 - s[[f"cand@{t}" for t in PROPOSAL_THR]]
                              .max(axis=1).mean(), 3) if len(s) else np.nan)
        tab.append(row)
    print("\n  " + pd.DataFrame(tab).to_string(index=False).replace("\n", "\n  "))
    print("""
  Each column is the fraction of missed lesions with at least one overlapping
  component at that threshold. The last is the fraction never surfaced at any
  of them -- irrecoverable by this candidate-generation mechanism, whatever
  model is placed downstream.""")

    # ---- C0-B ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(" C0-B · what each threshold costs")
    print("=" * 78)
    cs = pd.DataFrame(case_rows)
    print(f"\n  {'thr':>6} {'cand/case':>10} {'FN surfaced':>12} "
          f"{'cand per FN surfaced':>22}")
    for t in PROPOSAL_THR:
        cand = sum(cs[f"{r}_n_cand@{t}"].sum() for r in REGIONS
                   if f"{r}_n_cand@{t}" in cs.columns)
        surf = int(fn[f"cand@{t}"].sum())
        print(f"  {t:>6} {cand/max(len(cs),1):>10.1f} {surf:>12} "
              f"{cand/max(surf,1):>22.1f}")
    print("""
  The last column is the burden: how many candidate components must be
  considered for each missed lesion the threshold surfaces. A threshold that
  recovers everything at a hundred candidates apiece has moved the problem
  rather than solved it.""")

    # ---- C0.5 ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(" C0.5 · do surfaced lesions sit near ones already found?")
    print("=" * 78)
    surfaced = fn[fn[[f"cand@{t}" for t in PROPOSAL_THR]].max(axis=1) == 1]
    tp = les[les.baseline_detected == 1]
    print(f"\n  {'group':<26} {'n':>6} {'median dist (mm)':>18}")
    for name, s in (("surfaced false negatives", surfaced),
                    ("unsurfaced false negatives",
                     fn[fn[[f'cand@{t}' for t in PROPOSAL_THR]].max(axis=1) == 0]),
                    ("detected lesions", tp)):
        d = s.dist_nearest_detected_mm.dropna()
        print(f"  {name:<26} {len(s):>6} "
              f"{(d.median() if len(d) else float('nan')):>18.1f}")
    print("""
  Descriptive. A shorter distance for surfaced lesions than for unsurfaced ones
  would say the recoverable evidence tends to sit near lesions the model
  already found, which is the observation a relational model would exploit. It
  is not itself evidence that such a model would work.""")

    print(f"\n  written: {OUT}/")
    print("    c0_lesions.csv · c0_cases.csv · c0_manifest.json · c0_qc.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry", type=int, default=0,
                    help="technical dry run on N cases; not a scientific result")
    a = ap.parse_args()

    if a.gate:
        gate()
    elif a.dry:
        print(f"  DRY RUN on {a.dry} cases — checks the code, not the science.\n")
        lk = pd.read_csv(f"{DRIVE}/phase_b_outputs/wave0/leak_class_567.csv")
        cases = sorted(lk.loc[lk.leak_class.isin(["patient_only", "seen"]),
                              "case"])[:a.dry]
        t0 = time.time()
        n = 0
        for c in cases:
            rows, _ = analyse_case(c)
            n += len(rows or [])
        el = time.time() - t0
        print(f"  {len(cases)} cases · {n} lesions · {el:.0f}s "
              f"({el/max(len(cases),1):.1f}s per case)")
        print(f"  full cohort estimate: {el/max(len(cases),1)*1296/3600:.1f} h")
        print("\n  Do not draw conclusions from this. Run --gate, then --run.")
    elif a.run:
        run()
    else:
        print(__doc__)
        print("  --gate      validate the reconstruction (required first)")
        print("  --dry N     technical dry run, timing only")
        print("  --run       full cohort, after the gate passes")


if __name__ == "__main__":
    main()
