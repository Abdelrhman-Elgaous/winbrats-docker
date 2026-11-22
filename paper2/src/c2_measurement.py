# -*- coding: utf-8 -*-
"""
==============================================================================
 C2 MEASUREMENT · lesion contrast within matched case-region strata
==============================================================================
 Implements C2_DESIGN_frozen.md literally. Nothing here is a choice: every
 parameter below was decided and written down before this file existed, and
 the reasons are in that document rather than repeated here.

     shell            dilate(lesion, 5) - dilate(lesion, 1)
     excluded         other reference lesions
                      predicted components OF THE SAME REGION   (decision A)
                      background (intensity zero)
     lesion value     median
     reference value  median
     denominator      1.4826 x MAD(shell)
     contrast         (median_lesion - median_shell) / denominator
     guard            MAD == 0 -> missing, counted
     endpoint         t1c inferentially; other sequences descriptive

 PROVENANCE, NOT REIMPLEMENTATION
 --------------------------------
 Predicted components come from load_mean_prob -> build_nested_label ->
 prediction_components, the path that passed the C0 reconstruction gate at
 620/620. Reference lesions come from the same merge-by-dilation construction
 C1 used. Neither is rewritten here. Rewriting component logic is how the
 false-negative definition diverged five times before the gate caught it; the
 fix was to call the validated function, not to guard against the divergence.

 The cohort is read from c2_cohort.json, fixed before any contrast value
 existed, so it cannot be chosen by what it produces.

 This script measures. It does not test, model, or select anything.
==============================================================================
"""

import os
import sys
import json
import glob
import time
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
C0 = f"{DRIVE}/phase_c_outputs/c0"
C2 = f"{DRIVE}/phase_c_outputs/c2"
COHORT = f"{C2}/c2_cohort.json"
OUT = f"{C2}/c2_contrast.csv"

REGIONS = ["WT", "TC", "ET", "RC"]
CH = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}
GT_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)}
SEQS = ["t1c", "t1n", "t2w", "t2f"]      # t1c first: the inferential endpoint

SHELL_OUTER = 5
SHELL_INNER = 1
MISSED_MAX_P = 0.001
DIL_FACTOR = 1
VOL_THRESH = 2
MAD_SCALE = 1.4826

STRUCT = ndimage.generate_binary_structure(3, 2)
INV = np.argsort(ns.NPZ_TO_GT)

os.makedirs(C2, exist_ok=True)


# ---------------------------------------------------------------------------
#  construction, taken from the validated path
# ---------------------------------------------------------------------------

def merge_by_dilation(mask):
    """As in the C0 gate: merged across a one-voxel dilation, no size filter."""
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32), np.array([]), np.array([])
    dil = ndimage.binary_dilation(mask, structure=STRUCT, iterations=DIL_FACTOR)
    lab, _ = ndimage.label(dil, structure=np.ones((3, 3, 3)))
    lab = (lab * mask).astype(np.int32)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    return lab, ids, counts


def prediction_components(mask):
    """Prediction path: components at or below two voxels removed, gaps kept."""
    lab, ids, counts = merge_by_dilation(mask)
    if not len(ids):
        return lab
    small = counts <= VOL_THRESH
    if small.any():
        lab[np.isin(lab, ids[small])] = 0
    return lab


def sequence_path(case, seq):
    h = (glob.glob(f"{DRIVE}/**/{case}-{seq}.nii.gz", recursive=True)
         or glob.glob(f"{DRIVE}/**/{case}_{seq}.nii.gz", recursive=True))
    return h[0] if h else None


# ---------------------------------------------------------------------------

def contrast(lesion_mask, shell_mask, vol):
    """
    The frozen measure. Returns the value and the pieces it came from, so a
    surprising number can be traced without recomputing it.
    """
    if not shell_mask.any() or not lesion_mask.any():
        return None
    a, b = vol[lesion_mask], vol[shell_mask]
    med_l, med_s = float(np.median(a)), float(np.median(b))
    mad = float(np.median(np.abs(b - med_s)))
    denom = MAD_SCALE * mad
    return {
        "median_lesion": med_l, "median_shell": med_s,
        "mad_shell": mad, "n_shell": int(shell_mask.sum()),
        # MAD of zero means a uniform shell, which happens in small or edge
        # lesions. Recorded as missing rather than infinite (frozen guard).
        "contrast": (med_l - med_s) / denom if denom > 0 else np.nan,
    }


def analyse_case(case, lesions_here):
    """Every lesion of interest in one case, across the four sequences."""
    mean = ns.load_mean_prob(case)
    lab_pred_all = ns.build_nested_label(mean, suppressor=None)

    img = nib.load(f"{ns.GT_DIR}/{case}.nii.gz")
    gt = np.asanyarray(img.dataobj).transpose(INV)
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])

    vols = {}
    for s in SEQS:
        p = sequence_path(case, s)
        if p is None:
            continue
        v = np.asanyarray(nib.load(p).dataobj)
        vols[s] = v.transpose(INV) if v.shape != gt.shape else v
    if "t1c" not in vols:
        return []

    rows = []
    for region, group in lesions_here.groupby("region"):
        gt_mask = np.isin(gt, GT_LABELS[region])
        gt_lab, gt_ids, _ = merge_by_dilation(gt_mask)

        # decision A: predictions of THIS region only
        pred_lab = prediction_components(np.isin(lab_pred_all,
                                                 GT_LABELS[region]))
        other_pred = pred_lab > 0

        for _, les in group.iterrows():
            gid = int(les.gt_lesion)
            sel = gt_lab == gid
            if not sel.any():
                continue

            outer = ndimage.binary_dilation(sel, structure=STRUCT,
                                            iterations=SHELL_OUTER)
            inner = ndimage.binary_dilation(sel, structure=STRUCT,
                                            iterations=SHELL_INNER)
            shell = outer & ~inner
            shell &= ~(gt_mask & ~sel)     # other reference lesions
            shell &= ~other_pred           # same-region predictions
            shell &= vols["t1c"] > 0       # background

            rec = {"case": case, "region": region, "gt_lesion": gid,
                   "gt_volume_mm3": float(les.gt_volume_mm3),
                   "group": les.group,
                   "n_shell_voxels": int(shell.sum()),
                   "voxel_mm3": round(sx * sy * sz, 6)}

            for s in SEQS:
                if s not in vols:
                    continue
                c = contrast(sel, shell, vols[s])
                if c is None:
                    rec[f"contrast_{s}"] = np.nan
                    continue
                rec[f"contrast_{s}"] = c["contrast"]
                if s == "t1c":
                    rec.update({f"t1c_{k}": v for k, v in c.items()
                                if k != "contrast"})
            rows.append(rec)

    del mean, lab_pred_all, gt, vols
    return rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", type=int, default=0)
    a = ap.parse_args()

    print("=" * 78)
    print(" C2 MEASUREMENT · frozen design, decision A")
    print("=" * 78)

    if not os.path.exists(COHORT):
        sys.exit(f"missing {COHORT} — run the coverage check first")
    cohort = json.load(open(COHORT))
    cases = cohort["cases"]

    les = pd.read_csv(f"{C0}/c0_lesions.csv")
    det = les[les.baseline_detected == 1].copy()
    mis = les[(les.baseline_detected == 0) &
              (les.max_prob_in_lesion <= MISSED_MAX_P)].copy()
    det["group"], mis["group"] = "detected", "missed"

    # only strata containing both, as in C1
    keys = ["case", "region"]
    both = set(map(tuple, det[keys].values)) & set(map(tuple, mis[keys].values))
    allles = pd.concat([det, mis])
    allles = allles[[tuple(x) in both for x in allles[keys].values]]
    allles = allles[allles.case.isin(cases)]

    print(f"  cohort            : {len(cases)} cases (from the manifest)")
    print(f"  matched strata    : {len(both)}")
    print(f"  lesions to measure: {len(allles)}"
          f"   ({int((allles.group=='detected').sum())} detected, "
          f"{int((allles.group=='missed').sum())} missed)")
    print(f"  shell             : {SHELL_INNER}->{SHELL_OUTER} voxels, "
          f"same-region predictions excluded (A)")

    todo = sorted(allles.case.unique())
    if a.dry:
        todo = todo[:a.dry]
        print(f"\n  DRY RUN on {len(todo)} cases — timing only, not a result\n")

    rows, t0, failed = [], time.time(), 0
    for i, c in enumerate(todo, 1):
        try:
            rows += analyse_case(c, allles[allles.case == c])
        except Exception as e:
            failed += 1
            print(f"    {c}: {type(e).__name__}: {e}")
        if i % 10 == 0 or i == len(todo):
            if not a.dry:
                pd.DataFrame(rows).to_csv(OUT, index=False)
            el = time.time() - t0
            print(f"    {i}/{len(todo)}  {len(rows)} lesions  "
                  f"{el/60:.0f}m  eta {el/i*(len(todo)-i)/60:.0f}m")

    d = pd.DataFrame(rows)
    if a.dry:
        print(f"\n  {len(todo)} cases · {len(d)} lesions · "
              f"{(time.time()-t0)/max(len(todo),1):.1f}s per case")
        print(f"  full cohort estimate: "
              f"{(time.time()-t0)/max(len(todo),1)*len(cases)/3600:.1f} h")
        print("\n  Nothing concluded from a dry run.")
        return

    d.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print(" MEASUREMENT COMPLETE — no test performed")
    print("=" * 78)
    print(f"  lesions measured : {len(d)}")
    print(f"  cases failed     : {failed}")
    n_missing = int(d.contrast_t1c.isna().sum())
    print(f"  contrast missing : {n_missing}"
          f"   ({n_missing/max(len(d),1):.1%}, MAD = 0 in the shell)")
    print(f"  shell size       : median {d.n_shell_voxels.median():.0f} voxels"
          f"   (min {d.n_shell_voxels.min():.0f})")

    ok = d.dropna(subset=["contrast_t1c"])
    print(f"\n  t1c contrast, by group (descriptive; the test is separate):")
    print("  " + ok.groupby("group").contrast_t1c
          .describe()[["count", "25%", "50%", "75%"]]
          .round(3).to_string().replace("\n", "\n  "))

    print(f"""
  written: {OUT}

  Next: the analysis, which applies the frozen model —
    detected ~ contrast_t1c + log10(volume) + stratum
  and the volume-matched Wilcoxon check. Both were specified in advance; the
  values above are printed for sanity only and are not the result.""")


if __name__ == "__main__":
    main()
