# -*- coding: utf-8 -*-
"""
==============================================================================
 T3/T4 SCHEMA AND PROVENANCE SMOKE TEST  —  ONE CASE, READ-ONLY
==============================================================================
 RUNTIME: CPU, a few minutes. Writes nothing except a scratch file it deletes.
 Touches no B-4 artefact. B-4 is complete and frozen.

 WHY THIS RUNS BEFORE ANY RUNNER IS WRITTEN
 ------------------------------------------
 The T4 run is ~231 h of compute (1,085 pairs on the clean stratum). Three
 things must be true before committing to it, and all three are cheap to check
 on a single case:

   1. get_LesionWiseResults returns BOTH tables, with all 15 aggregate columns
      and all 9 per-lesion columns.
   2. baseline and a suppressed arm can be matched unambiguously for the same
      case and region -- otherwise no paired delta can be formed at all.
   3. THE DECIDING CHECK: does the per-lesion table reproduce the evaluator's
      own Num_TP / Num_FP / Num_FN after the same size filter is applied?

 Check 3 is what makes T3 possible or impossible, and it is the one that would
 be most expensive to discover late.

 WHY CHECK 3 MATTERS
 -------------------
 T4 (detection F1) needs counts per REGION, and results_df supplies them
 directly -- that is the source of truth and nothing is inferred.

 T3 (per-size-bin) needs counts per SIZE BIN, which results_df cannot give:
 it has no size axis. The only route is the per-lesion table, classifying a
 reference lesion as detected via `_len > 0` and unmatched via `_len == 0`.

 But that classification is an INFERENCE, not a measurement. The evaluator
 filters on gt_lesion_vol before counting (metric_df_thresh, line 761), so
 counting raw `_len` need not agree with Num_TP/Num_FN. Assuming the two agree
 is exactly the class of assumption that made D6 wrong for four months.

 So: reproduce the counts from the per-lesion table under the same filter and
 compare. Agreement licenses T3. Disagreement means per-size-bin detection is
 NOT obtainable from these tables, and we need to know that before spending
 two days of compute, not after.

 OUTPUT
 ------
 Printed report only. Nothing is written to the Drive.
==============================================================================
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

ROOT = ns.DRIVE

# A clean-stratum case with lesions in every region, chosen for coverage, not
# for its scores. Override on the command line if it is unrepresentative.
CASE = sys.argv[1] if len(sys.argv) > 1 else None

AGG_COLS = [
    "Labels", "Num_TP", "Num_FP", "Num_FN", "Sensitivity", "Specificity",
    "Legacy_Dice", "Legacy_HD95", "Legacy NSD @ 0.5", "Legacy NSD @ 1.0",
    "GT_Complete_Volume", "LesionWise_Score_Dice", "LesionWise_Score_HD95",
    "LesionWise_Score_NSD @ 0.5", "LesionWise_Score_NSD @ 1.0",
]
LES_COLS = [
    "predicted_lesion_numbers", "gt_lesion_numbers", "gt_lesion_vol",
    "dice_lesionwise", "hd95_lesionwise", "nsd05_lesionwise",
    "nsd10_lesionwise", "_len", "Label",
]

fails = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(label)


def score(case, suppressor, tag):
    """
    Call the evaluator directly and keep BOTH returned tables.

    official_score() currently discards the second one (`res_df, _ = ...`) and
    keeps four of the fifteen columns of the first. This test deliberately does
    not use it: the point is to see what the evaluator actually returns before
    deciding how official_score should change.
    """
    gt = f"{ns.GT_DIR}/{case}.nii.gz"
    mean = ns.load_mean_prob(case)
    lab = ns.build_nested_label(mean, suppressor=suppressor)
    del mean
    img, _ = ns.to_native_nifti(lab, gt)
    p = f"{ns.WORK_DIR}/schema_smoke_{tag}_{case}.nii.gz"
    nib.save(img, p)
    try:
        agg, les = ns.ensure_evaluator()(pred_file=p, gt_file=gt,
                                         challenge_name="BraTS-MET", output=None)
    finally:
        if os.path.exists(p):
            os.remove(p)
    return agg, les


def pick_case():
    """A clean-stratum case with reference lesions in all four regions."""
    lk = pd.read_csv(f"{ROOT}/phase_b_outputs/wave0/leak_class_567.csv")
    clean = set(lk[lk.leak_class == "clean"].case)
    pc = pd.read_csv(glob.glob(f"{ROOT}/**/per_case_lesionwise.csv",
                               recursive=True)[0])
    pc = pc[(pc.config == "3fold_124") & pc.case.isin(clean)]
    pc = pc.assign(n_gt=pc.num_tp + pc.num_fn)
    wide = pc.pivot_table(index="case", columns="region", values="n_gt")
    full = wide[(wide > 0).all(axis=1)]
    if not len(full):
        # no case has all four; fall back to the most regions covered
        wide["k"] = (wide > 0).sum(axis=1)
        return wide.sort_values("k", ascending=False).index[0]
    # among those, the cheapest, so the test stays quick
    sec = (pd.read_csv(f"{ROOT}/phase_b_outputs/wave1/exp3_suppressed_v3.csv")
             .groupby("case").seconds.first())
    return sec[sec.index.isin(full.index)].idxmin()


def main():
    print("=" * 74)
    print(" T3/T4 SCHEMA AND PROVENANCE SMOKE TEST — one case, read-only")
    print("=" * 74)

    os.environ["BRATS_FAST_METRIC"] = "1"
    ns.ensure_evaluator()

    case = CASE or pick_case()
    print(f"  case: {case}\n")

    # ---- 1. both tables, all columns --------------------------------------
    print("--- 1. schema ---")
    agg_b, les_b = score(case, None, "base")

    check(isinstance(agg_b, pd.DataFrame) and isinstance(les_b, pd.DataFrame),
          "two DataFrames returned")
    miss_a = [c for c in AGG_COLS if c not in agg_b.columns]
    check(not miss_a, f"aggregate has all 15 columns", f"missing {miss_a}" if miss_a else "")
    miss_l = [c for c in LES_COLS if c not in les_b.columns]
    check(not miss_l, f"per-lesion has all 9 columns", f"missing {miss_l}" if miss_l else "")
    check(len(agg_b) == 4, "aggregate has one row per region", f"rows={len(agg_b)}")
    check(len(les_b) > 0, "per-lesion table is non-empty", f"rows={len(les_b)}")

    print("\n  aggregate:")
    print("   ", agg_b[["Labels", "Num_TP", "Num_FP", "Num_FN",
                        "LesionWise_Score_Dice"]].to_string(index=False).replace("\n", "\n    "))
    print("\n  per-lesion (first 6):")
    print("   ", les_b[["Label", "gt_lesion_numbers", "gt_lesion_vol",
                        "dice_lesionwise", "_len"]].head(6)
            .to_string(index=False).replace("\n", "\n    "))

    # ---- 2. pairing a suppressed arm ---------------------------------------
    print("\n--- 2. pairing baseline with a suppressed arm ---")
    agg_s, les_s = score(case, ns.entropy_suppressor(0.50), "e50")

    check(set(agg_b.Labels) == set(agg_s.Labels),
          "same region set in both arms")
    check(agg_b.Labels.duplicated().sum() == 0,
          "region is a unique key in the aggregate table")
    gt_ids_b = set(zip(les_b.Label, les_b.gt_lesion_numbers))
    gt_ids_s = set(zip(les_s.Label, les_s.gt_lesion_numbers))
    check(gt_ids_b == gt_ids_s,
          "same (region, gt_lesion) keys in both arms",
          f"b\\s={len(gt_ids_b - gt_ids_s)} s\\b={len(gt_ids_s - gt_ids_b)}")
    check(les_b.duplicated(["Label", "gt_lesion_numbers"]).sum() == 0,
          "(region, gt_lesion) is unique within an arm")

    vol_b = les_b.set_index(["Label", "gt_lesion_numbers"]).gt_lesion_vol
    vol_s = les_s.set_index(["Label", "gt_lesion_numbers"]).gt_lesion_vol
    same_vol = vol_b.reindex(vol_s.index).eq(vol_s).all()
    check(bool(same_vol), "gt_lesion_vol is identical across arms",
          "(it must be: suppression changes predictions, never the reference)")

    # ---- 3. THE DECIDING CHECK --------------------------------------------
    print("\n--- 3. can the per-lesion table reproduce the evaluator's counts? ---")
    print("      (this is what decides whether T3 per-size-bin is possible)")

    src = f"{ns.EVAL_REPO}/metrics_MET.py"
    txt = open(src).read()
    line = next((l for l in txt.splitlines() if "metric_df_thresh =" in l), "")
    print(f"\n      filter in evaluator: {line.strip()[:100]}")

    # Try the plausible floors; report which, if any, reproduces the counts.
    print(f"\n      {'floor':>6}  {'region':>6}  {'TP les':>7} {'TP eval':>7}  "
          f"{'FN les':>7} {'FN eval':>7}  match")
    best = None
    for floor in (0, 5, 20, 27, 50):
        ok_all = True
        rows = []
        for _, r in agg_b.iterrows():
            reg = str(r["Labels"]).strip()
            sub = les_b[les_b.Label.astype(str).str.strip() == reg]
            sub = sub[sub.gt_lesion_vol >= floor]
            tp_l = int((sub._len > 0).sum())
            fn_l = int((sub._len == 0).sum())
            tp_e, fn_e = int(r["Num_TP"]), int(r["Num_FN"])
            m = (tp_l == tp_e) and (fn_l == fn_e)
            ok_all &= m
            rows.append((floor, reg, tp_l, tp_e, fn_l, fn_e, m))
        for f_, reg, a, b, c, d, m in rows:
            print(f"      {f_:>6}  {reg:>6}  {a:>7} {b:>7}  {c:>7} {d:>7}  "
                  f"{'ok' if m else 'NO'}")
        if ok_all and best is None:
            best = floor
        print()

    check(best is not None,
          "per-lesion counts reproduce Num_TP / Num_FN at some floor",
          f"floor = {best}" if best is not None else "no floor reproduces them")

    # FP has no per-lesion representation: the table is indexed by reference
    # lesion, and a false positive has no reference lesion to sit against.
    fp_total = int(agg_b.Num_FP.sum())
    print(f"      note: Num_FP totals {fp_total} and has NO per-lesion row by "
          f"construction —\n            the table is keyed on reference lesions, so an FP "
          f"has nothing to attach to.")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 74)
    print(" VERDICT")
    print("=" * 74)
    if fails:
        print(f"  {len(fails)} check(s) failed:")
        for f_ in fails:
            print(f"    - {f_}")
        print("\n  Do NOT write the T4 runner yet.")
        return

    print("  All checks passed.")
    print("\n  T4 (detection F1 per region): SUPPORTED.")
    print("    Num_TP / Num_FP / Num_FN come straight from the aggregate table.")
    print("    Nothing is inferred; that table is the source of truth.")
    if best is not None:
        print(f"\n  T3 (per-size-bin): SUPPORTED, with the floor set to {best}.")
        print("    The per-lesion table reproduces the evaluator's own counts under")
        print("    that filter, so classifying a reference lesion by `_len` is")
        print("    verified rather than assumed.")
        print("\n    CAVEAT: per-size-bin DETECTION is one-sided. Recall can be")
        print("    binned by lesion size because every reference lesion has a size;")
        print("    precision cannot, because false positives have no reference")
        print("    lesion and therefore no bin. Report per-bin recall, or bin FPs by")
        print("    their own predicted size and state that it is a different axis.")
    print("\n  Next: patch native_space.official_score to return and persist both")
    print("  tables and all 15 aggregate columns, record it in the protocol change")
    print("  log, then write the T4 runner.")


if __name__ == "__main__":
    main()
