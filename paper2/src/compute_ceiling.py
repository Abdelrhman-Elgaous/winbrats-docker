# -*- coding: utf-8 -*-
"""
==============================================================================
 GAP DECOMPOSITION AND THEORETICAL CEILING
==============================================================================
 RUNTIME: CPU, seconds. No dependency on B-4.
 PROTOCOL: PHASE_B_PROTOCOL.md v1.0, section 7 --
   "تحليل الفجوة لمكوناتها + السقف النظري قبل تفسير أي Δ"

 WHY THIS RUNS BEFORE B-6 IS READ
 --------------------------------
 The official lesion-wise Dice for one case and region is

     LesionWise_Dice  =  S / (N_gt + N_fp)

 where S is the sum of per-lesion Dice over MATCHED reference lesions, every
 unmatched reference lesion contributes 0 to S, and every false positive adds
 1 to the denominator without adding anything to S.

 Suppression can only ever remove false positives. So the best any suppression
 rule could possibly achieve -- an oracle that deletes every FP and no TP -- is

     ceiling  =  S / N_gt        i.e.   current x (N_gt + N_fp) / N_gt

 That is a HARD upper bound. If the ceiling sits close to the baseline, then a
 near-zero delta from B-4 is the arithmetically expected outcome and not a
 failure of the hypothesis. Knowing this before reading B-6 is the difference
 between reporting "uncertainty does not help" and reporting "there was almost
 nothing available for it to win."

 THE FLOOR ASYMMETRY -- READ THIS
 --------------------------------
 In metrics_MET.py the two counts are NOT filtered the same way:

     'Num_TP': len(gt_tp) - gt_tp_sub      <- subtraction applied
     'Num_FN': len(fn)   - fn_sub          <- subtraction applied
     'Num_FP': len(fp)                     <- NO subtraction

 Reference lesions below the scoring floor are excluded; predicted false
 positive components appear to be counted whatever their size. This script
 measures the consequence rather than assuming it, because it bears directly
 on D6 and on the removal of size@20 (see the D6 section of the report).

 OUTPUT
 ------
   ceiling_by_region.csv · ceiling_per_case.csv · CEILING_REPORT.md
==============================================================================
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

DRIVE   = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
OUT_DIR = f"{DRIVE}/phase_b_outputs/wave2"

REGIONS = ["WT", "TC", "ET", "RC"]
# The evaluator's actual floor: metrics_MET.py line 681, inside the BraTS-MET
# branch. Applied symmetrically to predicted components (line 319) and to
# reference lesions (743, 747, 761). This was 27 until 16 Aug, an undocumented
# assumption that entered the project through a comment in size_pilot.py and was
# never an evaluator value. The ceiling itself is unaffected -- it is computed
# from the evaluator's own per-case counts, not from this constant.
FLOOR_VOX = 2

# HISTORICAL AUDIT VARIABLE -- NOT AN EVALUATOR VALUE.
#
# 27 is permitted here for one purpose only: reproducing the descriptive size
# breakdown this project produced while it believed the floor was 27, so the
# historical analysis stays inspectable. It is forbidden as a current
# assumption about how anything is scored.
#
# ⛔ NOTHING DERIVED FROM THIS CONSTANT MAY ENTER THE PAPER. Every column it
# feeds is labelled "(legacy)" in the output for exactly that reason. The
# ceiling, the headroom and every per-region figure are computed from the
# evaluator's own per-case counts and never touch it.
LEGACY_ASSUMED_FLOOR = 27


def find(name):
    hits = glob.glob(f"{DRIVE}/**/{name}", recursive=True)
    if not hits:
        sys.exit(f"could not find {name} under {DRIVE}")
    if len(hits) > 1:
        print(f"  [!] {len(hits)} copies of {name}; using {hits[0]}")
    return hits[0]


def load(config):
    d10 = pd.read_csv(find("day10_lesion_characterization.csv"))
    d11 = pd.read_csv(find("day11_fp_lesions.csv"))
    pc = pd.read_csv(find("per_case_lesionwise.csv"))

    if "config" in pc.columns:
        if config not in set(pc.config):
            sys.exit(f"config '{config}' not in {sorted(set(pc.config))}")
        pc = pc[pc.config == config]

    print(f"  day10 : {len(d10):>6} GT lesions   · {d10.case.nunique()} cases")
    print(f"  day11 : {len(d11):>6} FP lesions   · {d11.case.nunique()} cases")
    print(f"  per-case: {pc.case.nunique()} cases · config '{config}'")
    return d10, d11, pc


def provenance(d10, d11, pc):
    """
    Check that the lesion tables describe the same run as the per-case scores.
    They were produced on different days, and if they disagree the ceiling is
    computed from mismatched numerator and denominator.
    """
    print("\n  --- provenance check: lesion tables vs per-case scores ---")
    d10s = d10[d10.size_vox > FLOOR_VOX]
    rows = []
    for reg in REGIONS:
        p = pc[pc.region == reg]
        rows.append({
            "region": reg,
            "TP_lesion_tbl": int(((d10s.region == reg) & (d10s.status == "TP")).sum()),
            "TP_per_case":   int(p.num_tp.sum()),
            "FN_lesion_tbl": int(((d10s.region == reg) & (d10s.status == "FN")).sum()),
            "FN_per_case":   int(p.num_fn.sum()),
            "FP_lesion_tbl": int((d11.region == reg).sum()),
            "FP_per_case":   int(p.num_fp.sum()),
        })
    chk = pd.DataFrame(rows)
    for k in ("TP", "FN", "FP"):
        chk[f"{k}_delta"] = chk[f"{k}_lesion_tbl"] - chk[f"{k}_per_case"]
    print(chk.to_string(index=False))

    worst = chk[[f"{k}_delta" for k in ("TP", "FN", "FP")]].abs().max().max()
    if worst > 5:
        print(f"\n  [!] largest disagreement is {worst} lesions. The lesion tables and "
              f"\n      the per-case scores may come from different runs or configs. "
              f"\n      Try --config with the other value before trusting the ceiling.")
    else:
        print(f"\n  agreement within {worst} lesion(s) — tables describe the same run.")
    return chk


def ceiling(pc):
    """
    Per case and region:  S = dice x (N_gt + N_fp),  ceiling = S / N_gt.

    N_gt is taken from the EVALUATOR's own counts (num_tp + num_fn), not from
    the lesion table, so numerator and denominator come from one source.
    """
    d = pc.copy()
    d["n_gt"] = d.num_tp + d.num_fn
    d["n_den"] = d.n_gt + d.num_fp
    d["S"] = d.lesionwise_dice * d.n_den
    d["ceiling"] = np.where(d.n_gt > 0, d.S / d.n_gt, d.lesionwise_dice)
    d["ceiling"] = d.ceiling.clip(upper=1.0)
    d["headroom"] = d.ceiling - d.lesionwise_dice

    # Cases with no reference lesion score 1.0 when nothing is predicted and
    # 0.0 when something is. For those the whole gap is FP-driven, so the
    # ceiling is 1.0 by construction.
    no_gt = d.n_gt == 0
    d.loc[no_gt, "ceiling"] = 1.0
    d.loc[no_gt, "headroom"] = 1.0 - d.loc[no_gt, "lesionwise_dice"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="3fold_124",
                    help="per_case_lesionwise config (default: the reported arm)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print(" GAP DECOMPOSITION AND THEORETICAL CEILING")
    print(" protocol section 7 — run BEFORE interpreting any B-6 delta")
    print("=" * 78)

    d10, d11, pc = load(a.config)
    chk = provenance(d10, d11, pc)
    per_case = ceiling(pc)

    # ---- ceiling by region ------------------------------------------------
    print("\n" + "=" * 78)
    print(" THEORETICAL CEILING — oracle removes every FP, keeps every TP")
    print("=" * 78)
    g = (per_case.groupby("region")
         .agg(n_cases=("case", "nunique"),
              baseline=("lesionwise_dice", "mean"),
              ceiling=("ceiling", "mean"),
              headroom=("headroom", "mean"),
              n_gt=("n_gt", "sum"), n_fp=("num_fp", "sum"))
         .reindex(REGIONS).round(4))
    g["fp_per_gt"] = (g.n_fp / g.n_gt).round(3)
    print(g.to_string())

    print("\n  headroom = the ENTIRE budget available to any suppression rule.")
    print("  A B-6 delta must be read against this column, not against zero.")

    # ---- the FN wall ------------------------------------------------------
    print("\n" + "=" * 78)
    print(" WHAT THE CEILING DOES NOT REACH")
    print("=" * 78)
    d10s = d10[d10.size_vox > FLOOR_VOX]
    fn = d10s[d10s.status == "FN"]
    print(f"  Scored reference lesions missed entirely (FN): {len(fn)}")
    print("  Suppression cannot recover a single one of these — it only ever")
    print("  removes predictions. They are outside the ceiling by construction.")
    print()
    print(d10s.groupby(["region", "status"]).size().unstack(fill_value=0).to_string())

    # ---- the floor asymmetry, and what it means for D6 --------------------
    print("\n" + "=" * 78)
    print(" FLOOR ASYMMETRY — bears on D6 and on the removal of size@20")
    print("=" * 78)
    small_gt = (d10.size_vox < LEGACY_ASSUMED_FLOOR).sum()
    small_fp = (d11.size_vox < LEGACY_ASSUMED_FLOOR).sum()
    print(f"  reference lesions below {LEGACY_ASSUMED_FLOOR} vox (legacy) : "
          f"{small_gt} / {len(d10)} ({small_gt/len(d10):.1%})  -> excluded by evaluator")
    print(f"  FP components below {LEGACY_ASSUMED_FLOOR} vox (legacy) : "
          f"{small_fp} / {len(d11)} ({small_fp/len(d11):.1%})  -> counted? see below")

    fp_small_by_reg = (d11.assign(small=d11.size_vox < LEGACY_ASSUMED_FLOOR)
                       .groupby("region")["small"].agg(["sum", "count"]))
    fp_small_by_reg.columns = ["fp_below_floor", "fp_total"]
    fp_small_by_reg["per_case_num_fp"] = pc.groupby("region").num_fp.sum()
    print()
    print(fp_small_by_reg.reindex(REGIONS).to_string())

    matches = (fp_small_by_reg["fp_total"] - fp_small_by_reg["per_case_num_fp"]).abs()
    if (matches <= 2).all():
        print("\n  The evaluator's FP count matches the FULL lesion-table count.")
        print()
        print("  [SUPERSEDED 16 Aug] This block once argued that small FPs escape")
        print("  the floor while reference lesions do not. Direct reading of")
        print("  metrics_MET.py settled it: lesion_volume_thresh = 2 VOXELS")
        print("  (line 681), applied to predicted components (319) and to")
        print("  reference lesions (743, 747, 761) alike. The floor is symmetric")
        print("  and it is 2 -- not the 27 this project assumed for months.")
        print("  Protocol section 13 carries the full correction.")
        print(f"  For reference, {small_fp} FP components sit below 27 voxels;")
        print("  that count is descriptive only and implies nothing about scoring.")
    else:
        print("\n  The evaluator's FP count is BELOW the lesion-table count, so some")
        print("  small components are filtered somewhere. D6 stands. Difference per")
        print("  region:")
        print(matches.reindex(REGIONS).to_string())

    # ---- headroom distribution -------------------------------------------
    print("\n" + "=" * 78)
    print(" WHERE THE HEADROOM ACTUALLY IS")
    print("=" * 78)
    for reg in REGIONS:
        r = per_case[per_case.region == reg]
        if not len(r):
            continue
        share = (r.headroom > 0.01).mean()
        top = r.nlargest(5, "headroom")[["case", "lesionwise_dice", "ceiling", "num_fp"]]
        print(f"\n  --- {reg} --- cases with headroom > 0.01: "
              f"{(r.headroom > 0.01).sum()} / {len(r)} ({share:.1%})")
        print(top.to_string(index=False))

    if a.no_write:
        print("\n  --no-write: nothing saved.")
        return

    # The written artefacts are what a paper would read, so assert here that
    # nothing legacy reached them. A comment can be ignored; this cannot.
    assert FLOOR_VOX == 2, "FLOOR_VOX must be the evaluator's value (2)"
    assert not any("legacy" in str(c).lower() for c in g.columns), \
        "legacy-floor column leaked into the per-region table"
    assert not any("legacy" in str(c).lower() for c in per_case.columns), \
        "legacy-floor column leaked into the per-case table"

    os.makedirs(OUT_DIR, exist_ok=True)
    g.to_csv(f"{OUT_DIR}/ceiling_by_region.csv")
    per_case.to_csv(f"{OUT_DIR}/ceiling_per_case.csv", index=False)
    chk.to_csv(f"{OUT_DIR}/ceiling_provenance.csv", index=False)

    md = ["# Gap decomposition and theoretical ceiling", "",
          f"**Config:** `{a.config}` · protocol section 7", "",
          "`LesionWise_Dice = S / (N_gt + N_fp)`. Suppression removes only false "
          "positives, so the oracle bound is `S / N_gt`.", "",
          "## Ceiling by region", "", g.to_markdown(), "",
          "`headroom` is the entire budget available to any suppression rule. "
          "A B-6 delta must be read against it, not against zero.", "",
          "## Provenance", "", chk.to_markdown(index=False), "",
          "## Floor asymmetry — LEGACY (27-voxel assumption, not for the paper)", "",
          f"- reference lesions below {LEGACY_ASSUMED_FLOOR} vox (legacy): {small_gt} / {len(d10)}",
          f"- FP components below {LEGACY_ASSUMED_FLOOR} vox (legacy): {small_fp} / {len(d11)}", "",
          fp_small_by_reg.reindex(REGIONS).to_markdown(), ""]
    open(f"{OUT_DIR}/CEILING_REPORT.md", "w", encoding="utf-8").write("\n".join(md))
    print(f"\n  written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
