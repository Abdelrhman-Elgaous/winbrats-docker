# -*- coding: utf-8 -*-
"""
==============================================================================
 FALSE-NEGATIVE RECOVERY CEILING  —  the counterpart to D16
==============================================================================
 Read-only. Seconds. Derived from the frozen evaluator counts; no new
 experiment, no model, nothing recomputed.

 D16 asked how much a perfect false-positive remover could gain. This asks the
 mirror question: how much could a perfect false-negative recoverer gain?

 THE DERIVATION
 --------------
 The evaluator scores a region as

     Dice = S / (N_gt + N_fp)

 where S sums per-lesion Dice over matched reference lesions. An oracle that
 recovers every missed lesion perfectly turns each false negative into a
 matched lesion contributing 1.0 to S, leaving N_gt and N_fp untouched:

     Dice_oracle = (S + N_fn) / (N_gt + N_fp)
     headroom    = N_fn / (N_gt + N_fp)

 Note what this does not need: the baseline Dice cancels out. The ceiling
 follows from the counts alone.

 WHY THE CLEAN STRATUM AND NOT ALL 567
 -------------------------------------
 D16 was first computed on 567 cases and reported alongside deltas measured on
 155. The two populations differ enough that the mismatch produced an apparent
 RC gain exceeding its own ceiling. Every number here is therefore computed on
 the clean stratum, the same population as the B-6 deltas. The 567 figures are
 printed too, but only to show the size of the difference.

 THE DECISION RULE, FIXED BEFORE LOOKING
 ---------------------------------------
   FN headroom > 3x MDE in the tumour regions
       -> a false-negative direction has room worth pursuing
   FN headroom around 1x MDE or below
       -> it does not, and that is worth knowing before building anything

 This is a feasibility bound, not a prediction. A wide ceiling says an approach
 is not excluded by arithmetic; it says nothing about whether any method can
 reach it.
==============================================================================
"""

import os
import re
import glob

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W0, W2 = f"{DRIVE}/phase_b_outputs/wave0", f"{DRIVE}/phase_b_outputs/wave2"
PAT = re.compile(r"\.w\d+of\d+\.csv$")
REGIONS = ["WT", "TC", "ET", "RC"]
MDE = 0.019

# D16, clean stratum — the FP side, for the comparison that matters
FP_HEADROOM = {"WT": 0.0395, "TC": 0.0280, "ET": 0.0247, "RC": 0.2968}


def load_counts():
    """Baseline counts per case and region, from the frozen T4 tables."""
    files = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv") if PAT.search(f)]
    if not files:
        raise SystemExit(f"no T4 count shards under {W2}")
    t4 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bad = int((t4.status.astype(str) != "ok").sum())
    if bad:
        raise SystemExit(f"{bad} non-ok rows in the count table")
    return t4[t4.setting == "baseline"]


def clean_cases():
    lk = pd.read_csv(f"{W0}/leak_class_567.csv")
    return set(lk.loc[lk.leak_class == "clean", "case"])


def ceiling(df, label):
    rows = []
    for reg in REGIONS:
        r = df[df.region == reg]
        tp, fn, fp = float(r.Num_TP.sum()), float(r.Num_FN.sum()), float(r.Num_FP.sum())
        n_gt = tp + fn
        den = n_gt + fp
        head = fn / den if den else np.nan
        rows.append({"region": reg, "n_cases": r.case.nunique(),
                     "N_gt": int(n_gt), "TP": int(tp), "FN": int(fn), "FP": int(fp),
                     "fn_headroom": round(head, 4),
                     "x_MDE": round(head / MDE, 1) if den else np.nan})
    d = pd.DataFrame(rows)
    print(f"\n  --- {label} ---")
    print("  " + d.to_string(index=False).replace("\n", "\n  "))
    return d


def main():
    print("=" * 78)
    print(" FALSE-NEGATIVE RECOVERY CEILING")
    print(" derived from frozen counts · no experiment · no result changed")
    print("=" * 78)

    base = load_counts()
    clean = clean_cases()
    print(f"  cases: {base.case.nunique()} total · {len(clean)} clean")

    full = ceiling(base, "all cases in the count table")
    cl = ceiling(base[base.case.isin(clean)], "CLEAN STRATUM — the figure that decides")

    # ---- the comparison the decision rests on --------------------------
    print("\n" + "=" * 78)
    print(" THE TWO SIDES OF THE ERROR BUDGET — clean stratum")
    print("=" * 78)
    cmp_rows = []
    for _, r in cl.iterrows():
        reg = r.region
        fp_h = FP_HEADROOM.get(reg, np.nan)
        cmp_rows.append({
            "region": reg,
            "FP removal": round(fp_h, 4),
            "FP ÷MDE": round(fp_h / MDE, 1),
            "FN recovery": r.fn_headroom,
            "FN ÷MDE": r.x_MDE,
            "ratio FN/FP": round(r.fn_headroom / fp_h, 1) if fp_h else np.nan,
        })
    c = pd.DataFrame(cmp_rows)
    print("\n  " + c.to_string(index=False).replace("\n", "\n  "))

    # ---- the pre-stated decision rule ----------------------------------
    print("\n" + "=" * 78)
    print(" DECISION (rule fixed before the numbers were seen)")
    print("=" * 78)
    tum = c[c.region != "RC"]
    worst = float(tum["FN ÷MDE"].min())
    best = float(tum["FN ÷MDE"].max())
    print(f"\n  tumour-region FN headroom: {worst:.1f}× to {best:.1f}× MDE")

    if worst > 3:
        print(f"""
  ABOVE THE THRESHOLD. Every tumour region has a false-negative ceiling above
  3x the minimum detectable effect, so a method aimed at recovering missed
  lesions is not excluded by arithmetic the way false-positive removal was.

  What this does and does not license:
    · it does NOT say a structured method will work
    · it does NOT predict any effect size
    · it says only that the room exists, which is what the false-positive
      side turned out to lack""")
    elif best < 1.5:
        print("""
  BELOW THE THRESHOLD. The false-negative side is as constrained as the
  false-positive side. A method aimed at either is bounded before it starts,
  and that is worth knowing now rather than after building one.""")
    else:
        print("""
  MIXED. Some regions clear the threshold and others do not. Any method should
  be aimed at the regions that do, and the choice stated in advance rather than
  after seeing which one responded.""")

    # ---- what the ratio means ------------------------------------------
    print("\n" + "=" * 78)
    print(" READING THIS")
    print("=" * 78)
    print("""
  The two ceilings answer different questions and are not alternatives. False
  positives sit in the denominator; false negatives sit in the numerator. A
  region can be tightly bounded on one side and wide open on the other, and the
  ratio column shows how asymmetric each region's error budget is.

  A ceiling is an upper bound under a perfect oracle. Nothing here estimates
  what any actual method would achieve, and no method has been proposed, let
  alone tested. The figure exists to say which direction is not foreclosed.""")

    # ---- provenance ----------------------------------------------------
    print("\n" + "=" * 78)
    print(f"""  Source: baseline rows of the frozen T4 count tables, restricted to the
  clean stratum. FP headroom from D16, clean stratum. MDE = {MDE} as
  pre-registered at n = 155.

  Phase B is unchanged by this: no result was recomputed, no threshold moved,
  no test rerun. This is a bound derived from counts that already existed.""")


if __name__ == "__main__":
    main()
