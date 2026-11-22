# -*- coding: utf-8 -*-
"""
==============================================================================
 CLEAN-STRATUM CEILING  —  the headroom of Figure 1 and Table "Oracle ceiling"
==============================================================================
 Added 22 November 2026, when this repository was prepared for publication.
 It computes nothing new. It writes to a table the per-region headroom that the
 manuscript reports for the clean stratum (n = 155), so that the four values
 used by figures/make_figures.py come from a file rather than from the
 fallback constants written into that script.

 Same arithmetic as compute_ceiling.py (protocol section 7):

     LesionWise_Dice = S / (N_gt + N_fp)
     ceiling         = S / N_gt             (oracle removes every FP, keeps every TP)
     headroom        = ceiling - baseline

 with N_gt = Num_TP + Num_FN and N_fp = Num_FP taken from the evaluator's own
 per-case counts. The only difference is the input: compute_ceiling.py reads
 the Phase A per-case table for all cases; this script reads the T4 run
 (t4_counts.w*of9.csv, baseline rows), which is the clean stratum scored with
 all fifteen evaluator columns.

 Expected output (checked when this file was written):
     WT 0.0395 · TC 0.0280 · ET 0.0247 · RC 0.2968
 i.e. 2.08, 1.47, 1.30 and 15.6 times the pre-registered MDE of 0.019.

 OUTPUT:  {BRATS_ROOT}/phase_b_outputs/wave2/ceiling_by_region_clean.csv
          {BRATS_ROOT}/phase_b_outputs/wave2/ceiling_per_case_clean.csv
==============================================================================
"""
import os
import glob

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W2 = f"{DRIVE}/phase_b_outputs/wave2"
REGIONS = ["WT", "TC", "ET", "RC"]
MDE = 0.019


def main():
    shards = sorted(glob.glob(f"{W2}/t4_counts.w*of9.csv"))
    assert len(shards) == 9, f"expected 9 T4 shards, found {len(shards)}"
    d = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    b = d[d.setting == "baseline"].copy()
    assert (b.status == "ok").all(), "a baseline row is not status == ok"
    assert b.case.nunique() == 155 and len(b) == 155 * 4, "clean stratum incomplete"

    b["n_gt"] = b.Num_TP + b.Num_FN
    b["n_den"] = b.n_gt + b.Num_FP
    b["S"] = b.LesionWise_Score_Dice * b.n_den
    b["ceiling"] = np.where(b.n_gt > 0, b.S / b.n_gt, b.LesionWise_Score_Dice)
    b["ceiling"] = b.ceiling.clip(upper=1.0)
    # no reference lesion: the whole gap is false-positive driven, ceiling 1.0
    b.loc[b.n_gt == 0, "ceiling"] = 1.0
    b["headroom"] = b.ceiling - b.LesionWise_Score_Dice

    g = (b.groupby("region")
          .agg(n_cases=("case", "nunique"),
               baseline=("LesionWise_Score_Dice", "mean"),
               ceiling=("ceiling", "mean"),
               headroom=("headroom", "mean"),
               n_gt=("n_gt", "sum"), n_fp=("Num_FP", "sum"))
          .reindex(REGIONS))
    g["headroom_over_mde"] = g.headroom / MDE
    print(g.round(4).to_string())

    g.round(6).to_csv(f"{W2}/ceiling_by_region_clean.csv")
    b[["case", "region", "Num_TP", "Num_FN", "Num_FP", "LesionWise_Score_Dice",
       "n_gt", "ceiling", "headroom"]].to_csv(
        f"{W2}/ceiling_per_case_clean.csv", index=False)
    print(f"\n  written to {W2}/ceiling_by_region_clean.csv")


if __name__ == "__main__":
    main()
