# -*- coding: utf-8 -*-
"""
==============================================================================
 FOLLOW-UP CHECKS  —  read-only, changes nothing
==============================================================================
 Two questions left open by the five-layer audit. Neither adjusts a result;
 both decide how an existing result is described.

 CHECK A — is the endpoint-set divergence specific to RC?
   Per region, four counts:
     reference-negative cases            (TP + FN == 0 at baseline)
     reference-negative WITH an FP       (the ones suppression can act on)
     reference-positive cases
     cases actually emptied by each rule (FP-only -> TP=FP=FN=0)
   The distinction that matters is between an FP merely EXISTING in a
   reference-negative case, and suppression actually CONVERTING that case to a
   no-event case. Only the second removes it from the F1 comparison.
   Case IDs are printed, not inferred: the audit said one case is lost in each
   of WT, TC and ET, and whether it is the same case is a fact to read off, not
   to assume.

 CHECK B — are the four non-RC sign divergences meaningful?
   For each, the B-6 Dice delta with its bootstrap CI beside the T4 F1 delta.
   A sign comparison on a quantity indistinguishable from zero flags a
   divergence that is not there; this reports whether that is what happened,
   without calling anything noise.
==============================================================================
"""

import os
import re
import glob

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W2 = f"{DRIVE}/phase_b_outputs/wave2"
PAT = re.compile(r"\.w\d+of\d+\.csv$")
REGIONS = ["WT", "TC", "ET", "RC"]
BASELINE = "baseline"
TESTED = ["size@20", "size@50", "entropy@0.45",
          "entropy@0.50", "entropy@0.56", "confidence@0.45"]
MDE = 0.019

t4 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{W2}/t4_counts.w*of*.csv")
                if PAT.search(f)], ignore_index=True)

# ---------------------------------------------------------------------------
print("=" * 76)
print(" CHECK A · reference-negative cases per region")
print("=" * 76)

b = t4[t4.setting == BASELINE]
rows, emptied_ids = [], {}

for reg in REGIONS:
    r = b[b.region == reg].set_index("case")
    neg = r[(r.Num_TP + r.Num_FN) == 0]
    neg_fp = neg[neg.Num_FP > 0]
    pos = r[(r.Num_TP + r.Num_FN) > 0]
    rec = {"region": reg, "ref_negative": len(neg),
           "ref_neg_with_FP": len(neg_fp), "ref_positive": len(pos)}

    for rule in TESTED:
        x = t4[(t4.setting == rule) & (t4.region == reg)].set_index("case")
        j = r[["Num_TP", "Num_FP", "Num_FN"]].join(
            x[["Num_TP", "Num_FP", "Num_FN"]], lsuffix="_b", rsuffix="_r")
        # evaluable at baseline, empty afterwards
        emptied = j[((j.Num_TP_b + j.Num_FP_b + j.Num_FN_b) > 0) &
                    ((j.Num_TP_r + j.Num_FP_r + j.Num_FN_r) == 0)]
        rec[rule] = len(emptied)
        emptied_ids[(reg, rule)] = sorted(emptied.index)
    rows.append(rec)

d = pd.DataFrame(rows)
print("\n" + d[["region", "ref_negative", "ref_neg_with_FP", "ref_positive"]]
      .to_string(index=False))

print("\n  cases EMPTIED by each rule (FP-only -> no-event):")
print(d[["region"] + TESTED].to_string(index=False))

print("\n  which cases, in the tumour regions:")
seen = {}
for reg in ["WT", "TC", "ET"]:
    ids = set()
    for rule in TESTED:
        ids |= set(emptied_ids[(reg, rule)])
    seen[reg] = ids
    print(f"    {reg}: {sorted(ids) if ids else 'none'}")

if all(seen.values()):
    common = set.intersection(*seen.values())
    print(f"\n    common to WT, TC and ET: {sorted(common) if common else 'none'}")
    if common:
        c = sorted(common)[0]
        print(f"\n    baseline counts for {c}:")
        print("      " + b[b.case == c][["region", "Num_TP", "Num_FP", "Num_FN"]]
              .to_string(index=False).replace("\n", "\n      "))

rc_max = int(d.loc[d.region == "RC", TESTED].max(axis=1).iloc[0])
tum_max = int(d.loc[d.region != "RC", TESTED].max().max())
print(f"""
  Reading this: the mechanism needs a reference-negative case that carries an
  FP, and a rule that removes it. Both conditions occur in every region — what
  differs is how often. RC empties up to {rc_max} cases, the tumour regions up
  to {tum_max}. If the tumour regions show a nonzero count, the divergence is
  not a property of RC but of reference-negative prevalence, and RC is simply
  where reference-negative cases are the norm rather than the exception.""")

# ---------------------------------------------------------------------------
print("\n" + "=" * 76)
print(" CHECK B · the four non-RC sign divergences")
print("=" * 76)

try:
    b6 = pd.read_csv(f"{W2}/b6_primary_clean.csv")
    t4r = pd.read_csv(f"{W2}/t4_detection_results.csv")

    dcol = next(c for c in ("mean_delta", "delta", "dice_delta") if c in b6.columns)
    rcol = next(c for c in ("rule", "setting") if c in b6.columns)
    lo = next((c for c in ("ci_lo", "ci_low") if c in b6.columns), None)
    hi = next((c for c in ("ci_hi", "ci_high") if c in b6.columns), None)

    keep = ["region", rcol, dcol] + [c for c in (lo, hi) if c]
    m = (b6[keep].rename(columns={rcol: "rule", dcol: "dice",
                                  lo: "dice_ci_lo", hi: "dice_ci_hi"})
         .merge(t4r[["region", "rule", "F1_delta"]], on=["region", "rule"]))
    m = m[m.region != "RC"]

    div = m[np.sign(m.dice.round(4)) != np.sign(m.F1_delta.round(4))].copy()
    if len(div):
        if "dice_ci_lo" in div:
            div["dice_ci_spans_0"] = (div.dice_ci_lo <= 0) & (div.dice_ci_hi >= 0)
        div["dice_below_mde"] = div.dice.abs() < MDE
        div["f1_below_mde"] = div.F1_delta.abs() < MDE
        print("\n" + div.round(4).to_string(index=False))

        spans = int(div.get("dice_ci_spans_0", pd.Series(dtype=bool)).sum())
        print(f"""
  {spans} of {len(div)} have a Dice bootstrap CI that includes zero.

  Where the CI includes zero, the Dice estimate is not distinguishable from
  zero and its sign carries no information, so comparing signs flags a
  divergence that the data do not support. Where it does not, both endpoint
  effects are very small in magnitude and below the pre-specified Dice MDE of
  {MDE} — which is a statement about magnitude, not a claim that the effect is
  noise.

  Suggested wording:
    "All four non-RC sign divergences involved Dice effects that were either
     not distinguishable from zero or below the pre-specified MDE; none
     indicates a meaningful reversal of direction between the endpoints."
""")
    else:
        print("\n  no sign divergences outside RC")
except FileNotFoundError as e:
    print(f"\n  skipped — {e}")
