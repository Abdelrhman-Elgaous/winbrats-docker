# -*- coding: utf-8 -*-
"""
==============================================================================
 ELIGIBILITY: HOW MUCH DOES THE POPULATION SHIFT CHANGE THE NUMBER?
==============================================================================
 Read-only, from the frozen T4 counts. Seconds. No new threshold, no
 optimisation, no reopening of Phase B.

 The existing analysis establishes THAT the evaluable population differs
 between arms: 20 of 24 rule-region comparisons were not conducted over the
 same set of cases as their baseline arm. What it does not report is how much
 that matters to the reported number.

 This computes, for each rule and region, the detection F1 on two populations:

     the intersection    cases evaluable under BOTH arms -- the honest
                         paired comparison
     each arm's own      cases evaluable under that arm alone -- what a study
                         reports if it takes each arm's mean at face value

 The difference between the two is the size of the artefact.

 WHAT THIS ESTABLISHES AND WHAT IT DOES NOT
 ------------------------------------------
 Establishes: the magnitude of the discrepancy between a paired comparison
 and a naive per-arm mean, in this study, on quantities already computed.

 Does NOT establish: that two rules with the same error trade-off would differ
 because of eligibility alone. That counterfactual is not in these data and is
 not constructed here. Nor does it establish anything about other studies;
 prevalence is a property of the task.

 Every statement from this stays qualified "in our evaluated setting".
==============================================================================
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W2 = f"{DRIVE}/phase_b_outputs/wave2"
OUT = f"{DRIVE}/paper2/eligibility_effect.csv"
PAT = re.compile(r"\.w\d+of\d+\.csv$")
REGIONS = ["WT", "TC", "ET", "RC"]
ORDER = ["baseline", "size@20", "size@50", "entropy@0.45",
         "entropy@0.50", "entropy@0.56", "confidence@0.45"]


def f1(tp, fp, fn):
    """Undefined when there are no events at all, which is the whole point."""
    d = 2 * tp + fp + fn
    return np.nan if d == 0 else 2 * tp / d


def main():
    files = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv") if PAT.search(f)]
    if not files:
        sys.exit(f"no count shards under {W2}")
    t4 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if (t4.status.astype(str) != "ok").any():
        sys.exit("non-ok rows present; not proceeding")

    t4["F1"] = [f1(a, b, c) for a, b, c in
                zip(t4.Num_TP, t4.Num_FP, t4.Num_FN)]

    print("=" * 78)
    print(" HOW MUCH DOES THE POPULATION SHIFT CHANGE THE REPORTED NUMBER?")
    print("=" * 78)
    print("""
  Two ways to summarise the same arm:

    paired      mean over cases evaluable under BOTH arms
    per-arm     mean over cases evaluable under that arm alone

  The first is the comparison the study reports. The second is what a reader
  gets by taking each arm's published mean and subtracting. Where the eligible
  sets differ, the two disagree, and the gap is the artefact.
""")

    rows = []
    for reg in REGIONS:
        sub = t4[t4.region == reg]
        base = sub[sub.setting == "baseline"].set_index("case").F1
        print(f"\n  {'-' * 72}")
        print(f"  {reg}")
        print(f"    {'rule':<16} {'paired':>16} {'per-arm':>16} "
              f"{'difference':>12} {'n pair':>7}")

        for rule in ORDER:
            if rule == "baseline":
                continue
            arm = sub[sub.setting == rule].set_index("case").F1
            common = base.index.intersection(arm.index)
            both = common[base[common].notna() & arm[common].notna()]

            paired = float(arm[both].mean() - base[both].mean()) if len(both) else np.nan
            per_arm = float(arm.dropna().mean() - base.dropna().mean())
            diff = per_arm - paired

            flag = ""
            if diff == diff and abs(diff) >= 0.019:      # the study's own MDE
                flag = "  exceeds the MDE"
            elif diff == diff and abs(diff) >= 0.005:
                flag = "  notable"

            print(f"    {rule:<16} {paired:>+16.4f} {per_arm:>+16.4f} "
                  f"{diff:>+12.4f} {len(both):>7}{flag}")

            rows.append({"region": reg, "rule": rule,
                         "paired_delta": paired, "per_arm_delta": per_arm,
                         "difference": diff, "n_paired": int(len(both)),
                         "n_baseline_eligible": int(base.notna().sum()),
                         "n_arm_eligible": int(arm.notna().sum())})

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    # ---- summary ----------------------------------------------------
    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    ok = d.dropna(subset=["difference"])
    over = ok[ok.difference.abs() >= 0.019]
    print(f"""
  rule-region comparisons examined : {len(ok)}
  where the two summaries differ    : {int((ok.difference.abs() > 1e-9).sum())}
  where the difference exceeds the
  study's own minimum detectable
  effect of 0.019                   : {len(over)}
  largest difference                : {ok.difference.abs().max():.4f}
    """)
    if len(over):
        print("  the comparisons where it matters most:")
        print("  " + over.reindex(over.difference.abs().sort_values(
            ascending=False).index)[
            ["region", "rule", "paired_delta", "per_arm_delta", "difference"]]
              .head(8).round(4).to_string(index=False).replace("\n", "\n  "))

    # ---- reading ----------------------------------------------------
    print("\n" + "=" * 78)
    print(" READING THIS")
    print("=" * 78)
    print(f"""
  In our evaluated setting, taking each arm's mean at face value rather than
  pairing on the common evaluable set changes the apparent effect by up to
  {ok.difference.abs().max():.4f} --- against a minimum detectable effect of 0.019. The
  discrepancy is not a disagreement about what suppression did to any
  prediction. It is a consequence of which cases the endpoint was defined on
  in each arm.

  This is a demonstration within one study, on one task. Whether it matters
  elsewhere depends on how many cases are reference-negative in the region
  concerned, which is a property of the task rather than of the endpoint, and
  nothing here establishes its prevalence beyond this cohort.

  Not constructed and not claimed: that two rules with the same error
  trade-off would differ because of eligibility alone. That comparison does
  not exist in these data.

  written: {OUT}""")


if __name__ == "__main__":
    main()
