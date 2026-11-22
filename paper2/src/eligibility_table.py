# -*- coding: utf-8 -*-
"""
==============================================================================
 ELIGIBILITY TABLE · what each comparison was actually computed on
==============================================================================
 Read-only, from the frozen T4 counts. Seconds. No new analysis.

 WHAT THIS SHOWS
 ---------------
 For each region and each suppression rule: how many cases were eligible for
 the detection endpoint, how many were not, and whether the rule's arm and the
 baseline arm were scored over the same set of cases.

 A case is eligible when its region has at least one of a true positive, a
 false positive or a false negative. With none of the three, F1 has no
 denominator and the case contributes nothing to a paired comparison.

 WHAT IT ESTABLISHES, AND WHAT IT DOES NOT
 -----------------------------------------
 Establishes, empirically:

     The intervention can change endpoint eligibility, thereby changing the
     set of cases contributing to the reported comparison.

 Does NOT establish, and is not extended to:

     The same TP/FP/FN trade-off can produce different apparent metric changes
     solely because of eligibility.

 The second is a counterfactual. It would require two interventions identical
 in their error trade-off and differing in eligibility, which these data do
 not contain. It is not simulated, because a simulation would show that a
 constructed scenario behaves as constructed — not that this is what happened
 here.

 Every statement drawn from this table is qualified "in our evaluated
 setting" until a wider claim is justified.
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
OUT = f"{DRIVE}/paper2/eligibility_table.csv"
PAT = re.compile(r"\.w\d+of\d+\.csv$")
REGIONS = ["WT", "TC", "ET", "RC"]
ORDER = ["baseline", "size@20", "size@50", "entropy@0.45",
         "entropy@0.50", "entropy@0.56", "confidence@0.45"]


def main():
    files = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv") if PAT.search(f)]
    if not files:
        sys.exit(f"no count shards under {W2}")
    t4 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bad = int((t4.status.astype(str) != "ok").sum())
    if bad:
        sys.exit(f"{bad} non-ok rows; not proceeding")

    # eligible: at least one of TP, FP, FN
    t4["eligible"] = ((t4.Num_TP + t4.Num_FP + t4.Num_FN) > 0).astype(int)

    print("=" * 78)
    print(" ELIGIBILITY BY REGION AND RULE")
    print("=" * 78)
    print(f"  cases: {t4.case.nunique()}   rules: {t4.setting.nunique()}")
    print("""
  A case is eligible for the detection endpoint when its region has at least
  one true positive, false positive or false negative. With none of the three,
  F1 is undefined and the case contributes nothing to a paired comparison.
""")

    rows = []
    for reg in REGIONS:
        sub = t4[t4.region == reg]
        base = sub[sub.setting == "baseline"].set_index("case").eligible
        n_all = len(base)

        print(f"\n  {'-' * 70}")
        print(f"  {reg}")
        print(f"  {'-' * 70}")
        print(f"    {'rule':<16} {'eligible':>9} {'not':>6} "
              f"{'paired with baseline':>21} {'lost':>6} {'gained':>7}")

        for rule in ORDER:
            arm = sub[sub.setting == rule]
            if not len(arm):
                continue
            a = arm.set_index("case").eligible.reindex(base.index)
            n_elig = int(a.sum())
            paired = int(((base == 1) & (a == 1)).sum())
            lost = int(((base == 1) & (a == 0)).sum())
            gained = int(((base == 0) & (a == 1)).sum())

            flag = "" if rule == "baseline" else (
                "  same" if lost == 0 and gained == 0 else "  DIFFERS")
            print(f"    {rule:<16} {n_elig:>9} {n_all - n_elig:>6} "
                  f"{paired:>21} {lost:>6} {gained:>7}{flag}")

            rows.append({"region": reg, "rule": rule,
                         "n_cases": n_all, "n_eligible": n_elig,
                         "n_not_eligible": n_all - n_elig,
                         "paired_with_baseline": paired,
                         "lost_vs_baseline": lost,
                         "gained_vs_baseline": gained,
                         "same_population": int(lost == 0 and gained == 0)})

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    # ---- the summary the claim rests on -----------------------------
    print("\n" + "=" * 78)
    print(" DOES THE COMPARISON USE THE SAME CASES IN BOTH ARMS?")
    print("=" * 78)
    nb = d[d.rule != "baseline"]
    print(f"\n  {'region':<8} {'rules compared':>15} {'same population':>17} "
          f"{'differing':>11} {'max cases lost':>16}")
    for reg in REGIONS:
        s = nb[nb.region == reg]
        print(f"  {reg:<8} {len(s):>15} {int(s.same_population.sum()):>17} "
              f"{int((~s.same_population.astype(bool)).sum()):>11} "
              f"{int(s.lost_vs_baseline.max()):>16}")

    diff = nb[nb.same_population == 0]
    print(f"""
  Of {len(nb)} rule-region comparisons, {len(diff)} were not conducted over the
  same set of cases as their baseline arm.""")

    if len(diff):
        print("\n  the comparisons affected:")
        print("  " + diff[["region", "rule", "paired_with_baseline",
                           "lost_vs_baseline"]]
              .to_string(index=False).replace("\n", "\n  "))

    # ---- reading ----------------------------------------------------
    print("\n" + "=" * 78)
    print(" READING THIS")
    print("=" * 78)
    print("""
  What the table shows: the evaluator is fixed and its eligibility rule is the
  same for every arm, yet the set of cases clearing that rule differs between
  the baseline and the suppressed arms. The cases lost are those where
  suppression removed the last predicted component in a region with no
  reference lesion, leaving no true positive, no false positive and no false
  negative.

  In our evaluated setting, therefore, a paired comparison between baseline
  and a suppression rule was not always conducted over the same effective case
  population -- and nothing in the reported endpoint indicates that.

  What the table does NOT show: that two rules with the same error trade-off
  would produce different apparent metric changes because of eligibility
  alone. That comparison does not exist in these data and is not constructed.

  Prevalence is a property of the task, not of the endpoint. The mechanism
  needs a region that is reference-negative in some cases; how often that
  happens depends on the task. Both figures are in the table above and should
  be quoted together.""")

    print(f"\n  written: {OUT}")


if __name__ == "__main__":
    main()
