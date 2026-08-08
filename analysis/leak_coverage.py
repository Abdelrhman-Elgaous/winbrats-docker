#!/usr/bin/env python3
"""
Training exposure across the ten candidate three-fold subsets.

Reproduces Table 3 from the released coverage file.

    python analysis/leak_coverage.py

Because each candidate excludes a different pair of folds, candidates differ in
how many evaluated cases their members trained on. If difficult or
resection-cavity-positive cases were unevenly distributed across folds, a subset
could in principle win partly because its members had seen the harder cases
assigned to an excluded fold. This table reports that exposure for every
candidate so the possibility can be assessed rather than assumed away.

Two mechanisms are separated:

  ensemble exposure   the candidate's members trained on the case itself
  patient exposure    splits are grouped per case rather than per patient, and
                      the cohort is partly longitudinal, so another timepoint of
                      the same patient may have been in training

The second was not reported in the original submission and is disclosed here.

The source computation reads splits_final.json from the challenge preprocessing
directory, which is not redistributable; the resulting counts are released
instead.
"""

import sys
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "results" / "leak_coverage_candidates.csv"
DEPLOYED = "124"


def main():
    if not CSV.exists():
        sys.exit(f"not found: {CSV}")
    d = pd.read_csv(CSV).sort_values("pct_clean", ascending=False)
    d["combo"] = d["combo"].astype(str).str.zfill(3)

    print(f"Training exposure, {int(d['n'].iloc[0])} evaluated cases\n")
    print(f"{'subset':<10}{'unseen':>8}{'%':>8}{'same-pat':>10}{'seen':>8}"
          f"{'unseen RC+':>12}")
    print("-" * 56)
    for _, r in d.iterrows():
        mark = "  <- deployed" if r.combo == DEPLOYED else ""
        label = "(" + ",".join(r.combo) + ")"
        print(f"{label:<10}{r.clean:>8}{r.pct_clean:>8.1f}{r.patient_only:>10}"
              f"{r.seen:>8}{r.rc_clean:>12}{mark}")
    print("-" * 56)

    spread = d.pct_clean.max() - d.pct_clean.min()
    print(f"\nspread across candidates : {spread:.1f} points "
          f"({d.pct_clean.min():.1f}-{d.pct_clean.max():.1f})")
    print(f"unseen RC-positive range : {d.rc_clean.min()}-{d.rc_clean.max()}")

    dep = d[d.combo == DEPLOYED]
    if len(dep):
        rank = int((d.pct_clean > dep.pct_clean.iloc[0]).sum()) + 1
        print(f"deployed subset          : {dep.pct_clean.iloc[0]:.1f}% unseen, "
              f"rank {rank} of {len(d)} by coverage, "
              f"{dep.rc_clean.iloc[0]} unseen RC-positive cases")
        print("\nThe deployed subset was evaluated under a less favourable exposure")
        print("profile than most competing candidates, so its advantage is unlikely")
        print("to be explained solely by differential training exposure. This is an")
        print("observational comparison: no per-case difficulty score is available,")
        print("so it bounds the exposure difference rather than establishing that")
        print("the exposed cases were easier or harder.")


if __name__ == "__main__":
    main()
