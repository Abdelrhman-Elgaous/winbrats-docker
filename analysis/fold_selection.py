#!/usr/bin/env python3
"""
Exhaustive three-fold subset selection.

Reproduces Table 1 and Figure 2 from the released per-case scores. No model
weights or imaging data are required: the search operated on saved out-of-fold
probability maps, and this script recomputes the ranking from the per-case Dice
values those maps produced.

    python analysis/fold_selection.py --summarise

Input: results/per_case_fold_selection.csv
       567 cases x 4 regions x 16 configurations x 2 label variants

Configurations are the five individual folds, all ten three-fold subsets
(C(5,3) = 10), and the full five-fold ensemble.

The `variant` column distinguishes two label-construction routes. `direct`
thresholds each region independently; `nested` applies the BraTS label hierarchy
with the resection cavity written last, which is what the deployed system does.
The paper reports `nested`; `direct` is included so the effect of label
construction is visible.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "results" / "per_case_fold_selection.csv"
REGIONS = ["WT", "TC", "ET", "RC"]
DEPLOYED = "3fold_124"
PUBLISHED = "nested"


def load():
    if not CSV.exists():
        sys.exit(f"not found: {CSV}")
    d = pd.read_csv(CSV)
    missing = {"case", "region", "combo", "variant", "dice"} - set(d.columns)
    if missing:
        sys.exit(f"missing columns: {sorted(missing)}")
    return d


def summarise(d, variant):
    """Mean Dice per region and overall, per configuration."""
    v = d[d.variant == variant]
    t = v.groupby(["combo", "region"])["dice"].mean().unstack("region")[REGIONS]
    t["mean"] = t.mean(axis=1)
    return t.sort_values("mean")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarise", action="store_true",
                    help="print the candidate ranking (Table 1)")
    ap.add_argument("--variant", default=PUBLISHED, choices=["nested", "direct"],
                    help=f"label construction (default: {PUBLISHED}, as published)")
    ap.add_argument("--three-fold-only", action="store_true",
                    help="restrict to the ten three-fold candidates")
    a = ap.parse_args()

    d = load()
    print(f"loaded {len(d):,} rows  |  {d.case.nunique()} cases  |  "
          f"{d.combo.nunique()} configurations  |  variants "
          f"{sorted(d.variant.unique())}\n")

    t = summarise(d, a.variant)
    if a.three_fold_only:
        t = t[t.index.str.startswith("3fold")]

    print(f"Mean voxel-overlap Dice, variant = {a.variant}")
    print("-" * 64)
    print(f"{'configuration':<14}" + "".join(f"{r:>9}" for r in REGIONS)
          + f"{'mean':>10}")
    print("-" * 64)
    for combo, row in t.iterrows():
        mark = "  <- deployed" if combo == DEPLOYED else ""
        print(f"{combo:<14}" + "".join(f"{row[r]:9.4f}" for r in REGIONS)
              + f"{row['mean']:10.4f}{mark}")
    print("-" * 64)

    three = t[t.index.str.startswith("3fold")]
    if len(three):
        print(f"\nten three-fold candidates span {three['mean'].min():.4f} to "
              f"{three['mean'].max():.4f}  "
              f"(spread {three['mean'].max() - three['mean'].min():.4f})")
        print(f"best three-fold subset: {three['mean'].idxmax()}")

    if DEPLOYED in t.index and "5fold" in t.index:
        print(f"deployed vs five-fold ensemble: "
              f"{t.loc[DEPLOYED, 'mean'] - t.loc['5fold', 'mean']:+.4f}")
        # Selecting on the mean can hide a region that regressed; check it.
        worse = [r for r in REGIONS if t.loc[DEPLOYED, r] < t.loc["5fold", r]]
        if worse:
            print(f"regions below five-fold: {', '.join(worse)}")


if __name__ == "__main__":
    main()
