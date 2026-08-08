#!/usr/bin/env python3
"""
Statistical comparison of the deployed three-fold ensemble against the full
five-fold ensemble.

Reproduces the tests reported in Section 4.1 and Section 4.5 from the released
per-case scores. No model weights or imaging data are required.

    python analysis/statistics.py

WHAT IS TESTED, AND WHY IN THIS ORDER
-------------------------------------
A non-significant difference is not evidence of equivalence, so two tests are
reported rather than one. The Wilcoxon signed-rank test asks whether a
difference is detectable. Two one-sided tests (TOST) against a pre-registered
margin ask whether any difference that exists is small enough not to matter.
Only the second can support the claim the paper makes.

The equivalence margins were fixed before the analysis: 0.01 Dice for the voxel
comparison, on the grounds that a difference below this is smaller than reported
inter-rater variability for this task; and 0.02 for the lesion-wise comparison,
because paired differences are substantially noisier under that metric.

Region-wise tests are Bonferroni-corrected across the four evaluation regions.

WHAT THIS SCRIPT DOES NOT ESTABLISH
-----------------------------------
The tests are computed on the same cohort used to select the subset, and the
pre-registered margin does not account for a maximum taken over ten candidates.
They quantify the comparison between the selected subset and the full ensemble
on this cohort; they do not carry the usual guarantee under selection. This is
stated as a limitation in the paper.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
LESIONWISE = RESULTS / "per_case_lesionwise.csv"
FOLDSEL = RESULTS / "per_case_fold_selection.csv"

REGIONS = ["WT", "TC", "ET", "RC"]
DEPLOYED, FULL = "3fold_124", "5fold"
MARGIN_VOXEL, MARGIN_LESIONWISE = 0.01, 0.02
ALPHA = 0.05
N_BOOT = 10000
SEED = 137

# Values reported in the manuscript, checked against what this script computes.
PUBLISHED = {
    "voxel_deployed": 0.7734, "voxel_full": 0.7725,
    "lesionwise_deployed": 0.6523, "lesionwise_full": 0.6494,
}


def tost(diff, margin):
    """
    Two one-sided tests. Equivalence is concluded when BOTH one-sided tests
    reject, so the reported p is the larger of the two.
    """
    lower = stats.wilcoxon(diff + margin, alternative="greater").pvalue
    upper = stats.wilcoxon(diff - margin, alternative="less").pvalue
    return max(lower, upper), lower, upper


def boot_ci(diff, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def paired(df, value_col, config_col, case_col="case"):
    """Per-case means for each arm, aligned on case."""
    a = df[df[config_col] == DEPLOYED].groupby(case_col)[value_col].mean()
    b = df[df[config_col] == FULL].groupby(case_col)[value_col].mean()
    common = a.index.intersection(b.index)
    return a[common], b[common]


def report(name, a, b, margin):
    diff = (a - b).values
    n = len(diff)
    n_tied = int((diff == 0).sum())

    print(f"\n{name}")
    print("=" * 68)
    print(f"  n                    : {n}  ({n - n_tied} non-zero, {n_tied} tied)")
    print(f"  {DEPLOYED:<20} : {a.mean():.4f}")
    print(f"  {FULL:<20} : {b.mean():.4f}")
    print(f"  difference           : {diff.mean():+.4f}")

    w = stats.wilcoxon(diff).pvalue if n_tied < n else float("nan")
    lo, hi = boot_ci(diff)
    print(f"  Wilcoxon p           : {w:.3f}"
          + ("  (no detectable difference)" if w > ALPHA else "  (difference detected)"))
    print(f"  95% CI of mean diff  : [{lo:+.4f}, {hi:+.4f}]")

    p, p_lo, p_hi = tost(diff, margin)
    verdict = "equivalent" if p < ALPHA else "NOT established"
    print(f"  TOST, margin {margin}    : p = {p:.4g}   -> {verdict}")
    print(f"    one-sided lower    : p = {p_lo:.4g}")
    print(f"    one-sided upper    : p = {p_hi:.4g}")
    return diff


def region_tests(df, value_col, config_col, margin):
    print(f"\n  region-wise, Bonferroni alpha = {ALPHA / len(REGIONS):.4f}")
    print("  " + "-" * 62)
    print(f"  {'region':<8}{'deployed':>10}{'full':>10}{'diff':>10}"
          f"{'Wilcoxon':>12}{'TOST':>12}")
    print("  " + "-" * 62)
    for r in REGIONS:
        sub = df[df.region == r]
        a, b = paired(sub, value_col, config_col)
        d = (a - b).values
        if (d == 0).all():
            print(f"  {r:<8}{a.mean():10.4f}{b.mean():10.4f}{d.mean():+10.4f}"
                  f"{'all tied':>12}{'--':>12}")
            continue
        w = stats.wilcoxon(d).pvalue
        p, _, _ = tost(d, margin)
        flag = " *" if w < ALPHA / len(REGIONS) else ""
        print(f"  {r:<8}{a.mean():10.4f}{b.mean():10.4f}{d.mean():+10.4f}"
              f"{w:12.4f}{p:12.4g}{flag}")
    print("  " + "-" * 62)
    print("  * significant after Bonferroni correction")


def check(label, computed, key):
    """Compare against the manuscript. A mismatch is worth knowing about."""
    pub = PUBLISHED[key]
    delta = abs(computed - pub)
    status = "matches" if delta < 5e-4 else f"DIFFERS by {delta:.4f}"
    print(f"  {label:<28} computed {computed:.4f}   published {pub:.4f}   {status}")
    return delta < 5e-4


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-voxel", action="store_true",
                    help="lesion-wise comparison only")
    a = ap.parse_args()

    ok = []

    if not a.skip_voxel:
        if not FOLDSEL.exists():
            sys.exit(f"not found: {FOLDSEL}")
        fs = pd.read_csv(FOLDSEL)
        fs = fs[fs.variant == "nested"]
        x, y = paired(fs, "dice", "combo")
        report("VOXEL-OVERLAP DICE  (selection criterion)", x, y, MARGIN_VOXEL)
        region_tests(fs, "dice", "combo", MARGIN_VOXEL)
        ok.append(("voxel", x.mean(), y.mean()))

    if not LESIONWISE.exists():
        sys.exit(f"not found: {LESIONWISE}")
    lw = pd.read_csv(LESIONWISE)
    x, y = paired(lw, "lesionwise_dice", "config")
    report("LESION-WISE DICE  (official evaluator)", x, y, MARGIN_LESIONWISE)
    region_tests(lw, "lesionwise_dice", "config", MARGIN_LESIONWISE)

    # The stricter margin is reported in the paper as also holding.
    d = (x - y).values
    p, _, _ = tost(d, MARGIN_VOXEL)
    print(f"\n  TOST at the stricter {MARGIN_VOXEL} margin: p = {p:.4g}"
          f"   -> {'equivalent' if p < ALPHA else 'NOT established'}")
    ok.append(("lesionwise", x.mean(), y.mean()))

    print("\n" + "=" * 68)
    print(" AGREEMENT WITH THE MANUSCRIPT")
    print("=" * 68)
    all_ok = True
    for kind, dep, full in ok:
        all_ok &= check(f"{kind}, {DEPLOYED}", dep, f"{kind}_deployed")
        all_ok &= check(f"{kind}, {FULL}", full, f"{kind}_full")
    if not all_ok:
        print("\n  One or more values differ from the manuscript. This script")
        print("  recomputes them from the released per-case scores, so a mismatch")
        print("  means the two were produced from different data.")


if __name__ == "__main__":
    main()
