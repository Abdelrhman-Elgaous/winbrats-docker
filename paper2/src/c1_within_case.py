# -*- coding: utf-8 -*-
"""
==============================================================================
 C1 · WITHIN-CASE COMPARISON — what distinguishes a rejected lesion?
==============================================================================
 Read-only, from c0_lesions.csv. Nothing recomputed, no model, no training.

 THE QUESTION, FIXED BEFORE RUNNING
 ----------------------------------
   Within the same case and the same region, what measurable lesion-level
   properties distinguish successfully detected lesions from lesions that
   receive near-zero model probability?

 D24 established that confidently missed lesions sit alongside detected ones
 in 98.7% of instances. That makes the case-region a natural matched stratum:
 the patient, the acquisition, the preprocessing, the resolution and the model
 are all held constant, and the only thing that varies is whether the lesion
 was found. It is the cleanest comparison available in this data.

 THE UNIT, FIXED BEFORE RUNNING
 ------------------------------
 Paired within case-region, not pooled across lesions. Lesions in one scan
 share a patient and a prediction; pooling them and reporting a lesion-level
 p-value would treat 8 lesions from one scan as 8 independent observations,
 which is the pseudoreplication this project has refused elsewhere.

 For each case-region containing at least one detected and one missed lesion,
 a single paired difference is formed from the medians of the two groups. The
 test is then over case-regions.

 THE FEATURES, FIXED BEFORE RUNNING
 ----------------------------------
 Only what the frozen output already carries, kept in two families that are
 not mixed:

   geometry     gt_volume_mm3, lesion count in the stratum
   probability  max_prob_in_lesion, mean_prob_in_lesion

 Probability features are reported separately and NOT used to claim that
 low-probability lesions differ from high-probability ones -- that is the
 definition of the groups, not a finding about them. They are reported because
 their distribution within the missed group is itself informative.

 Intensity, anatomical location and distance-to-boundary are not available in
 this table and are not approximated. Their absence is a limit on what can be
 concluded and is stated rather than worked around.

 WHAT THIS CANNOT DO
 -------------------
 It cannot establish why a lesion was rejected. It can say which of the
 available properties separate the two groups and which do not, and a feature
 that fails to separate them is as informative as one that does.
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
LES = f"{DRIVE}/phase_c_outputs/c0/c0_lesions.csv"
OUT = f"{DRIVE}/phase_c_outputs/c0"

MISSED_MAX_P = 0.001      # "confidently missed", as in D24
ALPHA = 0.05


def main():
    if not os.path.exists(LES):
        sys.exit(f"missing {LES}")
    les = pd.read_csv(LES)

    det = les[les.baseline_detected == 1].copy()
    mis = les[(les.baseline_detected == 0) &
              (les.max_prob_in_lesion <= MISSED_MAX_P)].copy()

    print("=" * 78)
    print(" C1 · WITHIN-CASE COMPARISON")
    print(" question, unit and features fixed before running")
    print("=" * 78)
    print(f"  detected lesions        : {len(det):,}")
    print(f"  confidently missed      : {len(mis):,}   (Pmax <= {MISSED_MAX_P})")

    # strata containing both groups
    keys = ["case", "region"]
    d_by = det.groupby(keys)
    m_by = mis.groupby(keys)
    both = sorted(set(d_by.groups) & set(m_by.groups))
    print(f"  case-regions with both  : {len(both):,}")
    print(f"  ── the matched strata; everything else is excluded, because a")
    print(f"     stratum with only one group carries no comparison")

    if not both:
        sys.exit("  no matched strata")

    rows = []
    for k in both:
        d, m = d_by.get_group(k), m_by.get_group(k)
        rows.append({
            "case": k[0], "region": k[1],
            "n_det": len(d), "n_mis": len(m),
            "det_vol": float(d.gt_volume_mm3.median()),
            "mis_vol": float(m.gt_volume_mm3.median()),
            "det_pmax": float(d.max_prob_in_lesion.median()),
            "mis_pmax": float(m.max_prob_in_lesion.median()),
            "det_pmean": float(d.mean_prob_in_lesion.median()),
            "mis_pmean": float(m.mean_prob_in_lesion.median()),
        })
    s = pd.DataFrame(rows)
    s["d_vol"] = s.det_vol - s.mis_vol
    s["log_ratio_vol"] = np.log10(s.det_vol.clip(lower=.1) /
                                  s.mis_vol.clip(lower=.1))

    # ---- 1 · geometry ----------------------------------------------------
    print("\n" + "=" * 78)
    print(" 1 · GEOMETRY — does volume separate them within a stratum?")
    print("=" * 78)
    w = stats.wilcoxon(s.det_vol, s.mis_vol)
    rng = np.random.default_rng(20260819)
    boot = s.d_vol.to_numpy()[rng.integers(0, len(s), size=(10000, len(s)))].mean(axis=1)
    print(f"""
  median volume, detected : {s.det_vol.median():>8.1f} mm3
  median volume, missed   : {s.mis_vol.median():>8.1f} mm3
  paired difference       : {s.d_vol.median():>8.1f} mm3 (median)
                            {s.d_vol.mean():>8.1f} mm3 (mean)
  95% CI of the mean      : [{np.percentile(boot,2.5):.1f}, {np.percentile(boot,97.5):.1f}]
  Wilcoxon                : p = {w.pvalue:.3g}   (n = {len(s)} strata)
  strata where detected are larger : {int((s.d_vol > 0).sum())}/{len(s)} ({(s.d_vol > 0).mean():.1%})""")

    if w.pvalue < ALPHA:
        print("""
  Volume separates the groups within strata. That does not make size the
  cause: larger lesions are easier to detect for many reasons, and the
  comparison cannot distinguish them.""")
    else:
        print("""
  Volume does not separate the groups within strata. Given that the same scan
  contains both, this weakens size as an explanation more directly than the
  cohort-level comparison in C0.8 could.""")

    # ---- 2 · how far apart, not just whether ----------------------------
    print("\n" + "=" * 78)
    print(" 2 · HOW LARGE IS THE SEPARATION?")
    print("=" * 78)
    print(f"\n  log10 volume ratio (detected / missed) across strata:")
    q = s.log_ratio_vol.quantile([.1, .25, .5, .75, .9])
    for k, v in q.items():
        print(f"    {k:>5.0%}   {v:>7.2f}   (ratio {10**v:>6.1f}x)")
    print(f"""
  A ratio near 1.0 would mean the two groups are the same size within a
  stratum. The median here is {10**s.log_ratio_vol.median():.1f}x.

  Note what is NOT concluded: many detected lesions are also small, and many
  missed lesions are large. The overlap matters more than the median for
  whether size could serve as a decision rule, and it is substantial.""")

    ov = pd.DataFrame({
        "detected": det.gt_volume_mm3.quantile([.1, .25, .5, .75, .9]).round(1),
        "missed": mis.gt_volume_mm3.quantile([.1, .25, .5, .75, .9]).round(1)})
    print("\n  volume distributions, all lesions:")
    print("  " + ov.to_string().replace("\n", "\n  "))

    # ---- 3 · the probability side ---------------------------------------
    print("\n" + "=" * 78)
    print(" 3 · PROBABILITY — reported, not tested")
    print("=" * 78)
    print(f"""
  detected  Pmax median {det.max_prob_in_lesion.median():.4f} · Pmean median {det.mean_prob_in_lesion.median():.4f}
  missed    Pmax median {mis.max_prob_in_lesion.median():.4f} · Pmean median {mis.mean_prob_in_lesion.median():.4f}

  These are not compared statistically. The groups were DEFINED by
  probability, so a difference between them is the definition restated, not a
  finding. They appear here because the absolute values matter: the missed
  group sits at {mis.max_prob_in_lesion.median():.5f}, which is a confident
  negative rather than an uncertain one.""")

    # ---- 4 · burden -------------------------------------------------------
    print("\n" + "=" * 78)
    print(" 4 · HOW MANY ARE MISSED PER STRATUM")
    print("=" * 78)
    s["frac_missed"] = s.n_mis / (s.n_mis + s.n_det)
    print(f"""
  lesions per stratum : detected median {s.n_det.median():.0f} · missed median {s.n_mis.median():.0f}
  fraction missed     : median {s.frac_missed.median():.1%} · mean {s.frac_missed.mean():.1%}
  strata losing over half : {int((s.frac_missed > 0.5).sum())}/{len(s)} ({(s.frac_missed > .5).mean():.1%})""")

    s.to_csv(f"{OUT}/c1_within_case.csv", index=False)

    # ---- what remains -----------------------------------------------------
    print("\n" + "=" * 78)
    print(" WHAT THIS DOES AND DOES NOT SETTLE")
    print("=" * 78)
    print(f"""
  Settled: whether the properties available in this table separate detected
  from missed lesions within a matched stratum, and by how much.

  Not settled, and not approximable from this table:
    · intensity and contrast characteristics of the lesion
    · anatomical location and proximity to structures
    · lesion subtype and treatment history
    · whether the reference annotation is correct

  Any of those could account for the rejection, and the next step is to
  measure one of them rather than to choose an architecture. Selecting a model
  now would be selecting it before knowing what it needs to represent.

  written: {OUT}/c1_within_case.csv""")


if __name__ == "__main__":
    main()
