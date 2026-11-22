# -*- coding: utf-8 -*-
"""
==============================================================================
 B-6 — PAIRED STATISTICAL TESTS
==============================================================================
 RUNTIME: CPU, ~1-2 min (10k bootstrap x 5 rules x 4 regions).
 INPUT:   exp3_suppressed_v3.csv (merged, complete B-4 output).
 PROTOCOL: PHASE_B_PROTOCOL.md v1.0 (Frozen 13 Aug 2026), sections 6, 7, 8.

 WHAT THIS DECIDES
 -----------------
 This is the script that answers H4 -- does lesion-level separability TRANSLATE
 into official-metric improvement -- on the pre-registered primary stratum.
 Everything it computes was specified before any B-4 number existed.

 THE PRIMARY COMPARISON, AND ONLY IT
 -----------------------------------
   stratum   clean (n=155)          -- protocol section 6
   test      paired Wilcoxon signed-rank, baseline vs each rule
   effect    mean paired delta + BCa-free percentile bootstrap 95% CI, 10k
   multiple  Holm across the 5 non-baseline rules, WITHIN each region

 patient_only and seen are SENSITIVITY, reported alongside and never in place
 of clean. Protocol section 8: an effect present only in patient_only is
 memorization and is declared as such.

 WHAT THIS REFUSES
 -----------------
 1. Incomplete tables. Same reason as B-5b: cheapest-first ordering makes a
    partial run enriched in low-lesion-count cases, i.e. the population where
    suppression cannot do harm. There is no --allow-partial here at all --
    B-5b exists for looking, B-6 exists for deciding, and a p-value computed on
    a biased subsample is worse than no p-value.

 2. Unclassified failures. A pair with status != "ok" has no number. Per
    protocol section 5 each must first be classified: Class 2 is a real Dice-0
    result and must be REINSTATED as zero before testing; Class 1/3 is a
    pipeline fault. Testing while they are merely absent silently drops the
    cases where suppression did the most -- the exact bias the whole guard
    policy exists to prevent.

 3. The MDE is not a significance threshold and is not treated as one. A delta
    below 0.019 is BELOW DETECTION, which is a statement about this sample's
    power, not about the effect being zero. Reported as such, always.

 WHAT IT DOES NOT DO
 -------------------
 T3 (per-size-bin) and T4 (detection F1) are NOT here. They need the per-lesion
 table that official_score discards -- see protocol section 12. B-6 covers the
 Dice endpoint only.

 OUTPUT
 ------
   b6_primary_clean.csv · b6_by_stratum.csv · b6_qc_sensitivity.csv
   B6_REPORT.md
==============================================================================
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

DRIVE   = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W1      = f"{DRIVE}/phase_b_outputs/wave1"
MERGED  = f"{W1}/exp3_suppressed_v3.csv"
QC_CSV  = f"{DRIVE}/brats_qc_taxonomy.csv"
OUT_DIR = f"{DRIVE}/phase_b_outputs/wave2"

BASELINE = "baseline"
RULES = ["baseline", "size@50",
         "entropy@0.45", "entropy@0.50", "entropy@0.56", "confidence@0.45"]
TESTED = [r for r in RULES if r != BASELINE]
REGIONS = ["WT", "TC", "ET", "RC"]
STRATA = ["clean", "patient_only", "seen"]

# Protocol section 6 -- pre-registered, before any result was seen.
MDE = 0.019
N_CLEAN = 155
ALPHA = 0.05
N_BOOT = 10_000
SEED = 20260813          # frozen: the protocol freeze date, so the CI is reproducible

EXPECTED_PAIRS = 567 * len(RULES)


# ---------------------------------------------------------------------------

def load():
    if not os.path.exists(MERGED):
        sys.exit(f"missing {MERGED}\n  run merge_shards.py after B-4 completes")

    df = pd.read_csv(MERGED)
    stale = set(df["setting"]) - set(RULES)
    if stale:
        sys.exit(f"table contains non-protocol rules: {stale}")

    n_pairs = len(df.groupby(["case", "setting"]))
    if n_pairs != EXPECTED_PAIRS:
        sys.exit(
            f"INCOMPLETE: {n_pairs} / {EXPECTED_PAIRS} pairs.\n"
            "  B-6 does not run on partial data and has no override.\n"
            "  Cheapest-first ordering makes a partial table enriched in\n"
            "  low-lesion-count cases -- the population where suppression cannot\n"
            "  do harm -- so any p-value here would be computed on a sample\n"
            "  biased in favour of the effect under test.\n"
            "  Use b5b_stratify.py --allow-partial to watch progress.")

    n_failed = df[df["status"].astype(str) != "ok"].groupby(
        ["case", "setting"]).ngroups
    if n_failed:
        bad = (df[df["status"].astype(str) != "ok"][["case", "setting"]]
               .drop_duplicates())
        sys.exit(
            f"{n_failed} pair(s) carry status != ok.\n"
            "  Classify each with guard3_diagnose.py BEFORE testing:\n"
            "    Class 2   -> a real Dice-0 result; reinstate the row as 0.0\n"
            "    Class 1/3 -> pipeline fault; fix and re-run that pair\n"
            "  Leaving them absent drops the cases where suppression removed the\n"
            "  most, which biases every test below in favour of suppression.\n"
            + "\n".join(f"    {c} · {s}" for c, s in bad.values[:15]))

    print(f"  cases : {df['case'].nunique()}   pairs : {n_pairs}   failed : 0")
    return df


def to_wide(df):
    """(case, region) x rule, with leak_class attached. Deltas are within-case."""
    w = df.pivot_table(index=["case", "region"], columns="setting",
                       values="lesionwise_dice", aggfunc="first").reset_index()
    meta = df[["case", "leak_class"]].drop_duplicates().set_index("case")["leak_class"]
    w["leak_class"] = w["case"].map(meta)
    return w


# ---------------------------------------------------------------------------
#  TESTS
# ---------------------------------------------------------------------------

def boot_ci(d, rng, n=N_BOOT, alpha=ALPHA):
    """Percentile bootstrap CI for the mean paired difference."""
    if len(d) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    run = 0.0
    for i, j in enumerate(order):
        adj = (m - i) * p[j]
        run = max(run, adj)
        out[j] = min(run, 1.0)
    return out


def test_stratum(w, mask, label, rng):
    """Paired Wilcoxon + bootstrap CI per rule x region, Holm within region."""
    sub = w if mask is None else w[mask]
    rows = []

    for region in REGIONS:
        r_sub = sub[sub.region == region]
        recs = []
        for rule in TESTED:
            if rule not in r_sub.columns:
                continue
            pair = r_sub[[BASELINE, rule]].dropna()
            d = (pair[rule] - pair[BASELINE]).to_numpy()

            if len(d) < 2 or np.allclose(d, 0):
                # Wilcoxon is undefined on an all-zero difference vector. That
                # is a real finding (the rule changed nothing here), not an
                # error, so it is recorded rather than skipped.
                stat, p = np.nan, np.nan
                lo, hi = (0.0, 0.0) if len(d) else (np.nan, np.nan)
            else:
                stat, p = stats.wilcoxon(d, zero_method="wilcox",
                                         alternative="two-sided")
                lo, hi = boot_ci(d, rng)

            recs.append({
                "stratum": label, "region": region, "rule": rule,
                "n": int(len(d)),
                "mean_delta": round(float(d.mean()), 5) if len(d) else np.nan,
                "median_delta": round(float(np.median(d)), 5) if len(d) else np.nan,
                "ci_lo": round(lo, 5) if lo == lo else np.nan,
                "ci_hi": round(hi, 5) if hi == hi else np.nan,
                "n_worse": int((d < 0).sum()), "n_better": int((d > 0).sum()),
                "n_tied": int((d == 0).sum()),
                "wilcoxon_p": p,
            })

        # Holm across the five rules, within this region.
        adj = holm([x["wilcoxon_p"] for x in recs])
        for x, a in zip(recs, adj):
            x["p_holm"] = a
            x["sig_holm"] = bool(a < ALPHA) if a == a else False
            md = x["mean_delta"]
            x["below_mde"] = bool(abs(md) < MDE) if md == md else False
        rows += recs

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def show(tab, title, note=""):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)
    cols = ["region", "rule", "n", "mean_delta", "ci_lo", "ci_hi",
            "n_worse", "n_better", "wilcoxon_p", "p_holm", "sig_holm", "below_mde"]
    for region in REGIONS:
        r = tab[tab.region == region]
        if not len(r):
            continue
        print(f"\n  --- {region} ---")
        print(r[cols].drop(columns=["region"]).to_string(index=False))
    if note:
        print("\n  " + note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true",
                    help="print only, write nothing")
    a = ap.parse_args()

    print("=" * 78)
    print(" B-6 — PAIRED TESTS")
    print(" protocol: PHASE_B_PROTOCOL.md v1.0 (Frozen 13 Aug 2026), sections 6-8")
    print("=" * 78)

    df = load()
    w = to_wide(df)
    rng = np.random.default_rng(SEED)

    # ---- PRIMARY: clean ---------------------------------------------------
    clean_mask = w.leak_class == "clean"
    n_clean = w.loc[clean_mask, "case"].nunique()
    primary = test_stratum(w, clean_mask, "clean", rng)

    show(primary,
         f"PRIMARY · clean stratum (n={n_clean} cases, protocol {N_CLEAN})",
         f"MDE = {MDE} Dice (pre-registered, n={N_CLEAN}, alpha={ALPHA}, power=0.80).\n"
         f"  below_mde=True means BELOW DETECTION in this sample — not zero.\n"
         f"  p_holm corrects across the {len(TESTED)} rules WITHIN each region.")

    # ---- SENSITIVITY: other strata ---------------------------------------
    other = pd.concat(
        [test_stratum(w, w.leak_class == s, s, np.random.default_rng(SEED))
         for s in STRATA if s != "clean"]
        + [test_stratum(w, None, "all", np.random.default_rng(SEED))],
        ignore_index=True)

    print("\n" + "=" * 78)
    print(" SENSITIVITY · other strata (never in place of clean)")
    print("=" * 78)
    for s in [x for x in STRATA if x != "clean"] + ["all"]:
        t = other[other.stratum == s]
        print(f"\n  --- {s} ---")
        print(t[["region", "rule", "n", "mean_delta", "p_holm", "sig_holm"]]
              .to_string(index=False))
    print("\n  Protocol section 8: an effect present ONLY in patient_only is\n"
          "  memorization and is declared as such, not as a gain.")

    # ---- 6.1 QC sensitivity ----------------------------------------------
    qc = None
    if os.path.exists(QC_CSV):
        qc_ids = set(pd.read_csv(QC_CSV)["case_id"])
        m = clean_mask & ~w["case"].isin(qc_ids)
        n_ex = w.loc[clean_mask, "case"].isin(qc_ids).sum()
        qc = test_stratum(w, m, "clean_minus_qc", np.random.default_rng(SEED))
        show(qc, f"6.1 · PRE-REGISTERED QC SENSITIVITY "
                 f"(n={w.loc[m, 'case'].nunique()}, {n_ex} rows excluded)",
             "Declared 13 Aug 2026, before B-4 execution. Reported ALONGSIDE the\n"
             "  primary result, never in place of it.")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(" READING THIS AGAINST SECTION 8")
    print("=" * 78)
    sig = primary[primary.sig_holm]
    if len(sig):
        print(f"  {len(sig)} rule-region cell(s) significant after Holm on clean:")
        for _, r in sig.iterrows():
            direction = "HARM" if r.mean_delta < 0 else "gain"
            print(f"    {r.region:3s} · {r.rule:16s} "
                  f"delta={r.mean_delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] "
                  f"p={r.p_holm:.4g}  {direction}")
        print("\n  A significant NEGATIVE delta is a publishable result: suppression\n"
              "  is harmful at these operating points (protocol section 8).")
    else:
        print("  No rule reaches significance on the clean stratum after Holm.")
        small = primary[primary.below_mde]
        print(f"  {len(small)} of {len(primary)} cells have |delta| < {MDE} — "
              "BELOW DETECTION,\n  which is not the same as zero. Protocol "
              "section 8: this is cell D,\n  and the outcome is 'separability "
              "does not translate' — write it up.")

    # ---- write ------------------------------------------------------------
    if a.no_write:
        print("\n  --no-write: nothing saved.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    primary.to_csv(f"{OUT_DIR}/b6_primary_clean.csv", index=False)
    other.to_csv(f"{OUT_DIR}/b6_by_stratum.csv", index=False)
    if qc is not None:
        qc.to_csv(f"{OUT_DIR}/b6_qc_sensitivity.csv", index=False)

    md = [
        "# B-6 — Paired statistical tests", "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Source:** `{os.path.basename(MERGED)}` · {df['case'].nunique()} cases",
        "**Protocol:** v1.0 (Frozen 13 Aug 2026), sections 6-8", "",
        f"Paired Wilcoxon signed-rank · percentile bootstrap 95% CI "
        f"({N_BOOT:,} resamples, seed {SEED}) · Holm across {len(TESTED)} rules "
        f"within each region.", "",
        f"Pre-registered MDE **{MDE}** Dice at n={N_CLEAN}. A delta below it is "
        "below detection in this sample, which is not the same as zero.", "",
        "T3 (per-size-bin) and T4 (detection F1) are not here — they need the "
        "per-lesion table described in protocol section 12.", "",
        f"## Primary — clean stratum (n={n_clean})", "",
        primary.to_markdown(index=False), "",
        "## Sensitivity — other strata", "",
        other.to_markdown(index=False), "",
    ]
    if qc is not None:
        md += ["## 6.1 — Pre-registered QC sensitivity", "",
               qc.to_markdown(index=False), ""]
    open(f"{OUT_DIR}/B6_REPORT.md", "w", encoding="utf-8").write("\n".join(md))

    print(f"\n  written to {OUT_DIR}/")
    print("    b6_primary_clean.csv · b6_by_stratum.csv · B6_REPORT.md")


if __name__ == "__main__":
    main()
