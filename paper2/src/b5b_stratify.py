# -*- coding: utf-8 -*-
"""
==============================================================================
 B-5b — STRATIFICATION BY leak_class
==============================================================================
 RUNTIME: CPU, seconds. INPUT: exp3_suppressed_v3.csv (merged B-4 output).
 PROTOCOL: PHASE_B_PROTOCOL.md v1.0 (Frozen 13 Aug 2026), sections 6, 6.1, 9.

 WHAT THIS IS
 ------------
 Descriptive layer only. It fills protocol tables T1 and T2 -- means, paired
 deltas, and counts -- and runs NO significance test. The tests are B-6.
 Keeping them apart matters: it makes it impossible to look at a p-value here
 and then decide which stratum to feature.

 WHAT IT REFUSES TO DO
 ---------------------
 1. It will not run on an incomplete table. A partial B-4 is enriched in cheap
    cases (--cheap-first order), and cheap means few lesions, which is exactly
    the population where suppression cannot do harm. Reading a partial run
    therefore produces a systematically optimistic picture of suppression --
    the quantity under test. Use --allow-partial for a progress peek; it
    stamps every table PARTIAL and refuses to write files.

 2. It will not silently drop failed pairs. A pair with status != "ok" is
    excluded from means (there is no number to average) but is REPORTED, and
    if any exist the script says the analysis is not final. Per protocol
    section 5, each must first be classified by guard3_diagnose.py: Class 2 is
    a real Dice-0 result and must be REINSTATED as zero, not left out; Class 1
    or 3 is a pipeline fault to fix and re-run.

 3. It will not compute a delta from unpaired cases. Every delta is
    within-case (rule minus baseline for the same case and region), so a case
    missing its baseline contributes nothing rather than contributing noise.

 OUTPUT
 ------
   b5b_T1_clean.csv · b5b_T2_by_leakclass.csv · b5b_qc_sensitivity.csv
   B5B_REPORT.md
==============================================================================
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

DRIVE   = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W1      = f"{DRIVE}/phase_b_outputs/wave1"
MERGED  = f"{W1}/exp3_suppressed_v3.csv"
QC_CSV  = f"{DRIVE}/brats_qc_taxonomy.csv"
OUT_DIR = f"{DRIVE}/phase_b_outputs/wave2"

BASELINE_RULE = "baseline"
PROTOCOL_RULES = [
    "baseline", "size@50",
    "entropy@0.45", "entropy@0.50", "entropy@0.56",
    "confidence@0.45",
]
REGIONS = ["WT", "TC", "ET", "RC"]
STRATA  = ["clean", "patient_only", "seen"]

# Protocol section 6: pre-registered, before any result was seen.
N_CLEAN, MDE_CLEAN = 155, 0.019
N_CLEAN_QC, MDE_CLEAN_QC = 149, 0.0194
EXPECTED_PAIRS = 567 * len(PROTOCOL_RULES)


# ---------------------------------------------------------------------------
#  LOAD
# ---------------------------------------------------------------------------

def load(allow_partial):
    if os.path.exists(MERGED):
        df = pd.read_csv(MERGED)
        print(f"  source       : {os.path.basename(MERGED)} (merged)")
    else:
        # Read the worker shards directly. Deliberately NOT via
        # merge_shards.py --allow-partial: that would put a file carrying the
        # FINAL name on disk with incomplete contents, and anyone finding it
        # later -- including you, in a month -- would read it as complete.
        # An in-memory union has no such afterlife.
        import glob as _g
        # Strict pattern: exactly ".w<k>of<n>.csv" and nothing after it.
        # A loose "*.w*of*.csv" also matches archived files such as
        # ".w0of3.smoketest.csv", whose rows then collide with the live shard
        # and trip the overlap guard for the wrong reason.
        import re as _re
        _pat = _re.compile(r"\.w\d+of\d+\.csv$")
        sh = sorted(f for f in _g.glob(MERGED.replace(".csv", ".w*of*.csv"))
                    if _pat.search(f))
        if not sh:
            sys.exit(f"missing {MERGED} and no shards — run B-4 first")
        print(f"  source       : {len(sh)} shard(s), in memory "
              f"(no merged file written)")
        for f in sh:
            print(f"                 {os.path.basename(f)}")
        df = pd.concat([pd.read_csv(f) for f in sh], ignore_index=True)
        dup = df.groupby(["case", "setting"]).size()
        dup = dup[dup > len(REGIONS)]
        if len(dup):
            sys.exit(f"{len(dup)} pair(s) appear in more than one shard — "
                     "workers overlapped, investigate before reading anything")
    for col in ("case", "setting", "region", "lesionwise_dice", "leak_class", "status"):
        if col not in df.columns:
            sys.exit(f"{MERGED} has no '{col}' column — wrong schema")

    stale = set(df["setting"]) - set(PROTOCOL_RULES)
    if stale:
        sys.exit(f"table contains non-protocol rules: {stale}")

    n_pairs = len(df.groupby(["case", "setting"]))
    n_cases = df["case"].nunique()
    ok = df[df["status"].astype(str) == "ok"].copy()
    n_failed = (len(df) - len(ok)) // len(REGIONS)

    print(f"  cases        : {n_cases}")
    print(f"  pairs        : {n_pairs} / {EXPECTED_PAIRS}")
    print(f"  failed pairs : {n_failed}")

    complete = (n_pairs == EXPECTED_PAIRS)
    if not complete:
        msg = (f"\n  INCOMPLETE — {EXPECTED_PAIRS - n_pairs} pair(s) missing.\n"
               "  B-4 runs cheapest-first, so a partial table is enriched in "
               "low-lesion-count cases,\n  which is precisely where suppression "
               "cannot do harm. Any mean read now is\n  biased IN FAVOUR of "
               "suppression — the quantity under test.")
        if not allow_partial:
            sys.exit(msg + "\n\n  refusing to run. Use --allow-partial for a "
                           "progress peek (writes nothing).")
        print(msg + "\n  --allow-partial: continuing, tables stamped PARTIAL, "
                    "nothing written.\n")

    if n_failed:
        print(f"\n  {n_failed} failed pair(s) are excluded from means but NOT "
              f"dropped from the record.")
        print("  Classify them with guard3_diagnose.py before treating anything "
              "here as final:")
        print("    Class 2 -> a real Dice-0 result; reinstate as zero")
        print("    Class 1/3 -> pipeline fault; fix and re-run")
        bad = df[df["status"].astype(str) != "ok"]
        for (c, s_), _ in list(bad.groupby(["case", "setting"]))[:10]:
            print(f"      {c} · {s_}")

    return ok, complete, n_failed


# ---------------------------------------------------------------------------
#  PAIRED DELTAS
# ---------------------------------------------------------------------------

def paired(ok):
    """
    Wide frame indexed by (case, region) with one column per rule, plus a
    delta column per non-baseline rule.

    Deltas are WITHIN-CASE by construction: a case that lost its baseline row
    yields NaN and is excluded from that comparison rather than being compared
    against some other case's baseline.
    """
    wide = ok.pivot_table(index=["case", "region"], columns="setting",
                          values="lesionwise_dice", aggfunc="first")

    if BASELINE_RULE not in wide.columns:
        sys.exit("no baseline rows — cannot compute deltas")

    meta = (ok[["case", "leak_class"]].drop_duplicates()
              .set_index("case")["leak_class"])
    wide = wide.reset_index()
    wide["leak_class"] = wide["case"].map(meta)

    for r in PROTOCOL_RULES:
        if r != BASELINE_RULE and r in wide.columns:
            wide[f"d_{r}"] = wide[r] - wide[BASELINE_RULE]

    return wide


def table(wide, mask=None, label=""):
    """Mean Dice per rule x region, and mean paired delta, for a subset."""
    w = wide if mask is None else wide[mask]
    rows = []
    for rule in PROTOCOL_RULES:
        if rule not in w.columns:
            continue
        rec = {"rule": rule}
        for reg in REGIONS:
            sub = w[w.region == reg]
            rec[reg] = round(sub[rule].mean(), 4) if len(sub) else np.nan
        vals = [rec[r] for r in REGIONS if pd.notna(rec.get(r))]
        rec["mean_4"] = round(float(np.mean(vals)), 4) if vals else np.nan
        rec["mean_WT_TC_ET"] = round(
            float(np.mean([rec[r] for r in ("WT", "TC", "ET")
                           if pd.notna(rec.get(r))])), 4)

        if rule != BASELINE_RULE and f"d_{rule}" in w.columns:
            d = w[f"d_{rule}"].dropna()
            rec["delta_mean"] = round(d.mean(), 4) if len(d) else np.nan
            rec["delta_sd"]   = round(d.std(), 4) if len(d) else np.nan
            rec["n_pairs"]    = int(len(d))
            rec["n_worse"]    = int((d < 0).sum())
            rec["n_better"]   = int((d > 0).sum())
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.insert(0, "stratum", label)
    return out


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="peek at an unfinished B-4 (writes nothing)")
    a = ap.parse_args()

    print("=" * 74)
    print(" B-5b — STRATIFICATION")
    print(" protocol: PHASE_B_PROTOCOL.md v1.0 (Frozen 13 Aug 2026)")
    print("=" * 74)

    ok, complete, n_failed = load(a.allow_partial)
    wide = paired(ok)
    stamp = "" if complete else "   [PARTIAL — NOT A RESULT]"

    # ---- T1: the clean stratum, the headline ------------------------------
    clean_mask = wide.leak_class == "clean"
    n_clean = wide.loc[clean_mask, "case"].nunique()
    t1 = table(wide, clean_mask, "clean")

    print("\n" + "=" * 74)
    print(f" T1 · CLEAN STRATUM — n={n_clean} cases (protocol: {N_CLEAN}){stamp}")
    print("=" * 74)
    print(t1.drop(columns=["stratum"]).to_string(index=False))
    print(f"\n  Pre-registered MDE at n={N_CLEAN}: {MDE_CLEAN} Dice.")
    print("  A delta below it is BELOW DETECTION — which is not the same as zero.")

    # ---- T2: all strata ---------------------------------------------------
    t2 = pd.concat([table(wide, wide.leak_class == s, s) for s in STRATA]
                   + [table(wide, None, "all")], ignore_index=True)

    print("\n" + "=" * 74)
    print(f" T2 · BY leak_class{stamp}")
    print("=" * 74)
    for s in STRATA + ["all"]:
        sub = t2[t2.stratum == s]
        n = (wide[wide.leak_class == s]["case"].nunique() if s != "all"
             else wide["case"].nunique())
        print(f"\n  --- {s} (n={n} cases) ---")
        print(sub.drop(columns=["stratum"])[
            ["rule"] + REGIONS + ["mean_4", "delta_mean", "n_worse", "n_better"]
        ].to_string(index=False))

    print("\n  Protocol section 8: an effect present ONLY in patient_only is "
          "memorization\n  and is declared as such, not as a gain.")

    # ---- QC sensitivity (protocol 6.1) ------------------------------------
    qc_tab = None
    if os.path.exists(QC_CSV):
        qc_ids = set(pd.read_csv(QC_CSV)["case_id"])
        m = clean_mask & ~wide["case"].isin(qc_ids)
        n_qc = wide.loc[clean_mask, "case"].isin(qc_ids).sum()
        qc_tab = table(wide, m, "clean_minus_qc")
        print("\n" + "=" * 74)
        print(f" 6.1 · PRE-REGISTERED SENSITIVITY — clean minus QC-flagged{stamp}")
        print("=" * 74)
        print(f"  excluded: {wide.loc[clean_mask, 'case'].isin(qc_ids).sum()} "
              f"case-region rows · n={wide.loc[m, 'case'].nunique()} cases "
              f"(protocol: {N_CLEAN_QC}, MDE {MDE_CLEAN_QC})")
        print(qc_tab.drop(columns=["stratum"])[
            ["rule"] + REGIONS + ["mean_4", "delta_mean"]].to_string(index=False))
    else:
        print(f"\n  [note] {QC_CSV} not found — 6.1 sensitivity skipped")

    # ---- write ------------------------------------------------------------
    if not complete:
        print("\n" + "=" * 74)
        print("  PARTIAL RUN — nothing written. These numbers are not a result.")
        print("=" * 74)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    t1.to_csv(f"{OUT_DIR}/b5b_T1_clean.csv", index=False)
    t2.to_csv(f"{OUT_DIR}/b5b_T2_by_leakclass.csv", index=False)
    if qc_tab is not None:
        qc_tab.to_csv(f"{OUT_DIR}/b5b_qc_sensitivity.csv", index=False)

    md = [
        "# B-5b — Stratification by leak_class", "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Source:** `{os.path.basename(MERGED)}` · "
        f"{wide['case'].nunique()} cases · {len(PROTOCOL_RULES)} rules",
        f"**Protocol:** v1.0 (Frozen 13 Aug 2026), sections 6, 6.1, 9", "",
        "Descriptive only — no significance test is run here. Tests are B-6.", "",
        f"## T1 · clean stratum (n={n_clean})", "",
        t1.drop(columns=["stratum"]).to_markdown(index=False), "",
        f"Pre-registered MDE: **{MDE_CLEAN}** Dice at n={N_CLEAN}. "
        "A delta below it is below detection, not zero.", "",
        "## T2 · by leak_class", "",
        t2.to_markdown(index=False), "",
    ]
    if qc_tab is not None:
        md += ["## 6.1 · Pre-registered QC sensitivity", "",
               qc_tab.drop(columns=["stratum"]).to_markdown(index=False), ""]
    if n_failed:
        md += ["## Failed pairs", "",
               f"{n_failed} pair(s) carry status != ok and are excluded from the "
               "means above. They must be classified with `guard3_diagnose.py` "
               "before these tables are final: Class 2 is a real Dice-0 result "
               "and must be reinstated as zero; Class 1 or 3 is a pipeline fault.", ""]

    open(f"{OUT_DIR}/B5B_REPORT.md", "w", encoding="utf-8").write("\n".join(md))

    print(f"\n  written to {OUT_DIR}/")
    print("    b5b_T1_clean.csv · b5b_T2_by_leakclass.csv · B5B_REPORT.md")
    if n_failed:
        print(f"\n  NOT FINAL — {n_failed} failed pair(s) still unclassified.")
    else:
        print("\n  Ready for B-6 (paired tests).")


if __name__ == "__main__":
    main()
