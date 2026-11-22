# -*- coding: utf-8 -*-
"""
==============================================================================
 MERGE B-4 WORKER SHARDS
==============================================================================
 Combines exp3_suppressed_v3.w{k}of{n}.csv into exp3_suppressed_v3.csv and
 refuses to produce a merged file that is not provably complete.

 WHY THIS IS NOT A ONE-LINE pd.concat
 ------------------------------------
 Every checkpoint write in b4_suppressed_arm.py is a full to_csv, not an
 append. If two workers ever shared a path, the loser's rows are gone and the
 file still parses cleanly -- the failure mode is a SHORT table, not a corrupt
 one. Silently short tables are what invalidated Exp3, so the merge asserts
 rather than trusts:

   1. no (case, rule) pair appears in more than one shard
   2. every pair appears exactly 4 times (one row per region)
   3. the union is exactly 567 x 6 = 3,402 pairs
   4. only rules from the frozen protocol are present
   5. failed pairs are reported, never dropped

 A merge that cannot satisfy 1-4 exits non-zero and writes nothing.
==============================================================================
"""

import os
import sys
import glob
import argparse

import pandas as pd

DRIVE    = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
OUT_DIR  = f"{DRIVE}/phase_b_outputs/wave1"
TARGET   = f"{OUT_DIR}/exp3_suppressed_v3.csv"
PATTERN  = f"{OUT_DIR}/exp3_suppressed_v3.w*of*.csv"

PROTOCOL_RULES = {
    "baseline", "size@50",
    "entropy@0.45", "entropy@0.50", "entropy@0.56",
    "confidence@0.45",
}
N_CASES   = 567
N_REGIONS = 4
EXPECTED_PAIRS = N_CASES * len(PROTOCOL_RULES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the merge even if incomplete (inspection only; "
                         "the result must NOT be analysed)")
    a = ap.parse_args()

    # Strict pattern. Accepts the case split alone (.wKofN.csv) and the case
    # split combined with the rule split (.wKofN.pJofM.csv). Rejects archived
    # files such as ".w0of3.smoketest.csv", whose rows would otherwise collide
    # with a live shard and trip the overlap guard for the wrong reason.
    import re
    _pat = re.compile(r"\.w\d+of\d+(\.p\d+of\d+)?\.csv$")
    shards = sorted(f for f in glob.glob(PATTERN) if _pat.search(f))
    if not shards:
        sys.exit(f"no shards matching {PATTERN}")

    print("=" * 74)
    print(" MERGE B-4 SHARDS")
    print("=" * 74)

    frames, seen_owner, problems = [], {}, []

    for path in shards:
        df = pd.read_csv(path)
        name = os.path.basename(path)

        if "status" not in df.columns:
            sys.exit(f"{name} has no 'status' column — wrong schema, refusing to merge")

        stale = set(df["setting"].astype(str)) - PROTOCOL_RULES
        if stale:
            sys.exit(f"{name} contains non-protocol rules: {stale}")

        pairs = set(zip(df["case"], df["setting"]))
        for pr in pairs:
            if pr in seen_owner:
                problems.append(f"{pr} in both {seen_owner[pr]} and {name}")
            else:
                seen_owner[pr] = name

        ok = int((df["status"].astype(str) == "ok").sum())
        print(f"  {name:<46} {len(df):>6} rows · {len(pairs):>5} pairs · "
              f"{(len(df)-ok)//N_REGIONS} failed")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    print("-" * 74)
    if problems:
        print(f"  OVERLAP: {len(problems)} pair(s) written by more than one shard")
        for p in problems[:10]:
            print(f"    {p}")
        sys.exit("  refusing to merge — shards are not disjoint")

    counts = merged.groupby(["case", "setting"]).size()
    wrong = counts[counts != N_REGIONS]
    if len(wrong):
        print(f"  {len(wrong)} pair(s) do not have exactly {N_REGIONS} region rows:")
        print(wrong.head(10).to_string())
        sys.exit("  refusing to merge — malformed pairs")

    n_pairs = len(counts)
    n_cases = merged["case"].nunique()
    ok = merged[merged["status"].astype(str) == "ok"]
    n_failed = (len(merged) - len(ok)) // N_REGIONS

    print(f"  cases         : {n_cases} / {N_CASES}")
    print(f"  pairs         : {n_pairs} / {EXPECTED_PAIRS}")
    print(f"  rows          : {len(merged)}")
    print(f"  failed pairs  : {n_failed}")

    complete = (n_pairs == EXPECTED_PAIRS and n_cases == N_CASES)
    if not complete:
        missing = EXPECTED_PAIRS - n_pairs
        print(f"\n  INCOMPLETE — {missing} pair(s) missing.")
        have = set(zip(merged["case"], merged["setting"]))
        gaps = {}
        for (c, r) in have:
            pass
        for c in merged["case"].unique():
            miss = [r for r in PROTOCOL_RULES if (c, r) not in have]
            if miss:
                gaps[c] = miss
        for c, m in list(gaps.items())[:10]:
            print(f"    {c}: missing {sorted(m)}")
        if not a.allow_partial:
            sys.exit("\n  not writing. Finish the run, or pass --allow-partial "
                     "for inspection only (the result must NOT be analysed).")

    if os.path.exists(TARGET):
        sys.exit(f"\n  {os.path.basename(TARGET)} already exists — archive it first.")

    merged.to_csv(TARGET, index=False)
    print(f"\n  written: {TARGET}")

    if n_failed:
        print(f"\n  {n_failed} failed pair(s) are IN the file with status set.")
        print("  Diagnose with guard3_diagnose.py before the statistical analysis:")
        print("  Class 2 is a real Dice-0 result and stays; Class 1 or 3 is a")
        print("  pipeline fault that must be fixed and re-run.")
        bad = merged[merged["status"].astype(str) != "ok"]
        for (c, s_), _ in list(bad.groupby(["case", "setting"]))[:15]:
            print(f"    {c} · {s_}")

    if complete and not n_failed:
        print("\n  COMPLETE — 3,402 pairs, no failures. Ready for B-5b.")


if __name__ == "__main__":
    main()
