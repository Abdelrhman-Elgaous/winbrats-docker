# -*- coding: utf-8 -*-
"""
==============================================================================
 T4 + T3 — DETECTION COUNTS AND PER-LESION TABLE  ·  clean stratum only
==============================================================================
 RUNTIME: CPU. 155 cases x 7 rules = 1,085 pairs. ~231 h serial; ~26 h across
 nine workers, realistically 40-60 h once the tail is included.
 PROTOCOL: PHASE_B_PROTOCOL.md v1.0 (Frozen 13 Aug 2026), section 12.

 WHAT THIS IS FOR
 ----------------
 B-4 answered the Dice question on all 567 cases. It could not answer the
 detection question, because native_space.official_score kept four of the
 fifteen columns the evaluator returns and discarded the per-lesion table
 entirely. Those columns are computed and thrown away on every call, and they
 cannot be recovered after the fact -- hence this separate run.

 TWO OUTPUTS, TWO SOURCES OF TRUTH, NEVER MIXED
 ----------------------------------------------
   t4_counts_*.csv    from results_df -- one row per (case, rule, region).
                      Num_TP / Num_FP / Num_FN plus Sensitivity, Specificity,
                      NSD @ 0.5 / @ 1.0 and HD95. This is the source of truth
                      for detection F1. Nothing is inferred from it.

   t3_lesions_*.csv   from final_lesionwise_metrics_df -- one row per
                      REFERENCE lesion. Carries gt_lesion_vol (a VOXEL COUNT),
                      dice_lesionwise, and `_len` (0 = undetected).
                      This is the size axis for T3.

 They are written as separate files on purpose. Detection counts and lesion
 sizes answer different questions, and a single wide table invites deriving one
 from the other.

 WHAT T3 CAN AND CANNOT DO  (protocol section 12.9)
 --------------------------------------------------
   Delta lesion-wise Dice by GT size bin    YES
   Recall / FN behaviour by GT size bin     YES
   Full F1 by GT size bin                   NO -- an FP has no reference
                                            lesion, so it has no size bin. Its
                                            denominator cannot be formed.
   FP analysis by predicted-component size  YES, but as a separate axis.

 SEVEN RULES
 -----------
 The six frozen protocol rules plus size@20. size@20 is here because the 4 Aug
 change log assigned it to the detection analysis, and this is that analysis --
 executing a prior assignment, not widening the search after seeing B-6. Its
 Dice values are retained for audit only and are excluded from the primary Dice
 analysis (protocol section 12).

 B-4 IS NOT TOUCHED. Different output filenames, different directory. Nothing
 here reads or writes exp3_suppressed_v3*.
==============================================================================
"""

import os
import sys
import json
import glob
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
for _p in (_HERE, "/content/drive/MyDrive/brats_2026_metastases/src/eval"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import native_space as ns


# ---------------------------------------------------------------------------
#  PATHS  --  wave2, entirely separate from B-4's wave1
# ---------------------------------------------------------------------------

OUT_DIR   = f"{ns.DRIVE}/phase_b_outputs/wave2"
T4_CSV    = f"{OUT_DIR}/t4_counts.csv"
T3_CSV    = f"{OUT_DIR}/t3_lesions.csv"
FAIL_LOG  = f"{OUT_DIR}/t4_failures.json"
LEAK_CSV  = f"{ns.DRIVE}/phase_b_outputs/wave0/leak_class_567.csv"

REGIONS   = ("WT", "TC", "ET", "RC")
CKPT_EVERY = 1          # a single tail evaluation can exceed 12 h; lose one

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  RULES  --  protocol section 12
# ---------------------------------------------------------------------------

RULES = {
    "baseline":        (None,         None),
    "size@20":         ("size",       20),      # section 12: detection analysis
    "size@50":         ("size",       50),
    "entropy@0.45":    ("entropy",    0.45),
    "entropy@0.50":    ("entropy",    0.50),
    "entropy@0.56":    ("entropy",    0.56),
    "confidence@0.45": ("confidence", 0.45),
}

T4_RULES = {
    "baseline", "size@20", "size@50",
    "entropy@0.45", "entropy@0.50", "entropy@0.56", "confidence@0.45",
}

assert set(RULES) == T4_RULES, (
    f"RULES drift vs PHASE_B_PROTOCOL.md section 12: {set(RULES) ^ T4_RULES}"
)

_SUPPRESSORS = {
    "size":       ns.size_suppressor,
    "entropy":    ns.entropy_suppressor,
    "confidence": ns.confidence_suppressor,
}


def make_suppressor(rule):
    kind, val = RULES[rule]
    return None if kind is None else _SUPPRESSORS[kind](val)


def _shard(path, worker, n_workers):
    if n_workers <= 1:
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}.w{worker}of{n_workers}{ext}"


def _all_shards(path):
    import re
    stem, ext = os.path.splitext(path)
    pat = re.compile(r"\.w\d+of\d+" + re.escape(ext) + r"$")
    return sorted([f for f in glob.glob(f"{stem}.w*of*{ext}") if pat.search(f)]
                  + ([path] if os.path.exists(path) else []))


# ---------------------------------------------------------------------------
#  COHORT
# ---------------------------------------------------------------------------

def clean_cases():
    """
    The 155 pre-registered clean-stratum cases, ordered cheapest first so a
    disconnect costs the least. Timing comes from B-4's completed table when
    it is available -- read only, never written.
    """
    if not os.path.exists(LEAK_CSV):
        sys.exit(f"cohort file missing: {LEAK_CSV}")
    lk = pd.read_csv(LEAK_CSV)
    cases = sorted(lk.loc[lk.leak_class == "clean", "case"])
    if len(cases) != 155:
        print(f"  [!] clean stratum has {len(cases)} cases, protocol says 155")

    b4 = f"{ns.DRIVE}/phase_b_outputs/wave1/exp3_suppressed_v3.csv"
    if os.path.exists(b4):
        sec = pd.read_csv(b4).groupby("case").seconds.first()
        cases.sort(key=lambda c: (float(sec.get(c, 1e9)), c))
        print("  order    : cheapest first (timings from the completed B-4 table)")
    else:
        print("  order    : alphabetical (no B-4 timings found)")
    return cases


# ---------------------------------------------------------------------------
#  RESUME
# ---------------------------------------------------------------------------

def load_done(t4_path):
    """
    Seen pairs from EVERY shard; rows to rewrite from THIS shard only.

    Copying a sibling's rows into this process's buffer would duplicate them at
    merge time, because each checkpoint is a full to_csv rather than an append.
    """
    seen, own = set(), []
    if os.path.exists(t4_path):
        d = pd.read_csv(t4_path)
        own = d.to_dict("records")
        seen |= set(zip(d.case, d.setting))
        n_bad = int((d.status.astype(str) != "ok").sum()) // len(REGIONS)
        print(f"  resume   : {len(d)} rows in {os.path.basename(t4_path)} "
              f"({n_bad} failed pair(s))")

    extra = 0
    for f in _all_shards(T4_CSV):
        if os.path.abspath(f) == os.path.abspath(t4_path):
            continue
        d = pd.read_csv(f)
        before = len(seen)
        seen |= set(zip(d.case, d.setting))
        extra += len(seen) - before
    if extra:
        print(f"  siblings : {extra} pair(s) done elsewhere — skipped, not copied")
    return seen, own


# ---------------------------------------------------------------------------
#  PREFLIGHT
# ---------------------------------------------------------------------------

def preflight(case):
    """
    One evaluation before the run proper, checking that the columns this whole
    exercise exists to capture are actually present. Cheap; a schema surprise
    discovered 30 h in is not.
    """
    print("\n  --- preflight: one evaluation, schema check ---")
    rows, agg, les = ns.official_score(case, suppressor=None, tag="t4_preflight",
                                       guards=False, full=True)

    need_agg = ["Labels", "Num_TP", "Num_FP", "Num_FN",
                "LesionWise_Score_Dice", "Sensitivity", "Specificity"]
    need_les = ["gt_lesion_numbers", "gt_lesion_vol", "dice_lesionwise", "_len", "Label"]

    miss_a = [c for c in need_agg if c not in agg.columns]
    miss_l = [c for c in need_les if c not in les.columns]
    if miss_a or miss_l:
        sys.exit(f"  SCHEMA FAILURE — aggregate missing {miss_a}, "
                 f"per-lesion missing {miss_l}\n"
                 "  native_space.official_score(full=True) is not returning what "
                 "T3/T4 need. Do not start the run.")
    if len(agg) != len(REGIONS):
        sys.exit(f"  aggregate has {len(agg)} rows, expected {len(REGIONS)}")

    print(f"  aggregate  : {len(agg)} rows x {len(agg.columns)} cols")
    print(f"  per-lesion : {len(les)} rows x {len(les.columns)} cols")
    print(f"  counts     : " + " · ".join(
        f"{r.Labels}: TP={int(r.Num_TP)} FP={int(r.Num_FP)} FN={int(r.Num_FN)}"
        for _, r in agg.iterrows()))
    print("  schema OK\n")


# ---------------------------------------------------------------------------
#  SCORING
# ---------------------------------------------------------------------------

def score_pair(case, rule, t0):
    """Both tables, flattened. Always returns rows for T4; T3 rows may be empty."""
    tag = f"t4_{rule.replace('@', '_at_').replace('.', 'p')}"
    _, agg, les = ns.official_score(case, suppressor=make_suppressor(rule),
                                    tag=tag, guards=False, full=True)
    dt = time.time() - t0

    t4 = []
    for _, r in agg.iterrows():
        rec = {"case": case, "setting": rule,
               "region": str(r["Labels"]).strip(),
               "seconds": round(dt, 2), "status": "ok"}
        for c in agg.columns:
            if c != "Labels":
                rec[c] = r[c]
        t4.append(rec)

    t3 = []
    for _, r in les.iterrows():
        rec = {"case": case, "setting": rule}
        for c in les.columns:
            rec[c] = r[c]
        # predicted_lesion_numbers is a list; store it as text so the CSV
        # round-trips without pandas turning it into something else.
        if "predicted_lesion_numbers" in rec:
            rec["predicted_lesion_numbers"] = str(rec["predicted_lesion_numbers"])
        t3.append(rec)

    return t4, t3


def fail_rows(case, rule, seconds, detail):
    """
    Metric columns stay EMPTY on failure, never 0. Zero is a real score here
    (suppression emptied the region); writing it for "did not run" would
    corrupt the counts this run exists to produce.
    """
    return [{"case": case, "setting": rule, "region": reg,
             "seconds": round(seconds, 2), "status": detail}
            for reg in REGIONS]


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="explicit case IDs — rescue only, applied after the "
                         "worker split")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-preflight", action="store_true")
    a = ap.parse_args()

    print("=" * 74)
    print(" T4 + T3 — DETECTION COUNTS AND PER-LESION TABLE")
    print(" protocol: PHASE_B_PROTOCOL.md section 12 · clean stratum only")
    print("=" * 74)

    # native_space caches the evaluator and chdir()s on its first call, so the
    # per-process scratch has to be set before anything triggers that.
    if a.n_workers > 1:
        if not 0 <= a.worker < a.n_workers:
            raise SystemExit(f"--worker must be in [0,{a.n_workers-1}]")
        ns.WORK_DIR = os.path.join(ns.WORK_DIR, f"t4_w{a.worker}")
        os.makedirs(ns.WORK_DIR, exist_ok=True)
        print(f"  scratch  : {ns.WORK_DIR}")

    os.environ.setdefault("BRATS_FAST_METRIC", "1")
    print(f"  fast     : BRATS_FAST_METRIC={os.environ['BRATS_FAST_METRIC']}")

    if not hasattr(ns.official_score, "__wrapped__"):
        import inspect
        if "full" not in inspect.signature(ns.official_score).parameters:
            sys.exit("  native_space.official_score has no `full` parameter — "
                     "the patched version is not installed.")

    cases = clean_cases()
    if a.n_workers > 1:
        cases = cases[a.worker::a.n_workers]
        print(f"  worker   : {a.worker} of {a.n_workers}")
    if a.cases:
        bad = [c for c in a.cases if c not in cases]
        if bad:
            raise SystemExit(f"not in this worker's slice: {bad}")
        cases = [c for c in cases if c in a.cases]
        print(f"  cases    : EXPLICIT — {cases}")

    sel = [r for r in RULES if (not a.rules or r in a.rules)]
    if a.rules:
        unknown = set(a.rules) - set(RULES)
        if unknown:
            raise SystemExit(f"unknown rules: {unknown}")
    print(f"  cases    : {len(cases)}")
    print(f"  rules    : {sel}")

    t4_path = _shard(T4_CSV, a.worker, a.n_workers)
    t3_path = _shard(T3_CSV, a.worker, a.n_workers)
    fl_path = _shard(FAIL_LOG, a.worker, a.n_workers)

    seen, t4_rows = load_done(t4_path)
    t3_rows = pd.read_csv(t3_path).to_dict("records") if os.path.exists(t3_path) else []

    todo = [(c, r) for c in cases for r in sel if (c, r) not in seen]
    if a.limit:
        todo = todo[:a.limit]
    print(f"  already  : {len(seen)} case-rule pairs")
    print(f"  to do    : {len(todo)}")
    print(f"  output   : {os.path.basename(t4_path)} · {os.path.basename(t3_path)}")

    if not todo:
        print("\n  nothing to do — already complete")
        return

    if not a.skip_preflight:
        preflight(todo[0][0])

    failures = json.load(open(fl_path)) if os.path.exists(fl_path) else []
    t_start = time.time()

    for i, (case, rule) in enumerate(todo, 1):
        t0 = time.time()
        try:
            t4, t3 = score_pair(case, rule, t0)
            t4_rows += t4
            t3_rows += t3
            dt = time.time() - t0
            wt = next((r for r in t4 if r["region"] == "WT"), {})
            print(f"  [{i}/{len(todo)}] {case} · {rule:<16} "
                  f"TP={int(wt.get('Num_TP', -1))} FP={int(wt.get('Num_FP', -1))} "
                  f"FN={int(wt.get('Num_FN', -1))} · {len(t3)} lesions · {dt:.0f}s")
        except Exception as e:
            dt = time.time() - t0
            detail = f"FAILED: {type(e).__name__}: {e}"
            t4_rows += fail_rows(case, rule, dt, detail)
            failures.append({"case": case, "rule": rule, "error": detail,
                             "when": datetime.now().isoformat(timespec="seconds")})
            print(f"  [{i}/{len(todo)}] {case} · {rule:<16} RECORDED AS FAILED — "
                  f"{str(e)[:70]}")

        if i % CKPT_EVERY == 0 or i == len(todo):
            pd.DataFrame(t4_rows).to_csv(t4_path, index=False)
            pd.DataFrame(t3_rows).to_csv(t3_path, index=False)
            json.dump(failures, open(fl_path, "w"), indent=2)

    # ---- summary ----------------------------------------------------------
    t4d = pd.DataFrame(t4_rows)
    ok = t4d[t4d.status.astype(str) == "ok"]
    n_fail = (len(t4d) - len(ok)) // len(REGIONS)

    print("\n" + "=" * 74)
    print(" SUMMARY")
    print("=" * 74)
    print(f"  T4 rows      : {len(t4d)}  ({n_fail} failed pair(s))")
    print(f"  T3 rows      : {len(t3_rows)}")
    print(f"  wall time    : {timedelta(seconds=int(time.time() - t_start))}")

    if len(ok):
        det = ok.groupby("setting")[["Num_TP", "Num_FP", "Num_FN"]].sum()
        det["precision"] = det.Num_TP / (det.Num_TP + det.Num_FP).replace(0, np.nan)
        det["recall"] = det.Num_TP / (det.Num_TP + det.Num_FN).replace(0, np.nan)
        det["F1"] = 2 * det.precision * det.recall / (det.precision + det.recall)
        print("\n  detection totals across all regions — NOT the analysis:")
        print(det.round(4).to_string())
        print("\n  The reported figures are per region, on the clean stratum,")
        print("  with paired tests. This block is a sanity check only.")

    if n_fail:
        print(f"\n  {n_fail} failed pair(s) are IN the table with status set.")
        print("  Diagnose before analysing: a Class-2 failure is a real result,")
        print("  a Class-1/3 failure is a pipeline fault to fix and re-run.")

    print(f"\n  {t4_path}\n  {t3_path}")


if __name__ == "__main__":
    main()
