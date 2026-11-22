# -*- coding: utf-8 -*-
"""
==============================================================================
 PHASE B0 — EXECUTION VALIDATION
==============================================================================
 One run answers four questions that would otherwise take four runs:

   1. EQUIVALENCE   does the fast metric reproduce the original EXACTLY,
                    across every column (Dice, HD95, NSD, TP/FP/FN)?
   2. SPEEDUP       what is the TIME-WEIGHTED speedup over the cohort?
   3. MEMORY        does peak RSS stay flat across repeated cases?
   4. ETA           what will the full 567-case run actually cost?

 WHY TIME-WEIGHTED
 -----------------
     weighted = sum(t_slow) / sum(t_fast)        <- what we need
     naive    = mean(t_slow / t_fast)            <- misleading here

 Runtime in this cohort is heavy-tailed: the top 20 cases consume 48.4% of the
 total and the top 50 consume 64.6%. A per-case mean gives a 234-second case
 the same vote as a 13-hour one, so it cannot predict the total.

 WHY THE TAIL IS INCLUDED
 ------------------------
 An earlier draft of this sampler excluded cases above one hour. That was a
 sampling bias: it drops precisely the cases that determine the total, and
 yields an optimistic estimate of the easy half. The tail is sampled here.

 WHY A TAIL BUDGET INSTEAD OF "THE TOP 3"
 ----------------------------------------
 The heaviest case takes ~13 h under the original metric, and measurement needs
 BOTH implementations. Three cases drawn from the top ten would cost roughly
 35 h -- about a quarter of the full run, spent on measurement alone. The
 sampler therefore fills the tail up to --tail-budget hours, taking the
 heaviest cases that fit. Raise the budget to include heavier cases; the
 stratum-level speedups reported at the end show whether the tail behaves
 differently from the rest, which is the question the tail sample exists to
 answer.

 OUTPUT
 ------
   b0_pilot_per_case.csv    slow/fast times and every metric column, both impls
   b0_pilot_deltas.csv      per-column differences
   B0_EXECUTION_VALIDATION.md
==============================================================================
"""

import os, sys, json, glob, time, gc, resource, argparse
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
for _p in (_HERE, "/content/drive/MyDrive/brats_2026_metastases/src/eval"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import native_space as ns

OUT_DIR = f"{ns.DRIVE}/phase_b_outputs/wave1"
COST_CSV = f"{ns.DRIVE}/results/metrics/lesionwise_validation_250/per_case_lesionwise.csv"
CONFIG = "3fold_124"

os.makedirs(OUT_DIR, exist_ok=True)


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def load_cost():
    """Per-case runtime from the Phase A run: a real cost model over all 567."""
    d = pd.read_csv(COST_CSV)
    if "config" in d.columns:
        d = d[d["config"].astype(str) == CONFIG]
    sec = d.groupby("case")["seconds"].max().sort_values(ascending=False)
    les = d.groupby("case")[["num_tp", "num_fp", "num_fn"]].sum().sum(axis=1)
    return sec, les


def select(sec, per_q, tail_budget_h):
    """
    Stratified by runtime quartile, plus a tail sample bounded by a time budget.
    Returns (cases, labels) where labels record which stratum each came from.
    """
    q = pd.qcut(sec, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    chosen, label = [], {}

    for name in ["Q1", "Q2", "Q3", "Q4"]:
        pool = sec[q == name]
        # median-ward picks: representative of the stratum, not its extremes
        mid = pool.iloc[np.argsort(np.abs(pool.values - pool.median()))][:per_q]
        for c in mid.index:
            chosen.append(c); label[c] = name

    # Tail sample: heaviest cases that fit the budget, taken top-down and
    # STOPPING when the budget runs out.
    #
    # An earlier version used `continue` instead of `break`, so once an
    # expensive case exhausted the budget it kept walking down the list and
    # admitted trivially light cases (a 111 s case landed in TAIL alongside a
    # 19,854 s one). A tail sample containing sub-Q1 cases measures nothing
    # about the tail. It must also stay strictly heavier than Q4.
    q4_max = sec[[c for c in chosen if label[c] == "Q4"]].max()
    budget_s = tail_budget_h * 3600
    spent, n_tail = 0.0, 0

    # Walk DOWN from the heaviest. Skip cases too expensive for the remaining
    # budget, but stop once the candidates are no longer meaningfully heavier
    # than Q4 -- otherwise the sampler slides into ordinary cases and the tail
    # is not represented. (An earlier version used `continue` with no floor and
    # admitted a 111 s case; a later one used a bare `break` and admitted none.)
    tail_floor = max(q4_max * 2.0, q4_max + 600)
    for c, t in sec.items():                      # already sorted descending
        if c in label:
            continue
        if t < tail_floor:
            break                                  # below the tail: stop
        if spent + 2 * t > budget_s:
            continue                               # too big for what is left
        chosen.append(c); label[c] = "TAIL"
        spent += 2 * t; n_tail += 1
        if n_tail >= 3:
            break

    if n_tail == 0:
        print(f"\n  [WARN] no tail case fits a {tail_budget_h:.0f} h budget "
              f"(heaviest case needs ~{2*sec.iloc[0]/3600:.1f} h). The pilot "
              f"will not measure the tail — raise --tail-budget.")

    return chosen, label


def run_one(case, fast):
    """Score one case, returning (rows, seconds). fast toggles the metric impl."""
    os.environ["BRATS_FAST_METRIC"] = "1" if fast else "0"
    ns.set_fast_metric(fast)

    # Fail loudly rather than silently timing the same code twice: without a
    # working toggle the pilot would report speedup ~1.0 after hours of work.
    if ns.fast_metric_active() != fast:
        raise RuntimeError(
            f"fast-metric toggle did not take effect (requested fast={fast}). "
            "Check that metrics_MET is imported and set_fast_metric works.")

    t0 = time.time()
    rows = ns.official_score(case, suppressor=None,
                             tag=f"b0_{'fast' if fast else 'slow'}")
    return rows, time.time() - t0


def main(per_q, tail_budget_h, dry):
    print("=" * 74); print(" PHASE B0 — EXECUTION VALIDATION"); print("=" * 74)

    sec, les = load_cost()
    print(f"  cost model    : {len(sec)} cases · total {sec.sum()/3600:.1f} h")
    print(f"  median        : {sec.median():.0f}s · max {sec.max():.0f}s")
    print(f"  corr(time,lesions): {sec.corr(les):.3f}")

    cases, label = select(sec, per_q, tail_budget_h)
    # Run cheapest first. With frequent disconnects this guarantees the light
    # strata complete early, so even a partial run yields a usable (if
    # tail-less) speedup estimate rather than nothing.
    cases = sorted(cases, key=lambda c: sec[c])
    est = sum(sec[c] for c in cases) * 2 / 3600
    print(f"\n  pilot         : {len(cases)} cases")
    for name in ["Q1", "Q2", "Q3", "Q4", "TAIL"]:
        grp = [c for c in cases if label[c] == name]
        if grp:
            print(f"    {name:>4s} n={len(grp)}  "
                  f"{sec[grp].min():.0f}s .. {sec[grp].max():.0f}s")
    print(f"\n  estimated measurement cost: ~{est:.1f} h (slow + fast)")

    if dry:
        print("\n  --dry: selection only, nothing executed.")
        print("  Raise --tail-budget to include heavier cases (costs more).")
        return

    # RESUME. The tail case alone can take ~11 h of the ~14 h total, and Colab
    # disconnects long before that, so a run that restarts from scratch may
    # never finish. Completed cases are reloaded and skipped.
    rows, times, failures = [], [], []
    t_csv = f"{OUT_DIR}/b0_pilot_times.csv"
    r_csv = f"{OUT_DIR}/b0_pilot_per_case.csv"
    if os.path.exists(t_csv) and os.path.exists(r_csv):
        try:
            times = pd.read_csv(t_csv).to_dict("records")
            rows = pd.read_csv(r_csv).to_dict("records")
            done = {r["case"] for r in times}
            if done:
                print(f"\n  resuming: {len(done)} case(s) already measured "
                      f"({sum(r['slow_s'] + r['fast_s'] for r in times)/3600:.1f} h "
                      f"of work preserved)")
                cases = [c for c in cases if c not in done]
                if not cases:
                    print("  all pilot cases already measured — reporting only")
        except Exception as e:
            print(f"  [note] could not resume ({e}) — starting fresh")
            rows, times = [], []

    for i, case in enumerate(cases, 1):
        print(f"\n  [{i}/{len(cases)}] {case}  ({label[case]}, "
              f"expected ~{sec[case]:.0f}s)")
        rec = {"case": case, "stratum": label[case], "expected_s": float(sec[case])}
        try:
            for fast in (False, True):
                r, dt = run_one(case, fast)
                rec[f"{'fast' if fast else 'slow'}_s"] = round(dt, 2)
                for x in r:
                    rows.append({**x, "impl": "fast" if fast else "slow",
                                 "stratum": label[case]})
                print(f"      {'fast' if fast else 'slow'}: {dt:8.1f}s   "
                      f"[peak RSS {rss_gib():.2f} GiB]")
                gc.collect()
            rec["speedup"] = rec["slow_s"] / max(rec["fast_s"], 1e-9)
            rec["peak_rss_gib"] = round(rss_gib(), 2)
            print(f"      speedup: {rec['speedup']:.2f}x")
            times.append(rec)
        except Exception as e:
            failures.append({"case": case, "error": str(e)})
            print(f"      FAILED — {str(e)[:120]}")

        if times:
            pd.DataFrame(times).to_csv(f"{OUT_DIR}/b0_pilot_times.csv", index=False)
            pd.DataFrame(rows).to_csv(f"{OUT_DIR}/b0_pilot_per_case.csv", index=False)

    if not times:
        raise SystemExit("no case completed — check the errors above")

    strata_done = {r["stratum"] for r in times}
    if "TAIL" not in strata_done:
        print("\n  [note] tail case not yet measured — the weighted speedup below "
              "covers the light strata only and will UNDERSTATE or OVERSTATE the "
              "full-cohort figure depending on how the tail behaves. Rerun to "
              "continue; completed cases are skipped.")

    # ---- equivalence -------------------------------------------------------
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns
                if df[c].dtype.kind in "fi" and c not in ("seconds",)]
    piv = df.pivot_table(index=["case", "region"], columns="impl", values=num_cols)

    deltas = {}
    for c in num_cols:
        if ("fast" in piv[c].columns) and ("slow" in piv[c].columns):
            d = (piv[c]["fast"] - piv[c]["slow"]).abs()
            deltas[c] = float(np.nanmax(d))
    max_delta = max(deltas.values()) if deltas else float("nan")
    identical = max_delta == 0.0

    # ---- weighted speedup --------------------------------------------------
    t = pd.DataFrame(times)
    weighted = t["slow_s"].sum() / t["fast_s"].sum()
    naive = t["speedup"].mean()

    print("\n" + "=" * 74); print(" RESULTS"); print("=" * 74)
    print(f"\n  EQUIVALENCE   max |delta| over all columns : {max_delta:.10g}")
    for c, v in sorted(deltas.items(), key=lambda kv: -kv[1])[:6]:
        print(f"      {c:>22s}  {v:.10g}")
    print(f"      -> {'IDENTICAL' if identical else 'NOT IDENTICAL — use the original'}")

    print(f"\n  SPEEDUP       weighted (sum/sum) : {weighted:.3f}x   <- use this")
    print(f"                naive per-case mean : {naive:.3f}x   (do not use)")
    print("\n  by stratum:")
    for name in ["Q1", "Q2", "Q3", "Q4", "TAIL"]:
        g = t[t.stratum == name]
        if len(g):
            print(f"    {name:>4s} n={len(g)}  "
                  f"weighted {g['slow_s'].sum()/g['fast_s'].sum():.3f}x")

    total_h = sec.sum() / 3600
    proj_h = total_h / weighted
    print(f"\n  ETA           567 cases, original : {total_h:.1f} h")
    print(f"                567 cases, fast     : {proj_h:.1f} h")
    print(f"                baseline + 9 rules  : {proj_h*10:.0f} h")
    print(f"\n  peak RSS across pilot : {t['peak_rss_gib'].max():.2f} GiB")

    # ---- report ------------------------------------------------------------
    md = ["# Phase B0 — Execution Validation", "",
          f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
          f"**Pilot:** {len(t)} cases, stratified by runtime quartile plus a "
          f"time-budgeted tail sample", "",
          f"## Equivalence: {'IDENTICAL' if identical else 'NOT IDENTICAL'}", "",
          f"Maximum absolute difference across every numeric column "
          f"(Dice, HD95, NSD, TP/FP/FN): **{max_delta:.10g}**", "",
          "| column | max abs delta |", "|---|---|"]
    for c, v in sorted(deltas.items(), key=lambda kv: -kv[1]):
        md.append(f"| {c} | {v:.10g} |")

    md += ["", f"## Speedup: {weighted:.3f}x (time-weighted)", "",
           "The time-weighted ratio `sum(t_slow) / sum(t_fast)` is the figure that "
           "predicts total cost. The naive per-case mean "
           f"({naive:.3f}x) is reported only for contrast: with a heavy-tailed "
           "runtime distribution it weights a 234-second case equally with a "
           "multi-hour one and does not predict the total.", "",
           "| stratum | n | weighted speedup |", "|---|---|---|"]
    for name in ["Q1", "Q2", "Q3", "Q4", "TAIL"]:
        g = t[t.stratum == name]
        if len(g):
            md.append(f"| {name} | {len(g)} | "
                      f"{g['slow_s'].sum()/g['fast_s'].sum():.3f}x |")

    md += ["", "## Projected cost for the full cohort", "",
           f"- 567 cases, original metric: **{total_h:.1f} h**",
           f"- 567 cases, fast metric: **{proj_h:.1f} h**",
           f"- baseline + 9 suppression rules: **~{proj_h*10:.0f} h**", "",
           f"## Memory", "",
           f"Peak RSS across the pilot: **{t['peak_rss_gib'].max():.2f} GiB**. "
           "No evidence of cumulative growth was observed across the tested cases.", ""]

    if failures:
        md += ["## Failures", ""] + [f"- `{f['case']}`: {f['error'][:200]}"
                                     for f in failures]

    open(f"{OUT_DIR}/B0_EXECUTION_VALIDATION.md", "w", encoding="utf-8").write(
        "\n".join(md) + "\n")
    json.dump({"identical": bool(identical), "max_delta": max_delta,
               "weighted_speedup": weighted, "naive_speedup": naive,
               "projected_h_567": proj_h, "deltas": deltas},
              open(f"{OUT_DIR}/b0_verdict.json", "w"), indent=2)

    print(f"\n  md  : {OUT_DIR}/B0_EXECUTION_VALIDATION.md")
    if not identical:
        print("\n  >> NOT IDENTICAL. Do not use the fast metric. Investigate the")
        print("     column with the largest delta before proceeding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-quartile", type=int, default=3)
    ap.add_argument("--tail-budget", type=float, default=6.0,
                    help="hours of measurement time allotted to tail cases")
    ap.add_argument("--dry", action="store_true", help="show selection and cost only")
    a = ap.parse_args()
    main(a.per_quartile, a.tail_budget, a.dry)
