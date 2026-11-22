# -*- coding: utf-8 -*-
"""
==============================================================================
 SIZE PILOT — is size@100 worth entering the full run?
==============================================================================
 RUNTIME: CPU. Budgeted; see --time-budget.

 THE QUESTION
 ------------
 The component-size analysis established that size_suppressor works and that
 size@20 CANNOT move lesion-wise Dice: it only removes components below the
 evaluator's minimum lesion size [CORRECTED 16 Aug 2026: this said '27-voxel'. BraTS-MET sets lesion_volume_thresh = 2, applied as 2 voxels to predicted components and 2 mm3 to reference lesions. The figure 27 was an undocumented project assumption that entered through this very comment — see protocol section 13], which were never scored. size@50
 and size@100 are the first thresholds that reach scorable components.

 But "removes more" is not "improves more". At thr=50 on BraTS-MET-01101-002,
 6 scorable WT components fall below the threshold -- and nothing yet says
 whether they are false positives (removing them helps) or true positives
 (removing them hurts). The cohort-level counts argue for caution: WT has
 FN=729 against FP=282, so this model misses lesions more often than it invents
 them, and small components are not obviously spurious.

 SAMPLE SELECTION -- AND WHY NOT THE OBVIOUS ONE
 -----------------------------------------------
 The tempting design is to pick cases that HAVE scorable components below the
 threshold, since only those can change. That is selection on the variable
 under test: it guarantees a visible effect and inflates whatever is measured.

 Instead the sample is stratified on LESION COUNT, taken from the published
 per-case file and fixed before any suppression is applied. Lesion count is
 correlated with the opportunity for suppression but is not the quantity being
 tested, so the sample stays neutral.

 The number of cases that turn out to contain scorable components is then a
 RESULT, not a selection criterion. If 17 of 20 cases have none, that is
 informative: the rule rarely intervenes at all, which is part of judging its
 value.

 WHAT IT REPORTS
 ---------------
   delta Dice per rule, per region, and per leak_class
   how many cases actually changed
   how many cases COULD have changed (scorable components present)
   the conditional effect: delta among cases that could change

 The last of these is the honest way to get the sensitive-subgroup number
 without selecting the sample on it.

 OUTPUT: size_pilot_per_case.csv · SIZE_PILOT_REPORT.md
==============================================================================
"""

import os, sys, json, glob, time, argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
for _p in (_HERE, "/content/drive/MyDrive/brats_2026_metastases/src/eval"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import native_space as ns

OUT_DIR   = f"{ns.DRIVE}/phase_b_outputs/wave1"
COST_CSV  = f"{ns.DRIVE}/results/metrics/lesionwise_validation_250/per_case_lesionwise.csv"
LEAK_CSV  = f"{ns.DRIVE}/phase_b_outputs/wave0/leak_class_567.csv"
CONFIG    = "3fold_124"

RULES     = {"baseline": None, "size@50": 50, "size@100": 100}
# The evaluator's floor is 2, not 27: metrics_MET.py line 681 sets
# lesion_volume_thresh = 2, applied as 2 voxels to predicted components
# (line 319) and as 2 mm3 to reference lesions (743, 747, 761). The
# figure 27 was an undocumented project assumption; see protocol §13.
# Nothing published depended on it — this fed a diagnostic column only.
SCORABLE  = 2
REGIONS   = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)}
CH        = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}

os.makedirs(OUT_DIR, exist_ok=True)


def load_tables():
    d = pd.read_csv(COST_CSV)
    if "config" in d.columns:
        d = d[d["config"].astype(str) == CONFIG]
    lesions = d.groupby("case")[["num_tp", "num_fp", "num_fn"]].sum().sum(axis=1)
    seconds = d.groupby("case")["seconds"].max()
    leak = {}
    if os.path.exists(LEAK_CSV):
        lk = pd.read_csv(LEAK_CSV)
        leak = dict(zip(lk["case"], lk["leak_class"]))
    return lesions, seconds, leak


def select(lesions, seconds, n_per_stratum, budget_h):
    """
    Stratify on LESION COUNT -- available before any suppression is applied and
    not the quantity under test. Within each stratum, take cases closest to the
    stratum median, cheapest first, until the time budget is spent.
    """
    q = pd.qcut(lesions, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    chosen, label = [], {}
    spent, budget_s = 0.0, budget_h * 3600

    for name in list(q.cat.categories):
        pool = lesions[q == name]
        order = pool.iloc[np.argsort(np.abs(pool.values - pool.median()))]
        taken = 0
        for c in order.index:
            cost = 3 * seconds.get(c, 300)          # three rules per case
            if spent + cost > budget_s:
                continue
            chosen.append(c); label[c] = name
            spent += cost; taken += 1
            if taken >= n_per_stratum:
                break
    return chosen, label, spent / 3600


def scorable_components(mean_prob, thr_voxels):
    """
    Components that are BOTH below the suppression threshold and above the
    evaluator's minimum lesion size -- the only ones whose removal can move
    lesion-wise Dice.
    """
    out = {}
    for name, ch in CH.items():
        lab, n = ndimage.label(mean_prob[ch] >= ns.THR[name])
        if n == 0:
            out[name] = 0; continue
        s = np.bincount(lab.ravel())[1:]
        out[name] = int(((s <= thr_voxels) & (s > SCORABLE)).sum())
    return out


def main(n_per_stratum, budget_h):
    print("=" * 74); print(" SIZE PILOT — baseline vs size@50 vs size@100"); print("=" * 74)

    lesions, seconds, leak = load_tables()
    cases, label, est = select(lesions, seconds, n_per_stratum, budget_h)

    print(f"  stratified on lesion count (not on suppression opportunity)")
    print(f"  sample        : {len(cases)} cases · estimated {est:.1f} h")
    for name in ["Q1", "Q2", "Q3", "Q4"]:
        g = [c for c in cases if label.get(c) == name]
        if g:
            print(f"    {name} n={len(g):>2}  lesions {int(lesions[g].min()):>3}"
                  f"..{int(lesions[g].max()):>4}")

    rows = []
    for i, c in enumerate(cases, 1):
        print(f"\n  [{i}/{len(cases)}] {c}  ({label[c]}, {int(lesions.get(c,0))} lesions)")
        try:
            mean = ns.load_mean_prob(c)
            opportunity = {t: scorable_components(mean, t) for t in (50, 100)}
            del mean
        except Exception as e:
            print(f"      skipped: {e}"); continue

        per_rule = {}
        ok = True
        for rule, thr in RULES.items():
            sup = ns.size_suppressor(thr) if thr else None
            try:
                t0 = time.time()
                res = ns.official_score(c, suppressor=sup, tag=f"pilot_{rule.replace('@','_at_')}")
                per_rule[rule] = {r["region"]: r["lesionwise_dice"] for r in res}
                print(f"      {rule:>10s}: " +
                      " ".join(f"{r['region']}={r['lesionwise_dice']:.4f}" for r in res) +
                      f"   {time.time()-t0:.0f}s")
            except Exception as e:
                print(f"      {rule:>10s}: FAILED — {str(e)[:90]}")
                ok = False; break
        if not ok:
            continue

        for rule in RULES:
            if rule == "baseline":
                continue
            thr = RULES[rule]
            for reg in REGIONS:
                b = per_rule["baseline"].get(reg)
                v = per_rule[rule].get(reg)
                if b is None or v is None:
                    continue
                rows.append({
                    "case": c, "stratum": label[c], "rule": rule, "region": reg,
                    "baseline": b, "suppressed": v, "delta": v - b,
                    "n_lesions": int(lesions.get(c, 0)),
                    "scorable_below_thr": opportunity[thr][reg],
                    "could_change": opportunity[thr][reg] > 0,
                    "leak_class": leak.get(c),
                })
        pd.DataFrame(rows).to_csv(f"{OUT_DIR}/size_pilot_per_case.csv", index=False)

    if not rows:
        raise SystemExit("no case completed")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 74); print(" RESULTS"); print("=" * 74)

    md = ["# Size pilot — baseline vs size@50 vs size@100", "",
          f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
          f"**Sample:** {df.case.nunique()} cases, stratified on lesion count and "
          f"fixed before any suppression was applied.", "",
          "The sample was deliberately NOT selected on the presence of scorable "
          "components below the threshold, which would be selection on the variable "
          "under test. How often the rule could intervene is reported below as a "
          "result.", ""]

    for rule in [r for r in RULES if r != "baseline"]:
        g = df[df.rule == rule]
        changed = int((g.delta.abs() > 1e-9).sum())
        could = int(g.could_change.sum())
        mean_all = g.delta.mean()
        cond = g[g.could_change]
        mean_cond = cond.delta.mean() if len(cond) else float("nan")

        print(f"\n  {rule}")
        print(f"    mean delta (all)              : {mean_all:+.5f}")
        print(f"    rows that could change        : {could}/{len(g)}")
        print(f"    rows that DID change          : {changed}/{len(g)}")
        print(f"    mean delta | could change     : {mean_cond:+.5f}")
        print(f"    by region:")
        for reg in REGIONS:
            r = g[g.region == reg]
            if len(r):
                print(f"      {reg}: delta={r.delta.mean():+.5f}  "
                      f"could={int(r.could_change.sum())}/{len(r)}")

        md += [f"## {rule}", "",
               f"- mean delta over all rows: **{mean_all:+.5f}**",
               f"- rows where the rule *could* change the score "
               f"(scorable components below threshold): **{could}/{len(g)}**",
               f"- rows where it *did*: **{changed}/{len(g)}**",
               f"- mean delta restricted to rows that could change: **{mean_cond:+.5f}**", "",
               "| region | mean delta | could change |", "|---|---|---|"]
        for reg in REGIONS:
            r = g[g.region == reg]
            if len(r):
                md.append(f"| {reg} | {r.delta.mean():+.5f} | "
                          f"{int(r.could_change.sum())}/{len(r)} |")
        md.append("")

    # decision
    best = {}
    for rule in [r for r in RULES if r != "baseline"]:
        best[rule] = df[df.rule == rule].delta.mean()
    winner = max(best, key=best.get)

    print("\n" + "-" * 66)
    if all(v <= 0 for v in best.values()):
        verdict = ("Both thresholds reduce lesion-wise Dice on this sample. Morphological "
                   "suppression is not beneficial on the ranked metric at these operating "
                   "points; carry ONE as a reference arm and report the finding rather "
                   "than searching for a threshold that turns positive.")
    elif abs(best["size@50"] - best["size@100"]) < 0.002:
        verdict = (f"The two thresholds are within 0.002 of each other "
                   f"({best['size@50']:+.5f} vs {best['size@100']:+.5f}). Carrying both "
                   f"into the full run buys little; keep {winner} and drop the other.")
    else:
        verdict = (f"{winner} is clearly stronger ({best[winner]:+.5f}). Carry it; the "
                   f"other can be dropped unless the threshold curve is itself of interest.")

    print(f"  VERDICT: {verdict}")
    md += ["## Decision", "", verdict, "",
           "> Note: a rule that removes more components is not thereby better. What "
           "matters is whether the removed components were false positives. This "
           "cohort has FN=729 against FP=282 on WT, so aggressive removal is not "
           "obviously safe.", ""]

    open(f"{OUT_DIR}/SIZE_PILOT_REPORT.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\n  csv : {OUT_DIR}/size_pilot_per_case.csv")
    print(f"  md  : {OUT_DIR}/SIZE_PILOT_REPORT.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--per-stratum", type=int, default=5)
    ap.add_argument("-b", "--time-budget", type=float, default=8.0)
    a = ap.parse_args([x for x in sys.argv[1:] if not x.startswith("-f")])
    main(a.per_stratum, a.time_budget)
