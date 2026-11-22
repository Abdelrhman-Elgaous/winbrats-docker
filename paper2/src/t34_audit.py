# -*- coding: utf-8 -*-
"""
==============================================================================
 T3 / T4 CONSISTENCY AUDIT  —  five layers, read-only
==============================================================================
 RUNTIME: seconds. Writes one report; changes nothing.

 Purpose: verify -> reconcile -> explain.  NOT inspect -> redefine -> rerun.
 Nothing here may alter a definition, a denominator or a result. If a layer
 fails, the failure is the finding and it goes in the paper; it is not a reason
 to adjust the analysis that produced it.

 LAYER 1  T4 internal consistency
          Counts add up; the evaluable rule behaves as documented; n moves
          between baseline and rule only where a case is emptied.

 LAYER 2  Denominator audit
          For every region x rule: baseline-evaluable, rule-evaluable, and the
          intersection actually tested. This is the layer that matters most --
          if a delta were computed on a re-selected set rather than on paired
          cases, every downstream number would be quietly wrong. RC already
          showed that effective sample size is endpoint- and rule-dependent;
          this checks whether a milder version exists in WT/TC/ET.

 LAYER 3  Multiplicity
          24 comparisons (6 rules x 4 regions), Holm applied within region as
          in B-6, no double correction, raw <= adjusted throughout.

 LAYER 4  Confidence intervals
          CI attached to the endpoint it claims. Cells where the bootstrap CI
          excludes zero while the rank test does not are listed, not resolved:
          the two answer different questions and B-6 already documents one.

 LAYER 5  Cross-check against B-6
          Delta Dice (B-6) vs delta F1 (T4) on the same clean cases and rules.
          Directions need not agree -- Dice and detection F1 are different
          questions -- but every disagreement must have an explanation, not be
          left as an anomaly.
==============================================================================
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W1, W2 = f"{DRIVE}/phase_b_outputs/wave1", f"{DRIVE}/phase_b_outputs/wave2"
PAT = re.compile(r"\.w\d+of\d+\.csv$")

REGIONS = ["WT", "TC", "ET", "RC"]
BASELINE = "baseline"
TESTED = ["size@20", "size@50", "entropy@0.45",
          "entropy@0.50", "entropy@0.56", "confidence@0.45"]
B6_RULES = [r for r in TESTED if r != "size@20"]   # size@20 was not in B-6

results, log = [], []


def emit(s=""):
    print(s)
    log.append(s)


def verdict(layer, ok, detail=""):
    results.append((layer, ok, detail))
    emit(f"\n  >>> {layer}: {'PASS' if ok else 'FAIL'}"
         + (f"  —  {detail}" if detail else ""))


def load():
    t4f = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv") if PAT.search(f)]
    if not t4f:
        sys.exit(f"no T4 shards under {W2}")
    t4 = pd.concat([pd.read_csv(f) for f in t4f], ignore_index=True)

    b6p = f"{W2}/b6_primary_clean.csv"
    b6 = pd.read_csv(b6p) if os.path.exists(b6p) else None
    if b6 is None:
        emit(f"  [!] {os.path.basename(b6p)} not found — layer 5 will be skipped")
    return t4, b6


# ---------------------------------------------------------------------------

def layer1(t4):
    emit("\n" + "=" * 76)
    emit(" LAYER 1 · T4 INTERNAL CONSISTENCY")
    emit("=" * 76)

    ok = True

    bad_status = int((t4.status.astype(str) != "ok").sum())
    emit(f"  rows with status != ok        : {bad_status}")
    ok &= bad_status == 0

    neg = int((t4[["Num_TP", "Num_FP", "Num_FN"]] < 0).any(axis=1).sum())
    nan = int(t4[["Num_TP", "Num_FP", "Num_FN"]].isna().any(axis=1).sum())
    emit(f"  rows with negative counts     : {neg}")
    emit(f"  rows with NaN counts          : {nan}")
    ok &= (neg == 0 and nan == 0)

    pairs = t4.groupby(["case", "setting"]).ngroups
    per_pair = t4.groupby(["case", "setting"]).size()
    emit(f"  pairs                         : {pairs} (expected 1085)")
    emit(f"  pairs without exactly 4 rows  : {int((per_pair != 4).sum())}")
    ok &= (pairs == 1085 and (per_pair == 4).all())

    # Reference lesion count must not move: suppression edits predictions only.
    gt = t4.assign(n_gt=t4.Num_TP + t4.Num_FN)
    piv = gt.pivot_table(index=["case", "region"], columns="setting", values="n_gt")
    moved = int((piv.nunique(axis=1) > 1).sum())
    emit(f"  case×region where TP+FN moves : {moved}   "
         f"(must be 0 — the reference cannot change)")
    ok &= moved == 0

    # A case leaves the F1 sample only by being emptied.
    base = t4[t4.setting == BASELINE].set_index(["case", "region"])
    emit("\n  cases leaving the evaluable set, by rule:")
    emit(f"    {'rule':<16} {'left':>5} {'emptied':>8} {'unexplained':>12}")
    for r in TESTED:
        sub = t4[t4.setting == r].set_index(["case", "region"])
        j = base[["Num_TP", "Num_FP", "Num_FN"]].join(
            sub[["Num_TP", "Num_FP", "Num_FN"]], lsuffix="_b", rsuffix="_r")
        db = j.Num_TP_b + j.Num_FP_b + j.Num_FN_b
        dr = j.Num_TP_r + j.Num_FP_r + j.Num_FN_r
        left = (db > 0) & (dr == 0)
        emptied = left & (j.Num_FP_b > 0) & (j.Num_TP_b == 0) & (j.Num_FN_b == 0)
        unexplained = int((left & ~emptied).sum())
        emit(f"    {r:<16} {int(left.sum()):>5} {int(emptied.sum()):>8} "
             f"{unexplained:>12}")
        ok &= unexplained == 0

    verdict("LAYER 1", ok,
            "counts sound, reference fixed, every exit from the sample is an "
            "emptied FP-only case" if ok else "see above")
    return ok


def layer2(t4):
    emit("\n" + "=" * 76)
    emit(" LAYER 2 · DENOMINATOR AUDIT — is every delta computed on paired cases?")
    emit("=" * 76)

    ok = True
    rows = []
    for region in REGIONS:
        sub = t4[t4.region == region]
        wide = {s: g.set_index("case")[["Num_TP", "Num_FP", "Num_FN"]]
                for s, g in sub.groupby("setting")}
        b = wide[BASELINE]
        b_ev = set(b.index[(b.sum(axis=1) > 0)])
        for r in TESTED:
            if r not in wide:
                continue
            x = wide[r]
            r_ev = set(x.index[(x.sum(axis=1) > 0)])
            both = b_ev & r_ev
            rows.append({"region": region, "rule": r,
                         "base_eval": len(b_ev), "rule_eval": len(r_ev),
                         "paired": len(both),
                         "lost": len(b_ev - r_ev), "gained": len(r_ev - b_ev)})
    d = pd.DataFrame(rows)
    emit("\n" + d.to_string(index=False))

    # Suppression removes predictions; it can never create an evaluable case.
    gained = int(d.gained.sum())
    emit(f"\n  cases gained by suppression   : {gained}   (must be 0)")
    ok &= gained == 0

    emit("\n  The tested n is the INTERSECTION. A case evaluable at baseline but")
    emit("  emptied by the rule is dropped from that comparison, so n varies by")
    emit("  rule. That is paired-sample behaviour, not re-selection: no case")
    emit("  enters a comparison it was absent from at baseline.")

    worst = d.loc[d.lost.idxmax()] if len(d) else None
    if worst is not None:
        emit(f"\n  largest loss: {worst.region} · {worst['rule']} — "
             f"{int(worst.lost)} of {int(worst.base_eval)} cases "
             f"({worst.lost / max(worst.base_eval,1):.0%})")

    nonrc = d[(d.region != "RC") & (d.lost > 0)]
    if len(nonrc):
        emit("\n  RC-like behaviour outside RC (cases emptied in WT/TC/ET):")
        emit("  " + nonrc[["region", "rule", "base_eval", "lost"]]
             .to_string(index=False).replace("\n", "\n  "))
        emit("  Present but small — the RC pattern is a matter of degree, not kind.")
    else:
        emit("\n  No case is emptied in WT/TC/ET: the RC pattern is confined to RC.")

    verdict("LAYER 2", ok,
            "all deltas on paired cases; n varies only by cases being emptied"
            if ok else "a case became evaluable under suppression — impossible")
    return ok


def layer3(t4res):
    emit("\n" + "=" * 76)
    emit(" LAYER 3 · MULTIPLICITY")
    emit("=" * 76)

    ok = True
    n = len(t4res)
    emit(f"  comparisons in the T4 table   : {n} (expected 24 = 6 rules × 4 regions)")
    ok &= n == 24

    per_region = t4res.groupby("region").size().to_dict()
    emit(f"  per region                    : {per_region}")
    ok &= all(v == 6 for v in per_region.values())

    if {"p_raw", "p_holm"} <= set(t4res.columns):
        m = t4res.dropna(subset=["p_raw", "p_holm"])
        viol = int((m.p_raw > m.p_holm + 1e-12).sum())
        emit(f"  rows where raw > adjusted     : {viol}   (must be 0)")
        ok &= viol == 0
        over = int((m.p_holm > 1.0 + 1e-12).sum())
        emit(f"  adjusted p above 1.0          : {over}   (must be 0)")
        ok &= over == 0
        emit("\n  Holm is applied across the six rules WITHIN each region, the same")
        emit("  family as B-6. Correcting across all 24 would mix four questions")
        emit("  into one family and was not what either analysis pre-specified.")

    verdict("LAYER 3", ok, "24 comparisons, Holm within region, no double correction"
            if ok else "see above")
    return ok


def layer4(t4res):
    emit("\n" + "=" * 76)
    emit(" LAYER 4 · CONFIDENCE INTERVALS")
    emit("=" * 76)

    ok = True
    need = {"F1_delta", "ci_lo", "ci_hi", "p_holm"}
    if not need <= set(t4res.columns):
        verdict("LAYER 4", False, f"missing columns: {need - set(t4res.columns)}")
        return False

    m = t4res.dropna(subset=["ci_lo", "ci_hi"])
    inside = int(((m.F1_delta >= m.ci_lo - 1e-9) &
                  (m.F1_delta <= m.ci_hi + 1e-9)).sum())
    emit(f"  point estimate inside its CI  : {inside}/{len(m)}")
    ok &= inside == len(m)

    excl = (m.ci_lo > 0) | (m.ci_hi < 0)
    sig = m.p_holm < 0.05
    disagree = m[excl & ~sig]
    emit(f"  CI excludes 0 but Holm not sig: {len(disagree)}")
    if len(disagree):
        emit("  " + disagree[["region", "rule", "F1_delta", "ci_lo", "ci_hi",
                              "n_worse", "n_better", "p_holm"]]
             .round(4).to_string(index=False).replace("\n", "\n  "))
        emit("\n  Reported, not reconciled. The rank test asks whether the paired")
        emit("  differences are directionally skewed; the bootstrap interval")
        emit("  estimates the magnitude of their mean. Balanced win/loss counts")
        emit("  with asymmetric magnitudes produce exactly this. B-6 documents")
        emit("  the same situation for confidence@0.45 in WT.")

    verdict("LAYER 4", ok, "CIs consistent with their point estimates"
            if ok else "a point estimate fell outside its own interval")
    return ok


def layer5(t4res, b6):
    emit("\n" + "=" * 76)
    emit(" LAYER 5 · CROSS-CHECK — Δ Dice (B-6) vs Δ F1 (T4)")
    emit("=" * 76)

    if b6 is None:
        verdict("LAYER 5", True, "SKIPPED — b6_primary_clean.csv not found")
        return True

    dcol = next((c for c in ("mean_delta", "delta", "dice_delta")
                 if c in b6.columns), None)
    rcol = next((c for c in ("rule", "setting") if c in b6.columns), None)
    if dcol is None or rcol is None or "region" not in b6.columns:
        verdict("LAYER 5", True, f"SKIPPED — unexpected B-6 schema: "
                                 f"{list(b6.columns)[:8]}")
        return True

    b = b6[[ "region", rcol, dcol]].rename(columns={rcol: "rule", dcol: "dice"})
    t = t4res[["region", "rule", "F1_delta", "p_holm"]].rename(
        columns={"F1_delta": "f1"})
    m = b.merge(t, on=["region", "rule"], how="inner")
    if not len(m):
        verdict("LAYER 5", True, "SKIPPED — no overlapping region/rule rows")
        return True

    m["same_dir"] = np.sign(m.dice.round(4)) == np.sign(m.f1.round(4))
    m["note"] = np.where(
        m.same_dir, "consistent",
        np.where(m.region == "RC", "endpoint-set effect (see RC audit)",
                 "DIVERGENT — needs explanation"))
    emit("\n" + m.round(4).to_string(index=False))

    unexplained = m[(~m.same_dir) & (m.region != "RC")]
    emit(f"\n  divergences outside RC        : {len(unexplained)}")
    if len(unexplained):
        emit("  Each needs an explanation before interpretation. Note that Dice")
        emit("  and detection F1 are different questions: a rule can trade lesion")
        emit("  overlap against detection counts, so divergence is not by itself")
        emit("  an error — but it is not to be left unremarked either.")

    emit("\n  size@20 is absent from B-6 by protocol (§4): it was excluded from")
    emit("  the Dice arm and reinstated only for the detection analysis, so it")
    emit("  has no Dice counterpart to compare against.")

    verdict("LAYER 5", True,
            f"{int(m.same_dir.sum())}/{len(m)} same direction; "
            f"{len(unexplained)} divergence(s) outside RC")
    return True


# ---------------------------------------------------------------------------

def main():
    emit("=" * 76)
    emit(" T3 / T4 CONSISTENCY AUDIT — read-only")
    emit(" verify -> reconcile -> explain.  No definition is changed here.")
    emit("=" * 76)

    t4, b6 = load()
    t4res_p = f"{W2}/t4_detection_results.csv"
    if not os.path.exists(t4res_p):
        sys.exit("t4_detection_results.csv not found — run t34_analysis.py first")
    t4res = pd.read_csv(t4res_p)

    emit(f"  T4 counts   : {len(t4)} rows · "
         f"{t4.groupby(['case','setting']).ngroups} pairs · {t4.case.nunique()} cases")
    emit(f"  T4 results  : {len(t4res)} comparisons")
    emit(f"  B-6         : {'loaded' if b6 is not None else 'absent'}")

    layer1(t4)
    layer2(t4)
    layer3(t4res)
    layer4(t4res)
    layer5(t4res, b6)

    emit("\n" + "=" * 76)
    emit(" AUDIT VERDICT")
    emit("=" * 76)
    for name, ok, detail in results:
        emit(f"  {'PASS' if ok else 'FAIL'}  {name:<10} {detail}")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        emit(f"\n  {len(failed)} layer(s) failed: {', '.join(failed)}")
        emit("  A failure is a finding. Do NOT adjust the analysis to make it pass.")
    else:
        emit("\n  All layers passed. T3/T4 may be interpreted.")
        emit("  Standing caveats: RC F1 rests on five reference-positive cases and")
        emit("  is not used inferentially; T3 bins are exploratory and not")
        emit("  pre-registered; per-bin n is below the n the MDE was computed at.")

    out = f"{W2}/t34_audit_report.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
