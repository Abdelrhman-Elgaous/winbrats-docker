# -*- coding: utf-8 -*-
"""
==============================================================================
 T4 + T3 ANALYSIS  —  detection counts and per-size-bin behaviour
==============================================================================
 RUNTIME: CPU, ~1 min. INPUT: wave2 shards from t4_run.py.
 PROTOCOL: PHASE_B_PROTOCOL.md sections 9, 12.9, 12.10.

 This script implements decisions that were frozen before any T3/T4 number was
 computed. It is not a place to make them. Every choice below has a protocol
 reference, and the guards refuse the alternatives.

 T4 — detection, per region
 --------------------------
   Num_TP / Num_FP / Num_FN come from results_df and are the source of truth;
   nothing is inferred from the per-lesion table.

   ENDPOINT (frozen): F1 is the single inferential endpoint. It is formed per
   case, compared baseline vs rule with a paired Wilcoxon, and Holm-corrected
   across the six rules within each region -- the same shape as B-6, so the two
   sections are read on the same terms.

   Precision and recall are reported as DESCRIPTIVE secondary measures: means
   and deltas, no test, no correction. Testing all three would add eighteen
   comparisons that the protocol never asked for, and Holm across a larger
   family would weaken the endpoint that was actually specified. Protocol
   section 12 asks for detection F1, not for three detection endpoints.

 T3 — per GT-volume bin  (section 12.10, frozen 17 Aug)
 ------------------------------------------------------
   Lesion-level Dice and recall are aggregated within each case by an
   UNWEIGHTED mean over that case's lesions in the bin, so a case with 30
   lesions and a case with 1 carry equal weight. The CASE is the statistical
   unit, restricted to cases with at least one GT lesion in that bin. A case
   with no lesion in a bin contributes NO observation and is not assigned zero,
   so n differs across bins by construction.

   Using the lesion as the unit would give n = 3,239, but lesions inside one
   case share a patient, a prediction and an annotator. That is
   pseudoreplication, and it would put this section on different terms from
   B-6 in the same paper.

 WHAT IS REFUSED
 ---------------
   F1 within a GT-size bin      an FP has no reference lesion, hence no bin;
                                its denominator cannot be formed (12.9)
   bins other than the four     27/100/500 are frozen exploratory bins (9)
   zero imputation              a missing observation is not a zero (12.10)
   FP size analysis here        needs predicted-component sizes, which neither
                                table carries -- see the FP section

 The four-point exploratory disclosure is printed directly above the T3 table
 so that anyone copying the table takes the disclosure with it.
==============================================================================
"""

import os
import sys
import glob
import re

import numpy as np
import pandas as pd
from scipy import stats

DRIVE   = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W2      = f"{DRIVE}/phase_b_outputs/wave2"
OUT_DIR = W2
PAT     = re.compile(r"\.w\d+of\d+\.csv$")

BASELINE = "baseline"
RULES = ["baseline", "size@20", "size@50",
         "entropy@0.45", "entropy@0.50", "entropy@0.56", "confidence@0.45"]
TESTED  = [r for r in RULES if r != BASELINE]
REGIONS = ["WT", "TC", "ET", "RC"]
ALPHA   = 0.05

# Section 9 -- frozen exploratory bins. The <= 2 mm3 band is evaluator-excluded
# and is not a scored bin: reference lesions at or below it never enter
# metric_df_thresh (line 761) or Num_TP/FN (743, 747).
BIN_EDGES  = [2.0, 27.0, 100.0, 500.0, np.inf]
BIN_LABELS = ["(2,27]", "(27,100]", "(100,500]", "(500,inf)"]

DISCLOSURE = """  These bins are descriptive/exploratory only and were not pre-registered.
  They were retained unchanged after results became available; no alternative
  bins were selected post hoc. They are not used for confirmatory hypothesis
  testing."""


# ---------------------------------------------------------------------------

def holm(pvals):
    """Holm-Bonferroni, order preserved, NaN passed through."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    order = idx[np.argsort(p[idx])]
    run, m = 0.0, len(order)
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        out[j] = min(run, 1.0)
    return out


def paired(base, rule_vals, label):
    """
    Paired Wilcoxon plus a percentile bootstrap CI for the mean difference,
    on whatever cases have both values. Returns a dict; never raises on a
    degenerate input, because "the rule changed nothing here" is a result.
    """
    d = (rule_vals - base).dropna().to_numpy()
    rec = {"n": int(len(d)),
           "mean_delta": float(np.mean(d)) if len(d) else np.nan,
           "median_delta": float(np.median(d)) if len(d) else np.nan,
           "n_worse": int((d < 0).sum()), "n_better": int((d > 0).sum()),
           "n_tied": int((d == 0).sum()), "p": np.nan,
           "ci_lo": np.nan, "ci_hi": np.nan}
    if len(d) >= 2 and not np.allclose(d, 0):
        rec["p"] = stats.wilcoxon(d, zero_method="wilcox")[1]
        rng = np.random.default_rng(20260813)
        boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        rec["ci_lo"], rec["ci_hi"] = (float(np.percentile(boot, 2.5)),
                                      float(np.percentile(boot, 97.5)))
    elif len(d) and np.allclose(d, 0):
        rec["ci_lo"] = rec["ci_hi"] = 0.0
    return rec


def load():
    t4 = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv") if PAT.search(f)]
    t3 = [f for f in glob.glob(f"{W2}/t3_lesions.w*of*.csv") if PAT.search(f)]
    if not t4:
        sys.exit(f"no T4 shards under {W2}")
    a = pd.concat([pd.read_csv(f) for f in t4], ignore_index=True)
    b = pd.concat([pd.read_csv(f) for f in t3], ignore_index=True) if t3 else pd.DataFrame()

    bad = a[a.status.astype(str) != "ok"]
    if len(bad):
        sys.exit(f"{bad.groupby(['case','setting']).ngroups} failed pair(s) in T4 — "
                 "diagnose before analysing")
    pairs = a.groupby(["case", "setting"]).ngroups
    print(f"  T4 : {len(a)} rows · {pairs} pairs · {a.case.nunique()} cases · "
          f"{a.setting.nunique()} rules")
    print(f"  T3 : {len(b)} lesion rows · {b.case.nunique()} cases")

    stale = set(a.setting) - set(RULES)
    if stale:
        sys.exit(f"non-protocol rules present: {stale}")
    return a, b


# ---------------------------------------------------------------------------
#  T4  —  detection, per region
# ---------------------------------------------------------------------------

def run_t4(t4):
    print("\n" + "=" * 76)
    print(" T4 · DETECTION — per region, counts from results_df")
    print("=" * 76)

    d = t4.copy()
    den_p = d.Num_TP + d.Num_FP
    den_r = d.Num_TP + d.Num_FN
    d["precision"] = np.where(den_p > 0, d.Num_TP / den_p.replace(0, np.nan), np.nan)
    d["recall"]    = np.where(den_r > 0, d.Num_TP / den_r.replace(0, np.nan), np.nan)
    den_f = 2 * d.Num_TP + d.Num_FP + d.Num_FN
    d["F1"] = np.where(den_f > 0, 2 * d.Num_TP / den_f.replace(0, np.nan), np.nan)

    # A case with no reference lesion AND no prediction has an undefined F1:
    # there is nothing to be right or wrong about. Such cases contribute no
    # observation, exactly as in section 12.10 -- they are not scored as 1.0,
    # which would reward the model for a region it never had to find.
    undef = int(den_f.eq(0).sum())
    print(f"  cases×regions with no GT and no prediction (F1 undefined, excluded): "
          f"{undef}")

    rows = []
    for region in REGIONS:
        sub = d[d.region == region]
        wide = {m: sub.pivot_table(index="case", columns="setting", values=m)
                for m in ("precision", "recall", "F1")}
        recs = []
        for rule in TESTED:
            if rule not in wide["F1"].columns:
                continue
            rec = {"region": region, "rule": rule}
            for m in ("F1", "precision", "recall"):
                w = wide[m][[BASELINE, rule]].dropna()
                r = paired(w[BASELINE], w[rule], f"{region}/{rule}/{m}")
                rec[f"{m}_base"] = float(w[BASELINE].mean()) if len(w) else np.nan
                rec[f"{m}_rule"] = float(w[rule].mean()) if len(w) else np.nan
                rec[f"{m}_delta"] = r["mean_delta"]
                if m == "F1":
                    rec.update({"n": r["n"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                                "n_worse": r["n_worse"], "n_better": r["n_better"],
                                "p_raw": r["p"]})
            recs.append(rec)
        for rec, a_ in zip(recs, holm([x["p_raw"] for x in recs])):
            rec["p_holm"] = a_
            rec["sig"] = bool(a_ < ALPHA) if a_ == a_ else False
        rows += recs

    t = pd.DataFrame(rows)
    for region in REGIONS:
        r = t[t.region == region]
        if not len(r):
            continue
        print(f"\n  --- {region} ---   baseline F1 = {r.F1_base.iloc[0]:.4f}")
        print(r[["rule", "n", "F1_delta", "ci_lo", "ci_hi",
                 "precision_delta", "recall_delta",
                 "n_worse", "n_better", "p_holm", "sig"]]
              .round(4).to_string(index=False))

    print("\n  F1 is the inferential endpoint: Wilcoxon + Holm across the six")
    print("  rules within each region. precision_delta and recall_delta are")
    print("  DESCRIPTIVE — no test, no correction (see the module docstring).")
    return t


# ---------------------------------------------------------------------------
#  T3  —  per GT-volume bin
# ---------------------------------------------------------------------------

def run_t3(t3):
    print("\n" + "=" * 76)
    print(" T3 · PER GT-VOLUME BIN — case is the statistical unit (section 12.10)")
    print("=" * 76)
    print(DISCLOSURE)

    if not len(t3):
        print("\n  no per-lesion table — T3 cannot run")
        return pd.DataFrame()

    d = t3.copy()
    # Counted directly off the baseline arm rather than divided out of the
    # total: dividing assumes every setting carries the same lesion rows, and
    # that assumption is not needed here even though it happens to hold.
    excluded = int((t3[t3.setting == BASELINE].gt_lesion_vol <= BIN_EDGES[0]).sum())
    d = d[d.gt_lesion_vol > BIN_EDGES[0]].copy()
    d["bin"] = pd.cut(d.gt_lesion_vol, bins=BIN_EDGES, labels=BIN_LABELS,
                      include_lowest=False)
    d["detected"] = (d._len > 0).astype(float)

    print(f"\n  ≤ {BIN_EDGES[0]:.0f} mm³ : {excluded} reference lesion(s) — "
          f"evaluator-excluded, not part of scored T3 analysis")
    counts = (d[d.setting == BASELINE].groupby("bin", observed=True)
              .size().reindex(BIN_LABELS).fillna(0).astype(int))
    print("  scored lesions per bin:", counts.to_dict())

    # UNWEIGHTED mean within case and bin (12.10). A case enters a bin only if
    # it has >= 1 lesion there; absence is a missing observation, never a zero.
    agg = (d.groupby(["case", "setting", "bin"], observed=True)
             .agg(dice=("dice_lesionwise", "mean"),
                  recall=("detected", "mean"),
                  n_les=("dice_lesionwise", "size"))
             .reset_index())

    rows = []
    for b in BIN_LABELS:
        sub = agg[agg.bin == b]
        if not len(sub):
            continue
        wide_d = sub.pivot_table(index="case", columns="setting", values="dice")
        wide_r = sub.pivot_table(index="case", columns="setting", values="recall")
        recs = []
        for rule in TESTED:
            if rule not in wide_d.columns:
                continue
            wd = wide_d[[BASELINE, rule]].dropna()
            wr = wide_r[[BASELINE, rule]].dropna()
            # Every case-bin cell that has a Dice mean also has a recall mean:
            # both come from the same groupby over the same lesions. Asserting
            # it rather than assuming it means a future change to the
            # aggregation cannot silently give the two endpoints different
            # denominators.
            assert len(wd) == len(wr) and set(wd.index) == set(wr.index), \
                f"Dice/recall case sets differ in bin={b}, rule={rule}: " \
                f"{len(wd)} vs {len(wr)}"
            rd = paired(wd[BASELINE], wd[rule], b)
            rr = paired(wr[BASELINE], wr[rule], b)
            recs.append({
                "bin": b, "rule": rule,
                "n_cases": rd["n"], "n_cases_recall": rr["n"],
                "dice_base": float(wd[BASELINE].mean()) if len(wd) else np.nan,
                "dice_delta": rd["mean_delta"],
                "dice_ci_lo": rd["ci_lo"], "dice_ci_hi": rd["ci_hi"],
                "dice_worse": rd["n_worse"], "dice_better": rd["n_better"],
                "p_dice_raw": rd["p"],
                "recall_base": float(wr[BASELINE].mean()) if len(wr) else np.nan,
                "recall_delta": rr["mean_delta"],
                "p_recall_raw": rr["p"],
            })
        for rec, pa, pb in zip(recs,
                               holm([x["p_dice_raw"] for x in recs]),
                               holm([x["p_recall_raw"] for x in recs])):
            rec["p_dice_holm"], rec["p_recall_holm"] = pa, pb
            rec["sig_dice"] = bool(pa < ALPHA) if pa == pa else False
        rows += recs

    t = pd.DataFrame(rows)
    for b in BIN_LABELS:
        r = t[t.bin == b]
        if not len(r):
            continue
        print(f"\n  --- {b} mm³ ---   n = {r.n_cases.iloc[0]} cases · "
              f"baseline Dice = {r.dice_base.iloc[0]:.4f} · "
              f"recall = {r.recall_base.iloc[0]:.4f}")
        print(r[["rule", "n_cases", "dice_delta", "dice_ci_lo", "dice_ci_hi",
                 "recall_delta", "dice_worse", "dice_better",
                 "p_dice_holm", "sig_dice"]].round(4).to_string(index=False))

    print("\n  n differs across bins by construction: a case enters a bin only")
    print("  if it has at least one reference lesion there (section 12.10).")
    print("  No F1 column: an FP has no reference lesion and therefore no bin,")
    print("  so a per-bin F1 denominator cannot be formed (section 12.9).")
    return t


# ---------------------------------------------------------------------------
#  FP by predicted-component size  —  separate axis, and not available here
# ---------------------------------------------------------------------------

def fp_note(t3):
    print("\n" + "=" * 76)
    print(" FP BY PREDICTED-COMPONENT SIZE — separate axis, NOT COMPUTED")
    print("=" * 76)
    print("""
  This analysis is in scope (section 12.9) but cannot be produced from the T3
  or T4 tables:

    · the per-lesion table is keyed on REFERENCE lesions, so a false positive
      has no row in it at all -- that is the same structural fact that rules
      out a per-bin F1;
    · results_df gives Num_FP per region, a count with no size axis;
    · day11_fp_lesions.csv does have FP sizes, but its provenance does not
      match this run -- the counts disagree with the evaluator's own by
      300-450 lesions, which is why it was excluded from the ceiling analysis
      too.

  Producing it needs a separate pass that records the size of each predicted
  component at the point of suppression. That is a new computation, not a
  re-reading of these tables, and it is left out rather than approximated.
""")


# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print(" T3 / T4 ANALYSIS — clean stratum")
    print(" protocol: sections 9, 12.9, 12.10 (all frozen before computation)")
    print("=" * 76)

    t4, t3 = load()

    # Guards. These refuse the alternatives the protocol rules out, so a later
    # edit cannot quietly reintroduce them.
    assert BIN_EDGES == [2.0, 27.0, 100.0, 500.0, np.inf], \
        "T3 bins are frozen (section 9) — do not edit"
    assert len(BIN_LABELS) == len(BIN_EDGES) - 1

    t4_res = run_t4(t4)
    t3_res = run_t3(t3)
    fp_note(t3)

    os.makedirs(OUT_DIR, exist_ok=True)
    t4_res.to_csv(f"{OUT_DIR}/t4_detection_results.csv", index=False)
    if len(t3_res):
        t3_res.to_csv(f"{OUT_DIR}/t3_bin_results.csv", index=False)

    # The disclosure has to survive the session. A CSV gets opened months later
    # by someone who never saw this log, so the terms the T3 table is valid
    # under are written next to it rather than printed and lost.
    with open(f"{OUT_DIR}/t34_analysis_report.txt", "w", encoding="utf-8") as fh:
        fh.write("T3 / T4 ANALYSIS — clean stratum (n = 155)\n")
        fh.write("protocol: PHASE_B_PROTOCOL.md sections 9, 12.9, 12.10\n")
        fh.write("=" * 76 + "\n\n")

        fh.write("T3 BIN DISCLOSURE — applies to t3_bin_results.csv\n")
        fh.write("-" * 76 + "\n")
        fh.write(DISCLOSURE.replace("  ", "", 1) + "\n\n")
        fh.write(f"Bins (mm3, reference-lesion volume): "
                 f"{' · '.join(BIN_LABELS)}\n")
        fh.write(f"<= {BIN_EDGES[0]:.0f} mm3: evaluator-excluded reference "
                 f"lesions; not part of scored T3 analysis.\n\n")

        fh.write("STATISTICAL UNIT (section 12.10, frozen before computation)\n")
        fh.write("-" * 76 + "\n")
        fh.write("Lesion-level Dice and recall are aggregated within each case by an\n"
                 "UNWEIGHTED mean over that case's lesions in the bin. The CASE is the\n"
                 "statistical unit, restricted to cases with >= 1 reference lesion in\n"
                 "that bin. Cases with no lesion in a bin contribute NO observation and\n"
                 "are not assigned zero, so n differs across bins by construction.\n\n")

        fh.write("ENDPOINTS\n")
        fh.write("-" * 76 + "\n")
        fh.write("T4: F1 is the single inferential endpoint (paired Wilcoxon, Holm\n"
                 "    across the six rules within each region). Precision and recall\n"
                 "    are descriptive — no test, no correction.\n")
        fh.write("T3: Dice and recall deltas per bin, same unit and correction shape.\n")
        fh.write("    No F1 per bin: a false positive has no reference lesion and\n"
                 "    therefore no bin, so its denominator cannot be formed (12.9).\n\n")

        fh.write("NOT COMPUTED\n")
        fh.write("-" * 76 + "\n")
        fh.write("FP by predicted-component size. Neither table carries predicted\n"
                 "component sizes, and day11_fp_lesions.csv is from a different run\n"
                 "(counts disagree with the evaluator's by 300-450 lesions). It needs\n"
                 "a separate pass and is left out rather than approximated.\n\n")

        fh.write("POWER\n")
        fh.write("-" * 76 + "\n")
        fh.write("The pre-registered MDE of 0.019 Dice was computed at n = 155. Per-bin\n"
                 "n is smaller, so power there is lower and a null result is\n"
                 "correspondingly weaker evidence.\n\n")

        fh.write("=" * 76 + "\n\nT4 — DETECTION (F1 inferential; precision/recall descriptive)\n\n")
        fh.write(t4_res.round(4).to_string(index=False) + "\n")
        if len(t3_res):
            fh.write("\n\n" + "=" * 76 + "\n\nT3 — PER GT-VOLUME BIN\n\n")
            fh.write(t3_res.round(4).to_string(index=False) + "\n")

    assert "F1" not in " ".join(t3_res.columns), \
        "a per-bin F1 column reached the T3 output — section 12.9 forbids it"

    print("\n" + "=" * 76)
    print(f"  written: t4_detection_results.csv"
          + (" · t3_bin_results.csv" if len(t3_res) else "")
          + " · t34_analysis_report.txt")
    print("  The report carries the bin disclosure, the statistical unit and the")
    print("  endpoint definitions, so the CSVs are never read without them.")
    print("""
  Reading these:
    · T4 deltas are per region and directly comparable with B-6's Dice deltas.
    · T3 deltas are per bin with the case as the unit; n varies by bin.
    · t34_analysis_report.txt states the terms the T3 table is valid under.
      Quote them with it.
    · The pre-registered MDE of 0.019 was computed at n = 155. Per-bin n is
      smaller, so power there is lower and a null is correspondingly weaker
      evidence.""")
    print("=" * 76)


if __name__ == "__main__":
    main()
