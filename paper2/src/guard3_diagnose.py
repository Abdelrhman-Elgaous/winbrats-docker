# -*- coding: utf-8 -*-
"""
==============================================================================
 GUARD-3 DIAGNOSIS AND RECOVERY
==============================================================================
 RUNTIME: CPU. WALL TIME: ~20-40 min for two cases.

 WHY THIS EXISTS
 ---------------
 Two pilot cases (00546, 00174) raised Guard 3 under size@50: whole tumour
 scored exactly zero while present in the ground truth. size_pilot.py skips a
 case entirely when any rule fails, so neither case appears in the results CSV
 -- not even its successful baseline row.

 That omission is not random. These are the cases where suppression removed the
 most, so dropping them biases the pilot mean IN FAVOUR of suppression, which is
 exactly the quantity under test. The pilot mean must be computed on all 20.

 CLASSIFICATION -- two independent stages
 ----------------------------------------
   A. SUPPRESSION CONSISTENCY
      Are the components actually removed the ones the rule says to remove?
      Compared as ID SETS, not counts: removing a wrong component while keeping
      another of the same size would pass a count check.

   B. RECONSTRUCTION CONSISTENCY
      Does the nested-label rebuild produce the region size predicted from the
      surviving components?

   Class 1  baseline itself fails            -> pipeline failure, STOP
   Class 2  A and B both hold                -> expected behaviour, record it
   Class 3  either fails                     -> unexpected, STOP

 THE ORACLE
 ----------
 Expected values are computed from ndimage.label on the thresholded probability
 map -- NOT by calling build_nested_label again. Re-running the function under
 test and comparing it with itself would pass even if that function were wrong.
 The oracle is not independent of the thresholding step, and does not need to
 be: the thresholding is not what is in doubt.

 A note on measurement: region sizes are counted as NESTED regions
 (WT = labels {1,2,3}), never as the bare value 2. Suppressing a TC component
 leaves its voxels labelled 2, so counting `label == 2` can INCREASE after
 suppression -- an artefact of write precedence, not of removal.

 OUTPUT
 ------
   guard3_diagnosis.csv · GUARD3_DIAGNOSIS.md
   and, for Class-2 cases, rows appended to size_pilot_per_case.csv
==============================================================================
"""

import os, sys, json, time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
for _p in (_HERE, "/content/drive/MyDrive/brats_2026_metastases/src/eval"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import native_space as ns

OUT_DIR  = f"{ns.DRIVE}/phase_b_outputs/wave1"
PILOT_CSV = f"{OUT_DIR}/size_pilot_per_case.csv"

FAILED   = ["BraTS-MET-00546-000", "BraTS-MET-00174-000"]
RULES    = {"baseline": None, "size@50": 50, "size@100": 100}
REGIONS  = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)}
CH       = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}
# The BraTS-MET evaluator's scoring floor is 2 VOXELS, set at metrics_MET.py
# line 681 inside the `challenge_name == 'BraTS-MET'` branch, and applied
# symmetrically to predicted components (line 319) and to reference lesions
# (743, 747, 761).
#
# This constant read 27 until 16 Aug 2026. That figure was a historical,
# undocumented project assumption -- it entered through a comment in
# size_pilot.py and propagated from there; it was never an evaluator value.
# See protocol section 13 for the full audit.
#
# Nothing published is affected. SCORABLE feeds one diagnostic column,
# `scorable_removed`, which never entered a decision: the Class 1/2/3 verdict
# is determined by the oracle/recovery logic below, not by this count. The two
# cases were classified Class 2 without reference to it.
SCORABLE = 2

os.makedirs(OUT_DIR, exist_ok=True)


def nested_voxels(label, region):
    """Count a region as its NESTED definition, never as a bare label value."""
    return int(np.isin(label, REGIONS[region]).sum())


def expected_from_components(mean_prob, region, thr_voxels):
    """
    ORACLE. Predict, from the component list alone, which components the rule
    should remove and how many voxels should survive in this region's channel.
    Independent of the suppression and reconstruction code being tested.
    """
    mask = mean_prob[CH[region]] >= ns.THR[region]
    lab, n = ndimage.label(mask)
    if n == 0:
        return {"ids_to_remove": set(), "n_components": 0,
                "surviving_voxels": 0, "removed_voxels": 0,
                "max_removed_size": 0, "scorable_removed": 0}

    sizes = np.bincount(lab.ravel())[1:]          # index i -> component i+1
    if thr_voxels is None:
        remove = np.zeros(len(sizes), dtype=bool)
    else:
        remove = sizes <= thr_voxels

    ids = {int(i + 1) for i in np.flatnonzero(remove)}
    return {
        "ids_to_remove": ids,
        "n_components": int(n),
        "surviving_voxels": int(sizes[~remove].sum()),
        "removed_voxels": int(sizes[remove].sum()),
        "max_removed_size": int(sizes[remove].max()) if remove.any() else 0,
        # Components above the evaluator's floor that the rule deleted, i.e.
        # ones that would actually have been scored. Diagnostic only.
        "scorable_removed": int((remove & (sizes > SCORABLE)).sum()),
    }


def actual_removed_ids(mean_prob, region, thr_voxels):
    """
    Which component IDs did the suppressor actually drop? Determined by applying
    the real suppressor callable to each component, so this exercises the code
    under test rather than restating the rule.
    """
    if thr_voxels is None:
        return set()
    mask = mean_prob[CH[region]] >= ns.THR[region]
    lab, n = ndimage.label(mask)
    sup = ns.size_suppressor(thr_voxels)
    dropped = set()
    ent = ns.binary_entropy(mean_prob[CH[region]])
    for cid in range(1, n + 1):
        comp = (lab == cid)
        if sup(region, comp, mean_prob[CH[region]], ent):
            dropped.add(cid)
    return dropped


def diagnose(case):
    print(f"\n{'='*70}\n{case}\n{'='*70}")
    rows = []

    mean = ns.load_mean_prob(case)

    # ---- is the baseline itself sound? (Class 1 test) ----------------------
    base_label = ns.build_nested_label(mean)
    base_region = {r: nested_voxels(base_label, r) for r in REGIONS}
    print(f"  baseline regions: {base_region}")

    baseline_ok = base_region["WT"] > 0
    if not baseline_ok:
        print("  [CLASS 1] baseline itself produces empty WT — pipeline failure")
        return [{"case": case, "rule": "baseline", "region": "WT",
                 "klass": "1", "detail": "baseline WT empty"}], None

    for rule, thr in RULES.items():
        if rule == "baseline":
            continue
        print(f"\n  --- {rule} ---")
        lab = ns.build_nested_label(mean, suppressor=ns.size_suppressor(thr))

        for region in REGIONS:
            exp = expected_from_components(mean, region, thr)
            act_ids = actual_removed_ids(mean, region, thr)
            actual = nested_voxels(lab, region)

            # A: suppression consistency — ID SETS, not counts
            a_ok = (act_ids == exp["ids_to_remove"])
            # B: reconstruction consistency
            b_ok = (actual == exp["surviving_voxels"])

            if not a_ok:
                klass, why = "3", "suppression inconsistency (removed IDs differ)"
            elif not b_ok:
                klass, why = "3", "reconstruction inconsistency"
            else:
                klass, why = "2", "expected behaviour"

            emptied = (base_region[region] > 0 and actual == 0)
            flag = "  <-- REGION EMPTIED" if emptied else ""
            mark = "OK " if klass == "2" else "!! "
            print(f"    {mark}{region}: comps={exp['n_components']:>3} "
                  f"removed={len(act_ids):>3} "
                  f"expected_voxels={exp['surviving_voxels']:>7} "
                  f"actual={actual:>7} "
                  f"delta={actual - exp['surviving_voxels']:>+6} "
                  f"class={klass}{flag}")
            if exp["max_removed_size"] > (thr or 0):
                print(f"        [!] largest removed component "
                      f"({exp['max_removed_size']}) exceeds threshold {thr}")

            rows.append({
                "case": case, "rule": rule, "region": region,
                "n_components": exp["n_components"],
                "expected_removed_ids": len(exp["ids_to_remove"]),
                "actual_removed_ids": len(act_ids),
                "ids_match": a_ok,
                "expected_region_voxels_from_components": exp["surviving_voxels"],
                "actual_region_voxels_after_reconstruction": actual,
                "delta_region_voxels": actual - exp["surviving_voxels"],
                "max_removed_component_size": exp["max_removed_size"],
                "scorable_components_removed": exp["scorable_removed"],
                "region_emptied": emptied,
                "klass": klass, "reason": why,
            })

    del mean
    return rows, base_region


def recover(case):
    """Re-score the case with guards OFF, so an emptied region records its true
    value (Dice 0) instead of vanishing from the sample."""
    print(f"\n  recovering {case} (guards disabled — the failure is now classified)")
    per_rule = {}
    for rule, thr in RULES.items():
        sup = ns.size_suppressor(thr) if thr else None
        t0 = time.time()
        res = ns.official_score(case, suppressor=sup,
                                tag=f"recov_{rule.replace('@','_at_')}", guards=False)
        per_rule[rule] = {r["region"]: r["lesionwise_dice"] for r in res}
        print(f"    {rule:>10s}: " +
              " ".join(f"{r['region']}={r['lesionwise_dice']:.4f}" for r in res) +
              f"   {time.time()-t0:.0f}s")

    out = []
    for rule in RULES:
        if rule == "baseline":
            continue
        for reg in REGIONS:
            b, v = per_rule["baseline"][reg], per_rule[rule][reg]
            out.append({"case": case, "stratum": "recovered", "rule": rule,
                        "region": reg, "baseline": b, "suppressed": v,
                        "delta": v - b, "recovered": True})
    return out


def main():
    print("=" * 70); print(" GUARD-3 DIAGNOSIS AND RECOVERY"); print("=" * 70)
    print(f"  cases: {FAILED}")

    all_rows, verdicts = [], {}
    for case in FAILED:
        try:
            rows, _ = diagnose(case)
        except Exception as e:
            print(f"  ERROR on {case}: {e}")
            continue
        all_rows += rows
        classes = {r["klass"] for r in rows}
        verdicts[case] = "3" if "3" in classes else ("1" if "1" in classes else "2")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(f"{OUT_DIR}/guard3_diagnosis.csv", index=False)

    print("\n" + "=" * 70); print(" VERDICT"); print("=" * 70)
    for c, v in verdicts.items():
        label = {"1": "CLASS 1 — pipeline failure",
                 "2": "CLASS 2 — expected suppression behaviour",
                 "3": "CLASS 3 — unexpected, investigate"}[v]
        print(f"  {c}: {label}")

    recovered = []
    if all(v == "2" for v in verdicts.values()) and verdicts:
        print("\n  all failures are Class 2 — recovering both cases so the pilot "
              "mean is computed on all 20")
        for c in verdicts:
            try:
                recovered += recover(c)
            except Exception as e:
                print(f"  recovery failed for {c}: {e}")

        if recovered and os.path.exists(PILOT_CSV):
            old = pd.read_csv(PILOT_CSV)
            merged = pd.concat([old, pd.DataFrame(recovered)], ignore_index=True)
            merged.to_csv(PILOT_CSV, index=False)
            print(f"\n  pilot CSV now holds {merged.case.nunique()} cases "
                  f"(was {old.case.nunique()})")
    else:
        print("\n  NOT recovering: at least one failure is not Class 2. "
              "Investigate before adding these cases to the analysis.")

    md = ["# Guard-3 diagnosis", "",
          f"**Generated:** {datetime.now().isoformat(timespec='seconds')}", "",
          "## Why these cases matter", "",
          "`size_pilot.py` skips a case entirely when any rule raises, so neither "
          "case reached the results CSV — not even its successful baseline row. "
          "The omission is not random: these are the cases where suppression "
          "removed the most, so excluding them biases the pilot mean in favour of "
          "suppression, the very quantity under test.", "",
          "## Classification", "",
          "| stage | test |", "|---|---|",
          "| A · suppression consistency | removed component **ID sets** match the rule |",
          "| B · reconstruction consistency | nested-region size matches the oracle prediction |",
          "", "| case | verdict |", "|---|---|"]
    for c, v in verdicts.items():
        md.append(f"| {c} | Class {v} |")
    md += ["", f"**Recovered into the pilot:** {'yes' if recovered else 'no'}", ""]
    open(f"{OUT_DIR}/GUARD3_DIAGNOSIS.md", "w", encoding="utf-8").write("\n".join(md) + "\n")

    print(f"\n  csv : {OUT_DIR}/guard3_diagnosis.csv")
    print(f"  md  : {OUT_DIR}/GUARD3_DIAGNOSIS.md")


if __name__ == "__main__":
    main()
