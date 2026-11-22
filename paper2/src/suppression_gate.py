# -*- coding: utf-8 -*-
"""
==============================================================================
 SUPPRESSION RECONSTRUCTION GATE
==============================================================================
 Figure 5 needs the suppressed prediction masks. Phase B scored the arms and
 kept the counts, not the masks, so one has to be rebuilt before it can be
 drawn -- and a rebuilt mask is not evidence until it reproduces the
 experiment.

 THIS SCRIPT IS A WRAPPER AROUND ns.official_score AND NOTHING ELSE
 ------------------------------------------------------------------
 That function is the path the arm itself took:

     official_score(case, suppressor=None, folds=FOLDS, thr=THR, ...)
       -> load_mean_prob(case, folds)
       -> build_nested_label(mean, thr, suppressor)
       -> to_native_nifti(label, gt_path)
       -> _guard_shape, _guard_orientation
       -> get_LesionWiseResults(..., challenge_name="BraTS-MET", output=None)
       -> _guard_exact_zero

 EIGHT FAILED ATTEMPTS PRECEDED THIS ONE, AND EVERY ONE FAILED THE SAME WAY:
 by rebuilding a step that official_score already performs.

   1  reimplemented the component definition, using the evaluator's
      26-connected dilated construction where the suppression rule uses
      scipy's default 6-connectivity.                             13/24

   2  used the right connectivity but suppressed after composing the nested
      label instead of per region before it.                      19/24

   3  called build_nested_label directly -- and got EXACTLY 19/24 again. That
      identity was the useful result: it proved the builder was never the
      difference, and that the WT story offered for attempt 2 was a plausible
      narrative rather than a diagnosis. What remained was the evaluator's own
      TP/FP matching, still reimplemented in the gate. Counting matched
      PREDICTED components is not counting matched REFERENCE lesions.

   4  searched the tree for an evaluator by filename and found
      evaluator_forensics.py, a diagnostic tool written for this project
      rather than the challenge metric.

   5  imported metrics_MET directly, skipping ns.ensure_evaluator and with it
      the repository clone, the dependency installs, the NumPy 2.x
      compatibility pass, the sys.path entry and the chdir the evaluator
      needs.                                             ModuleNotFoundError

   6  called get_LesionWiseResults without challenge_name. The evaluator's
      behaviour depends on it, so guessing would have produced numbers that
      look reasonable and belong to a different task.           TypeError

   7  passed a tag containing a dot. The evaluator truncates the prediction
      filename at the first dot to name its scratch directory, so
      gate_entropy0.45_<case>.nii.gz produced tmp_pred/gate_entropy0/, which
      nothing had created. t4_run.py passes the rule string itself, and the
      scratch parents exist in a session that has already run the arm.
                                                          FileNotFoundError

   8  called official_score correctly but read the wrong frame. It returns
      three with full=True -- rows, agg, les -- and the counts are in `agg`,
      the evaluator's own aggregate. `rows` is the Dice summary B-4 wrote and
      has no count columns. t4_run.py states this in its preflight message
      at line 224, which is where it should have been read from.  KeyError

 The lesson is not subtle and it cost eight rounds: when frozen code already
 performs a step, call it. Every reconstruction of a step is a second
 implementation that has to be kept in agreement with the first, and the
 whole point of this gate was to check agreement.

 WHAT THIS ESTABLISHES, AND WHAT IT CANNOT
 -----------------------------------------
 Passing means the same function, the same folds, the same thresholds, the
 same guards and the same evaluator reproduce the Phase B counts exactly. At
 that point the only thing that was ever in question -- whether a mask
 rebuilt now is the mask the experiment produced -- is settled, and figure 5
 can be drawn.

 It cannot establish that the archived probabilities are the ones the arm
 read. ENSEMBLE_DIR points into archive/inference_v1, a copy, and test-time
 augmentation appears nowhere in the protocol text. If the archive were the
 wrong inference run, both sides would agree while the mask was not the
 experiment's. That is stated rather than solved.
==============================================================================
"""

import os
import re
import sys
import json
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
W2 = f"{DRIVE}/phase_b_outputs/wave2"
OUT = f"{DRIVE}/paper2/figures/qualitative"
SHARD = re.compile(r"\.w\d+of\d+\.csv$")

REGIONS = ("WT", "TC", "ET", "RC")
RULE = "entropy@0.45"
ENTROPY_THR = 0.45
N_TEST = 4

os.makedirs(OUT, exist_ok=True)


def head(t):
    print("\n" + "=" * 78)
    print(f" {t}")
    print("=" * 78)


def counts_from(ret):
    """
    Per-region counts out of official_score(full=True).

    The function returns THREE frames, and the counts are in the second:

        rows, agg, les = ns.official_score(case, ..., full=True)

    `rows` is the per-region Dice summary -- case, setting, region,
    lesionwise_dice, legacy_dice, zero_warning -- which is what B-4 wrote.
    `agg` is the evaluator's own aggregate frame and carries Labels, Num_TP,
    Num_FP and Num_FN, which is what t4_run.py reads. Attempt 7 of this gate
    read `rows` and failed on the missing columns; t4_run.py says so in its
    own preflight message at line 224.
    """
    if isinstance(ret, (tuple, list)):
        if len(ret) < 2:
            raise ValueError(f"official_score returned {len(ret)} values; "
                             f"full=True should give rows, agg, les")
        agg = ret[1]
    else:
        agg = ret
    df = pd.DataFrame(agg) if not isinstance(agg, pd.DataFrame) else agg
    if not len(df):
        raise ValueError("the aggregate frame is empty")
    reg_col = next((c for c in df.columns
                    if c.lower() in ("labels", "label", "region")), None)
    if reg_col is None:
        raise KeyError(f"no region column in {list(df.columns)}")
    need = [c for c in ("Num_TP", "Num_FP", "Num_FN") if c not in df.columns]
    if need:
        raise KeyError(f"missing {need} in the aggregate frame; have "
                       f"{list(df.columns)}")
    long = {"WHOLE TUMOR": "WT", "TUMOR CORE": "TC",
            "ENHANCING TUMOR": "ET", "RESECTION CAVITY": "RC"}
    out = {}
    for _, r in df.iterrows():
        reg = long.get(str(r[reg_col]).strip().upper(),
                       str(r[reg_col]).strip().upper())
        if reg in REGIONS:
            out[reg] = (int(r["Num_TP"]), int(r["Num_FP"]), int(r["Num_FN"]))
    return out


def main():
    print("=" * 78)
    print(" SUPPRESSION RECONSTRUCTION GATE")
    print(f" rule under test: {RULE}")
    print("=" * 78)

    # ---- what the gate is standing on ------------------------------
    head("1 · THE PATH BEING USED")
    fn = getattr(ns, "official_score", None)
    if not callable(fn):
        print("""  ns.official_score not found.

  The gate will not rebuild the path by hand. Six attempts at that failed,
  each by reimplementing a step this function already performs, and a metric
  scored outside it is not the metric the analysis used.""")
        return
    print(f"""  ns.official_score(case, suppressor=...)

  folds  {ns.FOLDS}
  thr    {ns.THR}

  These are the module's own constants, the same ones the arm ran with, so
  they cannot drift from it. The evaluator, its challenge_name, the geometry
  guards and the native-space write are all inside that function.

  The probability archive is the one thing this cannot verify: ENSEMBLE_DIR
  is {ns.ENSEMBLE_DIR.replace(DRIVE, '…')}, a copy, and test-time
  augmentation is not recorded in the protocol text. If the archive is not
  the arm's inference run, both sides agree and the mask is still wrong.""")

    # ---- the cases, and why these -----------------------------------
    files = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv")
             if SHARD.search(f)]
    if not files:
        sys.exit(f"no count shards under {W2}")
    t4 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ref = t4[(t4.setting == RULE) & (t4.status.astype(str) == "ok")]
    base = t4[(t4.setting == "baseline") & (t4.status.astype(str) == "ok")]
    if not len(ref):
        sys.exit(f"{RULE} absent from the retained counts")
    rc = ref[ref.region == "RC"].merge(
        base[base.region == "RC"][["case", "Num_FP"]],
        on="case", suffixes=("", "_base"))
    rc["removed"] = rc.Num_FP_base - rc.Num_FP
    cases = rc.sort_values("removed", ascending=False).case.head(N_TEST).tolist()

    head("2 · SCORING")
    print(f"""
  {len(cases)} cases, those where this rule removed the most components: a
  case where suppression removed nothing cannot test a reconstruction of
  suppression.
""")
    # The evaluator writes intermediates into tmp_gt/<case> and
    # tmp_pred/<case> relative to WORK_DIR, which ns.ensure_evaluator()
    # chdirs into. _purge_case_temp removes those directories but does not
    # create them, so in a session that has never run the arm the parents are
    # absent. Creating them is idempotent and touches nothing else.
    work = getattr(ns, "WORK_DIR", None)
    if work:
        for sub in ("tmp_gt", "tmp_pred"):
            os.makedirs(os.path.join(work, sub), exist_ok=True)
    else:
        print("  note: ns.WORK_DIR not exposed; relying on the evaluator to "
              "create its own scratch")

    rows = []
    for case in cases:
        try:
            got = counts_from(fn(case,
                                 suppressor=ns.entropy_suppressor(ENTROPY_THR),
                                 tag=RULE,
                                 full=True))
        except Exception as e:
            print(f"  FAIL  {case}: {type(e).__name__}: {e}")
            rows.append({"case": case, "region": "-", "match": 0,
                         "note": f"{type(e).__name__}: {e}"})
            continue
        for reg in REGIONS:
            want = ref[(ref.case == case) & (ref.region == reg)]
            if not len(want) or reg not in got:
                continue
            w = want.iloc[0]
            tp, fp, fn_ = got[reg]
            ok = (tp == int(w.Num_TP) and fp == int(w.Num_FP)
                  and fn_ == int(w.Num_FN))
            rows.append({"case": case, "region": reg,
                         "rebuilt_TP": tp, "expected_TP": int(w.Num_TP),
                         "rebuilt_FP": fp, "expected_FP": int(w.Num_FP),
                         "rebuilt_FN": fn_, "expected_FN": int(w.Num_FN),
                         "match": int(ok), "note": ""})
            print(f"  {'ok  ' if ok else 'DIFF'} {case} {reg:3s} "
                  f"TP {tp}/{int(w.Num_TP)} · FP {fp}/{int(w.Num_FP)} · "
                  f"FN {fn_}/{int(w.Num_FN)}")

    d = pd.DataFrame(rows)
    scored = d[d.region != "-"] if len(d) else d

    head("VERDICT")
    if not len(scored):
        print("  nothing scored; the gate cannot pass on no evidence")
        d.to_csv(f"{OUT}/suppression_gate.csv", index=False)
        return
    n_ok = int(scored.match.sum())
    passed = n_ok == len(scored)
    print(f"""
  cells scored : {len(scored)}
  exact matches: {n_ok}
  mismatches   : {len(scored) - n_ok}""")

    d.to_csv(f"{OUT}/suppression_gate.csv", index=False)
    json.dump({"rule": RULE, "path": "ns.official_score",
               "folds": list(ns.FOLDS), "thr": dict(ns.THR),
               "cells": int(len(scored)), "matches": n_ok,
               "passed": bool(passed), "cases": cases},
              open(f"{OUT}/suppression_gate.json", "w"), indent=2, default=str)

    if passed:
        print("""
  PASSED. official_score, with the frozen suppressor, reproduces the Phase B
  counts exactly -- true positives, false positives and false negatives -- on
  every cell scored. The mask a figure would be drawn from is the mask the
  experiment produced, subject to the archive caveat above.

  Figure 5 can be built. Rebuild the prediction the same way, through
  official_score or through build_nested_label with the same suppressor, and
  draw from it.""")
    else:
        print("""
  FAILED. Figure 5 is not built.

  This is now a real disagreement rather than a coding difference in the
  gate: the path being scored is the arm's own, so the remaining candidates
  are about the inputs, not the logic.

    the probability archive. ENSEMBLE_DIR is a copy under
    archive/inference_v1. If inference was run more than once and the
    archived set is not the one the arm read, the folds match while the
    contents do not.

    the counts themselves. t4_counts came from a separate wave2 run; if that
    run used a different operating point or a different evaluator state, the
    comparison target is not what this rule produces.

  Do not adjust anything until it matches and then call it validated.

  And stopping here is an acceptable outcome. The suppression result is
  reported numerically in full -- four tables and the headroom comparison --
  and a qualitative panel is an aid, not the evidence. Dropping figure 5 with
  a stated reason is the same judgement that dropped the second model and the
  second dataset.""")

    print(f"\n  written: {OUT}/suppression_gate.csv · suppression_gate.json")


if __name__ == "__main__":
    main()
