# -*- coding: utf-8 -*-
"""
==============================================================================
 FP-FLOOR PROBE  —  does size@20 move the official lesion-wise Dice?
==============================================================================
 RUNTIME: CPU, ~5-10 min for a handful of cases.
 ISOLATED: writes ONLY to fp_floor_probe.csv. Touches no shard, no protocol
 file, and no B-4 output. B-4 production keeps running untouched.

 THE QUESTION
 ------------
 The 4 Aug removal of size@20 rested on D6:

   "Any suppression threshold that removes only connected components below the
    evaluator's minimum lesion size cannot change lesion-wise Dice."

 Reading metrics_MET.py, that no longer holds as stated:

   761:  metric_df_thresh = metric_df[metric_df['gt_lesion_vol'] >= ...]
   766:  sum(metric_df_thresh['dice_lesionwise']) / (len(metric_df_thresh) + len(fp))

 The volume floor is applied to `gt_lesion_vol` -- GROUND TRUTH lesions. The
 `fp` list (line 652) enters the denominator with no size filter at all. So a
 predicted false-positive component of any size, down to a single voxel, adds
 1 to the denominator and 0 to the numerator. Removing it therefore CAN raise
 the score.

 The corrected statement:

   The evaluation volume floor applies to GT lesions, not to predicted FP
   components. Suppression below the floor cannot recover filtered GT lesions,
   but it CAN change lesion-wise Dice by removing FP components from the
   denominator.

 WHY PROBE AT ALL IF THE CODE SETTLES IT
 ---------------------------------------
 Reading the code proves the mechanism exists. It does not prove this cohort
 contains a case where it fires. Before a frozen protocol decision is reversed,
 the effect should be demonstrated, not inferred -- an empirical row is far
 stronger in an audit, and in a paper, than a code-reading argument.

 WHY day11 IS NOT THE EVIDENCE
 -----------------------------
 The provenance check in compute_ceiling.py showed day10/day11 disagree with
 the per-case evaluator counts by 300-450 lesions under BOTH configs. They are
 a different run. They are used here ONLY to nominate candidate cases; the
 probe recomputes everything from the live prediction with the live evaluator,
 so the answer does not depend on them being consistent.

 READING THE RESULT
 ------------------
   Dice changes    -> the mechanism fires in this cohort. D6 is revised and
                      size@20 becomes a valid Dice-arm candidate again.
   Dice unchanged  -> this does NOT restore the old D6. Investigate why the
                      components day11 lists are not FP components for the live
                      evaluator: the likely cause is provenance, not theory.
==============================================================================
"""

import os
import sys
import glob
import time
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
for _p in (_HERE, "/content/drive/MyDrive/brats_2026_metastases/src/eval"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import native_space as ns

OUT_CSV = f"{ns.DRIVE}/phase_b_outputs/wave2/fp_floor_probe.csv"
REGIONS = ("WT", "TC", "ET", "RC")

# Probe arms only. NOT the protocol RULES -- this file never touches B-4.
ARMS = {"baseline": None, "size@20": 20, "size@50": 50}


def find(name):
    hits = glob.glob(f"{ns.DRIVE}/**/{name}", recursive=True)
    return hits[0] if hits else None


def candidates(n=8, min_small=2):
    """
    Nominate cases from day11: at least `min_small` FP components under 20
    voxels in the SAME region, so the probe has more than one chance to fire.
    Nomination only -- day11 is not used as evidence anywhere below.
    """
    p = find("day11_fp_lesions.csv")
    if p is None:
        sys.exit("day11_fp_lesions.csv not found — pass --cases explicitly")
    d11 = pd.read_csv(p)
    s = (d11[d11.size_vox < 20].groupby(["case", "region"]).size()
         .rename("n_small").reset_index())
    s = s[s.n_small >= min_small].sort_values("n_small", ascending=False)
    if not len(s):
        sys.exit(f"no case has >= {min_small} sub-20 FP components in one region")

    print("  candidate cases (from day11 — nomination only, not evidence):")
    print(s.head(n).to_string(index=False))
    return s.head(n)


def component_sizes(case):
    """
    Ground truth for the probe's own bookkeeping: how many predicted components
    would each arm actually delete? Computed from the live probability map, so
    it does not depend on day11 being current.
    """
    mean = ns.load_mean_prob(case)
    from scipy import ndimage
    out = {}
    for reg in REGIONS:
        ch = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}[reg]
        lab, k = ndimage.label(mean[ch] >= ns.THR[reg])
        sz = np.bincount(lab.ravel())[1:] if k else np.array([], dtype=int)
        out[reg] = {"n_comp": int(k),
                    "n_below_20": int((sz <= 20).sum()),
                    "n_below_50": int((sz <= 50).sum()),
                    "smallest": int(sz.min()) if k else 0}
    del mean
    return out


def probe(case):
    print(f"\n{'='*74}\n{case}\n{'='*74}")
    comp = component_sizes(case)
    for reg in REGIONS:
        c = comp[reg]
        print(f"  {reg}: {c['n_comp']:>3} components · "
              f"{c['n_below_20']:>3} at/below 20 vox · "
              f"{c['n_below_50']:>3} at/below 50 · smallest {c['smallest']}")

    scores, rows = {}, []
    for arm, thr in ARMS.items():
        t0 = time.time()
        sup = ns.size_suppressor(thr) if thr else None
        res = ns.official_score(case, suppressor=sup,
                                tag=f"probe_{arm.replace('@','_at_')}",
                                guards=False)
        scores[arm] = {r["region"]: r["lesionwise_dice"] for r in res}
        print(f"    {arm:>10s}: " +
              "  ".join(f"{r['region']}={r['lesionwise_dice']:.6f}" for r in res) +
              f"   {time.time()-t0:.0f}s")

    for arm in ARMS:
        if arm == "baseline":
            continue
        for reg in REGIONS:
            b, v = scores["baseline"][reg], scores[arm][reg]
            rows.append({
                "case": case, "arm": arm, "region": reg,
                "baseline": round(b, 6), "suppressed": round(v, 6),
                "delta": round(v - b, 6),
                "changed": abs(v - b) > 1e-9,
                "n_comp": comp[reg]["n_comp"],
                "n_below_20": comp[reg]["n_below_20"],
                "n_below_50": comp[reg]["n_below_50"],
                "smallest_comp": comp[reg]["smallest"],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=3, help="how many nominated cases")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    print("=" * 74)
    print(" FP-FLOOR PROBE — isolated. B-4 is not touched.")
    print(" question: can size@20 move the official lesion-wise Dice?")
    print("=" * 74)

    ns.ensure_evaluator()
    ns.set_fast_metric(True)

    if a.cases:
        cases = a.cases
        print(f"  cases: EXPLICIT — {cases}")
    else:
        cases = candidates(n=a.n * 3)["case"].drop_duplicates().head(a.n).tolist()
        print(f"\n  probing: {cases}")

    rows = []
    for c in cases:
        try:
            rows += probe(c)
        except Exception as e:
            print(f"  ERROR on {c}: {type(e).__name__}: {e}")

    if not rows:
        sys.exit("no results")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 74)
    print(" VERDICT")
    print("=" * 74)

    s20 = df[df.arm == "size@20"]
    # Only cells where the arm actually had something to delete can testify.
    live = s20[s20.n_below_20 > 0]
    moved = live[live.changed]

    print(f"  size@20 cells where sub-20 components existed : {len(live)}")
    print(f"  of those, cells where Dice changed            : {len(moved)}")

    if len(moved):
        print("\n  MECHANISM CONFIRMED — size@20 moves the official Dice.\n")
        print(moved[["case", "region", "baseline", "suppressed", "delta",
                     "n_below_20"]].to_string(index=False))
        print("\n  D6 as currently worded is falsified. Corrected statement:")
        print("    The evaluation volume floor applies to GT lesions, not to")
        print("    predicted FP components. Suppression below the floor cannot")
        print("    recover filtered GT lesions, but CAN change lesion-wise Dice")
        print("    by removing FP components from the denominator.")
        print("\n  Consequence: the 4 Aug mathematical exclusion of size@20 is")
        print("  invalid, so size@20 is a valid CANDIDATE for the Dice arm again.")
        print("  Whether it is actually run is a separate protocol decision.")
    elif len(live):
        print("\n  NO CHANGE, though sub-20 components were present and deleted.")
        print("  This does NOT restore the original D6. Before concluding anything,")
        print("  check whether those components register as FP for the evaluator at")
        print("  all — they may be absorbed into a TP match, or sit inside a larger")
        print("  region under nesting, in which case they were never in `fp`.")
    else:
        print("\n  INCONCLUSIVE — no sub-20 components in any probed case.")
        print("  day11 nominated these cases from a different run. Re-nominate")
        print("  using the live component counts printed above.")

    print("\n  size@50 is included as a positive control: it should move cells")
    print("  where components at/below 50 exist, whatever size@20 does.")
    ctrl = df[(df.arm == "size@50") & (df.n_below_50 > 0)]
    print(f"    control cells: {len(ctrl)} · changed: {int(ctrl.changed.sum())}")

    if a.no_write:
        print("\n  --no-write: nothing saved.")
        return
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  written: {OUT_CSV}")


if __name__ == "__main__":
    main()
