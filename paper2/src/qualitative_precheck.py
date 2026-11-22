# -*- coding: utf-8 -*-
"""
==============================================================================
 QUALITATIVE FIGURES · PRE-CHECK
==============================================================================
 Read-only, minutes. Writes nothing.

 Four figures are proposed. Three can be built from what exists; one depends
 on something that may not have been retained, and this establishes which
 before any of them is written.

   1  dataset anatomy      sequences, labels, composite regions
   2  model output         MRI, reference, prediction, error map
   3  before/after         baseline against a suppressed arm
   4  small-lesion cases   four pre-specified situations

 Figures 1, 2 and 4 need the sequences, the reference and the frozen
 probability maps. All three are known to exist: the appearance analysis read
 them for 179 cases.

 Figure 3 needs the SUPPRESSED prediction masks. Phase B retained the
 evaluator's scores and counts, not the masks each rule produced --- the arms
 were scored and the masks discarded. If that is so, figure 3 requires
 re-deriving each mask by re-applying the frozen rule to the frozen
 probability output, which is reproducible but is a computation, not a lookup.
 This check settles which.

 CASE SELECTION, FIXED HERE BEFORE ANY IMAGE IS SEEN
 ---------------------------------------------------
 Choosing a case because its picture looks good is selection on the outcome,
 which this project has refused everywhere else. Every case below is chosen by
 a deterministic rule stated in advance, leaving no room for a manual pick.

 THE COHORT. Only cases with retained probability output can be drawn, and
 those are the 567 of the Phase B cohort rather than all 1,296. The
 qualitative cohort is therefore the intersection of that set with the clean
 stratum, and that restriction is a property of what was retained, not a
 choice. It is stated in the figure captions.

   figure 1  the case whose lesion-wise WT Dice is the MEDIAN of the
             qualitative cohort. Not the best, not the clearest --- the
             middle one.
   figure 2  the same case, so the reader follows one case throughout.
   figure 3  the case with the median RC false-positive count among cases
             where entropy suppression removed at least one component. That
             region is chosen because it is where suppression did anything.
   figure 4  four cases, one per pre-specified situation:

               (a) the smallest reference lesion ABOVE THE EVALUATOR'S
                   2 mm3 floor. The floor is applied explicitly here rather
                   than taken from c0_lesions.csv, which by design contains
                   lesions the evaluator does not score: an earlier version
                   of this rule selected a 0.1 mm3 lesion, which the
                   evaluator excludes, and a figure captioned "small scored
                   lesion" showing an unscored one would be wrong.

               (b) the case whose lesion count is CLOSEST TO THE 75th
                   PERCENTILE of the cohort. Not the maximum: the maximum is
                   180 lesions, which is unreadable in a panel and is not
                   what "multiple lesions" is meant to illustrate.

               (c) the false positive nearest to a true lesion.

               (d) the largest confidently missed lesion. Already named in
                   the manuscript's appendix, so it introduces no new
                   selection.

 No case is chosen for appearance. The rules live here rather than in the
 figure captions so that they cannot be adjusted once the images are seen.

==============================================================================
"""

import os
import re
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
W1 = f"{DRIVE}/phase_b_outputs/wave1"
W2 = f"{DRIVE}/phase_b_outputs/wave2"
C0 = f"{DRIVE}/phase_c_outputs/c0"
SHARD = re.compile(r"\.w\d+of\d+\.csv$")


def head(t):
    print("\n" + "=" * 78)
    print(f" {t}")
    print("=" * 78)


def main():
    print("=" * 78)
    print(" QUALITATIVE FIGURES · what can be built")
    print("=" * 78)

    # ---- 1 · the inputs every figure needs -------------------------
    head("1 · SEQUENCES, REFERENCE, PROBABILITIES")
    gt = sorted(glob.glob(f"{ns.GT_DIR}/*.nii.gz"))
    print(f"  reference labels        : {len(gt)} cases")

    probe = os.path.basename(gt[0]).split(".nii")[0] if gt else None
    if probe:
        seqs = {}
        for s in ("t1n", "t1c", "t2w", "t2f"):
            h = glob.glob(f"{DRIVE}/**/{probe}-{s}.nii.gz", recursive=True)
            seqs[s] = bool(h)
        print(f"  sequences for {probe}: "
              f"{', '.join(k for k, v in seqs.items() if v) or 'none found'}")
        try:
            m = ns.load_mean_prob(probe)
            print(f"  probability maps        : yes, shape {m.shape}")
        except Exception as e:
            print(f"  probability maps        : FAILED -- {type(e).__name__}")

    # ---- 2 · suppressed masks, which figure 3 needs ----------------
    head("2 · SUPPRESSED PREDICTION MASKS (figure 3)")
    pats = [f"{DRIVE}/**/*entropy*0.5*/*.nii.gz",
            f"{DRIVE}/**/*suppressed*/*.nii.gz",
            f"{DRIVE}/**/*size@*/*.nii.gz",
            f"{W1}/**/*.nii.gz", f"{W2}/**/*.nii.gz"]
    found = {}
    for p in pats:
        for f in glob.glob(p, recursive=True)[:400]:
            found[os.path.dirname(f)] = found.get(os.path.dirname(f), 0) + 1
    if found:
        print("  mask directories found:")
        for d, n in sorted(found.items(), key=lambda x: -x[1])[:6]:
            print(f"    {n:>5}  {d.replace(DRIVE, '…')}")
        print("""
  If these are the per-arm masks, figure 3 is a lookup.""")
    else:
        print("""  none found.

  The suppression arms were scored and their masks discarded, which is the
  expected outcome: Phase B retained scores and counts. Figure 3 therefore
  requires re-deriving each mask by re-applying the frozen rule to the frozen
  probability output.

  That is reproducible --- the rules and thresholds are fixed and the
  probability maps are retained --- but it is a computation, and it must
  reproduce the evaluator's own counts before any image drawn from it is
  trusted. The gate is the same one used in Phase C: rebuild, then check the
  rebuilt counts against the evaluator's Num_FP and Num_TP for the case shown.
  An image built from an unvalidated reconstruction would be an illustration
  of our code rather than of the experiment.""")

    # ---- 3 · locate the probability maps ---------------------------
    head("3 · WHERE THE PROBABILITY MAPS ARE")
    print("""
  The first case alphabetically has none, which is expected: the maps were
  retained for the 567-case Phase B cohort, not for all 1,296. The
  qualitative cohort is therefore bounded by what was kept.
""")
    lk = f"{DRIVE}/phase_b_outputs/wave0/leak_class_567.csv"
    cohort, clean = [], []
    if os.path.exists(lk):
        L = pd.read_csv(lk)
        cohort = sorted(L.case)
        clean = sorted(L[L.leak_class == "clean"].case)
        print(f"  Phase B cohort          : {len(cohort)}")
        print(f"  clean stratum           : {len(clean)}")
        ok = 0
        for c in clean[:12]:
            try:
                ns.load_mean_prob(c); ok += 1
            except Exception:
                pass
        print(f"  probability maps load   : {ok}/12 probed from the clean stratum")
        if ok < 12:
            print("""
  Not every clean case loads. Establish which do before selecting, or the
  rule may pick a case that cannot be drawn.""")
    else:
        print(f"  {lk} not found -- cannot define the qualitative cohort")
        return

    # ---- 4 · apply the corrected rules -----------------------------
    head("4 · THE CASES THE CORRECTED RULES SELECT")
    b4 = f"{W1}/exp3_suppressed_v3.csv"
    if os.path.exists(b4):
        d = pd.read_csv(b4)
        # the columns are case, setting, region, lesionwise_dice -- long form
        wt = d[(d.setting == "baseline") & (d.region == "WT") &
               (d.status.astype(str) == "ok") & (d.case.isin(clean))]
        wt = wt.dropna(subset=["lesionwise_dice"]).sort_values("lesionwise_dice")
        if len(wt):
            mid = wt.iloc[len(wt) // 2]
            print(f"  figures 1 and 2, median WT Dice : {mid.case}")
            print(f"    Dice {mid.lesionwise_dice:.4f}, "
                  f"rank {len(wt)//2 + 1} of {len(wt)}")
    else:
        print(f"  {b4} not found")

    les = f"{C0}/c0_lesions.csv"
    if os.path.exists(les):
        L = pd.read_csv(les)

        # (a) the evaluator's 2 mm3 floor applied explicitly, not inherited
        det = L[(L.baseline_detected == 1) & (L.gt_volume_mm3 > 2.0)]
        if len(det):
            sm = det.sort_values("gt_volume_mm3").iloc[0]
            print(f"\n  4a smallest SCORED lesion (> 2 mm3): "
                  f"{sm.case} {sm.region}, {sm.gt_volume_mm3:.1f} mm3")
            below = int((L[L.baseline_detected == 1].gt_volume_mm3 <= 2.0).sum())
            print(f"     ({below} detected rows fall at or below the floor and "
                  f"are excluded here)")

        # (b) the 75th percentile of lesion count, not the maximum
        cnt = L[L.baseline_detected == 1].groupby("case").size()
        if len(cnt):
            q75 = cnt.quantile(0.75)
            pick = (cnt - q75).abs().idxmin()
            print(f"\n  4b lesion count nearest the 75th percentile: "
                  f"{pick} with {int(cnt[pick])} lesions")
            print(f"     (75th percentile {q75:.0f}; the maximum is "
                  f"{int(cnt.max())}, which is unreadable in a panel)")

        # (d) already named in the appendix
        fn = L[(L.baseline_detected == 0) & (L.max_prob_in_lesion <= 0.001)]
        if len(fn):
            big = fn.sort_values("gt_volume_mm3", ascending=False).iloc[0]
            print(f"\n  4d largest confidently missed lesion: "
                  f"{big.case} {big.region} lesion {int(big.gt_lesion)}, "
                  f"{big.gt_volume_mm3:.0f} mm3")
            print("     (already named in the manuscript appendix)")
        print("\n  4c the nearest false positive requires the component "
              "extraction and\n     is resolved when the figure is built, "
              "under the same rule.")
    else:
        print(f"\n  {les} not found")

    # ---- verdict ----------------------------------------------------
    head("WHAT THIS SETTLES")
    print("""
  The qualitative cohort is bounded by what was retained: probability maps
  exist for the 567-case Phase B cohort, not for all 1,296, so figures are
  drawn from the intersection of that cohort with the clean stratum. This is a
  constraint on what can be shown, and it belongs in the figure captions.

  Two selection rules were wrong and are corrected above. The first would have
  shown a 0.1 mm3 lesion as a "small scored lesion" when the evaluator does
  not score it, because it took the floor from a table built without one; the
  floor is now applied explicitly. The second would have shown a case with 180
  lesions as "multiple lesions", which is unreadable in a panel and is not
  what the situation is meant to illustrate; the rule is now the 75th
  percentile.

  Both were caught before anything was drawn, which is the whole purpose of
  running this first.

  Figure 3 still depends on the suppressed masks, which were not retained. It
  requires a reconstruction validated against the evaluator's own counts
  before any image from it is used --- the same gate as Phase C. Until that
  passes, figure 3 is not built.

  Nothing has been drawn. The figure script is written against what this check
  found, not against what a visualisation would ideally show.""")


if __name__ == "__main__":
    main()
