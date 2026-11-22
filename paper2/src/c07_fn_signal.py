# -*- coding: utf-8 -*-
"""
==============================================================================
 C0.7 · FN SIGNAL AVAILABILITY  —  absent signal, or present but weak?
==============================================================================
 Read-only, seconds, from c0_lesions.csv. No inference, no training, no
 threshold search, nothing recomputed.

 C0 established that four fifths of the missed lesions produce no overlapping
 component at any proposal threshold down to 0.02. That rules out candidate
 generation as a recovery route, but it does not by itself say why, and the
 two possible reasons point in opposite directions:

   A  the probability inside those lesions is essentially zero
      -> the signal is absent from the output. No post-processing can recover
         what was never represented, and the remaining room is in the model.

   B  the probability is low but non-zero
      -> the signal is present and thresholding plus connected components is
         what loses it. Representation-level recovery would still be open.

 One case showed Pmax = 0.0163 in a 40 mm3 lesion, which points at A. One case
 is not a distribution, so this reads all 1,875.

 The decision rule, fixed before looking, follows the reviewer's:
     Pmax > 0.02 in under 25% of missed lesions  -> A, signal absent
     Pmax > 0.02 in over  60%                    -> B, signal present
     between                                     -> mixed; report both and
                                                    make no architectural
                                                    claim from it
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
C0 = f"{DRIVE}/phase_c_outputs/c0"
LES = f"{C0}/c0_lesions.csv"

LOW, HIGH = 0.25, 0.60          # the decision thresholds, fixed in advance
KEY = 0.02                      # the lowest proposal threshold C0 swept


def main():
    if not os.path.exists(LES):
        sys.exit(f"missing {LES}")
    les = pd.read_csv(LES)
    fn = les[les.baseline_detected == 0].copy()
    cand_cols = [c for c in fn.columns if c.startswith("cand@")]
    fn["surfaced"] = fn[cand_cols].max(axis=1) > 0

    print("=" * 78)
    print(" C0.7 · FN SIGNAL AVAILABILITY")
    print(" from the frozen C0 table · nothing recomputed")
    print("=" * 78)
    print(f"  missed lesions   : {len(fn):,}")
    print(f"  cases            : {fn.case.nunique()}")
    print(f"  surfaced at all  : {int(fn.surfaced.sum()):,}"
          f"   ({fn.surfaced.mean():.1%})")

    # ---- 1 · the distribution ------------------------------------------
    print("\n" + "=" * 78)
    print(" 1 · MAXIMUM PROBABILITY INSIDE A MISSED LESION")
    print("=" * 78)
    q = fn.max_prob_in_lesion.quantile([.25, .5, .75, .9, .95, .99])
    print(f"\n  {'quantile':>10} {'Pmax':>10}")
    for k, v in q.items():
        print(f"  {k:>10.0%} {v:>10.4f}")
    print(f"  {'mean':>10} {fn.max_prob_in_lesion.mean():>10.4f}")
    print(f"  {'max':>10} {fn.max_prob_in_lesion.max():>10.4f}")

    print(f"\n  {'above':>10} {'fraction':>10} {'n':>8}")
    fracs = {}
    for t in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        f = float((fn.max_prob_in_lesion > t).mean())
        fracs[t] = f
        mark = "  <- decision" if t == KEY else ""
        print(f"  {t:>10.3f} {f:>10.1%} {int((fn.max_prob_in_lesion > t).sum()):>8}{mark}")

    # ---- 2 · surfaced against unsurfaced --------------------------------
    print("\n" + "=" * 78)
    print(" 2 · SURFACED AGAINST UNSURFACED")
    print("=" * 78)
    g = fn.groupby("surfaced").agg(
        n=("max_prob_in_lesion", "size"),
        med_Pmax=("max_prob_in_lesion", "median"),
        mean_Pmax=("max_prob_in_lesion", "mean"),
        med_Pmean=("mean_prob_in_lesion", "median"),
        med_vol_mm3=("gt_volume_mm3", "median"))
    print("\n  " + g.round(4).to_string().replace("\n", "\n  "))
    print("""
  If unsurfaced lesions carry a comparable Pmax to surfaced ones, the
  difference is in connectivity and component size rather than in the
  probability itself. If their Pmax is far lower, the two groups differ in
  whether the signal exists at all.""")

    # ---- 3 · by lesion size ---------------------------------------------
    print("\n" + "=" * 78)
    print(" 3 · BY LESION SIZE")
    print("=" * 78)
    fn["bin"] = pd.cut(fn.gt_volume_mm3, [0, 27, 100, 500, np.inf],
                       labels=["(0,27]", "(27,100]", "(100,500]", ">500"])
    b = fn.groupby("bin", observed=True).agg(
        n=("surfaced", "size"),
        surfaced=("surfaced", "mean"),
        med_Pmax=("max_prob_in_lesion", "median"),
        frac_above_key=("max_prob_in_lesion", lambda s: float((s > KEY).mean())))
    print("\n  " + b.round(3).to_string().replace("\n", "\n  "))

    # ---- 4 · by region ---------------------------------------------------
    print("\n" + "=" * 78)
    print(" 4 · BY REGION")
    print("=" * 78)
    r = fn.groupby("region").agg(
        n=("surfaced", "size"),
        surfaced=("surfaced", "mean"),
        med_Pmax=("max_prob_in_lesion", "median"),
        frac_above_key=("max_prob_in_lesion", lambda s: float((s > KEY).mean())))
    print("\n  " + r.round(3).to_string().replace("\n", "\n  "))

    # ---- verdict ----------------------------------------------------------
    key = fracs[KEY]
    print("\n" + "=" * 78)
    print(" VERDICT (rule fixed before the numbers were seen)")
    print("=" * 78)
    print(f"\n  Pmax above {KEY} in {key:.1%} of missed lesions")
    print(f"  thresholds: absent below {LOW:.0%} · present above {HIGH:.0%}\n")

    if key < LOW:
        print("""  A — THE SIGNAL IS ABSENT.

  The model assigns near-zero probability inside most of the lesions it
  misses. These are not lesions lost to thresholding or to a component-size
  rule: they are lesions the output does not represent. No post-processing of
  this output can recover them, because there is nothing in it to recover.

  That closes candidate recovery as a direction, and it closes it for a
  reason that is worth stating rather than a limitation to work around. What
  remains open is the model itself -- what it is trained to preserve, at what
  resolution, and with what loss -- which is a different programme from
  post-processing and should be scoped as one.""")
    elif key > HIGH:
        print("""  B — THE SIGNAL IS PRESENT BUT WEAK.

  Most missed lesions carry non-negligible probability. They are lost between
  the probability map and the component representation, not absent from the
  map. Recovery at the representation level is therefore not excluded, though
  C0 shows that thresholding plus connected components is not the way to do
  it, and any alternative would have to beat a burden of roughly 23 candidates
  per lesion surfaced.""")
    else:
        print(f"""  MIXED — {key:.1%} sits between the two thresholds.

  Neither reading is supported outright. Report the distribution and make no
  architectural claim from it. The size and region breakdowns above may show
  the split falling differently in different strata, which would be the more
  useful statement than a single fraction.""")

    print("""
 ─────────────────────────────────────────────────────────────────────────
  On the false-positive side, the wording matters too. Phase B found small
  but in places statistically detectable effects, against an oracle headroom
  of 1.3-2.1x the MDE in the tumour regions. The defensible statement is that
  the attainable benefit from false-positive suppression is strongly
  constrained by that headroom -- not that suppression is impossible.

  Read the two together and the claim is about limits, quantified from both
  sides, rather than about a method having failed.""")


if __name__ == "__main__":
    main()
