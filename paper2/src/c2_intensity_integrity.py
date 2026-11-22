# -*- coding: utf-8 -*-
"""
==============================================================================
 C2 · INTENSITY INTEGRITY  —  before any measurement
==============================================================================
 Read-only, seconds. Writes nothing.

 The coverage check found mean non-zero intensity varying by up to 2,401x
 across cases within a single sequence, with t1c maxima above two million in
 at least one case. That is evidence of differing intensity scaling. It is not
 by itself evidence that any volume is corrupt, and the two need separating
 before a measurement runs over them.

 The frozen measure is scale invariant under uniform multiplication: a factor
 applied to a whole volume appears in both the numerator difference and the
 MAD and cancels. So a rescaled volume is fine. A volume whose values are
 wrong is not, and the ratio would carry that error through silently.

 WHAT DISTINGUISHES THEM
   rescaling   the whole distribution shifts together; p50, p99 and max keep
               their relative spacing; the histogram has the usual shape
   corruption  a handful of voxels sit orders of magnitude above the rest, or
               the distribution has no tissue structure at all

 The check reports the shape of the distribution and the header scaling, and
 leaves the judgement explicit rather than applying a rule.

 ON THE THRESHOLDS BELOW
 -----------------------
 The factors used to compare against the controls (0.4, 2.5, 5x) are
 operational triage rules, not statistical criteria. There is no reference
 distribution for p99/p50, so nothing here tests whether a value is "normal" --
 only whether it is consistent with the ordinary cases alongside it. A case
 falling outside them is a case to look at, not a case shown to be wrong.

 This is data-integrity triage before an analysis, not a hypothesis test, and
 the numbers it produces carry no inferential weight.
==============================================================================
"""

import os
import sys
import glob
import json

import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
COHORT = f"{DRIVE}/phase_c_outputs/c2/c2_cohort.json"
SEQS = ["t1n", "t1c", "t2w", "t2f"]

# Cases flagged by the coverage check for extreme values, plus a few ordinary
# ones for comparison. A distribution only looks abnormal against a normal one.
FLAGGED = ["BraTS-MET-00604-000"]
CONTROLS = ["BraTS-MET-00014-000", "BraTS-MET-00137-000"]


def path(case, seq):
    h = (glob.glob(f"{DRIVE}/**/{case}-{seq}.nii.gz", recursive=True)
         or glob.glob(f"{DRIVE}/**/{case}_{seq}.nii.gz", recursive=True))
    return h[0] if h else None


def profile(case, seq):
    p = path(case, seq)
    if p is None:
        return None
    img = nib.load(p)
    a = np.asanyarray(img.dataobj)
    nz = a[a > 0]
    if not nz.size:
        return {"case": case, "seq": seq, "empty": True}
    q = np.percentile(nz, [1, 25, 50, 75, 90, 99, 99.9])
    return {
        "case": case, "seq": seq, "dtype": str(a.dtype),
        "slope": float(img.dataobj.slope), "inter": float(img.dataobj.inter),
        "p1": q[0], "p25": q[1], "p50": q[2], "p75": q[3],
        "p90": q[4], "p99": q[5], "p999": q[6], "max": float(nz.max()),
        # how far the extreme tail sits above the bulk: a rescaled volume keeps
        # this ratio ordinary, a corrupt one does not
        "max_over_p99": float(nz.max() / q[5]) if q[5] > 0 else np.inf,
        "p99_over_p50": float(q[5] / q[2]) if q[2] > 0 else np.inf,
        "n_above_p999": int((nz > q[6]).sum()),
        "n_nonzero": int(nz.size),
    }


def show(case, label):
    print(f"\n  {case}   [{label}]")
    print(f"    {'seq':<5} {'dtype':<9} {'slope':>6} "
          f"{'p50':>12} {'p99':>12} {'max':>14} "
          f"{'max/p99':>9} {'p99/p50':>9}")
    rows = []
    for s in SEQS:
        r = profile(case, s)
        if r is None:
            print(f"    {s:<5} not found")
            continue
        if r.get("empty"):
            print(f"    {s:<5} all zero")
            continue
        rows.append(r)
        print(f"    {s:<5} {r['dtype']:<9} {r['slope']:>6.2f} "
              f"{r['p50']:>12.1f} {r['p99']:>12.1f} {r['max']:>14.1f} "
              f"{r['max_over_p99']:>9.1f} {r['p99_over_p50']:>9.2f}")
    return rows


def main():
    print("=" * 78)
    print(" C2 · INTENSITY INTEGRITY")
    print(" separating rescaling from corruption, before measurement")
    print("=" * 78)

    all_rows = []
    for c in FLAGGED:
        all_rows += show(c, "flagged by coverage")
    for c in CONTROLS:
        all_rows += show(c, "control")

    if not all_rows:
        sys.exit("  nothing profiled")
    d = pd.DataFrame(all_rows)

    print("\n" + "=" * 78)
    print(" READING THE SHAPE, NOT THE SCALE")
    print("=" * 78)
    print("""
  Two ratios carry the judgement, and neither depends on the absolute scale:

    p99/p50    the spread of the tissue distribution. Similar across cases
               means the volumes have the same shape at different scales.
    max/p99    how far the single brightest voxel sits above the bulk. A few
               hundred is a bright vessel or an artefact voxel; orders of
               magnitude more is not tissue.
""")
    g = d.groupby("seq").agg(
        p99_p50_min=("p99_over_p50", "min"), p99_p50_max=("p99_over_p50", "max"),
        max_p99_min=("max_over_p99", "min"), max_p99_max=("max_over_p99", "max"))
    print("  " + g.round(2).to_string().replace("\n", "\n  "))

    flagged_rows = d[d.case.isin(FLAGGED)]
    ctrl_rows = d[d.case.isin(CONTROLS)]

    print("\n" + "=" * 78)
    print(" TRIAGE (operational rules, not statistical criteria)")
    print("=" * 78)
    for _, r in flagged_rows.iterrows():
        same_seq = ctrl_rows[ctrl_rows.seq == r.seq]
        if not len(same_seq):
            continue
        lo, hi = same_seq.p99_over_p50.min(), same_seq.p99_over_p50.max()
        shape_ok = lo * 0.4 <= r.p99_over_p50 <= hi * 2.5
        tail_ok = r.max_over_p99 <= max(50.0, same_seq.max_over_p99.max() * 5)
        print(f"\n  {r.case} · {r.seq}")
        print(f"    p99/p50 = {r.p99_over_p50:.2f}   controls {lo:.2f}–{hi:.2f}"
              f"   {'consistent' if shape_ok else 'DIFFERENT'}")
        print(f"    max/p99 = {r.max_over_p99:.1f}   controls "
              f"{same_seq.max_over_p99.min():.1f}–{same_seq.max_over_p99.max():.1f}"
              f"   {'ordinary' if tail_ok else 'EXTREME TAIL'}")
        print(f"    voxels above p99.9: {int(r.n_above_p999):,} of "
              f"{int(r.n_nonzero):,}")
        if shape_ok and tail_ok:
            print("    -> rescaled, not corrupt. The frozen measure is invariant")
            print("       to uniform multiplication, so this case is usable.")
        elif shape_ok:
            print("    -> distribution shape is ordinary but the extreme tail is")
            print("       not. The median and MAD are unaffected by a handful of")
            print("       outlying voxels, so the measure still holds; note it.")
        else:
            print("    -> the distribution differs in shape from the controls.")
            print("       That flags it for inspection; it does not establish that")
            print("       the volume is wrong. Do NOT assume the ratio protects")
            print("       against a shape difference. If the case is dropped, the")
            print("       exclusion is recorded, as the 729 were for C0.")

    print("""
  ─────────────────────────────────────────────────────────────────────────
  Whatever this shows, the measurement definition does not change. A case that
  cannot be measured reliably leaves the cohort with the exclusion recorded;
  the measure is not adjusted to accommodate it.""")


if __name__ == "__main__":
    main()
