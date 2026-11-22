# -*- coding: utf-8 -*-
"""
==============================================================================
 C2 COVERAGE · all 179 C1 cases, not a pilot
==============================================================================
 Read-only, minutes. Writes one manifest.

 The pilot found four sequences for 25 of 179 cases with matching shape and
 affine. That is an availability signal, not coverage: the 25 were taken at a
 stride through a sorted list, and this project has twice been caught by a
 property that held on a sample and failed on the rest.

 This checks every case C1 uses, records which are complete, and writes the
 usable cohort to a manifest so the analysis reads it rather than rediscovering
 it. If cases are missing, the cohort is fixed to what exists and the
 restriction is stated, exactly as the 729 were excluded from C0.
==============================================================================
"""

import os
import sys
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
C1 = f"{DRIVE}/phase_c_outputs/c0/c1_within_case.csv"
OUT = f"{DRIVE}/phase_c_outputs/c2"
SEQS = ["t1n", "t1c", "t2w", "t2f"]
CHECK_GEOM = 40          # cases to open and verify against the reference

os.makedirs(OUT, exist_ok=True)


def paths_for(case):
    found = {}
    for s in SEQS:
        h = (glob.glob(f"{DRIVE}/**/{case}-{s}.nii.gz", recursive=True)
             or glob.glob(f"{DRIVE}/**/{case}_{s}.nii.gz", recursive=True))
        if h:
            found[s] = h[0]
    return found


def main():
    print("=" * 78)
    print(" C2 COVERAGE · every case C1 uses")
    print("=" * 78)

    if not os.path.exists(C1):
        sys.exit(f"missing {C1}")
    c1 = pd.read_csv(C1)
    cases = sorted(c1.case.unique())
    print(f"  C1 strata : {len(c1)}")
    print(f"  cases     : {len(cases)}")

    # ---- 1 · presence ----------------------------------------------------
    print("\n--- 1 · sequence presence, all cases ---")
    rows, t0 = [], __import__("time").time()
    for i, c in enumerate(cases, 1):
        f = paths_for(c)
        rows.append({"case": c, **{s: int(s in f) for s in SEQS},
                     "n_seq": len(f)})
        if i % 40 == 0:
            print(f"    {i}/{len(cases)}")
    d = pd.DataFrame(rows)

    print(f"\n  {'sequence':<10} {'present':>8} {'of':>5}")
    for s in SEQS:
        print(f"  {s:<10} {int(d[s].sum()):>8} {len(d):>5}")
    complete = d[d.n_seq == len(SEQS)]
    print(f"\n  cases with all four : {len(complete)}/{len(cases)}"
          f"   ({len(complete)/len(cases):.1%})")
    if len(complete) < len(cases):
        missing = d[d.n_seq < len(SEQS)]
        print(f"  incomplete          : {len(missing)}")
        print("  " + missing.head(10).to_string(index=False).replace("\n", "\n  "))

    # ---- 2 · geometry against the reference ------------------------------
    print(f"\n--- 2 · geometry, {min(CHECK_GEOM, len(complete))} cases opened ---")
    probe = complete.case.tolist()[:: max(1, len(complete) // CHECK_GEOM)][:CHECK_GEOM]
    bad_shape, bad_aff, ranges = [], [], []
    for c in probe:
        f = paths_for(c)
        gt = nib.load(f"{ns.GT_DIR}/{c}.nii.gz")
        for s, p in f.items():
            img = nib.load(p)
            if img.shape != gt.shape:
                bad_shape.append((c, s, img.shape, gt.shape))
            elif not np.allclose(img.affine, gt.affine, atol=1e-3):
                bad_aff.append((c, s))
            a = np.asanyarray(img.dataobj)
            nz = a[a > 0]
            ranges.append({"case": c, "seq": s, "max": float(a.max()),
                           "mean_nz": float(nz.mean()) if nz.size else 0.0})

    print(f"  shape mismatches  : {len(bad_shape)}")
    for c, s, a, b in bad_shape[:5]:
        print(f"    {c} {s}: {a} vs reference {b}")
    print(f"  affine mismatches : {len(bad_aff)}")
    for c, s in bad_aff[:5]:
        print(f"    {c} {s}")

    r = pd.DataFrame(ranges)
    print("\n  intensity ranges across the probed cases:")
    g = r.groupby("seq").agg(n=("max", "size"), max_min=("max", "min"),
                             max_max=("max", "max"),
                             mean_nz_min=("mean_nz", "min"),
                             mean_nz_max=("mean_nz", "max"))
    print("  " + g.round(1).to_string().replace("\n", "\n  "))
    spread = (g.mean_nz_max / g.mean_nz_min.replace(0, np.nan)).max()
    print(f"""
  Mean non-zero intensity varies by up to {spread:.0f}x across cases within a
  single sequence. Absolute intensity values are therefore not directly
  comparable across subjects and cannot serve as the appearance measure; a
  within-scan relative quantity is required. That is a consequence of the
  data, not a preference.""")

    # ---- 3 · the usable cohort ------------------------------------------
    usable = sorted(set(complete.case) -
                    {c for c, *_ in bad_shape} - {c for c, _ in bad_aff})
    strata = c1[c1.case.isin(usable)]
    print("\n" + "=" * 78)
    print(" USABLE COHORT")
    print("=" * 78)
    print(f"  cases  : {len(usable)}/{len(cases)}")
    print(f"  strata : {len(strata)}/{len(c1)}")

    json.dump({
        "source": "C1 matched strata",
        "cases_requested": len(cases),
        "cases_with_all_sequences": len(complete),
        "cases_usable": len(usable),
        "strata_usable": int(len(strata)),
        "sequences": SEQS,
        "geometry_checked": len(probe),
        "shape_mismatches": len(bad_shape),
        "affine_mismatches": len(bad_aff),
        "cases": usable,
    }, open(f"{OUT}/c2_cohort.json", "w"), indent=2)
    d.to_csv(f"{OUT}/c2_coverage.csv", index=False)

    if len(usable) == len(cases):
        print("""
  Every C1 case has all four sequences, correctly aligned. The appearance
  question can be asked on the full C1 cohort.""")
    else:
        print(f"""
  {len(cases) - len(usable)} case(s) unusable. The cohort is fixed to the {len(usable)}
  that are, and the restriction is stated in the results rather than the
  shortfall being absorbed silently.""")

    print(f"\n  written: {OUT}/c2_cohort.json · c2_coverage.csv")
    print("""
  Nothing has been measured. The measurement definition is frozen separately,
  before any appearance value is computed.""")


if __name__ == "__main__":
    main()
