# -*- coding: utf-8 -*-
"""
==============================================================================
 C2 PRE-CHECK · ARE THE INPUT IMAGES AVAILABLE?
==============================================================================
 Read-only, minutes. Writes nothing.

 The next question is whether confidently missed lesions differ in image
 appearance from detected ones within the same case-region, after accounting
 for volume. Answering it needs the input images, and c0_lesions.csv contains
 none: it carries geometry and model probability, nothing about the scan.

 This establishes what is on disk before the analysis is designed around it.
 The pattern of assuming availability and discovering the gap mid-run has cost
 this project twice: the 729 cases without the Phase B ensemble, and the
 out-of-fold maps that answered a different question.

 It checks, per modality:
   presence for the C1 cohort
   intensity range and whether the volumes are already normalised
   voxel geometry against the reference, since a mismatch would silently
   sample the wrong location

 It does not measure any lesion. Choosing what to measure comes after knowing
 what exists, and the measurement design is fixed separately and in advance.
==============================================================================
"""

import os
import sys
import glob
from collections import Counter

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
C1 = f"{DRIVE}/phase_c_outputs/c0/c1_within_case.csv"

# BraTS names its four sequences with these suffixes; both conventions appear
# in different releases, so both are searched.
SUFFIXES = ["t1n", "t1c", "t2w", "t2f",          # BraTS 2023 onward
            "t1", "t1ce", "t2", "flair"]         # earlier convention
N_PROBE = 6


def find_image_roots():
    """Directories holding per-case sequence volumes, found not assumed."""
    pats = [f"{DRIVE}/**/imagesTr", f"{DRIVE}/**/*_t1c.nii.gz",
            f"{DRIVE}/**/*_t1ce.nii.gz", f"{DRIVE}/**/BraTS-MET-*/*.nii.gz"]
    dirs = Counter()
    for p in pats:
        for f in glob.glob(p, recursive=True):
            d = f if os.path.isdir(f) else os.path.dirname(f)
            dirs[d] += 1
    return dirs.most_common(15)


def sequences_for(case, roots):
    """Which of the four sequences can be found for this case, and where."""
    found = {}
    for suf in SUFFIXES:
        hits = glob.glob(f"{DRIVE}/**/{case}-{suf}.nii.gz", recursive=True)
        hits += glob.glob(f"{DRIVE}/**/{case}_{suf}.nii.gz", recursive=True)
        hits += glob.glob(f"{DRIVE}/**/{case}_{suf}_*.nii.gz", recursive=True)
        if hits:
            found[suf] = hits[0]
    return found


def main():
    print("=" * 78)
    print(" C2 PRE-CHECK · INPUT IMAGE AVAILABILITY")
    print(" nothing measured, nothing designed — availability only")
    print("=" * 78)

    if not os.path.exists(C1):
        sys.exit(f"missing {C1} — run C1 first")
    c1 = pd.read_csv(C1)
    cases = sorted(c1.case.unique())
    print(f"  C1 matched strata : {len(c1)}")
    print(f"  distinct cases    : {len(cases)}")

    # ---- 1 · where images might be ------------------------------------
    print("\n" + "=" * 78)
    print(" 1 · CANDIDATE IMAGE DIRECTORIES")
    print("=" * 78)
    roots = find_image_roots()
    if not roots:
        print("""
  None found under the drive.

  Intensity characterisation is not possible without the input volumes. Two
  routes, in order:
    · locate the preprocessed images used for training and inference
    · if they are genuinely absent, this question cannot be asked from what
      is retained, and the next candidate property is anatomical location,
      which can be derived from the reference labels alone
""")
        return
    print(f"\n  {'files':>7}  directory")
    for d, n in roots[:8]:
        print(f"  {n:>7}  {d.replace(DRIVE, '…')}")

    # ---- 2 · coverage for the cohort that matters ----------------------
    print("\n" + "=" * 78)
    print(" 2 · COVERAGE FOR THE C1 COHORT")
    print("=" * 78)
    probe = cases[:: max(1, len(cases) // 25)][:25]
    have = Counter()
    per_case = []
    for c in probe:
        f = sequences_for(c, roots)
        per_case.append(len(f))
        for k in f:
            have[k] += 1
    print(f"\n  probed {len(probe)} of {len(cases)} cases")
    print(f"\n  {'sequence':<10} {'found':>6}")
    for suf in SUFFIXES:
        if have.get(suf):
            print(f"  {suf:<10} {have[suf]:>6}")
    if not have:
        print("  none of the expected sequence names matched")
        print("\n  Filenames may follow another convention. Listing what is")
        print("  actually beside one case:")
        if roots:
            sample = glob.glob(f"{roots[0][0]}/*{probe[0]}*")[:8]
            for x in sample:
                print(f"    {os.path.basename(x)}")
        return
    full = sum(1 for n in per_case if n >= 4)
    print(f"\n  cases with at least four sequences: {full}/{len(probe)}")

    # ---- 3 · what the volumes contain ---------------------------------
    print("\n" + "=" * 78)
    print(" 3 · INTENSITY AND GEOMETRY")
    print("=" * 78)
    print("""
  Whether the volumes are raw or already normalised decides what a contrast
  measure can mean. A z-scored volume has no absolute scale, so any measure
  must be relative to tissue within the same scan either way -- but knowing
  which is the case decides whether that is a choice or a necessity.
""")
    for c in probe[:N_PROBE]:
        f = sequences_for(c, roots)
        if not f:
            continue
        print(f"  {c}")
        gt = nib.load(f"{ns.GT_DIR}/{c}.nii.gz")
        for suf, path in list(f.items())[:4]:
            img = nib.load(path)
            a = np.asanyarray(img.dataobj)
            same_shape = a.shape == gt.shape
            same_aff = np.allclose(img.affine, gt.affine, atol=1e-3)
            nz = a[a != 0]
            print(f"    {suf:<6} {str(a.shape):<18} "
                  f"range [{float(a.min()):>8.2f},{float(a.max()):>9.2f}] "
                  f"mean(nz) {float(nz.mean()) if nz.size else 0:>7.2f} "
                  f"| shape {'ok' if same_shape else 'DIFFERS'}"
                  f" affine {'ok' if same_aff else 'DIFFERS'}")
        print()

    # ---- verdict --------------------------------------------------------
    print("=" * 78)
    print(" WHAT THIS ESTABLISHES")
    print("=" * 78)
    print(f"""
  Availability only. Nothing has been measured and no comparison designed.

  If the sequences are present, aligned with the reference, and cover the C1
  cohort, then the appearance question can be asked. Its design -- which
  measure, which unit, which control for volume -- is fixed separately and
  before any number is computed, as C1's was.

  If they are absent or misaligned, the question cannot be asked from what is
  retained, and that is worth knowing now rather than after building an
  analysis around it.""")


if __name__ == "__main__":
    main()
