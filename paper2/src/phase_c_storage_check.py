# -*- coding: utf-8 -*-
"""
==============================================================================
 PHASE C · STORAGE CHECK  —  what actually exists, before any C0 code
==============================================================================
 Read-only. Minutes. Writes nothing.

 Phase A documents that out-of-fold probability maps were produced for all
 1,296 cases and used for fold selection. That establishes they existed, not
 that they are still on disk. This checks the disk.

 It also verifies the geometry convention rather than assuming it. The npz
 volumes and the reference NIfTIs are stored on different axis orders, and
 that mismatch has already produced one silent failure in this project and one
 crash. C0 will compare predicted components against reference lesions
 voxel-for-voxel, so an axis error there would not crash -- it would quietly
 report that lesions are unrecoverable.

 WHAT IT ANSWERS
   1  how many of the 1,296 cases have probability maps on disk, per fold
   2  how many have a reference label
   3  do the two align once NPZ_TO_GT is inverted, on real cases
   4  what the probability arrays actually contain (range, channels, dtype)
   5  which cases are usable for C0, and which strata they fall in

 If this comes back short, C0 runs on whatever is genuinely available and the
 restriction is stated, rather than the analysis being written against a
 cohort that is not there.
==============================================================================
"""

import os
import re
import sys
import glob
import json
from collections import Counter

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
try:
    import native_space as ns
    DRIVE = ns.DRIVE
    GT_DIR = ns.GT_DIR
    NPZ_TO_GT = ns.NPZ_TO_GT
    HAVE_NS = True
except Exception as e:
    print(f"  [!] native_space unavailable ({e}); using literal paths")
    DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
    GT_DIR = (f"{DRIVE}/workspace/preprocessing/nnUNet_raw/"
              f"Dataset137_BraTS2026MET/labelsTr")
    NPZ_TO_GT = (2, 1, 0)
    HAVE_NS = False

FOLDS = (1, 2, 4)          # the deployed subset from Phase A
N_PROBE = 5                # cases to open and inspect in full


def find_prob_roots():
    """
    Locate directories holding per-fold probability maps. Searched rather than
    assumed: the layout has changed at least once, and a hard-coded path that
    silently matches nothing is worse than a search that reports what it found.
    """
    pats = [f"{DRIVE}/**/fold_*/**/*.npz", f"{DRIVE}/**/*fold*[0-9]*/*.npz",
            f"{DRIVE}/**/ensemble*/**/*.npz", f"{DRIVE}/**/*.npz"]
    seen = {}
    for p in pats:
        for f in glob.glob(p, recursive=True):
            d = os.path.dirname(f)
            seen[d] = seen.get(d, 0) + 1
        if seen:
            break
    return sorted(seen.items(), key=lambda kv: -kv[1])[:12]


def case_ids_from(paths):
    ids = set()
    for f in paths:
        m = re.search(r"(BraTS-MET-\d+-\d+)", f)
        if m:
            ids.add(m.group(1))
    return ids


def main():
    print("=" * 78)
    print(" PHASE C · STORAGE CHECK — read-only")
    print("=" * 78)
    print(f"  drive       : {DRIVE}")
    print(f"  gt dir      : {GT_DIR}")
    print(f"  NPZ_TO_GT   : {NPZ_TO_GT}"
          + ("  (from native_space)" if HAVE_NS else "  (assumed)"))

    # ---- 1. reference labels -------------------------------------------
    print("\n" + "=" * 78)
    print(" 1 · REFERENCE LABELS")
    print("=" * 78)
    gt_files = sorted(glob.glob(f"{GT_DIR}/*.nii.gz"))
    gt_ids = case_ids_from(gt_files)
    print(f"  label files : {len(gt_files)}")
    print(f"  case ids    : {len(gt_ids)}   (Phase A cohort = 1,296)")
    if len(gt_ids) < 1296:
        print(f"  [!] {1296 - len(gt_ids)} short of the full training cohort")

    # ---- 2. probability maps -------------------------------------------
    print("\n" + "=" * 78)
    print(" 2 · PROBABILITY MAPS")
    print("=" * 78)
    roots = find_prob_roots()
    if not roots:
        print("  none found under the drive.")
        print("""
  C0 cannot run without them. Two options, in order of preference:
    · locate an archive of the out-of-fold predictions from Phase A
    · re-run out-of-fold inference, which is the expensive path and should
      only be taken once the storage search has genuinely failed""")
        return

    print(f"  {'files':>7}  directory")
    for d, n in roots:
        print(f"  {n:>7}  {d.replace(DRIVE, '…')}")

    npz = []
    for d, _ in roots:
        npz += glob.glob(f"{d}/*.npz")
    npz_ids = case_ids_from(npz)
    print(f"\n  npz files   : {len(npz)}")
    print(f"  case ids    : {len(npz_ids)}")

    both = gt_ids & npz_ids
    print(f"  with BOTH   : {len(both)}   ← the cohort C0 can actually use")
    if gt_ids - npz_ids:
        print(f"  gt only     : {len(gt_ids - npz_ids)}")
    if npz_ids - gt_ids:
        print(f"  npz only    : {len(npz_ids - gt_ids)}")

    # ---- 3. geometry, verified not assumed ------------------------------
    print("\n" + "=" * 78)
    print(" 3 · GEOMETRY — verified against real cases, not assumed")
    print("=" * 78)
    probe = sorted(both)[:N_PROBE]
    inv = np.argsort(NPZ_TO_GT)
    ok_geom = 0
    for c in probe:
        f = next((x for x in npz if c in x), None)
        if f is None:
            continue
        try:
            with np.load(f) as z:
                # Take the probability array by name, not by position. These
                # archives hold more than one entry and at least one of the
                # others is 1-D, so indexing z.files[0] picked up whichever
                # numpy happened to list first.
                key = ("probabilities" if "probabilities" in z.files
                       else next((k for k in z.files
                                  if z[k].ndim >= 3), z.files[0]))
                arr = z[key]
            g = np.asanyarray(nib.load(f"{GT_DIR}/{c}.nii.gz").dataobj)
            spatial = arr.shape[-3:]
            match = tuple(np.array(g.shape)[NPZ_TO_GT]) == tuple(spatial) or \
                    tuple(np.array(spatial)[inv]) == g.shape
            ok_geom += bool(match)
            print(f"  {c}   (key '{key}')")
            print(f"    npz   {arr.shape}  dtype {arr.dtype}  "
                  f"range [{float(arr.min()):.3f}, {float(arr.max()):.3f}]")
            print(f"    gt    {g.shape}   labels {sorted(np.unique(g))[:6]}")
            print(f"    align {'ok' if match else 'MISMATCH'}"
                  f"   (NPZ_TO_GT={NPZ_TO_GT})")
        except Exception as e:
            print(f"  {c}: {type(e).__name__}: {e}")

    if probe and ok_geom == len(probe):
        print(f"\n  All {ok_geom} probed cases align under NPZ_TO_GT. C0 must use")
        print("  this same mapping rather than transposing independently: a second")
        print("  transpose written by hand is a second thing that can disagree.")
    elif probe:
        print(f"\n  [!] {len(probe) - ok_geom} of {len(probe)} did not align.")
        print("  Do NOT write C0 until this is settled. A mismatch here does not")
        print("  crash — it reports lesions as unrecoverable when they are not.")

    # ---- 4. channel semantics -------------------------------------------
    print("\n" + "=" * 78)
    print(" 4 · WHAT THE ARRAYS CONTAIN")
    print("=" * 78)
    if probe:
        f = next((x for x in npz if probe[0] in x), None)
        if f:
            with np.load(f) as z:
                print(f"  keys        : {z.files}")
                for k in z.files:
                    print(f"    {k:<16} shape {z[k].shape}  dtype {z[k].dtype}")
                key = ("probabilities" if "probabilities" in z.files
                       else next((k for k in z.files if z[k].ndim >= 3),
                                 z.files[0]))
                a = z[key]
            print(f"  using key   : {key}")
            print(f"  shape       : {a.shape}")
            n_ch = a.shape[0] if a.ndim == 4 else 1
            print(f"  channels    : {n_ch}"
                  + ("   (expected 4: WT TC ET RC)" if n_ch == 4 else
                     "   [!] not 4 — check the ordering before use"))
            print(f"  value range : [{float(a.min()):.4f}, {float(a.max()):.4f}]"
                  + ("   probabilities" if 0 <= a.min() and a.max() <= 1.001
                     else "   [!] not in [0,1] — logits?"))
            sz = os.path.getsize(f) / 1e6
            print(f"  file size   : {sz:.1f} MB → ~{sz * len(both) / 1000:.1f} GB "
                  f"for {len(both)} cases")

    # ---- 5. strata --------------------------------------------------------
    print("\n" + "=" * 78)
    print(" 5 · STRATA — where the usable cases fall")
    print("=" * 78)
    lk = f"{DRIVE}/phase_b_outputs/wave0/leak_class_567.csv"
    if os.path.exists(lk):
        d = pd.read_csv(lk)
        inside = d[d.case.isin(both)]
        print("  of the 567 Phase B cases, usable here:")
        print("  " + inside.leak_class.value_counts().to_string().replace("\n", "\n  "))
        outside = len(both - set(d.case))
        print(f"\n  usable cases OUTSIDE the 567 : {outside}")
        print("""
  Those cases matter: they are candidates for a development split that does
  not touch the Phase B evaluation stratum. Development on the 155 would use
  the evaluation cohort to make design decisions, which is the failure this
  whole sequence is built to avoid.""")
    else:
        print(f"  leak_class_567.csv not found at {lk}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)
    usable = len(both)
    if usable == 0:
        print("  No case has both a probability map and a label. C0 cannot run.")
    elif usable < 600:
        print(f"""  {usable} usable cases — less than half the Phase A cohort.

  C0 can run on these, but the restriction must be stated and the development
  split drawn from them. Check first whether the missing predictions are
  archived elsewhere; regenerating them is expensive but preserves the full
  cohort.""")
    else:
        print(f"""  {usable} usable cases.

  Enough for C0 as a full-cohort feasibility characterisation, and enough to
  draw a patient-level development split that excludes the Phase B evaluation
  stratum entirely.

  Next, and not before: C0-A (does a lower threshold surface the missed
  lesions at all) and C0-B (at what cost in candidate count). No model, no
  features, no graph until those two are answered.""")


if __name__ == "__main__":
    main()
