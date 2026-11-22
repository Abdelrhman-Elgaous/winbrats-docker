# -*- coding: utf-8 -*-
"""
==============================================================================
 COHORT DESCRIPTION · full-cohort measurement for the data section
==============================================================================
 Read-only over the reference files. One pass. Writes one table.

 Two numbers in the data section draft rest on a 120-case stride sample:
 image geometry and resection-cavity prevalence. A stride sample through a
 sorted case list is not a random sample — if the sort correlates with
 acquisition site or date, it is systematically unrepresentative. The full
 measurement costs one pass, so the sample should not be what the paper
 reports.

 Everything here is measured. Nothing is inferred, and nothing about the
 challenge's provenance, preprocessing or annotation procedure is touched:
 those require the challenge documentation and are marked as such in the
 draft.
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

OUT = f"{ns.DRIVE}/paper2/cohort_description.csv"
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)}


def main():
    files = sorted(glob.glob(f"{ns.GT_DIR}/*.nii.gz"))
    print("=" * 78)
    print(" COHORT DESCRIPTION · full cohort, measured")
    print("=" * 78)
    print(f"  reference files: {len(files)}")

    rows = []
    for i, f in enumerate(files, 1):
        case = os.path.basename(f).split(".nii")[0]
        img = nib.load(f)
        z = img.header.get_zooms()[:3]
        a = np.asanyarray(img.dataobj)
        present = set(np.unique(a)) - {0}
        rec = {
            "case": case,
            "patient": case.rsplit("-", 1)[0],
            "shape": "x".join(str(v) for v in a.shape),
            "nx": a.shape[0], "ny": a.shape[1], "nz": a.shape[2],
            "sx": float(z[0]), "sy": float(z[1]), "sz": float(z[2]),
            "voxel_mm3": float(np.prod(z)),
            # Three distinct properties, easily conflated:
            #   isotropic      the three spacings are equal to each other
            #   is_1mm         each spacing is 1 mm  -> volume is also 1
            #   volume near 1  the PRODUCT is near 1, which (0.5, 1.0, 2.0)
            #                  satisfies while being strongly anisotropic
            # Only the second is what "1-mm isotropic" means.
            "isotropic": bool(np.allclose(z, z[0], atol=1e-3)),
            "is_1mm": bool(np.allclose(z, 1.0, atol=0.01)),
            "labels": ",".join(str(int(v)) for v in sorted(present)),
        }
        for reg, lab in REGIONS.items():
            m = np.isin(a, lab)
            rec[f"{reg}_voxels"] = int(m.sum())
            rec[f"{reg}_mm3"] = round(float(m.sum()) * rec["voxel_mm3"], 2)
            rec[f"has_{reg}"] = int(m.any())
        rows.append(rec)
        if i % 100 == 0:
            print(f"    {i}/{len(files)}")

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    # ---- patient structure ------------------------------------------
    print("\n" + "=" * 78)
    print(" PATIENT STRUCTURE")
    print("=" * 78)
    c = Counter(d.patient)
    print(f"  cases                        : {len(d)}")
    print(f"  unique patients              : {len(c)}")
    print(f"  patients with >1 timepoint   : {sum(1 for v in c.values() if v > 1)}")
    print(f"  maximum timepoints, one patient: {max(c.values())}")
    print(f"  timepoint distribution       : "
          f"{dict(sorted(Counter(c.values()).items()))}")

    # ---- geometry ----------------------------------------------------
    print("\n" + "=" * 78)
    print(" IMAGE GEOMETRY")
    print("=" * 78)
    print(f"  distinct array shapes        : {d['shape'].nunique()}")
    print("\n  most common:")
    for sh, n in d['shape'].value_counts().head(8).items():
        print(f"    {sh:<20} {n:>5}  ({n/len(d):.1%})")

    v = d.voxel_mm3
    print(f"""
  voxel volume, mm3
    range      {v.min():.4f} – {v.max():.4f}
    median     {v.median():.4f}
    quartiles  {v.quantile(.25):.4f} · {v.quantile(.75):.4f}
    ratio max/min   {v.max()/v.min():.1f}x

  isotropic (spacings equal)   : {int(d.isotropic.sum())}/{len(d)} ({d.isotropic.mean():.1%})
  1-mm isotropic (each ~1.0)   : {int(d.is_1mm.sum())}/{len(d)} ({d.is_1mm.mean():.1%})
  NOT 1-mm isotropic           : {int((~d.is_1mm).sum())}/{len(d)} ({(~d.is_1mm).mean():.1%})
  voxel VOLUME within 0.01 of 1: {int((v.sub(1).abs() <= 0.01).sum())}/{len(d)}"""
          f" ({(v.sub(1).abs() <= 0.01).mean():.1%})")

    both = int((d.is_1mm & (v.sub(1).abs() <= 0.01)).sum())
    vol_only = int(((~d.is_1mm) & (v.sub(1).abs() <= 0.01)).sum())
    print(f"""
  Volume near 1 but NOT 1-mm isotropic: {vol_only}
    These are the cases the two definitions disagree on -- anisotropic
    spacings whose product happens to be near 1, for example (0.5, 1.0, 2.0).
    Quoting a volume figure as an isotropy figure would misdescribe them, and
    the paper must use whichever quantity the surrounding sentence means.

  A single median with this spread is not a description. Report the range and
  the fraction away from it, since that spread is what the spacing-dependence
  finding rests on.""")

    print(f"\n  per-axis spacing, mm:")
    for ax in ("sx", "sy", "sz"):
        col = d[ax]
        print(f"    {ax}  range {col.min():.3f}-{col.max():.3f}  "
              f"median {col.median():.3f}")
    print("""
  Per-axis figures are given because through-plane spacing is often the
  outlier in clinical MRI, and a volume summary hides it entirely.""")

    # ---- region prevalence ------------------------------------------
    print("\n" + "=" * 78)
    print(" REGION PREVALENCE AND SIZE")
    print("=" * 78)
    print(f"\n  {'region':<8} {'cases present':>14} {'%':>7} "
          f"{'median mm3 where present':>26}")
    for reg in REGIONS:
        pres = d[d[f"has_{reg}"] == 1]
        print(f"  {reg:<8} {len(pres):>14} {len(pres)/len(d):>6.1%} "
              f"{pres[f'{reg}_mm3'].median():>26.1f}")

    print(f"""
  The resection cavity is the region that carries a caveat throughout, and
  this is why: it is present in a minority of cases, so any suppression rule
  acting on it acts mostly on cases where no cavity exists.""")

    # ---- label values -----------------------------------------------
    print("\n" + "=" * 78)
    print(" LABEL VALUES OBSERVED")
    print("=" * 78)
    print("\n  " + d.labels.value_counts().head(8).to_string()
          .replace("\n", "\n  "))
    print("""
  These are the numeric values present. The mapping from each value to a
  tissue name is NOT established by the data and must come from the challenge
  specification -- it is marked [VERIFY] in the draft and should not be
  written from memory.""")

    print(f"\n  written: {OUT}")
    print("""
  What this does not and cannot establish: contributing institutions,
  acquisition protocols, the preprocessing applied before release, how the
  annotations were produced or reviewed, the official partition, or patient
  demographics. Each requires the challenge documentation. The preprocessing
  question matters most, because it decides whether the geometric
  heterogeneity above is native variation or an artefact of the release.""")


if __name__ == "__main__":
    main()
