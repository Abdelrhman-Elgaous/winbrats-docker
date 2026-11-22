# -*- coding: utf-8 -*-
"""
==============================================================================
 EVALUATOR FORENSICS  —  two cases, every intermediate object printed
==============================================================================
 Read-only. Seconds. Writes nothing.

 The C0 gate disagreed with the official false-negative counts on 85 of 620
 comparisons, in BOTH directions. Two cases carry the two directions:

     00010 WT   mine FN 1  ·  official 3     official HIGHER
     00031 WT   mine FN 16 ·  official 14    official LOWER

 A single mechanism that explains both is an explanation. One that explains
 only one is a coincidence.

 A HYPOTHESIS, NOT A CONCLUSION
 ------------------------------
 After the merge step (metrics_MET 212-224), lesions that touch under dilation
 take the id of the merged component, so their original ids vanish while max()
 stays high. The scoring loop then runs over range(1, max+1) — every integer,
 not every present id — and an absent id yields an empty mask, no intersecting
 components, and `fn.append(gtcomp)`.

 That is a reading of the code. This script tests it rather than assuming it,
 by counting the gaps and checking whether they account for the discrepancy.

 WHAT THIS IS NOT FOR
 --------------------
 Not for fixing the evaluator. If its behaviour turns out to be surprising,
 the reconstruction must still reproduce it, because that behaviour is what
 produced every Phase B number. Changing it here would mean reproducing
 something else.
==============================================================================
"""

import sys
import glob

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

try:
    import cc3d
    HAVE_CC3D = True
except ImportError:
    HAVE_CC3D = False
    print("  [!] cc3d unavailable — falling back to ndimage 26-connectivity.")
    print("      Results should match, but the evaluator uses cc3d.\n")

ns.ensure_evaluator()

DRIVE = ns.DRIVE
OOF = f"{DRIVE}/archive/inference_v1/oof"
STRUCT = ndimage.generate_binary_structure(3, 2)
GT_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)}
CH = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}

CASES = [("BraTS-MET-00010-000", "WT", 3, 1, 10, 12),
         ("BraTS-MET-00031-000", "WT", 14, 16, 7, 5)]


def label26(mask):
    if HAVE_CC3D:
        return cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    return lab.astype(np.int32)


def combined_by_dilation(mask):
    """
    metrics_MET 271-279 and 212-224, followed exactly:
      components of the raw mask, components of the dilated mask, then each
      raw lesion relabelled with the id of the dilated component containing
      it. Merged lesions therefore share an id and the others are dropped.
    """
    cc = label26(mask)
    dil = ndimage.binary_dilation(mask, structure=STRUCT, iterations=1)
    dcc = label26(dil)

    comb = np.zeros_like(dcc)
    for comp in range(1, int(dcc.max()) + 1):
        tmp = (dcc == comp).astype(cc.dtype) * cc
        np.place(tmp, tmp > 0, comp)
        comb += tmp
    return cc, dcc, comb


def prediction_components(prob_ch, thr):
    """The prediction side, same construction (metrics_MET 307-323)."""
    mask = prob_ch >= thr
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32)
    cc = label26(mask)
    dil = ndimage.binary_dilation(mask, structure=STRUCT, iterations=1)
    dcc = label26(dil)
    comb = np.zeros_like(dcc)
    for comp in range(1, int(dcc.max()) + 1):
        tmp = (dcc == comp).astype(cc.dtype) * cc
        np.place(tmp, tmp > 0, comp)
        comb += tmp
    ids, counts = np.unique(comb[comb > 0], return_counts=True)
    drop = ids[counts <= 2]                    # the two-voxel floor, predictions
    if drop.size:
        comb[np.isin(comb, drop)] = 0
    return comb


def forensics(case, region, official_fn, mine_fn, official_tp, mine_tp):
    print("\n" + "=" * 78)
    print(f" {case} · {region}")
    print(f" official FN {official_fn} / TP {official_tp}   ·   "
          f"mine FN {mine_fn} / TP {mine_tp}   ·   "
          f"difference {official_fn - mine_fn:+d}")
    print("=" * 78)

    p = glob.glob(f"{OOF}/fold_*/{case}.npz")
    if not p:
        print("  no oof probability map")
        return None
    with np.load(p[0]) as z:
        prob = z["probabilities"]

    img = nib.load(f"{ns.GT_DIR}/{case}.nii.gz")
    gt = np.asanyarray(img.dataobj).transpose(np.argsort(ns.NPZ_TO_GT))
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    vox = sx * sy * sz

    mask = np.isin(gt, GT_LABELS[region])
    cc, dcc, comb = combined_by_dilation(mask)
    pred = prediction_components(prob[CH[region]], ns.THR[region])

    raw_ids = sorted(set(np.unique(cc)) - {0})
    comb_ids = sorted(set(np.unique(comb)) - {0})
    loop_ids = list(range(1, int(comb.max()) + 1))
    gaps = sorted(set(loop_ids) - set(comb_ids))

    print(f"\n  raw components            : {len(raw_ids)}   max id {max(raw_ids or [0])}")
    print(f"  components after dilation : {int(dcc.max())}")
    print(f"  ids present after merging : {len(comb_ids)}   max id {int(comb.max())}")
    print(f"  ids the loop visits       : {len(loop_ids)}   (range(1, max+1))")
    print(f"  gaps                      : {len(gaps)}   {gaps[:12]}")
    print(f"  merged away               : {len(raw_ids) - len(comb_ids)} lesion(s)")

    # walk the loop exactly as the evaluator does
    print(f"\n  {'id':>4} {'present':>8} {'voxels':>8} {'vol mm3':>9} "
          f"{'hits':>6}  verdict")
    tp_ids, fn_ids, gap_fn = [], [], []
    for gid in loop_ids:
        sel = comb == gid
        n = int(sel.sum())
        if n == 0:
            gap_fn.append(gid)
            fn_ids.append(gid)
            print(f"  {gid:>4} {'no':>8} {0:>8} {0.0:>9.1f} {0:>6}  "
                  f"FN  ← id absent after merging")
            continue
        sel_dil = ndimage.binary_dilation(sel, structure=STRUCT, iterations=1)
        hit = np.unique(pred[sel_dil])
        hit = hit[hit > 0]
        (tp_ids if hit.size else fn_ids).append(gid)
        print(f"  {gid:>4} {'yes':>8} {n:>8} {n * vox:>9.1f} {hit.size:>6}  "
              f"{'TP' if hit.size else 'FN'}")

    print(f"\n  reproduced: TP {len(tp_ids)} · FN {len(fn_ids)}")
    print(f"    of which FN from absent ids : {len(gap_fn)}")
    print(f"    of which FN from real misses: {len(fn_ids) - len(gap_fn)}")
    print(f"  official  : TP {official_tp} · FN {official_fn}")

    ok_fn = len(fn_ids) == official_fn
    ok_tp = len(tp_ids) == official_tp
    print(f"\n  FN {'MATCHES' if ok_fn else 'differs by ' + str(len(fn_ids) - official_fn)}"
          f"   ·   TP {'MATCHES' if ok_tp else 'differs by ' + str(len(tp_ids) - official_tp)}")

    return {"case": case, "region": region,
            "raw": len(raw_ids), "merged_ids": len(comb_ids),
            "loop_ids": len(loop_ids), "gaps": len(gaps),
            "repro_TP": len(tp_ids), "repro_FN": len(fn_ids),
            "gap_FN": len(gap_fn), "real_FN": len(fn_ids) - len(gap_fn),
            "official_TP": official_tp, "official_FN": official_fn,
            "fn_ok": ok_fn, "tp_ok": ok_tp}


def main():
    print("=" * 78)
    print(" EVALUATOR FORENSICS — testing the gap hypothesis on both directions")
    print("=" * 78)

    rows = [r for r in (forensics(*c) for c in CASES) if r]
    if not rows:
        return
    d = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print("\n" + d.to_string(index=False))

    both = bool(d.fn_ok.all())
    print("\n" + "=" * 78)
    if both:
        print(""" BOTH DIRECTIONS REPRODUCED.

 The same construction accounts for the case where the official count was
 higher and the case where it was lower, which is what distinguishes an
 explanation from a coincidence. Two components:

   merging   lesions touching after dilation share one id, so fewer lesions
             are scored than the raw component count suggests
   gaps      the ids vacated by merging are still visited by the scoring loop
             and each yields a false negative

 The two pull in opposite directions and their balance depends on how the
 lesions are arranged, which is why neither alone predicted the sign.

 NEXT: re-run the 155-case gate with this construction. Do not carry the
 mechanism into C0 until the full validation agrees.""")
    else:
        bad = d[~d.fn_ok]
        print(f""" NOT REPRODUCED on {len(bad)} of {len(d)} case(s).

 The gap mechanism is not the whole explanation. Do not adjust C0 and do not
 adopt this construction. What is still unaccounted for:""")
        print("\n" + bad.to_string(index=False))
        print("""
 Look next at: the order of the merge loop and whether ids can collide when
 two dilated components overlap the same raw lesion; whether the prediction
 side is built from the same construction used in Phase B; and whether the
 region mask itself is composed identically.""")

    if len(d) and d.gap_FN.sum():
        print(f"""
 ─────────────────────────────────────────────────────────────────────────
 A CONSEQUENCE WORTH FLAGGING

 {int(d.gap_FN.sum())} of {int(d.repro_FN.sum())} reproduced false negatives on these two cases
 come from ids vacated by merging rather than from lesions the model failed
 to find.

 If that proportion holds across the cohort, the false-negative ceiling
 computed for Phase C (10.6-11.1x MDE) counts both kinds together. The
 direction of the Phase C decision may well survive, but the figure behind it
 would need restating, and the two kinds should be counted separately before
 that decision is described as settled.

 Reproducing the evaluator faithfully and interpreting what it counts are
 different tasks. This script does the first.""")


if __name__ == "__main__":
    main()
