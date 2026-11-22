# -*- coding: utf-8 -*-
"""
==============================================================================
 GATE 2 · FP COMPONENT-SIZE  —  verification gate, then extraction
==============================================================================
 Option B: reproduce the evaluator's TP/FP assignment WITHOUT touching its
 code, and refuse to use the result until the reproduction is proven to match
 the evaluator's own counts.

 RUN THIS FIRST WITH --gate. It scores 2-3 cases, rebuilds the assignment, and
 compares component-for-component against Num_TP / Num_FP from results_df.

     match     -> the reproduction is verified; run again with --extract
     no match  -> STOP. The analysis is not available without modifying the
                  evaluator, and that is written up as a limitation rather
                  than worked around with an approximation.

 WHY A GATE AND NOT JUST A RUN
 -----------------------------
 The evaluator decides which predicted components are false positives inside
 get_LesionWiseResults. `fp` (line 652) is a list of component ids and it is
 never returned -- only len(fp) escapes, as Num_FP. So the label this analysis
 needs does not exist outside that function. Reproducing it is the only route
 that leaves the evaluator untouched, and a reproduction that has not been
 checked against the thing it reproduces is an assumption wearing a number.

 This is the same shape as the check that licensed T3: `_len` was only used to
 classify reference lesions after it was shown to reproduce Num_TP / Num_FN.

 TWO SIZE COLUMNS, NOT ONE
 -------------------------
 voxel_count is what size@N actually thresholds on. volume_mm3 is what that
 means physically. Voxel volume ranges 0.083-2.296 mm3 across this cohort and
 55% of cases are not 1.0, so size@50 removes anything under 7 mm3 in one case
 and under 115 mm3 in another (D19). Recording voxels alone would reproduce
 that error exactly.
==============================================================================
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

W2 = f"{ns.DRIVE}/phase_b_outputs/wave2"
OUT = f"{W2}/fp_components.csv"
REGIONS = ["WT", "TC", "ET", "RC"]
CH = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}

RULES = {
    "size@20": ("size", 20), "size@50": ("size", 50),
    "entropy@0.45": ("entropy", 0.45), "entropy@0.50": ("entropy", 0.50),
    "entropy@0.56": ("entropy", 0.56), "confidence@0.45": ("confidence", 0.45),
}
SUPP = {"size": ns.size_suppressor, "entropy": ns.entropy_suppressor,
        "confidence": ns.confidence_suppressor}

# The evaluator's own parameters, read rather than assumed (metrics_MET.py:679-681)
DIL_FACTOR = 1
VOL_THRESH = 2


def clean_cases():
    lk = pd.read_csv(f"{ns.DRIVE}/phase_b_outputs/wave0/leak_class_567.csv")
    return sorted(lk.loc[lk.leak_class == "clean", "case"])


def components(case):
    """
    TWO partitions of the same predicted voxels, because the pipeline uses two.

      suppression : ndimage.label(m)             -- 6-connectivity, no merging.
                    This is what build_nested_label iterates over, so it is the
                    partition a rule actually makes decisions on.
      evaluation  : 26-connectivity + dilation   -- what the evaluator merges
                    into "lesions" before counting Num_FP.

    They are not the same. An evaluator component can be several suppression
    components, so "which component did the rule remove" is ambiguous unless
    both are recorded and the mapping kept. Collapsing them would silently
    attribute a rule's decision to the wrong unit.
    """
    mean = ns.load_mean_prob(case)
    gt_img = nib.load(f"{ns.GT_DIR}/{case}.nii.gz")

    # The npz volumes and the GT NIfTI are stored on different axis orders.
    # native_space owns that mapping (NPZ_TO_GT, used by to_native_nifti), so
    # invert it here rather than transposing by hand: a second, independently
    # written transpose is a second thing that can silently disagree.
    inv = np.argsort(ns.NPZ_TO_GT)
    gt = np.asanyarray(gt_img.dataobj).transpose(inv)
    if gt.shape != mean[0].shape:
        raise RuntimeError(f"{case}: GT {gt.shape} != prob {mean[0].shape} after "
                           f"inverting NPZ_TO_GT={ns.NPZ_TO_GT}")

    # Voxel volume is order-independent (it is a product), but the zooms come
    # from the GT header and so are in GT order; only their product is used.
    sx, sy, sz = (float(v) for v in gt_img.header.get_zooms()[:3])
    vox_mm3 = sx * sy * sz

    ent = {}
    out = {}
    for reg in REGIONS:
        ch = CH[reg]
        pred = mean[ch] >= ns.THR[reg]

        # --- evaluation partition -------------------------------------
        struct = ndimage.generate_binary_structure(3, 2)
        dil = ndimage.binary_dilation(pred, structure=struct, iterations=DIL_FACTOR)
        lab_e, _ = ndimage.label(dil, structure=np.ones((3, 3, 3)))
        lab_e = lab_e * pred
        eids, ecounts = np.unique(lab_e[lab_e > 0], return_counts=True)
        keep = ecounts > VOL_THRESH              # evaluator drops <= 2 voxels
        eids, ecounts = eids[keep], ecounts[keep]

        # --- suppression partition, exactly as build_nested_label does --
        lab_s, n_s = ndimage.label(pred)
        sids, scounts = (np.unique(lab_s[lab_s > 0], return_counts=True)
                         if n_s else (np.array([], int), np.array([], int)))

        gt_reg = np.isin(gt, {"WT": (1, 2, 3), "TC": (1, 3),
                              "ET": (3,), "RC": (4,)}[reg])
        out[reg] = {"lab_e": lab_e, "eids": eids, "ecounts": ecounts,
                    "lab_s": lab_s, "sids": sids, "scounts": scounts,
                    "gt": gt_reg, "vox_mm3": vox_mm3,
                    "prob": mean[ch]}
        if n_s:
            ent[reg] = ns.binary_entropy(mean[ch])
    for reg in ent:
        out[reg]["ent"] = ent[reg]
    return out


def assign(comp):
    """
    A predicted component is a true positive if it overlaps any reference
    lesion of that region, a false positive otherwise -- the evaluator's own
    criterion, applied to the same dilation-merged components. Whether this
    reproduces its counts is exactly what the gate tests; nothing downstream
    uses it until it has.
    """
    lab, ids, gt = comp["lab_e"], comp["eids"], comp["gt"]
    return {int(i): bool((gt & (lab == i)).any()) for i in ids}


def gate(cases):
    """
    The gate only tests what the cases put in front of it. A case with no false
    positives contributes `0 == 0`, which is not evidence that the reproduction
    works -- it is evidence that there was nothing to reproduce. Earlier checks
    in this project were weakened exactly this way twice, so the cases here
    should be chosen for FP count, not taken off the top of an alphabetical
    list.
    """
    print("=" * 76)
    print(" GATE · does the reproduced assignment match the evaluator's counts?")
    print("=" * 76)
    os.environ.setdefault("BRATS_FAST_METRIC", "1")
    ns.ensure_evaluator()

    rows = []
    for case in cases:
        print(f"\n  {case}")
        _, agg, _ = ns.official_score(case, suppressor=None, tag="fpgate",
                                      guards=False, full=True)
        comp = components(case)
        for _, r in agg.iterrows():
            reg = str(r["Labels"]).strip()
            if reg not in comp:
                continue
            tp_fp = assign(comp[reg])
            mine_tp = sum(tp_fp.values())
            mine_fp = len(tp_fp) - mine_tp
            ev_tp, ev_fp = int(r["Num_TP"]), int(r["Num_FP"])
            ok = (mine_fp == ev_fp)
            rows.append({"case": case, "region": reg,
                         "mine_TP": mine_tp, "eval_TP": ev_tp,
                         "mine_FP": mine_fp, "eval_FP": ev_fp,
                         "FP_match": ok})
            print(f"    {reg}:  FP mine={mine_fp} eval={ev_fp}"
                  f"   TP mine={mine_tp} eval={ev_tp}   "
                  f"{'ok' if ok else 'MISMATCH'}")

    d = pd.DataFrame(rows)
    print("\n" + "=" * 76)
    n_ok = int(d.FP_match.sum())
    informative = d[(d.eval_FP > 0) | (d.mine_FP > 0)]
    n_inf_ok = int(informative.FP_match.sum())
    print(f"  FP counts matching       : {n_ok}/{len(d)}")
    print(f"  cells with any FP at all : {len(informative)}/{len(d)}"
          f"   ← only these test anything")
    print(f"  of those, matching       : {n_inf_ok}/{len(informative)}")
    print(f"  total FPs exercised      : {int(d.eval_FP.sum())}")

    if len(informative) < 3 or d.eval_FP.sum() < 10:
        print(f"""
  GATE INCONCLUSIVE. {len(informative)} informative cell(s), {int(d.eval_FP.sum())}
  false positives in total. Most comparisons were 0 == 0, which tests nothing.
  Re-run with cases chosen for FP count:

      --cases BraTS-MET-00271-000 BraTS-MET-00770-000 BraTS-MET-00685-000

  Do not treat this run as a pass.""")
        return False

    if n_ok == len(d):
        print("""
  GATE PASSED. The reproduction matches the evaluator's own FP counts on every
  region tested, so classifying components by overlap is verified rather than
  assumed. Run again with --extract.

  Note on TP: the evaluator's Num_TP counts matched REFERENCE lesions, not
  predicted components, so mine_TP and eval_TP need not agree and a difference
  there is not a failure. The FP column is the one this analysis depends on.""")
        return True

    print(f"""
  GATE FAILED on {len(d) - n_ok} of {len(d)} region(s).

  Do NOT proceed and do NOT loosen the criterion to make it pass. The correct
  outcome is to record that FP behaviour by predicted-component size is not
  obtainable from the evaluator as used here, for the same structural reason
  that per-bin F1 is not: a false positive has no reference-lesion row, and the
  component ids the evaluator works with are never returned.

  Obtaining it would require modifying the evaluator to return `fp`, which
  every result in this paper was produced without.""")
    print("\n" + d[~d.FP_match].to_string(index=False))
    return False


def extract(cases):
    """
    One row per SUPPRESSION component -- the unit a rule actually decides on --
    carrying the evaluator's FP verdict for the evaluation component it sits
    inside. Both ids are kept so the mapping can be inspected rather than
    trusted.
    """
    print("=" * 76)
    print(" EXTRACT · one row per predicted component")
    print("=" * 76)

    rows = []
    for i, case in enumerate(cases, 1):
        comp = components(case)
        for reg in REGIONS:
            c = comp[reg]
            if not len(c["sids"]):
                continue
            fp_of_eval = assign(c)               # keyed on evaluation ids
            lab_s, lab_e = c["lab_s"], c["lab_e"]
            prob = c["prob"]
            ent = c.get("ent")

            for cid, cnt in zip(c["sids"], c["scounts"]):
                mask = lab_s == cid
                # which evaluation component contains this one
                overlap = lab_e[mask]
                overlap = overlap[overlap > 0]
                eid = int(np.bincount(overlap).argmax()) if overlap.size else 0

                rec = {"case": case, "region": reg,
                       "supp_component_id": int(cid),
                       "eval_component_id": eid,
                       "voxel_count": int(cnt),
                       "volume_mm3": round(float(cnt) * c["vox_mm3"], 4),
                       "voxel_mm3": round(c["vox_mm3"], 6),
                       # unknown when the component falls below the evaluator's
                       # own 2-voxel floor and so has no evaluation counterpart
                       "is_fp": (int(not fp_of_eval[eid])
                                 if eid in fp_of_eval else -1)}

                for rule, (kind, val) in RULES.items():
                    f = SUPP[kind](val)
                    drop = f(reg, mask, prob, ent)
                    rec[rule] = "removed" if drop else "retained"
                rows.append(rec)
        del comp
        if i % 10 == 0 or i == len(cases):
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  {i}/{len(cases)} cases · {len(rows)} components")

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    print(f"\n  written: {OUT}")
    print(f"  {len(d)} suppression components")
    print(f"  false positives      : {int((d.is_fp == 1).sum())}")
    print(f"  true positives       : {int((d.is_fp == 0).sum())}")
    print(f"  no evaluation parent : {int((d.is_fp == -1).sum())}"
          f"   (below the evaluator's 2-voxel floor)")
    print(f"  voxel volume range   : {d.voxel_mm3.min():.3f}"
          f"-{d.voxel_mm3.max():.3f} mm3")

    fp = d[d.is_fp == 1]
    if len(fp):
        print("\n  false-positive component size, by rule decision:")
        print(f"    {'rule':<16} {'removed n':>10} {'median vox':>11}"
              f" {'median mm3':>11} | {'retained n':>11} {'median vox':>11}"
              f" {'median mm3':>11}")
        for rule in RULES:
            r = fp[fp[rule] == "removed"]
            k = fp[fp[rule] == "retained"]
            print(f"    {rule:<16} {len(r):>10} "
                  f"{(r.voxel_count.median() if len(r) else float('nan')):>11.1f} "
                  f"{(r.volume_mm3.median() if len(r) else float('nan')):>11.1f} |"
                  f" {len(k):>11} "
                  f"{(k.voxel_count.median() if len(k) else float('nan')):>11.1f} "
                  f"{(k.volume_mm3.median() if len(k) else float('nan')):>11.1f}")
        print("""
  Descriptive only. No endpoint was pre-registered for this analysis, so it is
  exploratory and no hypothesis is derived from it after the fact.

  Read the two median columns together: a rule that removes components of a
  consistent VOXEL size but a varying PHYSICAL size is spacing-dependent, which
  is the D19 mechanism seen directly rather than inferred from an association.""")
    return d


def analyse():
    """
    Exploratory analysis of the extracted table. No endpoint was pre-registered
    for this, so nothing here is a confirmatory test and no hypothesis is
    derived from it after the fact.

    The hierarchy is the one agreed before extraction:
      primary   how are false-positive component sizes distributed, and how
                does that distribution differ across rules
      derived   among baseline FPs, are removed components smaller than
                retained ones
    """
    if not os.path.exists(OUT):
        sys.exit(f"{OUT} not found — run --extract first")
    d = pd.read_csv(OUT)
    fp = d[d.is_fp == 1]

    print("=" * 78)
    print(" FP COMPONENT-SIZE — exploratory")
    print("=" * 78)
    print(f"  components   : {len(d)}  ({int((d.is_fp==1).sum())} FP · "
          f"{int((d.is_fp==0).sum())} TP · {int((d.is_fp==-1).sum())} unmapped)")
    print(f"  cases        : {d.case.nunique()}")
    print(f"  voxel volume : {d.voxel_mm3.min():.3f}–{d.voxel_mm3.max():.3f} mm3")

    # ---- 1. distribution of FP sizes, both units -----------------------
    print("\n--- 1 · false-positive size distribution ---")
    for unit in ("voxel_count", "volume_mm3"):
        q = fp[unit].quantile([.25, .5, .75, .9]).round(2)
        print(f"  {unit:<12} p25 {q.iloc[0]:>9} · median {q.iloc[1]:>9} · "
              f"p75 {q.iloc[2]:>9} · p90 {q.iloc[3]:>9}")

    print("\n  by region:")
    g = fp.groupby("region").agg(n=("voxel_count", "size"),
                                 med_vox=("voxel_count", "median"),
                                 med_mm3=("volume_mm3", "median")).round(1)
    print("  " + g.to_string().replace("\n", "\n  "))

    # ---- 2. removed vs retained ----------------------------------------
    print("\n--- 2 · among baseline FPs: removed vs retained ---")
    rows = []
    for rule in RULES:
        r = fp[fp[rule] == "removed"]
        k = fp[fp[rule] == "retained"]
        rec = {"rule": rule, "n_removed": len(r), "n_retained": len(k),
               "frac_removed": round(len(r) / max(len(fp), 1), 3)}
        for unit, tag in (("voxel_count", "vox"), ("volume_mm3", "mm3")):
            rec[f"rm_med_{tag}"] = round(r[unit].median(), 1) if len(r) else np.nan
            rec[f"kp_med_{tag}"] = round(k[unit].median(), 1) if len(k) else np.nan
        if len(r) > 1 and len(k) > 1:
            from scipy import stats
            rec["p_vox"] = round(stats.mannwhitneyu(
                r.voxel_count, k.voxel_count, alternative="two-sided")[1], 6)
        rows.append(rec)
    t = pd.DataFrame(rows)
    print("\n" + t.to_string(index=False))
    print("""
  The Mann-Whitney p is descriptive: these are components, not cases, and
  components within a case are not independent. It indexes separation between
  the two size distributions, nothing more, and is not corrected.""")

    # ---- 3. the D19 test ------------------------------------------------
    print("\n--- 3 · is the selection spacing-dependent? (D19, directly) ---")
    print("""
  A size rule thresholds voxels, so across cases with different voxel volumes
  it should remove a CONSTANT voxel size and a VARYING physical size. An
  entropy rule has no physical unit and should show no such pattern. D19
  inferred this from an association with the Dice delta; here it is visible in
  the removed components themselves.
""")
    fp2 = fp.copy()
    fp2["spacing"] = pd.qcut(fp2.voxel_mm3, 3, duplicates="drop")
    print(f"    {'rule':<16} {'spacing tertile':<22} {'n':>5} "
          f"{'med vox':>8} {'med mm3':>9}")
    for rule in RULES:
        r = fp2[fp2[rule] == "removed"]
        if not len(r):
            print(f"    {rule:<16} {'(removes nothing)':<22}")
            continue
        for tert, sub in r.groupby("spacing", observed=True):
            print(f"    {rule:<16} {str(tert):<22} {len(sub):>5} "
                  f"{sub.voxel_count.median():>8.1f} {sub.volume_mm3.median():>9.1f}")
    print("""
  Read down each rule: a flat 'med vox' with a rising 'med mm3' is a rule
  selecting on voxel count regardless of what that means physically. A flat
  'med mm3' would mean the opposite.""")

    # ---- 4. overlap between rules ---------------------------------------
    print("\n--- 4 · do the rules remove the same components? ---")
    names = list(RULES)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        ra = set(fp.index[fp[a] == "removed"])
        for b in names:
            rb = set(fp.index[fp[b] == "removed"])
            M.loc[a, b] = (len(ra & rb) / len(ra | rb)) if (ra | rb) else np.nan
    print("\n  Jaccard overlap of removed FP sets:")
    print("  " + M.round(2).to_string().replace("\n", "\n  "))
    print("""
  Low overlap between a size rule and an entropy rule at similar removal rates
  means they are selecting different components, not the same ones by another
  route. That is the component-level counterpart of the confidence control.""")

    print("\n" + "=" * 78)
    print("""  EXPLORATORY. Not pre-registered, no confirmatory test, no hypothesis
  derived after the fact. Reported alongside the frozen results, not within
  them.""")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="explicit cases for the gate — choose ones with many "
                         "false positives, or the test proves nothing")
    a = ap.parse_args()

    cases = clean_cases()
    if a.gate:
        if a.cases:
            bad = [c for c in a.cases if c not in cases]
            if bad:
                raise SystemExit(f"not in the clean stratum: {bad}")
            gate(a.cases)
        else:
            print("  [!] no --cases given; using the first few alphabetically.")
            print("      These may have no false positives, in which case the")
            print("      gate cannot conclude anything. Prefer --cases.\n")
            gate(cases[:a.n])
    elif a.analyse:
        analyse()
    elif a.extract:
        if not os.path.exists(f"{W2}/.fp_gate_passed"):
            print("  Run --gate first and confirm it passed.")
            print("  (touch .fp_gate_passed in wave2 once it has.)")
            return
        extract(cases)
    else:
        print(__doc__)
        print("  --gate     verify the reproduction · use --cases with FP-rich cases")
        print("  --extract  full extraction, only after the gate passes")
        print("  --analyse  exploratory analysis of the extracted table")


if __name__ == "__main__":
    main()
