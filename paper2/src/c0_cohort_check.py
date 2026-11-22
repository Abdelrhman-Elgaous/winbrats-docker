# -*- coding: utf-8 -*-
"""
==============================================================================
 C0 COHORT CHECK  —  which cases have the Phase B prediction source
==============================================================================
 Read-only. Minutes. Writes nothing.

 C0 asks whether the false negatives Phase B measured have sub-threshold
 evidence. That question is only about those false negatives if the
 predictions are produced the same way Phase B produced them: the mean of
 folds 1, 2 and 4, composed into a nested label. Out-of-fold single-fold maps
 answer a different question, and a case without the three-fold source cannot
 be in the primary analysis at all.

 So before fixing the cohort, this establishes which cases actually have that
 source on disk. It does not assume that because some .npz files exist for a
 case, the right ones do.

 The strata, from the frozen leakage classification:

     155  clean          LOCKED — no design decision may touch these
     165  patient_only   candidate for the primary cohort
     247  seen           candidate for the primary cohort
     729  outside the 567

 The 412 non-clean Phase B cases are the primary candidate. The 729 join only
 if they have the same source, which is what this checks rather than presumes.

 Nothing here decides anything. It reports what is available so the cohort can
 be fixed on evidence.
==============================================================================
"""

import os
import sys
import glob
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
LEAK = f"{DRIVE}/phase_b_outputs/wave0/leak_class_567.csv"
FOLDS = getattr(ns, "FOLDS", (1, 2, 4))
N_PROBE = 8


def strata():
    if not os.path.exists(LEAK):
        sys.exit(f"missing {LEAK}")
    d = pd.read_csv(LEAK)
    out = {k: sorted(d.loc[d.leak_class == k, "case"]) for k in d.leak_class.unique()}
    gt = sorted({os.path.basename(f).split(".nii")[0]
                 for f in glob.glob(f"{ns.GT_DIR}/*.nii.gz")})
    out["outside_567"] = sorted(set(gt) - set(d.case))
    return out


def sources_for(case):
    """Every directory holding a probability map for this case."""
    hits = glob.glob(f"{DRIVE}/**/{case}.npz", recursive=True)
    return {os.path.dirname(h).replace(DRIVE + "/", ""): h for h in hits}


def has_phase_b_source(case):
    """
    Can load_mean_prob reconstruct the Phase B ensemble for this case? Asked by
    calling it, because that is the function Phase B used and the only thing
    that settles it.
    """
    try:
        m = ns.load_mean_prob(case)
        ok = (m.ndim == 4 and m.shape[0] == 4
              and 0.0 <= float(m.min()) and float(m.max()) <= 1.0001)
        del m
        return ok, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("=" * 78)
    print(" C0 COHORT CHECK — availability of the Phase B prediction source")
    print("=" * 78)
    print(f"  ensemble folds (native_space): {FOLDS}")

    st = strata()
    print("\n  strata:")
    for k in ("clean", "patient_only", "seen", "outside_567"):
        print(f"    {k:<14} {len(st.get(k, [])):>5}")
    total = sum(len(v) for v in st.values())
    print(f"    {'total':<14} {total:>5}")

    # ---- where do the maps live, per stratum -------------------------
    print("\n" + "=" * 78)
    print(" 1 · WHICH DIRECTORIES HOLD MAPS, BY STRATUM")
    print("=" * 78)
    print("  (sampled — a directory listed for a stratum has maps for at least")
    print("   one of its cases, not necessarily all)\n")
    for k in ("clean", "patient_only", "seen", "outside_567"):
        cases = st.get(k, [])
        if not cases:
            continue
        seen_dirs = Counter()
        for c in cases[:: max(1, len(cases) // 25)][:25]:
            for d in sources_for(c):
                seen_dirs[d] += 1
        print(f"  {k}  (sampled {min(25, len(cases))} of {len(cases)})")
        for d, n in seen_dirs.most_common(6):
            print(f"      {n:>3}  {d}")
        if not seen_dirs:
            print("       none found")
        print()

    # ---- can the Phase B ensemble actually be built ------------------
    print("=" * 78)
    print(" 2 · CAN load_mean_prob BUILD THE PHASE B ENSEMBLE?")
    print("=" * 78)
    print("  Probed by calling the function Phase B used, not by inspecting paths.\n")

    rows = []
    for k in ("clean", "patient_only", "seen", "outside_567"):
        cases = st.get(k, [])
        if not cases:
            continue
        probe = cases[:: max(1, len(cases) // N_PROBE)][:N_PROBE]
        ok_n, errs = 0, []
        t0 = time.time()
        for c in probe:
            ok, err = has_phase_b_source(c)
            ok_n += ok
            if not ok:
                errs.append((c, err))
        rows.append({"stratum": k, "n": len(cases), "probed": len(probe),
                     "ok": ok_n, "secs": round(time.time() - t0, 1)})
        print(f"  {k:<14} {ok_n}/{len(probe)} probed cases loaded"
              f"   ({time.time()-t0:.0f}s)")
        for c, e in errs[:3]:
            print(f"      {c}: {e[:90]}")

    d = pd.DataFrame(rows)

    # ---- what this means for the cohort ------------------------------
    print("\n" + "=" * 78)
    print(" 3 · WHAT THIS MEANS FOR THE C0 COHORT")
    print("=" * 78)

    def rate(k):
        r = d[d.stratum == k]
        return (r.ok.iloc[0] / r.probed.iloc[0]) if len(r) and r.probed.iloc[0] else 0.0

    non_clean = len(st.get("patient_only", [])) + len(st.get("seen", []))
    outside_ok = rate("outside_567")

    print(f"""
  Primary candidate — the non-clean Phase B strata:

      patient_only {len(st.get('patient_only', [])):>4}
      seen         {len(st.get('seen', [])):>4}
      ────────────────
      total        {non_clean:>4}

  These have the Phase B ensemble by construction: they are the cases Phase B
  scored. They are NOT unbiased — the deployed folds saw them, wholly or
  through another timepoint of the same patient. They are the non-test-stratum
  cohort, and should be described that way and not as independent.

  The 155 clean cases stay locked. They may be used to validate a
  reconstruction against official counts, which is a technical check, and for
  a final comparison once everything else is fixed. They may not be used to
  choose a threshold, a feature or a model.""")

    if outside_ok >= 0.95:
        print(f"""
  The {len(st.get('outside_567', []))} cases outside the 567 loaded in {outside_ok:.0%} of probes, so the
  same source appears to be available for them too. Before adding them,
  confirm on the full set rather than a probe: a partial failure discovered
  mid-run costs more than the check.""")
    elif outside_ok > 0:
        print(f"""
  The cases outside the 567 loaded in only {outside_ok:.0%} of probes. They cannot
  be assumed to have the Phase B source, which is what this check was for.
  Keep the primary cohort at {non_clean} unless the failures are diagnosed.""")
    else:
        print(f"""
  No case outside the 567 could be loaded through load_mean_prob. The primary
  cohort is therefore the {non_clean} non-clean Phase B cases, and the wider
  cohort is available only for the out-of-fold secondary analysis, which asks
  a different question and must be reported as such.""")

    print("""
  A secondary analysis on the out-of-fold maps remains worth doing: it asks
  whether sub-threshold evidence persists when the model genuinely has not
  seen the case. Agreement between the two would be stronger than either
  alone; disagreement would itself say something about exposure. But it is a
  different question and cannot stand in for the primary one, because the
  false negatives Phase B measured were produced by the ensemble.""")


if __name__ == "__main__":
    main()
