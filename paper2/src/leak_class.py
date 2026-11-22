# -*- coding: utf-8 -*-
"""
==============================================================================
 LEAK CLASS — three-way leakage stratification
==============================================================================
 RUNTIME: CPU, seconds. Reads splits_final.json only.

 TWO INDEPENDENT LEAKAGE MECHANISMS
 ----------------------------------
 1. ENSEMBLE LEAKAGE (known, reported in Paper 1)
    A 3-fold ensemble prediction for a cross-validation case needs outputs from
    folds that saw that case during training. For 247 of the 567 cases, all
    three deployed folds had seen it; for 320 the held-out fold is present.

 2. PATIENT LEAKAGE (found 2026-08-03)
    The dataset is partly longitudinal: 287 of 810 patients contribute more
    than one timepoint, sharing an identifier stem. splits_final.json is
    grouped PER CASE, not per patient, so every fold has 114-129 patients with
    timepoints on both sides of the split. A model can then recognise a
    patient's anatomy from a different session rather than generalising.

 Newman-Norlund et al. (BraTS-METS 2026) grouped folds by patient specifically
 to avoid this, calling it "a subtle but important correctness requirement for
 this dataset". Since a published team treats it as material, silence about it
 is a gap rather than a non-issue.

 WHY IT MATTERS MORE FOR PHASE B THAN FOR PAPER 1
 ------------------------------------------------
 Paper 1 compares 3-fold against 5-fold; both arms carry the same leakage, so
 the comparison stands and only the absolute values are optimistic. Phase B
 claims that uncertainty carries information about false positives. If the
 model recognises a patient from another timepoint, confidence is high and
 variance is low for reasons unrelated to genuine reliability -- exactly the
 mechanism that motivated the ensemble-leakage stratification.

 THE THREE CLASSES
 -----------------
   clean         held-out fold present AND no other timepoint of this patient
                 seen by any deployed fold          -> THE NUMBER TO PUBLISH
   patient_only  held-out fold present, but another timepoint was in training
   seen          the case itself was seen by all deployed folds

 OUTPUT: leak_class_567.csv · LEAK_CLASS_REPORT.md
==============================================================================
"""

import os, sys, json, glob
from collections import defaultdict
from datetime import datetime

import pandas as pd

DRIVE   = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
DATASET = "Dataset137_BraTS2026MET"
PREPROC = f"{DRIVE}/workspace/preprocessing/nnUNet_preprocessed/{DATASET}"
ENSEMBLE_DIR = f"{DRIVE}/archive/inference_v1/ensemble"
OUT_DIR = f"{DRIVE}/phase_b_outputs/wave0"

FOLDS_DEPLOYED = (1, 2, 4)
MIN_STRATUM_N  = 100          # below this, a stratum cannot carry the headline claim

os.makedirs(OUT_DIR, exist_ok=True)


def patient_of(case):
    """BraTS-MET-01101-002 -> BraTS-MET-01101 (the identifier stem)."""
    return case.rsplit("-", 1)[0]


def main():
    print("=" * 74); print(" LEAK CLASS — three-way stratification"); print("=" * 74)

    sp = f"{PREPROC}/splits_final.json"
    if not os.path.exists(sp):
        alt = glob.glob(f"{DRIVE}/**/splits_final.json", recursive=True)
        if not alt:
            raise SystemExit(f"splits_final.json not found under {DRIVE}")
        sp = alt[0]
    splits = json.load(open(sp))

    # cohort = cases with saved probabilities in every deployed fold
    per_fold = {f: {os.path.basename(p)[:-4]
                    for p in glob.glob(f"{ENSEMBLE_DIR}/fold_{f}/*.npz")}
                for f in FOLDS_DEPLOYED}
    cohort = sorted(set.intersection(*per_fold.values()))
    print(f"  cohort            : {len(cohort)} cases")

    val_fold = {c: i for i, s in enumerate(splits) for c in s["val"]}
    train_of = {i: set(s["train"]) for i, s in enumerate(splits)}

    # which patients does each deployed fold see during training?
    patients_trained = {f: {patient_of(c) for c in train_of[f]} for f in FOLDS_DEPLOYED}

    pats = defaultdict(list)
    for c in cohort:
        pats[patient_of(c)].append(c)
    multi = {p: cs for p, cs in pats.items() if len(cs) > 1}
    print(f"  patients (cohort) : {len(pats)}")
    print(f"  with >1 timepoint : {len(multi)}")

    rows = []
    for c in cohort:
        vf = val_fold.get(c)
        pat = patient_of(c)
        leak_free = vf in FOLDS_DEPLOYED if vf is not None else None

        # did any deployed fold train on ANOTHER timepoint of this patient?
        others = [x for x in pats[pat] if x != c]
        pat_seen = False
        for f in FOLDS_DEPLOYED:
            if any(x in train_of[f] for x in others):
                pat_seen = True
                break

        if leak_free is None:
            klass = "unknown"
        elif not leak_free:
            klass = "seen"
        elif pat_seen:
            klass = "patient_only"
        else:
            klass = "clean"

        rows.append({"case": c, "patient": pat, "val_fold": vf,
                     "leak_free": leak_free, "other_timepoint_trained": pat_seen,
                     "n_timepoints": len(pats[pat]), "leak_class": klass})

    df = pd.DataFrame(rows)
    out = f"{OUT_DIR}/leak_class_567.csv"
    df.to_csv(out, index=False)

    counts = df.leak_class.value_counts().to_dict()
    print("\n  " + "-" * 50)
    for k in ["clean", "patient_only", "seen", "unknown"]:
        n = counts.get(k, 0)
        if n:
            print(f"  {k:>14s} : {n:>4d}  ({100*n/len(df):.1f}%)")
    print("  " + "-" * 50)

    # cross-check against the previously reported 320 / 247
    lf = int(df.leak_free.sum())
    print(f"\n  leak_free total   : {lf}  (Paper 1 states 320)"
          + ("  ✓" if lf == 320 else "  <-- MISMATCH"))

    n_clean = counts.get("clean", 0)
    print("\n" + "=" * 74); print(" VERDICT"); print("=" * 74)

    if n_clean >= MIN_STRATUM_N:
        verdict = "USABLE"
        note = (f"The clean stratum holds {n_clean} cases, enough to carry the "
                f"headline claim with the other strata reported as sensitivity "
                f"analyses.")
    elif n_clean > 0:
        verdict = "TOO SMALL"
        note = (f"Only {n_clean} clean cases. A paired test on this stratum will "
                f"be underpowered, so it cannot carry the headline result alone. "
                f"Report the full cohort as primary and the clean stratum as a "
                f"consistency check, stating the power limitation explicitly. Do "
                f"NOT silently fall back to the full cohort without saying so.")
    else:
        verdict = "NONE"
        note = ("No case is free of both leakage mechanisms. Every Phase B result "
                "is then subject to patient-level leakage and this must be stated "
                "as a limitation in both papers.")

    print(f"\n  clean stratum: {n_clean}  ->  {verdict}")
    print(f"  {note}")

    md = ["# Leakage classification (three-way)", "",
          f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
          f"**Deployed folds:** {FOLDS_DEPLOYED}", "",
          "## Two independent mechanisms", "",
          "| mechanism | description | status |",
          "|---|---|---|",
          "| ensemble leakage | deployed folds saw the case itself | reported in Paper 1 (320/247) |",
          "| **patient leakage** | another timepoint of the same patient was in training | **found 2026-08-03** |",
          "",
          f"`splits_final.json` is grouped per case, not per patient. "
          f"{len(multi)} patients in this cohort contribute more than one timepoint.",
          "",
          "## Distribution", "",
          "| class | n | % | meaning |", "|---|---|---|---|",
          f"| clean | {counts.get('clean',0)} | {100*counts.get('clean',0)/len(df):.1f}% | held-out fold present, no other timepoint trained |",
          f"| patient_only | {counts.get('patient_only',0)} | {100*counts.get('patient_only',0)/len(df):.1f}% | held-out fold present, another timepoint trained |",
          f"| seen | {counts.get('seen',0)} | {100*counts.get('seen',0)/len(df):.1f}% | the case itself was seen |",
          "",
          f"## Verdict: {verdict}", "", note, "",
          "## Consequences", "",
          "- **Paper 1:** the 3-fold vs 5-fold comparison is unaffected, since both "
          "arms carry identical leakage. The absolute values are more optimistic than "
          "the manuscript currently states, which attributes optimism to ensemble "
          "leakage alone. A Limitations sentence should cover patient leakage too.",
          "- **Phase B:** every reported number is stratified by `leak_class`, and "
          "the reporting rule follows the verdict above.",
          "- Reference: Newman-Norlund et al. group folds by patient for this exact "
          "reason, describing it as a correctness requirement for this dataset.", ""]

    open(f"{OUT_DIR}/LEAK_CLASS_REPORT.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\n  csv : {out}")
    print(f"  md  : {OUT_DIR}/LEAK_CLASS_REPORT.md")


if __name__ == "__main__":
    main()
