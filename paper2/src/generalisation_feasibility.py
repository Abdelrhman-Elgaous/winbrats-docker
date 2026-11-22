# -*- coding: utf-8 -*-
"""
==============================================================================
 GENERALISATION FEASIBILITY AUDIT  —  before any replication is designed
==============================================================================
 Read-only. Writes one manifest. No experiment, no model, no measurement.

 Every finding in Phases B and C rests on one task and one segmentation
 system. The defensible claim is therefore about the failure behaviour of this
 system on BraTS-METS, not about lesion-level post-processing in general, and
 that ceiling on the claim is what a second cohort would raise.

 But a second cohort only raises it if the comparison is legitimate. If the
 endpoint, the lesion semantics or the region composition differ, a replication
 becomes an experiment on other data rather than evidence that the phenomenon
 generalises — and it would cost weeks to produce a number that does not
 support the claim it was run for.

 So this checks feasibility first, against eight conditions fixed here:

   1  the lesion-wise endpoint is the same or comparable
   2  probability outputs suitable for candidate analysis exist or can be made
   3  the candidate-recoverability analysis transfers unchanged
   4  voxel spacing varies enough to test the spacing dependence
   5  region and lesion definitions do not change the question
   6  exposure can be characterised, so the comparison is not confounded
   7  a genuinely independent cohort can be identified
   8  the computational cost is realistic

 A NO on 1, 3 or 5 is disqualifying: the analysis would answer a different
 question. A NO on 2, 6, 7 or 8 is a cost, not a barrier.

 Nothing here is run on the new data. It inspects what exists and what the
 challenge specification says.
==============================================================================
"""

import os
import sys
import json
import glob
from collections import Counter

import numpy as np
import pandas as pd
import nibabel as nib

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
OUT = f"{DRIVE}/generalisation"
EVAL_REPO = "/content/brats_repos_inspect/capellan_2025"

# What the current work depends on, so the audit knows what to look for
CURRENT = {
    "task": "BraTS-METS 2026",
    "regions": ["WT", "TC", "ET", "RC"],
    "gt_labels": {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)},
    "lesion_volume_thresh": 2,
    "dil_factor": 1,
    "connectivity": 26,
    "spacing_range_mm3": (0.083, 2.296),
    "thresholds": {"WT": 0.40, "TC": 0.35, "ET": 0.45, "RC": 0.60},
}

os.makedirs(OUT, exist_ok=True)
findings = []


def record(n, question, verdict, detail, disqualifying=False):
    findings.append({"n": n, "question": question, "verdict": verdict,
                     "detail": detail, "disqualifying": disqualifying})
    mark = {"YES": "yes", "NO": "NO", "PARTIAL": "partial",
            "UNKNOWN": "unknown"}[verdict]
    flag = "  [DISQUALIFYING]" if disqualifying and verdict == "NO" else ""
    print(f"\n  {n} · {question}")
    print(f"      {mark}{flag}")
    for line in detail.splitlines():
        print(f"      {line}")


# ---------------------------------------------------------------------------
def check_endpoint():
    """
    1 · Is the lesion-wise endpoint the same? Read from the evaluator, which
    handles several challenges, rather than assumed from the challenge name.
    """
    p = f"{EVAL_REPO}/metrics_MET.py"
    if not os.path.exists(p):
        record(1, "Is the lesion-wise endpoint comparable?", "UNKNOWN",
               f"evaluator not found at {p}", True)
        return
    src = open(p, encoding="utf-8").read()

    # the challenge-specific branches the evaluator carries
    challenges = sorted(set(
        c.strip("\"'") for c in
        __import__("re").findall(r"challenge_name\s*==\s*(['\"][^'\"]+['\"])", src)))
    tissue = __import__("re").findall(r"tissue_list\s*=\s*(\[[^\]]+\])", src)

    detail = f"challenges handled: {challenges or 'none found by pattern'}\n"
    if tissue:
        detail += f"tissue lists: {tissue[:4]}\n"
    detail += ("The same evaluator scoring both tasks would make the endpoint\n"
               "identical rather than merely comparable, which is the\n"
               "strongest form of this condition.")
    v = "YES" if len(challenges) > 1 else "PARTIAL"
    record(1, "Is the lesion-wise endpoint comparable?", v, detail, True)


def check_probabilities():
    """2 · Do probability outputs exist, or would inference be needed?"""
    roots = Counter()
    for pat in (f"{DRIVE}/**/*GLI*/**/*.npz", f"{DRIVE}/**/*gli*/**/*.npz",
                f"{DRIVE}/**/*glioma*/**/*.npz"):
        for f in glob.glob(pat, recursive=True):
            roots[os.path.dirname(f)] += 1
    if roots:
        detail = "probability maps already on disk:\n"
        for d, n in roots.most_common(4):
            detail += f"  {n:>5}  {d.replace(DRIVE, '…')}\n"
        v = "YES"
    else:
        detail = ("none on disk. Inference would be required: a trained model\n"
                  "for the new task, then out-of-fold or held-out prediction.\n"
                  "This is the main cost driver and is a cost, not a barrier —\n"
                  "unless no trained model for the task exists at all.")
        v = "NO"
    record(2, "Are probability outputs available?", v, detail)


def check_transfer():
    """
    3 · Does the analysis code transfer? Checked by what it depends on, not
    by whether it would run: it would run and produce wrong numbers if the
    region composition differed.
    """
    deps = {
        "region composition from nested labels": "GT_LABELS per region",
        "four-channel probability output": "channel index per region",
        "operating thresholds per region": "fixed in the earlier phase",
        "evaluator component construction": "dilation merge, 2-voxel floor",
        "leakage stratification": "case and patient level exposure",
    }
    detail = "the analysis is parameterised by:\n"
    for k, v in deps.items():
        detail += f"  · {k}  ({v})\n"
    detail += ("""
Each is a constant in the code, not a hard-coded assumption, so the
scripts transfer if the new task supplies each one. What does NOT
transfer is any conclusion that depends on the RC region existing:
a task without a resection cavity cannot reproduce the one region
where suppression helped.""")
    record(3, "Does the analysis transfer unchanged?", "PARTIAL", detail, True)


def check_spacing():
    """4 · Is there enough spacing variation to test D19?"""
    imgs = []
    for pat in (f"{DRIVE}/**/*GLI*/**/*-t1c.nii.gz",
                f"{DRIVE}/**/*gli*/**/*_t1ce.nii.gz"):
        imgs += glob.glob(pat, recursive=True)[:40]
    if not imgs:
        record(4, "Does voxel spacing vary enough to test D19?", "UNKNOWN",
               """no images found to measure. Note that BraTS glioma releases
are conventionally resampled to 1 mm isotropic, which would
give NO spacing variation and make D19 untestable there.
That is worth confirming before counting D19 as replicable.""")
        return
    vols = []
    for f in imgs[:40]:
        z = nib.load(f).header.get_zooms()[:3]
        vols.append(float(np.prod(z)))
    v = np.array(vols)
    detail = (f"{len(v)} images: voxel volume {v.min():.3f}–{v.max():.3f} mm3\n"
              f"non-isotropic-1mm: {int((np.abs(v - 1) > 0.01).sum())}/{len(v)}\n"
              f"current task range: {CURRENT['spacing_range_mm3']}")
    ok = (v.max() / max(v.min(), 1e-6)) > 3
    record(4, "Does voxel spacing vary enough to test D19?",
           "YES" if ok else "NO",
           detail + ("\nenough spread to test the spacing dependence"
                     if ok else
                     "\ntoo uniform: D19 could not be tested on this data"))


def check_regions():
    """5 · Do region and lesion definitions preserve the question?"""
    detail = """The current work has four regions, one of which (RC) is
reference-negative in 150 of 155 clean cases and is where the
only positive suppression effect appeared.

A glioma task typically has WT, TC and ET but no resection
cavity in pre-treatment data. Consequences:

  transferable  headroom asymmetry, candidate availability,
                volume separation, contrast after volume
  NOT           anything about RC, including the one region
                where uncertainty suppression gained ground
                and the endpoint-eligibility finding (D20),
                which surfaced because RC is reference-negative
                in almost every case

So a glioma replication would test the negative findings and the
characterisation, but could not reproduce D20 or the RC result.
That is a real limit on what the replication would buy."""
    record(5, "Do region definitions preserve the question?", "PARTIAL",
           detail, True)


def check_exposure():
    """6 · Can exposure be characterised, so the comparison is not confounded?"""
    detail = """The current stratification (clean 155, patient_only 165,
seen 247) required knowing which folds saw which cases and
which patients contributed several timepoints.

For a replication this needs either:
  · a model trained with a known split, so exposure is
    derivable, or
  · a genuinely held-out cohort the model never saw

Without one, the replication inherits the same limitation the
current work has — findings on partially exposed cases — and
would not strengthen the claim on that axis."""
    record(6, "Can exposure be characterised?", "UNKNOWN", detail)


def check_independent():
    """7 · Is a genuinely independent cohort identifiable?"""
    detail = """Note what this must NOT be: the 155 clean cases of the
current task. They were used as the validation reference for
the false-negative reconstruction, and Phase C's design was
fixed on the 412 before results existed.

Re-running D22-D26 on the 155 now, knowing the results, would
create exactly the problem the whole protocol was built to
avoid. A clean replication needs a cohort that took no part in
any design decision — which means new data, not a held-back
slice of the same data used a second way."""
    record(7, "Is a genuinely independent cohort identifiable?", "UNKNOWN",
           detail)


def check_cost():
    """8 · Is the computational cost realistic?"""
    detail = """Measured costs from the current work, per case:

  official evaluation      minutes to hours  (the reason a proxy
                           was used for selection)
  candidate analysis       ~57 s
  contrast measurement     ~54 s
  false-negative gate      ~35 s

Full runs: C0 took 14.7 h over 412 cases; C2 took 4.9 h over 179.

A minimal replication — headroom, candidate availability,
volume separation, contrast — is roughly 20-30 h of compute
over a few hundred cases, plus inference if probability maps
must be generated. That is realistic. Reproducing the whole of
Phases B and C is not, and is not necessary: the replication
tests the claims that carry the paper, not the debugging."""
    record(8, "Is the computational cost realistic?", "YES", detail)


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(" GENERALISATION FEASIBILITY AUDIT")
    print(" eight conditions, fixed before any of them was checked")
    print("=" * 78)
    print(f"""
  Why this comes before a replication: the current claim ceiling is

    "characterisation of the failure behaviour of this segmentation
     system on {CURRENT['task']}"

  and not a statement about lesion-level post-processing in general. A second
  cohort raises that ceiling only if the comparison is legitimate. If the
  endpoint or the region semantics differ, weeks of compute would produce a
  number that does not support the claim it was run for.""")

    for fn in (check_endpoint, check_probabilities, check_transfer,
               check_spacing, check_regions, check_exposure,
               check_independent, check_cost):
        try:
            fn()
        except Exception as e:
            record(0, fn.__name__, "UNKNOWN", f"{type(e).__name__}: {e}")

    d = pd.DataFrame(findings)
    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print("\n  " + d[["n", "verdict", "question"]].to_string(index=False)
          .replace("\n", "\n  "))

    blocked = d[(d.verdict == "NO") & d.disqualifying]
    unknown = d[d.verdict == "UNKNOWN"]

    print("\n" + "=" * 78)
    print(" WHAT THIS SETTLES")
    print("=" * 78)

    if len(blocked):
        print(f"""
  {len(blocked)} disqualifying condition(s) failed. A replication on this data
  would answer a different question, and the weeks it costs would not raise
  the claim ceiling.

  The alternative is to fix the scope honestly: a rigorous characterisation of
  lesion-level failure and post-processing limits on {CURRENT['task']}, with
  the single-task restriction stated in the limitations rather than papered
  over. That is a defensible paper; it is a narrower one.""")
    elif len(unknown):
        print(f"""
  No disqualifying condition failed, but {len(unknown)} could not be settled
  from what is on disk. Those are the questions to answer before committing
  compute — mostly about what data and trained models are actually available
  for the second task.

  Note especially the two structural limits found above, which hold even if
  everything else is available:

    · a pre-treatment glioma task has no resection cavity, so the one region
      where suppression helped cannot be reproduced
    · D20 (endpoint eligibility) surfaced because RC is reference-negative in
      almost every case, and would not surface without such a region

  So a replication would test the negative findings and the characterisation,
  not the whole paper. Worth knowing before it is described as validating the
  work.""")
    else:
        print("""
  All eight conditions met. A minimal replication is worth designing, and its
  scope should be fixed in advance — which claims it tests, on what cohort,
  with what endpoints — exactly as C0 and C2 were.""")

    json.dump(findings, open(f"{OUT}/feasibility_audit.json", "w"), indent=2)
    print(f"""
  written: {OUT}/feasibility_audit.json

  Nothing was run on any new data. This audit does not authorise a
  replication; it establishes whether one could answer the question, which is
  a different thing from whether it would look impressive.""")


if __name__ == "__main__":
    main()
