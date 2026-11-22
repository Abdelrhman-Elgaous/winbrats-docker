# EXP-1 — run order

Every step refers to a section of `EXP1_PROTOCOL.md`. Do not reorder them: the
reproduction control exists so that EXP-1 measures the loss and not the
environment.

## 0 · Before freezing the protocol   (§7, Appendix A)

    python exp1.py calibrate --labels /path/to/labelsTr --out calibration_record.json

Recomputes the α table on all 1,296 training labels. The protocol's table came
from a 412-case subset; this closes that assumption. Commit the JSON next to
the protocol, then date and sign it.

## 1 · Weight maps   (§6)

Run nnU-Net preprocessing first, then attach the weight channel **in
nnU-Net's own space**:

    python exp1.py weights-pp --preproc $nnUNet_preprocessed/DatasetXXX/nnUNetPlans_3d_fullres

This labels the already-cropped, already-resampled segmentation, computes each
lesion's physical volume from the preprocessed spacing in the companion .pkl,
and writes the weight map as `seg` channel 1. The map is aligned to the label
by construction, so there is no resampling step that can silently misalign.

The command is idempotent and prints the largest weight it applied; that
number should sit well under the cap of 8.

`exp1.py weights --labels ... --out ...` writes full-resolution NIfTI maps
instead. Use it only if you need the maps as standalone images; it then leaves
you to resample them onto the preprocessed grid yourself, which is the step
`weights-pp` exists to avoid.

## 2 · Prove the plumbing works   — do this before spending GPU time

    python exp1.py selftest

Builds a synthetic case with one small and one large lesion and checks the
small lesion's share of the gradient actually moves. A trainer that loads but
never applies the weights would make EXP-1 look like a failed hypothesis when
only the wiring was wrong. Exit code 1 means stop.

## 3 · EXP-0, reproduction control   (§4)

Train fold 1 with the **original** loss in the current environment, predict
out-of-fold, score with the study's evaluator, then:

    python exp1.py gate --mode exp0 --baseline archived_fold1.csv --new reproduced_fold1.csv

Needs `case, region, lesionwise_dice`. Passes when every region has
|Δ| < 0.019 and a bootstrap CI containing zero. Exit code 1 means the
environment differs: stop and investigate. Do not start EXP-1.

## 4 · EXP-1 pilot   (§9–§11)

Train fold 1 with `nnUNetTrainerSizeAware`, everything else identical. Score
both arms on the same validation cases, then:

    python exp1.py gate --mode exp1 \
        --baseline reproduced_fold1_lesions.csv \
        --new sizeaware_fold1_lesions.csv \
        --fp-base 2.10 --fp-new 2.45

Needs `case, gt_lesion, gt_volume_mm3, surfaced, detected` — one row per
reference lesion, the **same** lesions in both arms. The script refuses to run
if the row counts disagree, because matching on case alone cross-joins lesions
within a case and silently inflates every denominator.

`--fp-*` are false positives per case, both measured in this pilot. No figure
is carried in from another cohort.

## 5 · The gate decides

Exit 0 → three folds (1, 2, 4), then rerun the preceding study's analysis on
the new ensemble, then reapply the suppression rules unchanged and two-sided.

Exit 1 → stop. No additional loss variant may be introduced (§11, §15).

---

### What is version-sensitive

`nnUNetTrainerSizeAware.py` has three hooks marked `HOOK 1/2/3`. nnU-Net v2's
trainer API has moved between releases, so check them against your installed
version. `selftest` is what tells you whether you got it right, and it does not
need nnU-Net installed — only torch.
