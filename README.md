# WIN_BRATS — BraTS-METS 2026

Inference container and analysis code for:

> **Runtime-Aware Ensemble Optimization for Brain Metastasis Segmentation under
> Strict Deployment Constraints**
> A. El-Gaghous, M. S. F. Saraya, S. M. Ayyad
> Faculty of Engineering, Mansoura University

Team `WIN_BRATS` · MICCAI 2026 Challenge BraTS-METS, Task 1.

---

## What this repository is

The challenge requires each submission to run as a Docker container on a single
GPU within an eight-hour budget for the 303-case test set. A conventional
five-fold nnU-Net ensemble exceeds that budget. This repository contains the
deployed inference container, together with the code and per-case scores behind
the reported comparisons.

The system is a fixed nnU-Net v2 ResEncL model. No architecture was modified and
no retraining was performed; the contribution is in ensemble composition and
deployment.

---

## Layout

```
Dockerfile              container definition (linux/amd64)
requirements.txt        pinned dependencies
infer_logic.py          inference: 3-fold ensemble, thresholds, nested labels
run_bratsmets.py        container entry point
.github/workflows/      build and push to the challenge registry

analysis/               code behind the reported numbers
  fold_selection.py       exhaustive search over all ten 3-fold subsets
  statistics.py           paired Wilcoxon, TOST, Bonferroni correction
  leak_coverage.py        per-candidate training exposure (Table 3)

results/
  per_case_lesionwise.csv        567 cases x 4 regions x 2 configurations
  per_case_fold_selection.csv    voxel Dice for all ten candidates
  leak_coverage_candidates.csv   exposure per candidate subset
```

---

## Reproducing the reported results

Every table in the paper derives from a file in `results/`. Model checkpoints
and BraTS imaging data are not redistributed (see below), so these files are the
reproducible layer: the numbers can be recomputed from them without access to
the weights.

| Paper | Source | Command |
|---|---|---|
| Table 1 (candidate search) | `per_case_fold_selection.csv` | `python analysis/fold_selection.py --summarise` |
| Table 2 (accuracy, equivalence) | `per_case_lesionwise.csv` | `python analysis/statistics.py` |
| Table 3 (training exposure) | `leak_coverage_candidates.csv` | `python analysis/leak_coverage.py` |
| Table 4 (leaderboard) | official evaluation server | — |
| Table 5 (runtime) | measured on an L4, see below | — |

### `per_case_lesionwise.csv`

| column | meaning |
|---|---|
| `case` | BraTS-METS case identifier |
| `config` | `3fold_124` (deployed) or `5fold` |
| `region` | WT, TC, ET, RC |
| `lesionwise_dice` | official lesion-wise Dice |
| `legacy_dice` | voxel-overlap Dice (`Legacy_Dice` column of the evaluator) |
| `num_tp`, `num_fp`, `num_fn` | lesion counts from the official evaluator |
| `seconds` | evaluation wall time for that case |

The paired comparison between the deployed 3-fold ensemble and the 5-fold
ensemble reported in the paper is computed directly from this file.

Scores were produced with the BraTS-2025 metrics implementation
(`dani-capellan/BraTS-2025-Metrics`, commit `c7db9c9`, unmodified).

---

## Running inference

### Build

```bash
docker build --platform linux/amd64 -t winbrats:latest .
```

### Weights

Not included: five checkpoints at roughly 820 MB each exceed repository limits.
They are provided to the challenge organisers through the MLCube submission. The
container expects them at `/opt/weights` in nnU-Net's standard layout:

```
weights/Dataset137_BraTS2026MET/
  nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/
    plans.json
    dataset.json
    fold_{0,1,2,3,4}/checkpoint_best.pth
```

Only folds 1, 2 and 4 are loaded at inference; the others are present for
completeness.

### Run

```bash
docker run --rm --gpus all \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  -v /path/to/weights:/opt/weights:ro \
  winbrats:latest
```

Input: one directory per case containing four NIfTI files with `t1n`, `t1c`,
`t2w`, `t2f` in their names. Output: one flat `<case>.nii.gz` per case, in the
original image geometry.

---

## Configuration

These values are fixed and were determined before the candidate search. They are
not tuned per case.

| | |
|---|---|
| Model | nnU-Net v2, ResEncL (`nnUNetResEncUNetLPlans`), 3d_fullres |
| Training | 1000 epochs, region-based sigmoid, five-fold cross-validation |
| Deployed folds | 1, 2, 4 |
| Checkpoint | `checkpoint_best.pth` |
| Test-time augmentation | enabled (mirroring) |
| Channel order | T1N-0000, T1C-0001, T2W-0002, T2F-0003 |
| Probability channels | WT, TC, ET, RC |
| Thresholds | WT 0.40, TC 0.35, ET 0.45, RC 0.60 |
| Label write order | WT-2, TC-1, ET-3, RC-4 (RC last) |
| Output transpose | `(2,1,0)`, saved with the reference affine and header |

The write order and the transpose both matter. Regions are nested, so later
writes overwrite earlier ones; RC is written last so that it takes precedence
deterministically. The transpose maps from nnU-Net's internal axis order back to
the input image space, and the saved volume carries the reference NIfTI's affine
and header. A shape check aborts the run if the two disagree, rather than
writing a misaligned volume.

---

## Runtime

Measured on an NVIDIA L4 (a proxy for the challenge A10G), first case discarded
as warm-up, for the 303-case test set:

| configuration | s/case | 303 cases |
|---|---|---|
| 5-fold, command line | 251.0 | 21.1 h |
| 3-fold, command line | 150.2 | 12.6 h |
| 3-fold, persistent | 76.0 | 6.4 h |

Two changes account for the reduction, and each can be isolated by holding the
other fixed. With inference mode fixed, going from five folds to three gives
40.2%. With fold count fixed, replacing repeated command-line invocation by a
persistent in-memory predictor gives a further 49.4%.

---

## Not included

- **Model checkpoints** — size; supplied to the organisers via MLCube.
- **BraTS imaging data and annotations** — governed by the challenge data
  agreement (Synapse `syn74274097`).

---

## Citation

```bibtex
@inproceedings{elgaghous2026runtime,
  title     = {Runtime-Aware Ensemble Optimization for Brain Metastasis
               Segmentation under Strict Deployment Constraints},
  author    = {El-Gaghous, Abdelrahman and Saraya, Mohamed Sabry Fouad
               and Ayyad, Sarah M.},
  booktitle = {MICCAI 2026 Challenge: Brain Tumor Segmentation --- Metastases},
  year      = {2026}
}
```

## License

Apache 2.0. See `LICENSE`.

## Acknowledgments

Data obtained through the BraTS Challenge project, Synapse ID `syn74274097`. We
thank the BraTS-METS organising committee and the contributing institutions.
