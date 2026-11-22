# Paper 2 — Beyond Error Headroom

Analysis code, frozen protocol and per-case result tables for:

> **Beyond Error Headroom: The Limits of Post-Processing in Lesion-Level Brain
> Metastasis Segmentation**
> A. El-Gaghous, M. S. Saraya, S. M. Ayyad
> Faculty of Engineering, Mansoura University · manuscript under review

This folder is self-contained. The inference container at the top level of the
repository belongs to Paper 1; see [How the two papers relate](#how-the-two-papers-relate).

---

## What is here

```
paper2/
  protocol/
    PHASE_B_PROTOCOL.md        frozen protocol, v1.0, 13 Aug 2026 (binding record;
                               written during the study in Arabic and English)
    PHASE_B_PROTOCOL_EN.md     full English translation, 22 Nov 2026
  src/                         Phase B (suppression) and Phase C (missed lesions)
  exp1/                        the prospective size-aware training contrast
    nnUNetTrainerSizeAware.py    the trainer
    exp1.py                      weight maps, calibration, self-test, the gate
    exp1_lesions.py              the per-lesion endpoint table for both arms
    config/                      splits, plans and dataset.json of Dataset137
  figures/                     every figure script, plus the TikZ sources
  data/                        every table the paper's numbers come from (7.8 MB)
  requirements.txt
```

`data/` mirrors the folder layout the scripts were run in, so every script reads
its inputs and writes its outputs without edits once `BRATS_ROOT` points at it.

---

## Quick check: recompute the paper's numbers from the tables

The table-level analyses need only CPU and the files in `data/`.

```bash
cd paper2
pip install -r requirements.txt
export BRATS_ROOT="$PWD/data"
export PYTHONPATH="$PWD/src"

python src/b6_stats.py            # Phase B paired tests       -> wave2/b6_*.csv
python src/b5b_stratify.py        # by leakage stratum          -> wave2/b5b_*.csv
python src/t34_analysis.py        # detection and size bins     -> wave2/t3_*, t4_*
python src/ceiling_clean.py       # oracle ceiling, clean       -> wave2/ceiling_*_clean.csv
python src/eligibility_table.py   # evaluable population        -> paper2/eligibility_table.csv
python src/eligibility_effect.py  # eligibility discrepancy     -> paper2/eligibility_effect.csv
python src/fn_recovery_ceiling.py # missed-lesion headroom      (printed)
python src/c07_fn_signal.py       # candidate availability      (printed)
python src/c2_analysis.py         # contrast model              (printed)
python figures/make_figures.py    # data figures                -> paper2/figures/
```

These were re-run from this folder on 22 November 2026 before publication.
`b6_primary_clean.csv`, `b5b_T1_clean.csv`, `t3_bin_results.csv`,
`t4_detection_results.csv`, `eligibility_table.csv` and `eligibility_effect.csv`
were reproduced exactly (maximum absolute difference 0, or 3e-16 from floating
point), including the bootstrap intervals, which are seeded.

---

## Where each result comes from

| Paper | Table in `data/` | Script |
|---|---|---|
| Cohort and geometry (Tables 1, 2, 4) | `paper2/cohort_description.csv` | `src/cohort_description.py` |
| Leakage strata, `clean` n = 155 | `phase_b_outputs/wave0/leak_class_567.csv` | `src/leak_class.py` |
| All six rules, 567 cases × 4 regions | `phase_b_outputs/wave1/exp3_suppressed_v3.csv` | `src/b4_suppressed_arm.py`, `src/merge_shards.py` |
| Oracle ceiling and headroom (Table 7, Fig. 9) | `phase_b_outputs/wave2/ceiling_by_region_clean.csv` | `src/ceiling_clean.py` |
| Effect of each rule, clean stratum (Tables 8–11) | `phase_b_outputs/wave2/b6_primary_clean.csv` | `src/b6_stats.py` |
| By leakage stratum; QC sensitivity | `b6_by_stratum.csv`, `b6_qc_sensitivity.csv`, `b5b_*.csv` | `src/b6_stats.py`, `src/b5b_stratify.py` |
| Detection counts, all 15 evaluator columns | `phase_b_outputs/wave2/t4_counts.w*of9.csv` | `src/t4_run.py` |
| Detection F1, precision and recall (Table 12, Fig. 10) | `phase_b_outputs/wave2/t4_detection_results.csv` | `src/t34_analysis.py` |
| Reference-negative cases emptied by suppression (Table 13) | `phase_b_outputs/wave2/t4_counts.w*of9.csv` | `src/t34_audit.py`, `src/t34_followup.py` |
| Endpoint eligibility (Tables 14–17, Fig. 11) | `paper2/eligibility_table.csv`, `paper2/eligibility_effect.csv` | `src/eligibility_table.py`, `src/eligibility_effect.py` |
| Effect by lesion volume (Table 18, Fig. 12) | `phase_b_outputs/wave2/t3_bin_results.csv` (from `t3_lesions.w*of9.csv`) | `src/t34_analysis.py` |
| False-positive component selection (Table 19, Fig. 14) | `phase_b_outputs/wave2/fp_components.csv` | `src/fp_component_size.py` |
| Missed lesions: candidate availability and cost (Tables 20–21, Fig. 15) | `phase_c_outputs/c0/c0_lesions.csv`, `c0_cases.csv` | `src/c0_fn_recoverability.py`, `src/c07_fn_signal.py` |
| Reconstruction gate, 620 of 620 cells | `phase_c_outputs/c0/c0_gate_validation.csv` | `src/c0_fn_recoverability.py` |
| Missed-lesion headroom (10–11 × MDE) | printed | `src/fn_recovery_ceiling.py` |
| Probability in and around six large missed lesions (Table 22) | from the probability maps; see the note below | — |
| Volume percentiles (Table 23) | `phase_c_outputs/c0/c1_within_case.csv` | `src/c1_within_case.py` |
| T1c contrast model (Table 24) | `phase_c_outputs/c2/c2_contrast.csv` | `src/c2_measurement.py`, `src/c2_analysis.py` |
| Size-aware gate (Table 25, Fig. 16) | `paper2/exp1_gate_result.json` | `exp1/exp1.py gate` |
| Size-aware per-lesion endpoints, both arms | `paper2/exp1_{baseline,sizeaware}_lesions.csv` (+ `_cases.csv`) | `exp1/exp1_lesions.py` |
| False positives by region (Table 26) | `paper2/exp1_*_lesions_cases.csv`, columns `*_FP` | — |
| Region-overlap Dice (Table 27) | `paper2/exp1_*_nnunet_summary.json` | nnU-Net's own validation |
| Weight coverage by region | `paper2/exp1_coverage_region.csv`, `weight_coverage*.json` | `exp1/exp1.py coverage` |
| Suppression-mask reconstruction (Fig. 13 c) | `paper2/suppression_gate.csv` | `src/suppression_gate.py` |

Figure scripts: `figures/make_figures.py` draws Figs. 9–12, 14 and 15 from the
tables above. `figures/make_figure1_final.py` … `make_figure4_final.py` and
`make_planes_figure.py` draw the image panels (Figs. 4, 5, 7, 8, 13); they need
the BraTS images. The schematics carry no measured data:
`make_metric_schematic.py` (Fig. 1), `make_graphical_abstract.py` (Fig. 2),
`make_nnunet_schematic.py` (Fig. 6), and the TikZ sources
`Figure1_framework_block.tex` (Fig. 3) and `fig_gate_body.tex` (Fig. 16).

---

## Re-running from the images

The steps above start from the stored tables. Starting from images needs three
things this repository does not redistribute:

1. **The BraTS-METS 2026 training release** (Synapse `syn74274097`), under the
   challenge data agreement.
2. **The probability maps** of the frozen ensemble (folds 1, 2, 4, test-time
   augmentation off), which come from the Paper 1 checkpoints. Checkpoints are
   not redistributed (about 820 MB each).
3. **The official evaluator**, `dani-capellan/BraTS-2025-Metrics`, module
   `metrics_MET`. `src/native_space.py` clones it on first use; pin it to the
   commit the study used before scoring anything:

   ```bash
   git -C <clone> checkout c7db9c9
   ```

`src/native_space.py` is the single place where a prediction is converted to
reference space and scored; every Phase B and Phase C script goes through it.
Phase B was run as `b4_suppressed_arm.py --worker W --n-workers 3` and merged
with `merge_shards.py`; T4 as `t4_run.py` on nine workers. The full execution
set-up is in §11 of the protocol.

For the size-aware contrast, `exp1/README.md` gives the run order: calibration,
weight maps in nnU-Net's preprocessed space (`weights-pp`), the self-test, the
reproduction control, training of fold 1 in both arms with
`nnUNetTrainerSizeAware`, the lesion tables, then the gate. Fold definitions are
`exp1/config/splits_final.json`.

---

## Notes a reader may need

- **The frozen protocol is partly in Arabic.** It was written that way during
  the study. `PHASE_B_PROTOCOL_EN.md` is a complete translation made for this
  release; the original stays the binding record.
- **Hard-coded paths.** The scripts were run on Google Colab against Google
  Drive. The project root was made overridable through `BRATS_ROOT` for this
  release (default unchanged); no other line of the analysis code was edited.
  `src/ceiling_clean.py` was added for this release (see its header): before it,
  the four clean-stratum headroom values were documented constants in
  `figures/make_figures.py`. The script recomputes them from the T4 counts and
  matches them exactly (WT 0.0395, TC 0.0280, ET 0.0247, RC 0.2968).
- **Criterion C of the gate.** `exp1_gate_result.json` holds the false-positive
  ratio as entered at compute time, 1.4485, from the rounded per-case means 1.36
  and 1.97. Recomputed from the per-case tables it is 510 / 353 = 1.4448. Both
  are above the pre-specified bound of 1.25; the paper reports 1.44.
- **`size@20` in the T4 tables.** The T4 run scored seven rules, the six of the
  Dice analysis plus `size@20`, which was assigned to the detection analysis
  before the run (protocol §12.3). Its Dice values are kept for audit and are not
  part of the Dice analysis.
- **Shards.** `t4_counts.w*of9.csv` and `t3_lesions.w*of9.csv` are the nine
  worker outputs of the T4 run, kept as written; `t34_analysis.py` reads them
  directly.
- **Table 22** lists six values read directly from the probability maps (maximum
  probability inside each lesion and in a five-voxel shell around it). They need
  the probability maps, which are not redistributed, and no table of them is
  stored here; the six rows in the paper are the complete record.
- `brats_qc_taxonomy.csv` is a categorisation of the organisers' public
  training-set QC log, used for the pre-registered QC sensitivity (protocol §6.1).

---

## How the two papers relate

| | Paper 1 (top level of this repository) | Paper 2 (this folder) |
|---|---|---|
| Question | Ensemble composition and runtime under the challenge's time limit | How much post-processing can recover under lesion-wise scoring |
| Model | nnU-Net v2 ResEncL, folds 1, 2, 4, test-time augmentation on | The same three folds, probabilities saved with test-time augmentation off |
| Cases | 567-case cohort and the challenge test set | The same 567-case cohort; the `clean` stratum (155) is the primary population |
| Training | none | one prospective contrast: fold 1 retrained with and without the size-aware loss |
| Tables | `results/` | `paper2/data/` |

The 567-case cohort and the leakage classes are shared. Paper 1's
`results/per_case_lesionwise.csv` is the regression target that Paper 2's
evaluation module was checked against (`src/b2b_regression_test.py`).

---

## Citation

```bibtex
@article{elgaghous2026headroom,
  title   = {Beyond Error Headroom: The Limits of Post-Processing in
             Lesion-Level Brain Metastasis Segmentation},
  author  = {El-Gaghous, Abdelrahman and Saraya, Mohamed Sabry
             and Ayyad, Sarah M.},
  note    = {Manuscript under review},
  year    = {2026}
}
```

## License

Apache 2.0, as for the rest of the repository. See `../LICENSE`.
