results/

Per-case scores behind every reported comparison. These are the reproducible layer: model checkpoints and imaging data are not redistributed, so the tables in the paper are recomputed from these files rather than from the models.

file	rows	reproduces
per_case_fold_selection.csv	72,576	Table 1, Figure 2
per_case_lesionwise.csv	4,536	Table 2, Section 4.5
leak_coverage_candidates.csv	10	Table 3
per_case_fold_selection.csv

567 cases x 4 regions x 16 configurations x 2 label variants.

column	meaning
case	BraTS-METS case identifier
region	WT, TC, ET, RC
combo	1fold_N, 3fold_ABC, or 5fold
variant	nested (as deployed and published) or direct
dice	voxel-overlap Dice against ground truth

nested applies the BraTS label hierarchy with the resection cavity written last; direct thresholds each region independently. The paper reports nested.

per_case_lesionwise.csv

567 cases x 4 regions x 2 configurations, scored with the official evaluator (dani-capellan/BraTS-2025-Metrics, commit c7db9c9, unmodified).

column	meaning
case	BraTS-METS case identifier
config	3fold_124 (deployed) or 5fold
region	WT, TC, ET, RC
lesionwise_dice	official lesion-wise Dice
legacy_dice	voxel-overlap Dice (Legacy_Dice column)
num_tp, num_fp, num_fn	lesion counts from the evaluator
seconds	evaluation wall time for that case
leak_coverage_candidates.csv

Training exposure for each of the ten candidate three-fold subsets, computed from splits_final.json. Cases are classified as fully unseen, same-patient exposure (another timepoint of the same patient was in training), or seen.
