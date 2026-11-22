# PHASE B — PROTOCOL v1.0 (FROZEN) — English translation

> **About this file.** This is an English translation of `PHASE_B_PROTOCOL.md`,
> prepared on 22 November 2026 for publication of the repository. The original was
> written during the study in mixed Egyptian Arabic and English. **The original
> file is the frozen record and is binding; where the two differ, the original
> governs.** Nothing has been added, removed or reinterpreted: section numbers,
> section order (including the original's out-of-sequence §13, §12, §11),
> tables, code and dates follow the original. English passages of the original
> are kept verbatim.

---

**Reference document during execution · to be converted into Methods**
**Status: FROZEN** · started 4 Aug 2026
**Freeze date: 13 Aug 2026** — before execution of B-4
**B-4 started: 14 Aug 2026** · 3 workers · separate shards

> ⛔ **The protocol is frozen.** Any change from now on is recorded in §10 with
> its date and reason, and declared in the paper as a protocol deviation.

> The protocol describes **what will be tested**, not what has been shown. There
> are no results here — results go to `DISCOVERIES.md` and then to the Results
> section. The two open items (Guard-3 policy · the `size@100` decision) were
> settled from the 20-case pilot report — §4 and §5.

> **Difference from `PHASE_B_MASTER_PLAN.md`:** the plan says **what, why and
> when**. This document says **what is frozen** — the protocol that execution is
> bound to and that cannot change after results appear. The tables are
> intentionally empty and are filled in with numbers when the experiments finish.

---

## 1. The scientific question

> Does predictive uncertainty provide information about lesion-level false
> positives **beyond confidence and beyond lesion morphology**, and does that
> translate into improved performance under the official BraTS lesion-wise
> metric?

**Independent variable:** the suppression signal (morphology / confidence / entropy)
**Dependent variable:** official lesion-wise Dice · detection F1
**Held fixed:** the model · the weights · the probabilities · the thresholds

---

## 2. Hypotheses

| # | Hypothesis | Tested in |
|---|---|---|
| H1 | Predictive uncertainty is associated with lesion-level false positives | Exp1 |
| H2 | Uncertainty carries information **beyond confidence** | Exp2 |
| H3 | Uncertainty carries information **beyond lesion morphology** | Exp2b |
| H4 | Lesion-level separability **translates** into official-metric improvement | Exp3 |
| H5 | Any observed gain is not attributable to leakage | Exp4 |
| H6 | The effect varies by region and by lesion size | Exp5 |

*(The status of each hypothesis is recorded in Results — not here.)*

**⚠️ H4 is independent of H2/H3.** Lesion-level separability **does not imply**
an improvement in the metric. This separation is cell D of the outcome space.

---

## 3. Data — frozen

| Item | Value |
|---|---|
| Cohort | 567 (hard subset: C1 ∪ C2, defined before fold selection) |
| Probabilities | `archive/inference_v1/ensemble/fold_{1,2,4}` · **tta = OFF** |
| Thresholds | WT 0.40 · TC 0.35 · ET 0.45 · RC 0.60 |
| Label order | WT=2 → TC=1 → ET=3 → **RC=4 last** |
| Geometry | `transpose(2,1,0)` + `ref.affine/header` from the GT |
| Evaluator | Capellán BraTS-2025-Metrics `c7db9c9` · `metrics_MET` |
| Fast metric | `get_touching_labels_fast` — exact equivalence (`delta = 0`) |

**No item above may be changed after the run starts.**

---

## 4. Suppression rules

| Rule | Type | Threshold | The question it answers |
|---|---|---|---|
| `baseline` | — | — | the reference |
| `size@50` | morphology | 50 voxel | H3 — first threshold above the evaluation floor |
| `entropy@0.45` | uncertainty | 0.45 | H4 |
| `entropy@0.50` | uncertainty | 0.50 | H4 |
| `entropy@0.56` | uncertainty | 0.56 | H4 — the Layer 1 point |
| `confidence@0.45` | confidence | 0.45 | H2 — the dominant signal, as a control |

**Removed rules:**

| Rule | Date | Reason |
|---|---|---|
| `size@20` | 4 Aug · **exclusion invalidated 15 Aug** | The original argument (`20 < 27` = the evaluation floor) is **invalid**: the number 27 is an **undocumented analytical assumption**, and the actual floor in BraTS-MET is **2 voxels** — §13. The rule operates ten times above the floor and has been shown experimentally to move the Dice (9/9 cells). It is a **candidate** again; running it requires an explicit protocol decision |
| `size@100` | **13 Aug** | **The 20-case pilot:** effect −0.0925 against −0.0591 for `size@50`. In the 24 cells where they differed, `size@100` was worse in **20**. Worse in every region except RC (identical). The harm is monotone in the threshold — H3 is answered by `size@50` |

**Final count: 6 rules** → 567 × 6 = **3,402 tasks**.

**Management rule:** any rule not tied to a question in the table is removed.

```python
PROTOCOL_RULES = {"baseline", "size@50",
                  "entropy@0.45", "entropy@0.50", "entropy@0.56",
                  "confidence@0.45"}
assert set(RULES) == PROTOCOL_RULES, f"RULES drift vs §4: {set(RULES) ^ PROTOCOL_RULES}"
```

> The assert is mandatory in any script that reads `RULES`. The B-4 run of
> 13 Aug started with `size@20` loaded and `size@100` absent — it was stopped
> before writing, and the assert prevents a repeat.

---

## 5. Guards

| # | Check | Behaviour |
|---|---|---|
| 1 | `pred.shape == gt.shape` | raise |
| 2 | `axcodes(pred) == axcodes(gt)` | raise |
| 3 | `WT == 0` while present in the GT | raise |
| 3b | another region = 0 while WT is intact | warning + `zero_warning` |

**Governing principle:** *a guard is responsible for detecting pipeline
problems, not for detecting hard cases.* Any size threshold inside a guard is an
arbitrary number ("why 500?" has no answer). Lesion size is a scientific
question and belongs in B-6.

### Guard-3 failure diagnosis — design frozen · **executed 13 Aug 2026**

```
A. Suppression consistency
   list of removed components == expected (by ID)

B. Reconstruction consistency
   expected_region_voxels_from_components ==
   actual_region_voxels_after_reconstruction
```

| Class | Meaning | Action |
|---|---|---|
| 1 | pipeline failure (the baseline itself fails) | STOP |
| 2 | expected suppression behaviour (A ✓ and B ✓) | recorded |
| 3 | unexpected behaviour | STOP |

**Oracle:** directly from `ndimage.label` — independent of suppression and
reconstruction (not independent of thresholding, and that is not required).

**Execution result (13 Aug 2026):** cases `BraTS-MET-00546-000` and
`BraTS-MET-00174-000` → **Class 2** (A ✓ and B ✓). Suppression removed every
WT/TC/ET component — expected behaviour, not a fault in the guard or in the
reconstruction. Both cases were recovered with `guards=False` at their true
values (`size@50` and `size@100`: WT = 0.0000 for both; baseline 0.4430 and
0.3774), and the pilot became **20 complete cases**.

> The recovery was necessary: these two cases are the ones where suppression
> removed the most, so dropping them **biases the pilot mean in favour of
> suppression** — the very quantity under test.

**Policy for the 567 — fixed 13 Aug 2026:**

```
guards = False  +  record a row with status
```

A case on which a guard fails **is recorded at its true value** (Dice = 0 if
the region was erased), **and is not skipped**. Consistent with the governing
rule: *record the failure as a row with `status` — do not skip the case.*

**Rationale:** the cases where suppression erases the most are exactly the ones
that raise Guard-3. Skipping them **biases the mean in favour of suppression** —
the quantity under test. The pilot demonstrated this in practice: the two
dropped cases came back with WT = 0.0000 under `size@50` and `size@100`.

The classification (Class 1/2/3) continues to run as a **post-hoc diagnosis**
on the recorded rows, not as a gate that stops the run.

---

## 6. Stratification

| Stratum | n | Role |
|---|---|---|
| `clean` | 155 | **the primary figure** |
| `patient_only` | 165 | sensitivity |
| `seen` | 247 | sensitivity |

**Pre-registered MDE:** at n=155 · SD of differences 0.082 · α=0.05 ·
power=0.80 → **≈ 0.019 Dice**.

> ⚠️ Δ below 0.019 = **below the detection limit**, not **zero**.

---

### 6.1 Pre-registered sensitivity — the QC cases

**Declared 13 Aug 2026, before execution of B-4.**

The organisers published a review log on the training set
(`BraTS_Training_QC_public.xlsx`) documenting ground-truth corrections for
**115 timepoints across 68 patients**, range `01058`–`01353`. The corrections
include missed lesions added · excess labels removed (mostly leptomeningeal
enhancement mislabelled as ET) · RC↔NETC confusion · sub-region completion ·
boundary adjustment · merging/splitting of lesion instances · co-registration
shifts.

**Six of the 155 cases in the `clean` stratum are flagged:**

```
BraTS-MET-01058-001   missed metastasis added
BraTS-MET-01060-001   missed metastasis added
BraTS-MET-01070-000   sub-region completed
BraTS-MET-01094-002   over-label removed (leptomeningeal)
BraTS-MET-01176-003   sub-region completed
BraTS-MET-01253-002   sub-region completed
```

**None of them involves RC↔NETC confusion.**

**Registered analysis:** the primary comparison is repeated excluding the six
cases (n = 155 → 149). The registered MDE moves from 0.019 to **0.0194**
(α = 0.05 · power = 0.80 · SD = 0.082). It is presented as a sensitivity
analysis **beside** the primary result, **not in place of it**.

**Scope:** no change to the cohort definition, the operating thresholds,
`RULES` or the primary endpoint. **The `clean` stratum is not redefined** — the
155 and its MDE were registered before the QC log was released and remain the
primary basis.

**Why it is declared now:** the QC log was released after the cohort was fixed
and before B-4 was executed. Declaring it in advance distinguishes it from a
subset chosen after seeing an unfavourable result — consistent with the
governing rule in §8.

#### Audit note (context — not an analytical commitment)

QC status is **not independent** of the leakage strata. Within the 567:
`clean` 3.9% · `patient_only` 23.0% · `seen` 10.5%.

The reason is structural: **82.9%** of QC cases have more than one timepoint,
against **47.7%** of the remaining cases — and `patient_only` by definition
requires another timepoint of the same patient in training. That is, patient
leakage and correlated label noise **have the same cause**: `splits_final.json`
is split per case, not per patient. And the `clean` stratum excludes both at
once.

**No enrichment claim** against a dataset-level base rate: the QC log covers
`01058`–`01353` only, and there is no evidence that cases outside that range
were reviewed — so their absence from the log does not show that they are
clean.

---

## 7. Statistical analysis

- Paired Wilcoxon (baseline against each rule)
- Bootstrap 95% CI for the effect size (10k resamples)
- Multiple-comparison correction: Holm across rules
- Per-region · per-size-bin · per-leak_class
- **Detection F1** beside the Dice (suppression raises precision and lowers recall)
- **Decomposition of the gap into its components** + the theoretical ceiling
  before interpreting any Δ

---

## 8. Success / failure criteria — registered before the result

| Outcome | Interpretation | Action |
|---|---|---|
| significant Δ on `clean` | uncertainty adds and translates | full Wave 2 |
| Δ ≈ 0 (above the MDE) | separability does not translate — **cell D** | write up immediately |
| Δ < MDE | below the detection limit | declared as such, explicitly |
| effect only in `patient_only` | memorisation | declared as such |
| negative Δ | suppression is harmful at these operating points | a publishable result |

> ⛔ **Forbidden:** searching for a threshold that produces a positive number
> after a negative result appears. That is overfitting to the OOF.

---

## 9. Tables (empty — filled after the run)

### T1 · Main result — the clean stratum (n=155)

| Rule | WT | TC | ET | RC | Mean | Δ vs baseline | p | 95% CI |
|---|---|---|---|---|---|---|---|---|
| baseline | | | | | | — | — | — |
| size@50 | | | | | | | | |
| entropy@0.45 | | | | | | | | |
| entropy@0.50 | | | | | | | | |
| entropy@0.56 | | | | | | | | |
| confidence@0.45 | | | | | | | | |

### T2 · By leak_class

| Rule | clean (155) | patient_only (165) | seen (247) | all (567) |
|---|---|---|---|---|

### T3 · By size bin

| Size bin (GT lesion, **mm³**) | n lesions | Δ Dice | Δ recall |
|---|---|---|---|
| ≤ 2 mm³ — *evaluator-excluded reference lesions; not part of scored T3 analysis* | — | — | — |
| 2–27 | | | |
| 27–100 | | | |
| 100–500 | | | |
| > 500 | | | |

> ⚠️ **The unit of `gt_lesion_vol` is mm³ — confirmed from the code and by
> measurement (17 Aug).** `gt_lesion_vol` is a **physical volume**, not a voxel
> count: the values are fractional (`232.905159` · `23.462190` · `0.184642`),
> and the smallest value in the cohort equals exactly one voxel of its own case
> (`0.1846 ÷ 0.4297×0.4297×1.0 = 1.0000000`). So the count is multiplied by the
> voxel volume.
>
> Accordingly, the constant `lesion_volume_thresh = 2` **is applied as 2 mm³ on
> the reference path** (lines 744 · 748 · 761) — while on the prediction path it
> is applied as **2 voxels** (line 319, a comparison with `counts`). The full
> approved wording is in §13.3.
>
> And the T3 bin edges (27 · 100 · 500) are **mm³**, because they are applied to
> `gt_lesion_vol`.
>
> **Correction of a correction:** the 16 Aug version changed the description
> from `mm³` to `voxels`, based on reading `np.unique(..., return_counts=True)`
> on line 317 — that function does produce the count, but the value is
> multiplied by the voxel volume before it is stored. The original description
> (`mm³`) was correct and has been restored.
>
> **As a consequence:** our `size@N` rules are defined in **voxels**, and voxel
> volume varies 28× across the cohort — so their physical effect is not
> homogeneous. The effect is measured and significant in the three regions:
> **D19**.
>
> ⚠️ **The two are not to be confused.** The asymmetry of the evaluator's units
> (2 voxels ↔ 2 mm³) is a finding with its own implications, and is recorded as
> a methodological correction. The spacing sensitivity of `size@N` holds
> regardless of it, because our rule removes by voxel count in our own code.
>
> **The bins: EXPLORATORY — decision of 17 Aug, before T3 was run.**
>
> The source of the edges 27 · 100 · 500 mm³ **is undocumented**, and they were
> probably inherited from the same 27 assumption (§13.4). The decision was taken
> to keep them as they are, with full disclosure, rather than replace them with
> clinical edges now — because choosing alternatives **after** seeing the B-6
> results is choosing bins after seeing the result, which is worse than using
> declared inherited edges.
>
> **The disclosure required in any presentation of T3 results, verbatim:**
>
> 1. These are **analytical bins** only.
> 2. **They were not pre-registered.**
> 3. **They were not chosen** on the basis of T3 or T4 results.
> 4. **They are not used** to test any confirmatory hypothesis.
>
> ⛔ **Replacing them with clinical edges is forbidden from now on.** Any change
> to them from this moment falls under the §8 rule: an experiment is not
> re-specified after its result has been seen.
>
> ⛔ **Inventing provenance is forbidden.** It is not to be written that the
> edges derive from the literature or from a clinical standard unless the
> reference is stated explicitly — and we have no reference.
>
> **The text required in the paper, verbatim:**
>
> > *These bins are descriptive/exploratory only and were not pre-registered.
> > They were retained unchanged after results became available; no alternative
> > bins were selected post hoc.*
>
> **And the answer to the reviewer's question** *"Why 27, 100 and 500 mm³?"*:
>
> > *They were retained as exploratory analytical bins; their original
> > provenance could not be established, and they were not used for
> > confirmatory hypothesis testing.*
>
> This is a scientifically stronger answer than constructing a clinical
> justification after the fact.
>
> **The bins are analytical/clinical bins, not evaluator bins.** The edges
> 27 · 100 · 500 are an analytical choice and stay as they are. What was
> corrected is the **description**: the `< 27` category was described as
> *"not evaluated"*, and that is **false** — it is fully evaluated. The only
> category excluded from the measured path is **`≤ 2 mm³`** (§13.2).
>
> **GT lesions of volume ≤ 2 mm³ are excluded from the evaluator's measured
> path, and therefore do not enter the comparative T3 analysis.** The row is
> listed in the table for completeness only, with empty cells: neither Δ Dice
> nor Δ recall is computed for it, because its lesions are never recorded in
> `metric_df_thresh` (line 761) and do not enter `Num_TP/FN` (743 · 747).
> Listing it with numbers would have suggested that T3 measures something and
> then declares it unmeasured.
>
> **The `Δ F1` column was replaced by `Δ recall`.** Per-bin F1 **cannot be
> computed**: its denominator needs the FPs, and an FP has no reference lesion
> and no reference volume to place it in a bin. Details in §12.9.

### T4 · Detection F1

| Rule | WT | TC | ET | RC |
|---|---|---|---|---|

---

## 10. Protocol Change Log

| Date | Change | Reason |
|---|---|---|
| 30 Jul | `leak_free` (binary) → `leak_class` (three-way) | Patient leakage discovered: 118 patients with more than one timepoint · 165 of the 320 "clean" cases were contaminated |
| 30 Jul | Sweep reduced from 9 rules → 6 | Layer 1 identified the region; points outside it generate numbers, not information |
| 30 Jul | `combined` scorer removed | It doubles the threshold space and burdens the multiple-comparison correction |
| 1 Aug | The baseline is computed inside the run instead of read from a CSV | B-2b proved exact agreement; cost 17% in return for a structural guarantee |
| 1 Aug | Guard 3: 500-voxel threshold removed → `WT == 0` only | An arbitrary number ("why 500?" has no answer); lesion size is a scientific question and belongs in B-6 |
| 3 Aug | Proposal to sweep on a 150-case subset withdrawn | The 567 are themselves a hard subset — any subset of them is a selection within a selection |
| **4 Aug** | **`size@20` removed from the Dice experiments** | ~~Mathematically impossible for it to move the Dice: 20 < 27 (the evaluation floor)~~ · **The argument is invalid — the actual floor is 2 voxels, and 27 is an undocumented assumption (§13.4)**. The rule moved to the detection-F1 analysis, and that assignment is what justifies its presence in T4 |
| 4 Aug | `size@50` added | First threshold above the evaluation floor |
| 4 Aug | `size@100` — under evaluation | Decided from the pilot, not before it |
| **13 Aug** | **Guard-3 diagnosis executed** — both cases Class 2 · recovered with `guards=False` · the pilot became 20 cases | Dropping them biased the mean in favour of suppression — the quantity under test |
| **13 Aug** | **§6.1 — pre-registered sensitivity for the QC cases** (exclude 6 from `clean` · n=149 · MDE 0.0194) | The organisers' QC log was released after the cohort was fixed and before B-4; declaring it in advance prevents it being read as post hoc |
| 13 Aug | QC × D9 audit — no change to the protocol | The 24 RC-confusion cases explain neither TP nor FP; both hypotheses withdrawn (`QC_LABEL_NOISE_ANALYSIS.md` §9) |
| **13 Aug** | **`size@100` removed** | The 20-case pilot: −0.0925 vs −0.0591 · worse in 20 of 24 differing cells · harm monotone in the threshold |
| **13 Aug** | **Guard-3 policy on the 567 fixed**: `guards=False` + a `status` row | Skipping failed cases biases the mean in favour of suppression — shown in the pilot |
| **13 Aug** | `assert set(RULES) == PROTOCOL_RULES` mandatory | The B-4 run started with `size@20` loaded and `size@100` absent — stopped before writing |
| **13 Aug** | `exp3_suppressed_v2.csv` archived | It contains rows with the abandoned `leak_free` schema (pre-30 Jul) mixed with `leak_class` rows · and the removed `size@20` rule |
| **13 Aug** | **Protocol frozen — v1.0** | Both open items settled |
| 14 Aug | **Execution set-up — §11 · not a protocol change** | Everything below concerns *how* the evaluation is executed, not *what* is measured; the final table is identical however these change. Recorded for provenance |
| 14 Aug | `native_space.py`: `_purge_case_temp()` before every evaluation + check of the prediction file size | An interrupted session left a zero-byte scratch file; the evaluator reused it → 5 failed pairs from one case (`01278-003`). Deletion confined to `WORK_DIR` |
| 14 Aug | `b4_suppressed_arm.py`: separate shards per worker (`_shard_path` / `_fail_path`) + two guards against mixing single/sharded | Every `to_csv` write is a full write, not an append; a shared path = the last writer erases the rest, and the damage is silent |
| 14 Aug | ETA calculation corrected — floor of 1.0 on the case cost | Division by `cost≈0` for empty cases produced a spurious "ETA 2101 h" |
| 14 Aug | `CKPT_EVERY` from 10 → 1 · `--cheap-first` ordering | Colab session interruption is the dominant risk over ~370 hours |
| 14 Aug | **B-4 started** — 3 workers · 3,402 pairs | — |
| **15 Aug** | **§13 — D6 reworded · the exclusion of `size@20` invalidated** | The volume floor is applied to `gt_lesion_vol` only; `len(fp)` enters the denominator unfiltered. Isolated probe: the Dice moved in 9/9 cells. The rule is a **candidate**, not included — including it requires an explicit prior decision |
| **15 Aug** | **§12.6–12.8** — saving both tables mandatory · T3 at lesion level via `gt_lesion_vol` · no case level | `get_LesionWiseResults` returns two tables; the second (one row per lesion) is discarded with an underscore. The first holds per-region aggregates and is not enough for T3. Grouping cases into bins changes T3's question and would need a threshold invented now |
| **15 Aug** | **§12 — T4 separated from B-4** · a later run on `clean` (155) with **7 rules** (the six + `size@20`) | `official_score` takes 4 of the 15 columns the evaluator returns; `Num_TP/FP/FN`, `Sensitivity/Specificity`, `NSD@0.5/1.0` and `HD95` are dropped. The fix is one line, **not applied during a frozen run**, and the columns **do not appear retroactively** → a limited re-run. Declared before seeing any result from B-4 |
| ~~16 Aug~~ | ~~Unit corrected: `mm³` → `voxels`~~ | **Withdrawn 17 Aug** — it was a wrong correction |
| ~~17 Aug~~ | ~~Floor = 2 mm³~~ | **Withdrawn the same day** — the unit differs between the two paths |
| **17 Aug** | **Approved wording: `lesion_volume_thresh = 2` = 2 voxels for the prediction · 2 mm³ for the reference** | `gt_lesion_vol` is fractional, and the smallest value = exactly one voxel of its case (ratio 1.0000000). The count is multiplied by the voxel volume. The original description was correct |
| **17 Aug** | **§12.10 — unit of analysis in T3 = the case · unweighted mean · no zero imputation** | Consistent with B-6; using the lesion as the unit (n=3,239) is pseudoreplication, because lesions of one case are not independent. Registered **before** computing any T3 number |
| **17 Aug** | **The `≤ 2 mm³` row in T3: not measured** — described as *evaluator-excluded; not part of scored T3 analysis*, with empty cells | Reference lesions ≤ 2 mm³ are outside `metric_df_thresh` and outside `Num_TP/FN`; showing them with a number suggests T3 measures them and then declares them unmeasured |
| **17 Aug** | **T3 bins = exploratory · not pre-registered · kept as they are** | The source of 27·100·500 is undocumented, but replacing them after seeing the B-6 results = choosing bins after seeing the result. The four-point disclosure is mandatory in any presentation (§9). Replacing them or claiming provenance for them is **forbidden** |
| **18 Aug** | **Analysis of FP components — exploratory · outside the freeze** | The gate counted on FP-rich cases (9/9 cells · 77 false positives · exact agreement with `Num_FP`); a first run on FP-poor cases was rejected because 10/12 were `0 == 0`. Results in **D21**: different selection patterns (Jaccard: size↔size 0.84 · size↔entropy 0.53–0.63 · confidence 0.14–0.23). **The physical-size table across spacing bins is descriptive and does not support D19** — every rule shows the same rise, because the FP population changes with spacing. The test design was not redone after it contradicted the expectation |
| **18 Aug** | **Inserting reported results in their place — no new analysis** | A claim-to-evidence check found 12 numbers in the Discussion not shown in the Results (D19 · D21 · RC). They were inserted in their places: D19 as a subsection after §8, D21 as an exploratory section with two tables, and the RC result inside §4 beside its ceiling. **No number changed and no test was added** — the results had been computed and recorded in `DISCOVERIES.md` and only lacked a place of presentation. The rule distinguished: `existing result → proper reporting location`, not `new analysis → new result`. Check afterwards: **5/5** |
| **18 Aug** | **🔒 Methods frozen** | Translated from the frozen protocol, 13 sections. Zero leakage from Results (design quantities only: n=155 · MDE · 3,402 · 1,085). §11 documents three corrections to earlier assumptions and interpretations, separating a **methodological provenance error** from **numerical corruption** — no published number depended on any of them. §13 states that `n` differs between endpoints |
| **18 Aug** | **🔒 Results frozen — B-4 · B-5b · B-6 · T3 · T4** | 67 numbers drawn from `RESULTS_DRAFT.md` itself and matched to their sources (zero mismatch · zero unparsed) · a five-layer audit of the detection tables · the endpoint-eligibility mechanism verified in all four regions · four provenance claims checked by hand, which found **one overclaim** in the QC-sensitivity sentence, corrected (the sign agrees in 18/20, not 20/20, and the largest difference is 0.0239, above the MDE). **Any change after today needs an independent scientific reason — not because a checking tool objected** |
| **17 Aug** | **D19 — spacing dependence of the `size@N` rules** | Voxel volume ranges 0.083–2.296 mm³ (55% of cases ≠ 1.0), so `size@50` cuts 7–115 mm³ depending on the case. Spearman significant in all three regions; `entropy@0.50` as a control: ρ=+0.071 p=0.38 |
| **16 Aug** | **§13 rewritten — the actual floor is 2 voxels on both sides · the number 27 is an undocumented assumption** | Direct reading: `lesion_volume_thresh = 2` (line 681, `BraTS-MET` branch), filtering prediction components (319) and reference lesions (743·747·761) with the same threshold. The number 27 appears only twice, in our own code (`size_pilot.py:12` comment · `guard3_diagnose.py:75` diagnostic constant), and does not exist in the evaluator. **No published number needs recomputation** |
| **16 Aug** | **§9 · T3** — the bins stay · the description *"not evaluated"* moves from `< 27` to `≤ 2` · `Δ F1` → `Δ recall` | The `< 27` category is fully evaluated; per-bin F1 cannot be computed, because an FP has no reference lesion (§12.9) |
| **16 Aug** | **`native_space.official_score`: a `full=True` mode returns both tables and the 15 columns** | The default `full=False` is **unchanged** — verified by AST comparison: one function changed and the rest is identical line by line. The addition is optional because `b5b`, `b6` and `merge_shards` are written against the current schema, and the 3,402 pairs of B-4 were produced with it. B-4 is complete, so the freeze has served its purpose. The columns **do not appear retroactively** → T3/T4 need a separate run (§12) |

> Any change after the protocol is frozen is recorded here with its date and
> reason — **without exception**.

---

## 13. The evaluation floor — correction of D6

**15 Aug 2026 · settled experimentally · does not affect the running B-4**

### 13.1 The old wording and why it fell

D6 as recorded:

> *Any suppression threshold that removes connected components below the
> evaluation floor **cannot** move the lesion-wise Dice.*

The argument was: threshold 20 is below the evaluation floor of 27, hence zero
effect.

**The number 27 does not exist in the evaluator.**

### 13.2 What the code actually says — direct reading (16 Aug)

```
680:  if challenge_name == 'BraTS-MET':
681:      lesion_volume_thresh = 2

      # filtering of prediction components — the loop over every region
317:  labels, counts = np.unique(pred_mat_combinedByDilation, return_counts=True)
319:  labels_to_remove = labels[counts <= lesion_volume_thresh]
322:  pred_mat_combinedByDilation = np.where(mask, pred_mat_combinedByDilation, 0)

      # filtering of reference lesions
743:  fn_sub    = metric_df[(metric_df['_len'] == 0) &
744:                        (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)]
747:  gt_tp_sub = metric_df[(metric_df['_len'] != 0) & ... <= lesion_volume_thresh]
761:  metric_df_thresh = metric_df[metric_df['gt_lesion_vol'] > lesion_volume_thresh]
```

**The same constant `2`, in two different units depending on the path:**
prediction components (line 319, inside the loop over every region and outside
any condition) and reference lesions (743 · 747 · 761).

### 13.3 The corrected wording

> **The official evaluator does not have a single physical-unit lesion floor.
> The numeric threshold `lesion_volume_thresh = 2` is applied as **2 voxels**
> to predicted connected components but as **2 mm³** to reference-lesion
> volumes.**
>
> ⛔ **This is the approved wording.** Neither "evaluator floor = 2 mm³" nor
> "= 2 voxels" is to be written on its own — both are incomplete.
>
> The evidence: line 319 compares `counts` (from
> `np.unique(..., return_counts=True)`) with the constant directly; line 596
> computes `gt_vol = np.sum(gt_tmp > 0) * sx * sy * sz` and enters it into
> `metric_pairs` as `gt_lesion_vol`, which lines 744 · 748 · 761 compare with
> the same constant.

**The withdrawn wording (15 Aug):** *"the floor is applied to reference
lesions, not to prediction components"*. The second half is **false** — line
319 filters prediction components with the same threshold. This correction came
from a deeper direct reading, not from a new result.

### 13.4 Origin of the number 27 — an undocumented analytical assumption

**The number 27 does not appear in `metrics_MET.py` at all.** A complete trace
within `src/eval/` gives only two occurrences, both from our own code:

```
size_pilot.py:12        "evaluator's 27-voxel minimum lesion size"   ← comment, no reference
guard3_diagnose.py:75   SCORABLE = 27                                 ← diagnostic constant
```

The first occurrence is a comment asserting the number with no source; from it
the number passed to the second, then to this protocol and to `DISCOVERIES.md`.
That is, it is **an undocumented analytical assumption, not an evaluator
threshold** — repeated until it became taken for granted.

> **Limits of the effect.** Not one published number needs recomputation. Every
> Dice value came from the evaluator itself with its correct threshold (2); the
> error was in **documenting the reason** for some design decisions, not in the
> numbers. And `SCORABLE` in `guard3_diagnose.py` feeds a diagnostic column
> (`scorable_removed`) that entered no judgement — the Class 2 classification of
> the two cases did not depend on it.

### 13.5 The experimental evidence

`fp_floor_probe.py` — an isolated file that touched no B-4 output. Three cases,
`baseline` against `size@20`, with sizes computed from the live probability map
(not from `day11`, whose provenance was shown not to match).

**The Dice moved in 9 of 9 cells that contained components ≤ 20 voxels.**

| Case | components ≤20 | Δ WT | Δ TC | Δ ET |
|---|---|---|---|---|
| `00770-000` | 12–16 | +0.011 | +0.037 | −0.070 |
| `00271-000` | 14–18 | +0.054 | +0.044 | +0.025 |
| `00685-000` | 70–91 | −0.259 | −0.406 | −0.386 |

`size@50` as a positive control: also moved in 9 of 9.

**The permitted wording of the result:**

> *The isolated probe revealed substantial bidirectional sensitivity to
> morphological suppression, with larger degradation observed in cases
> containing many small predicted components.*

⛔ **Forbidden** to claim monotonicity across `20 → 50 → 100`. `size@100` **was
not measured in this probe** at all, and the twenty pilot points and the
three-case probe are not the same sample. There is not a single point at which
all three thresholds were measured together.

### 13.6 The effect — precisely

| Item | Status |
|---|---|
| D6 | **reworded** — §13.2 |
| The mathematical exclusion of `size@20` | **invalidated** |
| `size@20` in the candidate set | **yes** |
| `size@20` in the Dice experiment | **no — requires an explicit protocol decision that has not been taken** |
| The running B-4 · the six rules · the cohort · the thresholds | **no change at all** |

> **D6 revised; the mathematical exclusion of `size@20` is invalidated.
> `size@20` is restored to the candidate set. Any execution of this candidate
> requires an explicit protocol decision made BEFORE inspecting its outcome.**

**Why it is not included automatically:** being a valid candidate ≠ being a
necessary experiment. It was reconsidered **after** seeing that it moves, so
including it without an independent question to justify it turns it into a
**post-hoc widening of the search space** — which the governing rule in §8
forbids. The question that must be answered first: does `size@20` answer a
question on the experiment map, or is it an extra point on a morphological
curve that `size@50` already answers?

### 13.7 What the probe did not measure

RC. All three cases have `0 components` in RC (a correct absence, Dice = 1.0),
so the probe is **completely silent** on the effect of `size@20` on RC — the
only region in which suppression showed a benefit in the pilot (+0.1429 at 50
and 100).

---

## 12. T4 — detection F1 · separated from B-4

**Declared 15 Aug 2026, before seeing any result from B-4.**

### 12.1 The problem

`get_LesionWiseResults` returns **15 columns**:

```
Labels · Num_TP · Num_FP · Num_FN · Sensitivity · Specificity
Legacy_Dice · Legacy_HD95 · Legacy NSD @ 0.5 · Legacy NSD @ 1.0
GT_Complete_Volume · LesionWise_Score_Dice · LesionWise_Score_HD95
LesionWise_Score_NSD @ 0.5 · LesionWise_Score_NSD @ 1.0
```

`native_space.official_score` takes **four** of them (`Labels` ·
`LesionWise_Score_Dice` · `Legacy_Dice` + the zero-warning) **and discards the
remaining eleven**. Among what is discarded: the lesion counters on which the
detection F1 is built, the NSD — **an official metric of the challenge** — and
the HD95.

Discovered 15 Aug by a direct schema check on `res_df` (a single evaluation,
read-only, a temporary file under a separate name, then deleted — without
touching the B-4 run).

### 12.2 Why it is not fixed now

The fix is **one line**, and yet:

1. B-4 is running on **frozen** code. Modifying the file that produces the
   scores during the run breaks provenance even if the output is identical.
2. The columns **do not appear retroactively**. The 3,402 pairs being computed
   now will keep four columns whatever is done later. So the fix alone is not
   enough — a re-run is needed.

### 12.3 The decision

**T4 is separated from B-4 and run after it, on the `clean` stratum only.**

| Item | Value |
|---|---|
| Cohort | `clean` only — **155 cases** |
| Rules | **7** — the six in §4 **+ `size@20`** |
| Size | 155 × 7 = **1,085 pairs** (~90 hours on 3 workers) |
| Outputs | **both tables together** — §12.6 |
| Dependency | after B-4, the merge and the diagnosis are complete |

**Why `clean` only:** T4 is a **supporting** measure in §7, not the primary
figure. And the clean stratum is the one on which the conclusion is built in
§8. Running the full 567 triples the cost for information that supports rather
than decides.

**Why `size@20` returns:** it was removed from the Dice experiments with the
explicit argument that it *"moves to the detection-F1 analysis"* (§4 · Change
Log 4 Aug), and **T4 is the analysis it was assigned to**. If T4 were run
without it, the rule would not move anywhere — it would be lost.

> **Update 16 Aug (§13):** its original argument (`20 < 27`) is invalid, and
> the actual floor is **2 voxels on both sides**. So its presence in T4 rests on
> **its earlier protocol assignment to this analysis**, not on the B-6 result —
> and that is the difference between honouring an old commitment and a post-hoc
> widening of the search space.
>
> **`size@20` was retained for T4 because its prior protocol assignment explicitly
> designated it for the detection/F1 analysis. Its inclusion is not motivated by the
> observed B-6 Dice results. The resulting Dice values are retained for
> audit/provenance only and are excluded from the primary Dice analysis.**
>
> ⛔ **It is not included in the Dice arm** (T1/T2/B-6). The resulting Dice
> values are kept for audit only. Even if they come out excellent or poor, **the
> Dice experiment is not redefined after seeing the results**.

### 12.4 A side benefit

The same change also recovers: `Sensitivity` · `Specificity` ·
`LesionWise_Score_NSD @ 0.5` · `@ 1.0` · `HD95` — for every arm.

The NSD is an official metric (the recorded result for RC: 0.474), and is
currently available for the baseline only. So this run is not one item — it is
**four metrics** from the same line.

### 12.6 Saving both tables — mandatory

`get_LesionWiseResults` returns **a tuple of two tables**:

```python
results_df, final_lesionwise_metrics_df = get_LesionWiseResults(...)
```

and `official_score` takes the first and discards the second with an
underscore:

```python
res_df, _ = get_LesionWiseResults(...)      # <-- the second table is lost here
```

| Table | Unit | Feeds |
|---|---|---|
| `results_df` | **one row per region** (4 rows) — aggregates | T4 |
| `final_lesionwise_metrics_df` | **one row per reference lesion** | **T3** |

The first table holds per-region aggregates — it gives `Num_TP/FP/FN` at region
level, **and is not enough on its own for T3**, because T3's unit is the lesion,
not the region.

Columns of the second table (verified directly 15 Aug on
`BraTS-MET-00003-000`):

```
predicted_lesion_numbers · gt_lesion_numbers · gt_lesion_vol
dice_lesionwise · hd95_lesionwise · nsd05_lesionwise · nsd10_lesionwise
_len · Label
```

**The T4/T3 run must save both.** Saving `results_df` alone reproduces the very
problem that §12 exists for.

### 12.7 T3 — at lesion level, without exception

**The unit is the reference lesion, not the case.** Each lesion enters a bin
directly by `gt_lesion_vol`.

The bins are **pre-registered in §9 · T3** and are not reworded:

```
≤ 2 mm³ (not evaluated) · 2–27 · 27–100 · 100–500 · > 500  **[mm³]**
```

**Per-bin `Δ Dice`:** the mean `dice_lesionwise` of the lesions in the bin,
under each rule, minus the baseline — paired at lesion level (same
`gt_lesion_numbers` for the same case and the same `Label`).

**Per-bin `Δ F1`:** from the `_len` column.

```
_len == 0   →  lesion not detected (FN)
_len >= 1   →  lesion detected (TP)
_len >  1   →  more than one prediction matched the same lesion
```

A lesion whose `_len` changes from `≥1` under the baseline to `0` under a rule =
**a true lesion removed by suppression**, which is the quantity `size@20` is
there to measure.

> **There is no case-level stratification in T3.** Assigning a whole case to a
> bin (by its largest lesion, its median, or the number of lesions under a
> threshold) **changes T3's question**: a case with a single 500-voxel lesion
> and a case with the same lesion plus ten small ones fall into the same bin
> although their response to suppression is completely different. And any
> threshold chosen now for the grouping would be a number invented after the run
> started. `gt_lesion_vol` is available for every lesion, so grouping is
> unnecessary in the first place.

### 12.10 Statistical unit of analysis in T3 — fixed before execution

**Registered 17 Aug 2026, before computing any T3 number.**

> **T3 statistical unit.** For each GT-volume bin, lesion-level Dice and recall
> will first be aggregated within each case by taking the **unweighted mean**
> across that case's lesions in that bin — every lesion within a case
> contributes equally, so a case with 30 lesions and a case with 1 lesion carry
> the same weight in the test. Within a bin, per-case recall is the proportion
> of that case's lesions detected (`_len > 0`); in cases with few lesions in a
> bin this is a coarse, near-binary quantity, which the rank-based test
> accommodates but which should be stated. Paired comparisons between baseline
> and each suppression rule will then use the **case** as the statistical unit,
> restricted to cases containing **at least one GT lesion in that bin**. Cases
> with no lesions in a bin are **not** assigned zero — they contribute no
> observation, so `n` differs across bins by construction. This preserves
> consistency with B-6 and avoids treating lesions within the same case as
> independent observations.

**Why the case and not the lesion:**

| | |
|---|---|
| The lesion as the unit | n = 3,239 — but lesions within one case are **not independent**: same patient, same prediction, same annotation. Using them suggests statistical power that does not exist (pseudoreplication) |
| **The case as the unit** ✅ | n is smaller and varies between bins, but it is **consistent with B-6** (paired Wilcoxon on 155 cases) and does not violate independence of observations |

**Three implicit decisions closed explicitly here:**

1. **Weighting within a case: unweighted.** Every lesion within a case has
   weight one. The alternative (weighting by volume) is rejected — the bin
   itself already constrains volume, so weighting would double the same axis.
2. **`n` varies between bins — by construction.** It is not a shortfall to be
   compensated.
3. **No zero imputation.** A case with no lesion in a bin = **a missing
   observation**, not a result of zero. Imputing zero would invent data that
   were not measured.

⛔ **This decision is frozen.** Any change to the unit of analysis or to the
aggregation method after seeing the T3 results falls under §8.

### 12.9 Scope of T3 — what can and cannot be computed

The smoke test (16 Aug · two cases from `clean`) showed that the lesion table
reproduces the evaluator's own counters, so classifying a reference lesion by
`_len` is **verified, not assumed**. But it bounds the scope of T3 precisely:

| Item | Available? | Reason |
|---|---|---|
| Δ lesion-wise Dice by reference-lesion volume bin | ✅ | `gt_lesion_vol` + `dice_lesionwise` for every lesion |
| recall / FN behaviour by bin | ✅ | `_len == 0` ⟺ an undetected reference lesion |
| **Full per-bin F1** | ❌ | **The denominator needs FPs, and an FP has no reference lesion — so it has no bin** |
| FP analysis by predicted-component size | ✅ | **An independent axis**, shown separately and not merged into the T3 table |

> ⛔ **Forbidden** to invent a denominator for per-bin F1. The lesion table is
> indexed by reference lesions; a false positive has no row in it by
> construction. Any per-bin F1 would require distributing FPs into bins that are
> not theirs.

**T4 is not affected:** `Num_TP / Num_FP / Num_FN` come directly from the
per-region aggregate table — a source of truth, with no inference. **T3 is a
stratification layer · T4 is a detection-statistics layer**, and they are not
mixed.

### 12.8 A note on `_len > 1`

`predicted_lesion_numbers` sometimes shows more than one identifier for the
same reference lesion (observed: `[3, 6]` with `_len = 2`) — two separate
predictions matched one lesion.

This is the **instance-definition ambiguity** documented in the organisers' QC
log (`QC_LABEL_NOISE_ANALYSIS.md` §2 · category `H_instance_topology`), and it
measures itself here. Recorded as a descriptive observation; **not an open
question in Phase B**.

### 12.5 What does **not** change

Nothing is touched in: the cohort (567) · the thresholds · the six rules in §4 ·
the hypotheses · the §8 criteria · the `clean` stratum (155) and its MDE.

T4 is an **addition** after B-4, not a replacement for it nor a condition of it.
And the T3 bins in §9 were registered before the freeze and do not change.

---

## 11. Running B-4 — execution set-up

> **This section is not part of the scientific protocol.** Everything in it
> concerns *how* the evaluation is executed, not *what* is measured. It has no
> effect on the result: every (case, rule) pair is computed completely
> independently with the same probabilities, thresholds, suppressor and
> evaluator, so the final table is identical whatever changes here. Recorded
> for provenance only.

### 11.1 Size

```
567 cases × 6 rules = 3,402 pairs
Time per evaluation: 75 s (empty case) → 14,218 s (largest case) · range ~190×
Realistic estimate: 340–450 hours on a single worker
```

### 11.2 Parallelism — 3 workers · separate shards

```bash
# in each separate Colab session, W = 0 or 1 or 2
!nohup python -u "{root}/src/eval/b4_suppressed_arm.py" --cheap-first \
    --worker W --n-workers 3 \
    > "{root}/phase_b_outputs/wave1/b4_wW.log" 2>&1 &
```

Each worker takes `cases[W::3]` (round-robin over the cost order, so heavy cases
are spread evenly) and writes to a **separate file**:

```
exp3_suppressed_v3.w0of3.csv     189 cases × 6 = 1,134 pairs
exp3_suppressed_v3.w1of3.csv     1,134
exp3_suppressed_v3.w2of3.csv     1,134
```

**Why separate shards:** every checkpoint write is a **full** `to_csv`, not an
`append`, and every worker holds its own `rows` in memory. Pointing more than
one worker at the same path makes the last writer **erase** the others' work —
and the damage is **silent**: the file remains a valid CSV, just missing rows at
random, discovered by counting after 300 hours. Two mutual guards in the script
prevent mixing the single-worker file with the shards in either direction.

### 11.3 Merge and verification

```bash
!python "{root}/src/eval/merge_shards.py"
```

It refuses to write an incomplete file. It checks that: the shards are disjoint
(no pair in more than one) · every pair has exactly 4 rows · the total is 3,402
· all rules are from the protocol.

### 11.4 The scratch guard in `native_space.py`

`ensure_evaluator()` does `os.chdir(WORK_DIR)` because the evaluator writes
intermediates into the cwd (`tmp_gt/<case>/`) **and reuses them if it finds
them**.

Colab killed the session in the middle of a write (14 Aug), so a
connected-components file was left at zero bytes, and the next run treated it as
cached → `ImageFileError: Empty file` on the baseline, and then **every** rule
for that case failed in turn. One case (`BraTS-MET-01278-003`) produced 5 failed
pairs that looked like a model or geometry problem and were neither.

`_purge_case_temp(case)` is called at the start of `official_score`: it deletes
`tmp_gt/<case>` and `tmp_pred/<case>` and any zero-byte `.nii.gz` in `WORK_DIR`,
**unconditionally** — because detection is what failed last time. And there is a
check after `nib.save` that the prediction file was actually written and its
size is > 0. All deletion is confined to `WORK_DIR`; `GT_DIR` and
`ENSEMBLE_DIR` are never touched.

### 11.5 Case order

`--cheap-first` (cheapest first). The ordering **is not a protocol item** — it
is not in §3, §4 or §7 — because it has no effect on the result.

The "most expensive first · fail fast" logic was right while testing the code.
After the code was frozen, the real risk became **Colab session interruption**,
and cheapest-first accumulates results much faster in each session.

**`CKPT_EVERY = 1`** — save after every evaluation. The cost of the write is
under 0.3% of evaluation time, and the maximum loss on interruption became one
evaluation instead of ten.

### 11.6 Resumption

On session interruption: new session → mount → **the same command with the same
worker number**. Each worker reads its resumption state from its own shard.
Verified in practice (13 Aug): 48 rows on disk → `already: 12` → `to do: 0` →
`nothing to do`.

### 11.7 Failure handling

A failed pair is written to the CSV as **4 rows** with a `status` column and
**empty** metric columns (not zero — zero is a real result, meaning suppression
erased the region).

> An **absent** row = "not reached yet". A row with `status = FAILED` = "reached
> and failed". The two were indistinguishable before, and that is what
> invalidated Exp3.

**Before the statistical analysis:** any failed pair is diagnosed with
`guard3_diagnose.py`. Class 2 = a genuine Dice-0 result and stays · Class 1 or 3
= a pipeline fault that must be fixed and re-run.

---

### Open items — all closed

```
✅ Guard-3 policy on the 567   guards=False + status row          (13 Aug)
✅ size@100 decision            removed — harm monotone in threshold (13 Aug)
```

### Execution status (as of the freeze)

```
🔄 B-4        3,402 pairs · 3 workers · started 14 Aug
⬜ Merge      merge_shards.py — after the 3 shards complete
⬜ Failure diagnosis  guard3_diagnose.py on any pair with status = FAILED
⬜ B-5b       stratification by leak_class
⬜ B-6        paired tests · per-region · per-size-bin
⬜ T4 + T3    a separate run — §12 · clean × 7 rules · both tables · after B-4
⬜ ★G4        decision point — §8
```

---

*Protocol v1.0 · Frozen 13 Aug 2026 · any change from now on = protocol deviation*
