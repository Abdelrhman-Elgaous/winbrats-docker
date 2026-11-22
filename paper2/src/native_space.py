# -*- coding: utf-8 -*-
"""
==============================================================================
 B-2 — UNIFIED EVALUATION MODULE          src/eval/native_space.py
==============================================================================
 THE ONE PLACE where a prediction crosses from npz space into GT space.

 RULE (non-negotiable): every Phase B experiment that computes an official
 metric MUST call official_score() from this module. No transpose is ever
 written inline again. A single verified path is the only thing that keeps the
 Exp3 failure from recurring.

 THE VERIFIED ROUTE
 ------------------
     probabilities (C,D,H,W)  in npz space
       -> build_nested_label()      WT=2, TC=1, ET=3, RC=4  (RC written last)
       -> transpose(2,1,0)          npz space -> GT space
       -> Nifti1Image(label, ref.affine, ref.header)   ref = nib.load(GT)
       -> get_LesionWiseResults(pred_file=..., gt_file=...)

 Established by Phase_A_Integrity_Verification.md (Dice = 1.0000 against the
 submission representation) and confirmed by B-1 on 2026-07-30: transpose
 (2,1,0) won on every tested case with margins 0.6943 - 0.9392.

 THREE FAIL-LOUD GUARDS — they raise, they never auto-fix
 --------------------------------------------------------
   1. SHAPE        pred.shape == gt.shape
   2. ORIENTATION  axcodes(pred) == axcodes(gt)
   3. EXACT-ZERO   a region present in GT must not score exactly 0.0000

 Guard 3 was introduced to detect failures that shape and orientation checks do
 not reveal. Case BraTS-MET-00632-000 illustrates the gap: pred and GT have
 IDENTICAL shape (512,212,512) and identical axcodes, yet the identity
 permutation scores Dice 0.0000 while (2,1,0) scores 0.9392. Shape agreement is
 necessary but not sufficient.

 FROZEN INPUTS (do not change in Phase B)
 ----------------------------------------
   folds       (1, 2, 4)
   thresholds  WT 0.40 · TC 0.35 · ET 0.45 · RC 0.60
   channels    0=WT 1=TC 2=ET 3=RC
   probability maps: tta = OFF for all five folds (ensemble manifest)
==============================================================================
"""

import os, re, sys, gc, glob, shutil, subprocess
import numpy as np
import nibabel as nib
from scipy import ndimage

DRIVE        = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
DATASET      = "Dataset137_BraTS2026MET"
ENSEMBLE_DIR = f"{DRIVE}/archive/inference_v1/ensemble"
GT_DIR       = f"{DRIVE}/workspace/preprocessing/nnUNet_raw/{DATASET}/labelsTr"
WORK_DIR     = "/content/nifti_work"
EVAL_REPO     = "/content/brats_repos_inspect/capellan_2025"
EVAL_REPO_URL = "https://github.com/dani-capellan/BraTS-2025-Metrics.git"

FOLDS     = (1, 2, 4)
PROB_KEY  = "probabilities"
CH        = {"WT": 0, "TC": 1, "ET": 2, "RC": 3}
THR       = {"WT": 0.40, "TC": 0.35, "ET": 0.45, "RC": 0.60}
NPZ_TO_GT = (2, 1, 0)          # verified: Phase A integrity + B-1 blast radius
EPS       = 1e-7

# GT label -> which evaluation regions it belongs to
GT_REGION_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "RC": (4,)}

os.makedirs(WORK_DIR, exist_ok=True)

_EVALUATOR = None


def ensure_evaluator():
    """
    Clone and prepare the official BraTS-2025 evaluator; return
    get_LesionWiseResults. Cached after the first call.

    The module is `metrics_MET`, NOT `metrics`. The repo also needs a NumPy 2.x
    compatibility pass (np.Inf / np.NaN were removed). Both are done exactly as
    in phase_b_0_audit.py so this module scores identically to earlier runs.
    """
    global _EVALUATOR
    if _EVALUATOR is not None:
        return _EVALUATOR

    if not os.path.exists(EVAL_REPO):
        os.makedirs(os.path.dirname(EVAL_REPO), exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", EVAL_REPO_URL, EVAL_REPO], check=True)

    for pkg, mod in [("connected-components-3d", "cc3d"), ("medpy", "medpy"),
                     ("SimpleITK", "SimpleITK")]:
        try:
            __import__(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg],
                           capture_output=True, timeout=600)

    for pf in glob.glob(f"{EVAL_REPO}/**/*.py", recursive=True):
        txt = open(pf, errors="replace").read()
        new = re.sub(r"\bnp\.Inf\b", "np.inf", txt)
        new = re.sub(r"\bnp\.NaN\b", "np.nan", new)
        new = re.sub(r"\bnp\.NAN\b", "np.nan", new)
        if new != txt:
            open(pf, "w").write(new)

    if EVAL_REPO not in sys.path:
        sys.path.insert(0, EVAL_REPO)
    os.chdir(WORK_DIR)          # the evaluator writes temp files into cwd

    from metrics_MET import get_LesionWiseResults
    _EVALUATOR = get_LesionWiseResults
    return _EVALUATOR


_ORIGINAL_TOUCHING = None


def get_touching_labels_fast(nifti_img1, nifti_img2):
    """
    Same result as the original, with each dilation computed once per label
    instead of once per label PAIR. Verified in Wave 0 at two levels
    (call-by-call and every output column).

    Early exit: when either image contributes no labels, no pair can form, so
    the original's loops never execute and it returns an empty set. Returning
    early reproduces that exactly and skips dilations that cannot contribute.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    data1 = nib.load(nifti_img1).get_fdata()
    data2 = nib.load(nifti_img2).get_fdata()

    labels1 = np.unique(data1)[1:]
    labels2 = np.unique(data2)[1:]

    touching_labels = set()
    if len(labels1) == 0 or len(labels2) == 0:
        return touching_labels

    structure = generate_binary_structure(3, 2)

    for label1 in labels1:                       # pass A
        dilated1 = binary_dilation(data1 == label1, structure)
        for label2 in labels2:
            if np.any(dilated1 & (data2 == label2)):
                touching_labels.add((label1, label2))

    for label2 in labels2:                       # pass B
        dilated2 = binary_dilation(data2 == label2, structure)
        for label1 in labels1:
            if np.any(dilated2 & (data1 == label1)):
                touching_labels.add((label1, label2))

    return touching_labels


def set_fast_metric(enabled):
    """
    Swap get_touching_labels on the metrics_MET module. Returns the state set.

    Monkey-patching IS valid for this specific function because it is DEFINED
    IN metrics_MET and called through the module namespace. The supervisor's
    warning against monkey-patching applies to names the module imports
    directly -- e.g. `from medpy.metric.binary import __surface_distances` --
    where rebinding the attribute silently captures nothing. Those were
    instrumented by editing a copy of the source instead.
    """
    global _ORIGINAL_TOUCHING
    ensure_evaluator()
    import metrics_MET as MM

    if _ORIGINAL_TOUCHING is None:
        _ORIGINAL_TOUCHING = MM.get_touching_labels

    MM.get_touching_labels = (get_touching_labels_fast if enabled
                              else _ORIGINAL_TOUCHING)
    return enabled


def fast_metric_active():
    """True if the fast implementation is currently installed."""
    import metrics_MET as MM
    return MM.get_touching_labels is get_touching_labels_fast


class GeometryError(RuntimeError):
    """Raised when a guard fails. Never caught-and-patched inside this module."""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_mean_prob(case, folds=FOLDS):
    """
    Mean probability over the deployed folds -> (C, D, H, W) in npz space.

    MEMORY: accumulates in place instead of holding every fold at once. The
    naive version (build a list, then np.mean) peaks at n_folds + 1 copies,
    which for the largest volumes here -- (4, 420, 512, 512) float32, 1.64 GiB
    each -- is about 6.6 GiB and will kill a Colab session. Streaming keeps the
    peak at two copies (~3.3 GiB), and it matters more in B-4, where several
    workers run concurrently.
    """
    acc = None
    for f in folds:
        p = f"{ENSEMBLE_DIR}/fold_{f}/{case}.npz"
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing probability map: {p}")
        with np.load(p) as z:
            a = z[PROB_KEY]
            if acc is None:
                acc = a.astype(np.float32, copy=True)
            else:
                acc += a.astype(np.float32, copy=False)
            del a
        gc.collect()
    acc /= float(len(folds))      # in place — no extra allocation
    return acc


def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


# ---------------------------------------------------------------------------
# label construction (with the optional suppression hook used by B-4)
# ---------------------------------------------------------------------------
def build_nested_label(mean_prob, thr=THR, suppressor=None):
    """
    Threshold each region, optionally drop whole connected components, then
    compose the nested BraTS label IN NPZ SPACE.

    suppressor: callable(region, component_mask, mean_prob, entropy) -> bool
                returning True to REMOVE that component. None = baseline.

    Label precedence follows the deployed pipeline exactly: WT=2 first, then
    TC=1, ET=3, and RC=4 written LAST so it takes precedence deterministically.
    """
    # Entropy is computed PER CHANNEL, only when a suppressor needs it. The
    # earlier version built it for all four channels up front, which added a
    # full extra copy of the volume for no reason on the baseline arm.
    masks = {}

    for reg, ch in CH.items():
        m = mean_prob[ch] >= thr[reg]
        if suppressor is not None and m.any():
            ent_ch = binary_entropy(mean_prob[ch])
            lab, n = ndimage.label(m)
            for cid in range(1, n + 1):
                comp = (lab == cid)
                if suppressor(reg, comp, mean_prob[ch], ent_ch):
                    m[comp] = False
            del ent_ch, lab
            gc.collect()
        masks[reg] = m

    out = np.zeros(masks["WT"].shape, np.uint8)
    out[masks["WT"]] = 2
    out[masks["TC"]] = 1
    out[masks["ET"]] = 3
    out[masks["RC"]] = 4          # last on purpose
    return out


def to_native_nifti(label_npz_space, gt_path):
    """npz space -> GT space, carrying the GT's own affine and header."""
    ref = nib.load(gt_path)
    lab = label_npz_space.transpose(NPZ_TO_GT)
    return nib.Nifti1Image(lab.astype(np.uint8), ref.affine, ref.header), ref


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _guard_shape(pred_img, ref, case):
    if pred_img.shape != ref.shape:
        raise GeometryError(
            f"[{case}] GUARD 1 SHAPE MISMATCH: pred {pred_img.shape} vs gt {ref.shape}. "
            f"Aborting — this is exactly the failure that invalidated Exp3.")


def _guard_orientation(pred_img, ref, case):
    a, b = nib.aff2axcodes(pred_img.affine), nib.aff2axcodes(ref.affine)
    if a != b:
        raise GeometryError(f"[{case}] GUARD 2 ORIENTATION MISMATCH: pred {a} vs gt {b}")


def _guard_exact_zero(res_df, gt_data, case, pred_label=None):
    """
    Detect PIPELINE faults. Not hard cases.

    A zero on whole tumour while WT is present in GT is a pipeline signature:
    all four regions are built from ONE array through ONE transpose, so a
    whole-tumour miss cannot come from localisation error alone. That raises.

    A zero on TC/ET/RC while WT is scored normally is a MODEL failure, not a
    geometry one. BraTS-MET-01287-002 is the reference example: a 68-voxel ET
    lesion, 741 voxels predicted 35 voxels away, zero overlap -- while WT still
    scored 698/942. The pipeline placed the volume correctly; the model put the
    enhancing core in the wrong part of the right lesion.

    An earlier version also raised on "large" structures above a 500-voxel
    threshold. That number had no source, and a guard is the wrong place to
    encode a judgement about which lesions are big enough to matter. Lesion
    size belongs in the scientific analysis (B-6 size bins), where it can be
    justified against the challenge protocol. This function answers one
    question only: is the pipeline broken?

    Non-WT zeros are returned as warnings and recorded per row, so a systematic
    fault would surface as a PATTERN across hundreds of cases rather than
    halting the run on a single one.
    """
    zeros = []
    for _, row in res_df.iterrows():
        reg = str(row["Labels"]).strip()
        if reg not in GT_REGION_LABELS:
            continue
        present_vox = int(np.isin(gt_data, GT_REGION_LABELS[reg]).sum())
        if present_vox and float(row["LesionWise_Score_Dice"]) == 0.0:
            zeros.append({"case": case, "region": reg, "gt_voxels": present_vox})

    if any(z["region"] == "WT" for z in zeros):
        raise GeometryError(
            f"[{case}] GUARD 3 EXACT-ZERO on WT while present in GT: {zeros}. "
            f"All regions share one transpose, so a whole-tumour miss indicates "
            f"MISALIGNMENT rather than localisation error. Abort.")

    for z in zeros:
        print(f"      [warn] {case}: {z['region']} Dice=0.0000 on "
              f"{z['gt_voxels']} GT voxels — WT scored normally, so this is a "
              f"model failure, not a pipeline fault")
    return zeros


# ---------------------------------------------------------------------------
# the single entry point
# ---------------------------------------------------------------------------
def _purge_case_temp(case):
    """
    Clear the evaluator's per-case scratch before scoring.

    WHY THIS EXISTS
    ---------------
    ensure_evaluator() does os.chdir(WORK_DIR) because the Capellan evaluator
    writes its intermediates into the current directory -- tmp_gt/<case>/ and
    tmp_pred/<case>/ -- and reuses them when they are already present.

    That reuse is fine within a healthy session and fatal across a broken one.
    Colab kills the runtime mid-write; a connected-component file is left at
    zero bytes; the next run finds it, treats it as cached, and nibabel raises
    `ImageFileError: Empty file`. Every later rule for that case then fails too,
    so ONE truncated file produces a cluster of failures that look like a model
    or geometry problem and are neither. Observed on BraTS-MET-01278-003,
    14 Aug 2026: five consecutive failed pairs from a single stale file.

    Rebuilding the scratch costs seconds. Trusting it costs a run. Any run that
    spans dozens of Colab sessions will hit this, so the purge is unconditional
    rather than conditional on detecting damage -- detection is the part that
    failed last time.

    Only scratch for THIS case is touched. Nothing under GT_DIR, ENSEMBLE_DIR
    or DRIVE is ever removed, and no evaluation logic is affected.
    """
    for sub in ("tmp_gt", "tmp_pred"):
        d = os.path.join(WORK_DIR, sub, case)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    # Zero-byte leftovers directly in WORK_DIR (killed nib.save calls).
    try:
        for f in os.listdir(WORK_DIR):
            if not f.endswith(".nii.gz"):
                continue
            fp = os.path.join(WORK_DIR, f)
            if os.path.isfile(fp) and os.path.getsize(fp) == 0:
                os.remove(fp)
    except OSError:
        pass


def official_score(case, suppressor=None, folds=FOLDS, thr=THR,
                   tag="baseline", keep_nifti=False, guards=True,
                   full=False):
    """
    Score one case with the official BraTS-MET lesion-wise evaluator.

    DEFAULT (full=False) -- unchanged, byte for byte
    ------------------------------------------------
    Returns a list of dicts: case, setting, region, lesionwise_dice,
    legacy_dice, zero_warning. Every existing caller keeps its current schema;
    b5b_stratify, b6_stats and merge_shards all depend on it, and B-4's 3,402
    completed pairs were produced under it.

    full=True -- for T3 / T4 (protocol section 12)
    ----------------------------------------------
    Returns (rows, agg, les):

      rows  the same list as above, so downstream code is unaffected
      agg   results_df, ALL FIFTEEN aggregate columns, one row per region:
              Labels · Num_TP · Num_FP · Num_FN · Sensitivity · Specificity
              Legacy_Dice · Legacy_HD95 · Legacy NSD @ 0.5 · Legacy NSD @ 1.0
              GT_Complete_Volume · LesionWise_Score_Dice
              LesionWise_Score_HD95 · LesionWise_Score_NSD @ 0.5 · @ 1.0
            -> SOURCE OF TRUTH for T4 detection counts. Nothing is inferred.
      les   final_lesionwise_metrics_df, one row per REFERENCE lesion:
              predicted_lesion_numbers · gt_lesion_numbers · gt_lesion_vol
              dice_lesionwise · hd95_lesionwise · nsd05_lesionwise
              nsd10_lesionwise · _len · Label
            -> the size axis for T3. `gt_lesion_vol` is a VOXEL COUNT.

    WHY THIS IS AN ADDITION, NOT A CHANGE
    -------------------------------------
    The evaluator has always returned both tables. This function discarded the
    second (`res_df, _ = ...`) and kept 4 of the 15 columns of the first, so
    Num_TP/FP/FN, Sensitivity, Specificity, NSD and HD95 were computed and then
    thrown away on every one of 3,402 evaluations. They cannot be recovered
    retrospectively, which is why T3/T4 need their own run (protocol §12).

    Making `full` opt-in keeps the default path identical. Changing the default
    return type would silently alter every caller written against it.

    A NOTE ON THE SCORING FLOOR
    ---------------------------
    BraTS-MET uses lesion_volume_thresh = 2 VOXELS (metrics_MET.py:681),
    applied to predicted components (line 319) and reference lesions
    (743, 747, 761) alike. The figure 27 that appeared in this project's
    comments and constants was an undocumented assumption, never an evaluator
    value -- see protocol section 13.

    Raises GeometryError if any guard fails. That is deliberate: a run that
    silently skips bad cases produces a table you cannot trust.
    """
    get_LesionWiseResults = ensure_evaluator()

    # B0 toggles the implementation through the environment so that a single
    # run can time both and compare every output column.
    _want_fast = os.environ.get("BRATS_FAST_METRIC")
    if _want_fast is not None:
        set_fast_metric(_want_fast == "1")

    gt_path = f"{GT_DIR}/{case}.nii.gz"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"missing GT: {gt_path}")

    _purge_case_temp(case)

    mean = load_mean_prob(case, folds)
    label = build_nested_label(mean, thr, suppressor)
    del mean                       # ~1.6 GiB on the largest cases
    gc.collect()
    pred_img, ref = to_native_nifti(label, gt_path)

    if guards:
        _guard_shape(pred_img, ref, case)
        _guard_orientation(pred_img, ref, case)

    pred_path = f"{WORK_DIR}/{tag}_{case}.nii.gz"
    nib.save(pred_img, pred_path)
    if not os.path.exists(pred_path) or os.path.getsize(pred_path) == 0:
        raise IOError(f"prediction not written or empty: {pred_path}")

    try:
        res_df, les_df = get_LesionWiseResults(pred_file=pred_path,
                                               gt_file=gt_path,
                                               challenge_name="BraTS-MET",
                                               output=None)
        zero_warnings = []
        if guards:
            zero_warnings = _guard_exact_zero(res_df, np.asarray(ref.dataobj), case)

        rows = []
        for _, r in res_df.iterrows():
            rows.append({
                "case": case,
                "setting": tag,
                "region": str(r["Labels"]).strip(),
                "lesionwise_dice": float(r["LesionWise_Score_Dice"]),
                "legacy_dice": float(r["Legacy_Dice"]) if "Legacy_Dice" in r else None,
                "zero_warning": any(w["region"] == str(r["Labels"]).strip()
                                    for w in zero_warnings),
            })
        if full:
            # Copy before returning: the evaluator reuses its frames across
            # calls, so handing out a reference would let a later call mutate
            # a table a caller is still holding.
            return rows, res_df.copy(), les_df.copy()
        return rows
    finally:
        if not keep_nifti and os.path.exists(pred_path):
            os.remove(pred_path)
        del pred_img, label
        gc.collect()


# ---------------------------------------------------------------------------
# suppression rules for B-4 (kept here so every arm shares one label builder)
# ---------------------------------------------------------------------------
def entropy_suppressor(threshold):
    """Drop a component whose mean entropy exceeds `threshold`."""
    def f(region, comp, prob_ch, ent_ch):
        return float(ent_ch[comp].mean()) > threshold
    return f


def size_suppressor(min_voxels):
    """
    Drop a connected component at or below `min_voxels`.

    The simplest possible suppression signal: small components are the most
    likely false positives. It is free to compute and fully interpretable, so
    it is the baseline any uncertainty-based rule has to beat before the extra
    complexity is worth carrying.
    """
    def f(region, comp, prob_ch, ent_ch):
        return int(comp.sum()) <= min_voxels
    return f


def confidence_suppressor(threshold):
    """Drop a component whose mean (1 - probability) exceeds `threshold`."""
    def f(region, comp, prob_ch, ent_ch):
        return float(1.0 - prob_ch[comp].mean()) > threshold
    return f


DEFAULT_SANITY_CASES = ["BraTS-MET-00632-000", "BraTS-MET-00003-000"]


def peak_rss_gb():
    """Resident memory in GiB — cheap way to watch for growth across cases."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return float("nan")


def sanity(cases=None):
    """Guard-checked path on a few cases. Call this directly from a notebook."""
    for c in (cases or DEFAULT_SANITY_CASES):
        print(f"\n=== {c} ===  [peak RSS so far {peak_rss_gb():.2f} GiB]")
        for r in official_score(c, tag="sanity"):
            print(f"  {r['region']:>4s}  lesionwise={r['lesionwise_dice']:.4f}  "
                  f"legacy={r['legacy_dice']}")


if __name__ == "__main__":
    # In a notebook, sys.argv carries the kernel's own flags (-f /path/kernel.json),
    # so keep only real case names.
    sanity([a for a in sys.argv[1:] if a.startswith("BraTS-")] or None)
