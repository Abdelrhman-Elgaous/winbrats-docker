# -*- coding: utf-8 -*-
"""
==============================================================================
 QUALITATIVE VISUALISATION PIPELINE · BraTS-METS 2026
==============================================================================
 Built to the visualisation brief. Priority order, applied throughout:

     SCIENTIFIC VALIDITY > REPRODUCIBILITY > COMMUNICATION > READABILITY
     > AESTHETICS

 THE VISUAL STORY THIS PACKAGE SUPPORTS
 --------------------------------------
   fig 1   DATA         what does one BraTS-METS case contain?
   fig 2   LABELS       how do native labels become evaluation regions?
   fig 3   RESULTS      what does the model predict, and where does it err?
   fig 4   REGIMES      which lesion regimes are difficult?
   S1      SPATIAL      the 3D extent of one segmentation
   fig 5   SUPPRESSION  NOT PRODUCED -- see the closing note

 WHAT THIS SCRIPT DOES
 ---------------------
   1  audits the actual files rather than assuming paths
   2  validates shape, affine, orientation and spacing per case
   3  validates the label set against the region definitions
   4  resolves every case from its frozen quantitative rule
   5  chooses every slice by a printed deterministic rule
   6  renders three genuinely different candidates per major figure
   7  writes a machine-readable manifest and a validation report

 CASE SELECTIONS ARE FROZEN. Each was resolved quantitatively before any
 image was seen, and two earlier rules were corrected before drawing:

   fig 1,2,3  BraTS-MET-00629-000  median WT lesion-wise Dice of the clean
                                   stratum, 0.6517, rank 78/155
   fig 4a     BraTS-MET-01350-004  smallest detected reference lesion above
                                   4.0 mm3, deliberately clear of the
                                   evaluator's 2.0 mm3 floor. The 2.0029 mm3
                                   case is NOT used: it clears the floor by
                                   0.15%, and a panel captioned "small scored
                                   lesion" should not show a case whose
                                   eligibility turns on rounding.
   fig 4b     BraTS-MET-00015-000  lesion count nearest the 75th percentile,
                                   18 lesions. The maximum is 180, rejected
                                   as unreadable, not as unrepresentative.
   fig 4c     from fp_components.csv, the largest false-positive component on
                                   which the six rules disagree
   fig 4d     BraTS-MET-00138-000  largest confidently missed lesion,
                                   1209 mm3, already named in the appendix

 THE PROBABILITY COHORT IS SMALLER THAN THE DATASET. Maps were retained for
 the 567-case Phase B cohort, not all 1,296, so any figure needing a
 prediction is restricted to that cohort intersected with the clean stratum.
 A missing map stops the affected figure; no case is silently substituted.

 ANATOMICAL PLANES COME FROM THE AFFINE, not from array position. These
 volumes are RAS with superior-inferior at file axis 2, which the analysis
 transpose moves to array axis 0. An earlier version named array axis 2
 "axial" and was showing sagittal slices.

 PANEL ASPECT COMES FROM THE HEADER ZOOMS. The voxel is 1.172 x 1.172 x 1.5
 mm, so a through-plane panel drawn at aspect 1.0 is squashed by 22%.
==============================================================================
"""

import os
import re
import sys
import json
import glob
import datetime

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

# ===========================================================================
#  CONFIGURATION
# ===========================================================================

DRIVE = ns.DRIVE
GT_DIR = ns.GT_DIR
W1 = f"{DRIVE}/phase_b_outputs/wave1"
W2 = f"{DRIVE}/phase_b_outputs/wave2"
C0 = f"{DRIVE}/phase_c_outputs/c0"
OUT = f"{DRIVE}/paper2/figures/qualitative"

SEQS = ["t1n", "t1c", "t2w", "t2f"]          # exact dataset names, not renamed

# Two separate colour systems, as the brief requires. A reader must never
# confuse a reference class with a prediction error, so the two palettes do
# not overlap in hue.
LABEL_COLOUR = {1: ("NETC", "#e07b39"), 2: ("SNFH", "#8a7fc4"),
                3: ("ET", "#17a673"), 4: ("RC", "#d94f8a")}
REGION_DEF = {"WT": ((1, 2, 3), "#00a8b5"), "TC": ((1, 3), "#b845c9"),
              "ET": ((3,), "#2f6fe0"), "RC": ((4,), "#d94f8a")}
# §11: the legend states the terms in full. "TP" alone is compact but a
# reader should not have to reconstruct it, and "detected" or "missed area"
# are the vague forms the brief rules out.
ERROR_COLOUR = {"TP": ("#1f6fb2", "True positive"),    # blue
                "FN": ("#d1342c", "False negative"),   # red
                "FP": ("#f0921f", "False positive")}   # orange

FROZEN = {
    "main":  ("BraTS-MET-00629-000",
              "median WT lesion-wise Dice of the clean stratum",
              "WT Dice 0.6517, rank 78/155"),
    "small": ("BraTS-MET-01350-004",
              "smallest detected reference lesion with GT volume > 4.0 mm3",
              "4.1250 mm3, evaluator floor 2.0 mm3"),
    "multi": ("BraTS-MET-00015-000",
              "case nearest the 75th percentile of lesion count",
              "18 lesions; cohort maximum 180 rejected as unreadable"),
    "missed": ("BraTS-MET-00138-000",
               "largest confidently missed lesion, already in the appendix",
               "1209 mm3, WT lesion 1"),
}

# Typography, one scale across every figure.
S = {"dpi": 600, "title": 7.4, "label": 6.8, "small": 6.0, "tiny": 5.2,
     "hair": 0.45, "panel": 8.4}
COL1, COL2 = 3.35, 7.0          # single- and two-column widths, inches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": S["label"],
    "savefig.dpi": S["dpi"], "figure.dpi": S["dpi"],
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.linewidth": S["hair"], "axes.edgecolor": "#333333",
    "pdf.fonttype": 42, "ps.fonttype": 42,      # editable vector text
})

INV = tuple(np.argsort(ns.NPZ_TO_GT))
MANIFEST = []
REPORT = []
os.makedirs(OUT, exist_ok=True)


def log(section, text=""):
    REPORT.append(f"{section}\n{text}" if text else section)
    print(text if text else section)


def head(t):
    line = "=" * 78
    print(f"\n{line}\n {t}\n{line}")
    REPORT.append(f"\n{line}\n {t}\n{line}")


# ===========================================================================
#  1 · FILE AUDIT.  Discover paths; do not assume them.
# ===========================================================================

def find_sequences(case):
    """Locate the four sequences for a case by searching, not by guessing a
    layout. Returns only what exists."""
    found = {}
    for s in SEQS:
        for pat in (f"{DRIVE}/**/{case}-{s}.nii.gz",
                    f"{DRIVE}/**/{case}_{s}.nii.gz"):
            h = glob.glob(pat, recursive=True)
            if h:
                found[s] = h[0]
                break
    return found


def audit():
    head("1 · FILE AUDIT")
    gt = sorted(glob.glob(f"{GT_DIR}/*.nii.gz"))
    log("", f"  reference labels      : {len(gt)} cases")

    lk = f"{DRIVE}/phase_b_outputs/wave0/leak_class_567.csv"
    cohort = clean = []
    if os.path.exists(lk):
        L = pd.read_csv(lk)
        cohort = sorted(L.case)
        clean = sorted(L[L.leak_class == "clean"].case)
        log("", f"  Phase B cohort        : {len(cohort)}")
        log("", f"  clean stratum         : {len(clean)}")
    else:
        log("", f"  leak_class_567.csv not found at {lk}")

    csvs = {}
    for name, path in (("B-4 Dice", f"{W1}/exp3_suppressed_v3.csv"),
                       ("T4 counts", f"{W2}/t4_counts.w1of1.csv"),
                       ("FP components", f"{W2}/fp_components.csv"),
                       ("C0 lesions", f"{C0}/c0_lesions.csv")):
        ok = os.path.exists(path)
        csvs[name] = path if ok else None
        log("", f"  {name:21s} : {'found' if ok else 'MISSING'}")

    log("", "\n  frozen cases, and whether each has what its figure needs:")
    status = {}
    for key, (case, rule, val) in FROZEN.items():
        ref_ok = os.path.exists(f"{GT_DIR}/{case}.nii.gz")
        seqs = find_sequences(case)
        try:
            ns.load_mean_prob(case)
            prob_ok = True
        except Exception:
            prob_ok = False
        status[key] = {"case": case, "reference": ref_ok,
                       "sequences": sorted(seqs), "probabilities": prob_ok}
        log("", f"    {key:7s} {case}  ref {'y' if ref_ok else 'N'} · "
                f"seq {len(seqs)}/4 · prob {'y' if prob_ok else 'N'}")
        if not prob_ok:
            log("", f"            no probability map: any figure for this case "
                    f"that needs a prediction STOPS")
    return {"gt_count": len(gt), "cohort": len(cohort), "clean": len(clean),
            "csvs": csvs, "frozen": status}


# ===========================================================================
#  2 · SPATIAL AND LABEL VALIDATION.  A figure is drawn only after this.
# ===========================================================================

def plane_axes(affine):
    """Anatomical plane to TRANSPOSED array axis, derived from the affine."""
    codes = nib.aff2axcodes(affine)
    file_axis = {}
    for i, c in enumerate(codes):
        if c in "SI":
            file_axis["axial"] = i
        elif c in "AP":
            file_axis["coronal"] = i
        elif c in "RL":
            file_axis["sagittal"] = i
    to_array = {f: k for k, f in enumerate(INV)}
    return {p: to_array[a] for p, a in file_axis.items()}, codes


def load_validated(case, need_pred=False):
    """
    Load a case and validate it before anything is drawn.

    Checked: the reference exists; all four sequences exist; every sequence
    matches the reference in shape and affine; the label set is a subset of
    the four defined labels; and, when a prediction is needed, that the
    probability map loads and the built label matches the reference shape.

    Raises on any failure. A silent substitution or a blind transpose is
    exactly what the brief forbids, so a failure stops the figure.
    """
    ref_path = f"{GT_DIR}/{case}.nii.gz"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"reference missing: {ref_path}")
    ref = nib.load(ref_path)
    gt_file = np.asanyarray(ref.dataobj)
    checks = {"reference": "PASS"}

    seq_paths = find_sequences(case)
    missing = [s for s in SEQS if s not in seq_paths]
    if missing:
        raise FileNotFoundError(f"{case}: sequences missing {missing}")

    vols = {}
    for s in SEQS:
        img = nib.load(seq_paths[s])
        if img.shape != ref.shape:
            raise ValueError(f"{case} {s}: shape {img.shape} != reference "
                             f"{ref.shape}")
        if not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ValueError(f"{case} {s}: affine differs from the reference")
        vols[s] = np.asanyarray(img.dataobj).transpose(INV)
    checks["sequences"] = f"PASS ({len(vols)}/4, shape and affine)"

    labels = set(np.unique(gt_file).tolist()) - {0}
    unknown = labels - set(LABEL_COLOUR)
    if unknown:
        raise ValueError(f"{case}: unexpected label values {sorted(unknown)}")
    checks["labels"] = f"PASS (present: {sorted(int(l) for l in labels)})"

    gt = gt_file.transpose(INV)
    planes, codes = plane_axes(ref.affine)
    zooms = tuple(float(v) for v in ref.header.get_zooms()[:3])
    spacing = tuple(zooms[INV[k]] for k in range(3))
    checks["orientation"] = f"PASS (axcodes {codes})"
    checks["spacing"] = (f"PASS (file {zooms}, array order {spacing}, "
                         f"anisotropy {max(spacing)/min(spacing):.3f})")
    checks["transpose"] = (f"DOCUMENTED: array = file.transpose{INV}, "
                           f"from argsort(NPZ_TO_GT)")

    pred = None
    if need_pred:
        prob = ns.load_mean_prob(case)          # raises if absent
        pred = ns.build_nested_label(prob, suppressor=None)
        if pred.shape != gt.shape:
            raise ValueError(f"{case}: prediction {pred.shape} != reference "
                             f"{gt.shape}")
        checks["prediction"] = "PASS (built from retained probabilities)"

    return {"gt": gt, "vols": vols, "pred": pred, "planes": planes,
            "codes": codes, "spacing": spacing, "checks": checks,
            "affine": ref.affine, "seq_paths": seq_paths,
            "ref_path": ref_path}


# ===========================================================================
#  3 · DETERMINISTIC SLICE RULES
# ===========================================================================

def slice_max_area(mask, axis):
    """The slice with the largest mask cross-section. Deterministic."""
    if not mask.any():
        return None
    other = tuple(i for i in range(3) if i != axis)
    return int(np.argmax(mask.sum(axis=other)))


def slice_at_centroid(mask, axis):
    """The slice through the mask centroid. Deterministic."""
    if not mask.any():
        return None
    return int(round(float(np.argwhere(mask)[:, axis].mean())))


def geometry(axis, spacing):
    """mm per display pixel and the imshow aspect for a slice along `axis`.

    The displayed image is rot90'd, so the array's two remaining axes appear
    as (x, y) = (rows, reversed cols). Aspect is vertical over horizontal
    physical size, which is 1.0 only when both in-plane.
    """
    rem = [i for i in range(3) if i != axis]
    r, c = rem
    return spacing[r], spacing[c], spacing[c] / spacing[r]


# ===========================================================================
#  4 · RENDERING PRIMITIVES
# ===========================================================================

def window(vol, extra=None, lo=0.5, hi=99.5):
    """
    Robust percentile windowing, applied identically to every sequence and
    every figure.

    The release is not intensity-normalised: mean non-zero intensity varies by
    up to 2,401-fold across cases, so a fixed window renders most of them
    black or white. The percentiles are fixed rather than tuned per image,
    which is the difference between a documented strategy and adjusting each
    picture until it looks good.
    """
    ref = vol[vol > 0]
    if extra is not None and extra.any():
        ref = np.concatenate([ref, vol[extra]])
    if not ref.size:
        return vol
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((vol - a) / max(b - a, 1e-6), 0, 1)


def bbox(*masks, pad=12):
    comb = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        comb |= m.astype(bool)
    if not comb.any():
        return (slice(None), slice(None))
    i = np.argwhere(comb)
    lo = np.maximum(i.min(0) - pad, 0)
    hi = np.minimum(i.max(0) + pad + 1, comb.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def mri(ax, img2d, aspect, title=None):
    """Anatomy: continuous interpolation, since nearest leaves blocky voxel
    edges that read as a low-resolution export."""
    ax.imshow(np.rot90(img2d), cmap="gray", vmin=0, vmax=1,
              interpolation="bicubic", aspect=aspect, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(S["hair"]); sp.set_color("#333333")
    if title:
        ax.set_title(title, fontsize=S["title"], pad=3.0)


# §13: reference and prediction are always drawn with the same alpha and the
# same line weight. Making one heavier would put a thumb on the comparison.
SEG_ALPHA = 0.40
SEG_LW = 0.75


def seg_fill(ax, mask2d, colour, aspect, alpha=SEG_ALPHA):
    """Segmentation: nearest-neighbour always. Interpolating a mask invents
    partial membership that does not exist, and the alpha is controlled so the
    anatomy stays visible underneath."""
    if not mask2d.any():
        return
    rgba = np.zeros(mask2d.shape + (4,))
    rgba[mask2d.astype(bool)] = to_rgba(colour, alpha)
    ax.imshow(np.rot90(rgba), interpolation="nearest", aspect=aspect,
              rasterized=True)


def seg_edge(ax, mask2d, colour, aspect, lw=SEG_LW):
    """An anti-aliased boundary, so extent is legible without hiding the
    anatomy a flood fill would cover."""
    if not mask2d.any():
        return
    ax.contour(np.rot90(mask2d.astype(float)), levels=[0.5], colors=[colour],
               linewidths=lw, antialiased=True)
    ax.set_aspect(aspect)


def scalebar(ax, mm_per_px, mm=20, avoid=None):
    """
    A scale bar on every anatomical panel. Lesion size carries several of this
    paper's results, and a figure about size with no scale asks the reader to
    take the sizes on trust. Stroked so it is legible on bright or dark
    tissue without knowing which.
    """
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w = mm / mm_per_px
    if w > 0.55 * abs(x1 - x0):
        mm = 10
        w = mm / mm_per_px
    stroke = [pe.withStroke(linewidth=1.1, foreground="#0a0a0a")]
    # §20: do not let the bar cover a lesion. Given the displayed mask, put
    # the bar in whichever bottom corner holds less of it.
    side = "right"
    if avoid is not None and avoid.any():
        disp = np.rot90(avoid.astype(bool))
        band = disp[int(0.75 * disp.shape[0]):, :]
        if band.size:
            half = band.shape[1] // 2
            side = "left" if band[:, half:].sum() > band[:, :half].sum() \
                   else "right"
    if side == "left":
        xl = min(x0, x1) + 0.05 * abs(x1 - x0)
    else:
        xl = max(x0, x1) - 0.05 * abs(x1 - x0) - w
    yb = max(y0, y1) - 0.055 * abs(y1 - y0)
    bar = Rectangle((xl, yb), w, 0.013 * abs(y1 - y0), facecolor="white",
                    edgecolor="none", zorder=8)
    bar.set_path_effects(stroke)
    ax.add_patch(bar)
    # label ABOVE the bar: below it sits at the axes edge and a tight
    # bounding box clips the descenders
    t = ax.text(xl + w / 2, yb + 0.025 * abs(y1 - y0), f"{mm} mm",
                color="white", fontsize=S["tiny"], ha="center", va="bottom",
                zorder=8)
    t.set_path_effects(stroke)


def letter(ax, ch):
    """§21: panel labels as (a), (b), (c), matching the manuscript, and not
    oversized -- the brief rules out large panel letters as well as large
    titles."""
    txt = ch if ch.startswith("(") or ch.startswith("S") else f"({ch})"
    ax.text(-0.012, 1.0, txt, transform=ax.transAxes, fontsize=S["panel"],
            fontweight="bold", ha="right", va="top")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    return f"{OUT}/{name}"


def record(**kw):
    MANIFEST.append(kw)


# ===========================================================================
#  FIGURE 1 · DATASET OVERVIEW
#
#  A dataset figure, not a results figure: four sequences and the native
#  reference labels, and nothing about prediction, error or suppression.
# ===========================================================================

def figure1(d, case):
    a = d["planes"]["axial"]
    gt, vols, sp = d["gt"], d["vols"], d["spacing"]

    # Slice rule, printed and recorded: the axial slice with the largest
    # enhancing-tumour cross-section. ET rather than WT, because a slice rich
    # in WT can miss the inner compartments entirely and the figure is about
    # what the labels are.
    inner = np.isin(gt, (3,))
    rule = ("axial slice of maximum enhancing-tumour area; "
            "falls back to maximum any-label area if ET is absent")
    z = slice_max_area(inner, a)
    if z is None:
        z = slice_max_area(gt > 0, a)
        rule += " [fallback used: no ET in this case]"
    box = bbox(np.take(gt > 0, z, axis=a), pad=8)
    mmx, mmy, asp = geometry(a, sp)
    base = {s: np.take(window(vols[s], gt > 0), z, axis=a)[box] for s in SEQS}
    lab = {l: np.take(gt == l, z, axis=a)[box] for l in LABEL_COLOUR}
    paths = {}

    # ---- candidate A · conservative, one row, minimal ------------------
    fig, ax = plt.subplots(1, 5, figsize=(COL2, 1.72))
    fig.subplots_adjust(wspace=0.03, right=0.875)
    for k, s in enumerate(SEQS):
        mri(ax[k], base[s], asp, s.upper().replace("T1N", "T1n")
            .replace("T1C", "T1c").replace("T2W", "T2w").replace("T2F", "T2f"))
    mri(ax[4], base["t1c"], asp, "Reference labels")
    for l, (nm, c) in LABEL_COLOUR.items():
        seg_fill(ax[4], lab[l], c, asp, 0.48)
        seg_edge(ax[4], lab[l], c, asp, 0.5)
    scalebar(ax[0], mmx); letter(ax[0], "a")
    ax[4].legend(handles=[Patch(facecolor=to_rgba(c, 0.48), edgecolor=c,
                                linewidth=0.5, label=n)
                          for n, c in LABEL_COLOUR.values()],
                 loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False,
                 fontsize=S["small"], handlelength=1.0, handleheight=0.85,
                 labelspacing=0.45, borderpad=0.1)
    paths["A"] = save(fig, "fig1_dataset_candA")

    # ---- candidate B · dense, 2x3 with labels twice --------------------
    # Information-dense: the same slice as a 2x3 block, with the labels shown
    # both alone and over T1c, so a reader can see the annotation on its own
    # terms and in anatomical context without switching figures.
    fig = plt.figure(figsize=(COL2 * 0.74, 2.95))
    gs = fig.add_gridspec(2, 3, wspace=0.03, hspace=0.10)
    order = ["t1n", "t1c", "t2w", "t2f"]
    names = ["T1n", "T1c", "T2w", "T2f"]
    for k in range(4):
        axk = fig.add_subplot(gs[k // 3, k % 3])
        mri(axk, base[order[k]], asp, names[k])
        if k == 0:
            scalebar(axk, mmx); letter(axk, "a")
    ax5 = fig.add_subplot(gs[1, 1])
    mri(ax5, np.zeros_like(base["t1c"]), asp, "Labels")
    for l, (nm, c) in LABEL_COLOUR.items():
        seg_fill(ax5, lab[l], c, asp, 0.95)
    ax6 = fig.add_subplot(gs[1, 2])
    mri(ax6, base["t1c"], asp, "Labels on T1c")
    for l, (nm, c) in LABEL_COLOUR.items():
        seg_fill(ax6, lab[l], c, asp, 0.45)
        seg_edge(ax6, lab[l], c, asp, 0.5)
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.55), edgecolor=c,
                              linewidth=0.5, label=n)
                        for n, c in LABEL_COLOUR.values()],
               loc="lower center", ncol=4, frameon=False,
               fontsize=S["small"], bbox_to_anchor=(0.5, -0.035),
               handlelength=1.0, columnspacing=1.3)
    paths["B"] = save(fig, "fig1_dataset_candB")

    # ---- candidate C · anatomy-led, wider crop, outlines only ----------
    # Anatomy-focused: a wider field of view and boundaries instead of fills,
    # so the lesion is read in its anatomical setting rather than as a
    # coloured patch. The trade is that small compartments are harder to see.
    wide = bbox(np.take(gt > 0, z, axis=a), pad=36)
    wb = {s: np.take(window(vols[s], gt > 0), z, axis=a)[wide] for s in SEQS}
    wl = {l: np.take(gt == l, z, axis=a)[wide] for l in LABEL_COLOUR}
    fig, ax = plt.subplots(1, 5, figsize=(COL2, 1.95))
    fig.subplots_adjust(wspace=0.025, right=0.875)
    for k, s in enumerate(SEQS):
        mri(ax[k], wb[s], asp, ["T1n", "T1c", "T2w", "T2f"][k])
    mri(ax[4], wb["t1c"], asp, "Reference labels")
    for l, (nm, c) in LABEL_COLOUR.items():
        seg_edge(ax[4], wl[l], c, asp, 0.95)
    scalebar(ax[0], mmx); letter(ax[0], "a")
    ax[4].legend(handles=[plt.Line2D([], [], color=c, lw=1.1, label=n)
                          for n, c in LABEL_COLOUR.values()],
                 loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False,
                 fontsize=S["small"], handlelength=1.2, labelspacing=0.5)
    paths["C"] = save(fig, "fig1_dataset_candC")

    for cand in paths:
        record(figure="1", candidate=cand, panel="a", case=case,
               region="native labels", selection_rule=FROZEN["main"][1],
               quantitative_value=FROZEN["main"][2], slice_rule=rule,
               slice_index=z, plane="axial", array_axis=a,
               aspect=round(asp, 4), mm_per_px=round(mmx, 4),
               source_files=[d["ref_path"]] + [d["seq_paths"][s] for s in SEQS],
               spatial_validation="PASS", label_validation=d["checks"]["labels"],
               output=paths[cand])
    log("", f"  fig 1  {case} · axial slice {z} (axis {a}) · aspect "
            f"{asp:.3f} · 3 candidates")
    return z, box, a, asp, mmx, rule


# ===========================================================================
#  FIGURE 2 · NATIVE LABELS TO EVALUATION REGIONS
#
#  A methods figure. It answers one question -- how the four native labels
#  compose into the four scored regions -- and shows no prediction.
# ===========================================================================

def figure2(d, case, z, box, a, asp, mmx, slice_rule):
    gt = d["gt"]
    base = np.take(window(d["vols"]["t1c"], gt > 0), z, axis=a)[box]
    lab = {l: np.take(gt == l, z, axis=a)[box] for l in LABEL_COLOUR}
    reg = {r: np.take(np.isin(gt, ls), z, axis=a)[box]
           for r, (ls, _) in REGION_DEF.items()}

    def caption(r):
        ls = REGION_DEF[r][0]
        comp = " + ".join(LABEL_COLOUR[l][0] for l in ls)
        # write the relation, not the right-hand side alone, and say when a
        # region is off this slice rather than leaving a blank panel: "absent
        # on this slice" and not "absent", which would imply the whole case
        cap = f"{r} = {comp}" if comp != r else f"{r} = {r}"
        return cap + ("" if reg[r].any() else "\n(absent on this slice)")

    paths = {}

    # ---- candidate A · labels row over regions row ---------------------
    fig, ax = plt.subplots(2, 4, figsize=(COL2 * 0.82, 3.55))
    fig.subplots_adjust(wspace=0.03, hspace=0.22, top=0.90, bottom=0.10)
    for k, (l, (nm, c)) in enumerate(LABEL_COLOUR.items()):
        mri(ax[0][k], base, asp, nm)
        seg_fill(ax[0][k], lab[l], c, asp, 0.50)
        seg_edge(ax[0][k], lab[l], c, asp, 0.7)
        if not lab[l].any():
            ax[0][k].set_xlabel("(absent on this slice)",
                                fontsize=S["tiny"], labelpad=2)
    for k, (r, (ls, c)) in enumerate(REGION_DEF.items()):
        mri(ax[1][k], base, asp, r)
        seg_fill(ax[1][k], reg[r], c, asp, 0.44)
        seg_edge(ax[1][k], reg[r], c, asp, 0.8)
        ax[1][k].set_xlabel(caption(r), fontsize=S["small"], labelpad=2.5)
    ax[0][0].text(-0.04, 1.14, "Native labels", transform=ax[0][0].transAxes,
                  fontsize=S["title"], fontweight="bold", ha="left")
    ax[1][0].text(-0.04, 1.10, "Evaluation regions",
                  transform=ax[1][0].transAxes, fontsize=S["title"],
                  fontweight="bold", ha="left")
    scalebar(ax[0][0], mmx); letter(ax[0][0], "b")
    paths["A"] = save(fig, "fig2_regions_candA")

    # ---- candidate B · nesting made explicit ---------------------------
    # The regions are nested by construction, so this candidate draws them
    # over one another on a single panel with the composition written beside
    # it. Denser, and it makes the set relationship visible rather than
    # leaving the reader to infer it from four separate panels.
    fig = plt.figure(figsize=(COL2 * 0.62, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.06)
    axn = fig.add_subplot(gs[0, 0])
    mri(axn, base, asp, "Nested regions on T1c")
    for r in ("WT", "TC", "ET"):
        seg_edge(axn, reg[r], REGION_DEF[r][1], asp, 1.0)
    if reg["RC"].any():
        seg_edge(axn, reg["RC"], REGION_DEF["RC"][1], asp, 1.0)
    scalebar(axn, mmx); letter(axn, "b")
    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
    y = 0.92
    for r, (ls, c) in REGION_DEF.items():
        comp = " + ".join(LABEL_COLOUR[l][0] for l in ls)
        txt = f"{r} = {comp}" if comp != r else f"{r} = {r}"
        if not reg[r].any():
            txt += "  (absent on this slice)"
        axt.plot([0.0, 0.055], [y, y], color=c, lw=1.4,
                 transform=axt.transAxes, clip_on=False)
        axt.text(0.085, y, txt, transform=axt.transAxes, va="center",
                 fontsize=S["small"])
        y -= 0.155
    axt.text(0.0, y - 0.02,
             "Regions are nested: WT contains TC, TC contains ET.\n"
             "RC is scored separately.",
             transform=axt.transAxes, va="top", fontsize=S["tiny"],
             color="#555555")
    paths["B"] = save(fig, "fig2_regions_candB")

    # ---- candidate C · single-column stack -----------------------------
    # For a single-column layout: the four regions stacked vertically at
    # column width, which survives journal scaling better than a wide row.
    fig, ax = plt.subplots(4, 1, figsize=(COL1 * 0.62, 4.6))
    fig.subplots_adjust(hspace=0.30, top=0.96, bottom=0.03)
    for k, (r, (ls, c)) in enumerate(REGION_DEF.items()):
        mri(ax[k], base, asp, None)
        seg_fill(ax[k], reg[r], c, asp, 0.44)
        seg_edge(ax[k], reg[r], c, asp, 0.85)
        ax[k].set_ylabel(r, fontsize=S["title"], labelpad=3)
        ax[k].set_xlabel(caption(r).replace("\n", "  "),
                         fontsize=S["tiny"], labelpad=2)
    scalebar(ax[0], mmx); letter(ax[0], "b")
    paths["C"] = save(fig, "fig2_regions_candC")

    present = [r for r in REGION_DEF if reg[r].any()]
    for cand in paths:
        record(figure="2", candidate=cand, panel="b", case=case,
               region="all four", selection_rule=FROZEN["main"][1],
               quantitative_value=FROZEN["main"][2], slice_rule=slice_rule,
               slice_index=z, plane="axial", array_axis=a,
               aspect=round(asp, 4), mm_per_px=round(mmx, 4),
               regions_on_slice=present,
               source_files=[d["ref_path"], d["seq_paths"]["t1c"]],
               spatial_validation="PASS",
               label_validation=d["checks"]["labels"], output=paths[cand])
    log("", f"  fig 2  same slice · regions present {present} · 3 candidates")


# ===========================================================================
#  FIGURE 3 · MAIN QUALITATIVE RESULT
#
#  The strongest qualitative figure: reference against prediction, with the
#  three error classes over the anatomy rather than on black.
# ===========================================================================

def figure3(d, case, a, slice_rule_base):
    gt, pred, sp = d["gt"], d["pred"], d["spacing"]
    if pred is None:
        log("", "  fig 3  STOPPED: no prediction for this case")
        return
    mmx, mmy, asp = geometry(a, sp)
    rows = [r for r in ("WT", "TC", "ET") if np.isin(gt, REGION_DEF[r][0]).any()]
    paths = {}

    def panels(reg):
        ls, col = REGION_DEF[reg]
        g, p = np.isin(gt, ls), np.isin(pred, ls)
        # the SAME slice for reference and prediction, by rule: maximum area
        # of their union, so neither side selects the slice
        z = slice_max_area(g | p, a)
        bx = bbox(np.take(g | p, z, axis=a), pad=14)
        return (np.take(window(d["vols"]["t1c"], g), z, axis=a)[bx],
                np.take(g, z, axis=a)[bx], np.take(p, z, axis=a)[bx],
                z, col)

    # ---- candidate A · one region, four columns, conservative ----------
    # WT only: the region the paper's headline numbers are reported on, at
    # four columns. Minimal and highly readable, at the cost of not showing
    # how the error changes between nested regions.
    g0, gm, pm, z0, col = panels("WT")
    fig, ax = plt.subplots(1, 4, figsize=(COL2, 1.95))
    fig.subplots_adjust(wspace=0.03, bottom=0.12)
    mri(ax[0], g0, asp, "T1c")
    mri(ax[1], g0, asp, "Reference")
    seg_fill(ax[1], gm, col, asp, 0.40); seg_edge(ax[1], gm, col, asp)
    mri(ax[2], g0, asp, "Prediction")
    seg_fill(ax[2], pm, col, asp, 0.40); seg_edge(ax[2], pm, col, asp)
    mri(ax[3], g0, asp, "Error")
    for m, k, al in ((gm & pm, "TP", 0.42), (gm & ~pm, "FN", 0.68),
                     (~gm & pm, "FP", 0.68)):
        seg_fill(ax[3], m, ERROR_COLOUR[k][0], asp, al)
        seg_edge(ax[3], m, ERROR_COLOUR[k][0], asp, 0.55)
    scalebar(ax[0], mmx); letter(ax[0], "c")
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERROR_COLOUR.values()],
               loc="lower center", ncol=3, frameon=False, fontsize=S["small"],
               bbox_to_anchor=(0.5, -0.02), handlelength=1.1,
               columnspacing=1.6)
    paths["A"] = save(fig, "fig3_result_candA")

    # ---- candidate B · all regions as rows, dense ----------------------
    # Every scored tumour region as a row. Denser, and it shows something
    # candidate A cannot: that the error changes character between regions --
    # in particular that ET is a thin rim, which is why a voxel-count filter
    # damages it most.
    # §22: the previous version left excessive vertical whitespace. The
    # panel height now follows the crop's own aspect instead of a fixed
    # multiple, and hspace is near zero because the rows share columns.
    fig, ax = plt.subplots(len(rows), 4, figsize=(COL2 * 0.92,
                                                  1.44 * len(rows) + 0.30),
                           squeeze=False)
    fig.subplots_adjust(wspace=0.025, hspace=0.02, bottom=0.085, top=0.955,
                        left=0.055, right=0.995)
    # §22: one crop for every row. Per-region crops gave each row a
    # different shape and the grid padded to the tallest, which is where the
    # vertical whitespace came from. A shared field of view also makes the
    # rows genuinely comparable rather than each zoomed to its own extent.
    allg = np.isin(gt, REGION_DEF["WT"][0]) | np.isin(pred, REGION_DEF["WT"][0])
    zs = {}
    shared_z = slice_max_area(allg, a)
    shared_box = bbox(np.take(allg, shared_z, axis=a), pad=14)
    for i, reg in enumerate(rows):
        ls_i, col = REGION_DEF[reg]
        g_i, p_i = np.isin(gt, ls_i), np.isin(pred, ls_i)
        # same slice and same crop for every row and every column
        z = shared_z
        zs[reg] = z
        b = np.take(window(d["vols"]["t1c"], g_i), z, axis=a)[shared_box]
        gm = np.take(g_i, z, axis=a)[shared_box]
        pm = np.take(p_i, z, axis=a)[shared_box]
        mri(ax[i][0], b, asp, "T1c" if i == 0 else None)
        mri(ax[i][1], b, asp, "Reference" if i == 0 else None)
        seg_fill(ax[i][1], gm, col, asp, 0.40); seg_edge(ax[i][1], gm, col, asp)
        mri(ax[i][2], b, asp, "Prediction" if i == 0 else None)
        seg_fill(ax[i][2], pm, col, asp, 0.40); seg_edge(ax[i][2], pm, col, asp)
        mri(ax[i][3], b, asp, "Error" if i == 0 else None)
        for m, k, al in ((gm & pm, "TP", 0.42), (gm & ~pm, "FN", 0.68),
                         (~gm & pm, "FP", 0.68)):
            seg_fill(ax[i][3], m, ERROR_COLOUR[k][0], asp, al)
            seg_edge(ax[i][3], m, ERROR_COLOUR[k][0], asp, 0.55)
        # §12: a compact unrotated row label rather than a rotated ylabel,
        # which consumes horizontal space the panels need
        ax[i][0].text(-0.055, 0.5, reg, transform=ax[i][0].transAxes,
                      fontsize=S["title"], fontweight="bold",
                      ha="right", va="center")
        # A shared slice keeps the rows comparable but can leave an inner
        # region with nothing on it. An unexplained blank row reads as a
        # segmentation failure, so it is labelled as what it is.
        if not (gm.any() or pm.any()):
            ax[i][1].text(0.5, 0.04, "absent on this slice",
                          transform=ax[i][1].transAxes, ha="center",
                          va="bottom", fontsize=S["tiny"], color="white",
                          path_effects=[pe.withStroke(linewidth=1.1,
                                                      foreground="#0a0a0a")])
        if i == 0:
            scalebar(ax[i][0], mmx); letter(ax[i][0], "c")
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERROR_COLOUR.values()],
               loc="lower center", ncol=3, frameon=False, fontsize=S["small"],
               bbox_to_anchor=(0.5, -0.012), handlelength=1.1,
               columnspacing=1.6)
    paths["B"] = save(fig, "fig3_result_candB")

    # ---- candidate C · error-emphasised, four columns ------------------
    # §10: the brief asks for an error-focused layout rather than a large
    # zoomed inset, which it allows only if the inset gives a genuine
    # scientific benefit. Here it did not: the disagreement in this case is
    # whole missed lesions rather than boundary detail, and at 20 mm across
    # they are already legible. So the emphasis is given by column width --
    # the error panel is wider than the other three -- and not by an inset
    # the reader has to track separately.
    g1, gm1, pm1, z1, col1 = panels("WT")
    fig, ax = plt.subplots(1, 4, figsize=(COL2, 2.0),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    fig.subplots_adjust(wspace=0.035, bottom=0.13)
    mri(ax[0], g1, asp, "T1c")
    mri(ax[1], g1, asp, "Reference")
    seg_fill(ax[1], gm1, col1, asp); seg_edge(ax[1], gm1, col1, asp)
    mri(ax[2], g1, asp, "Prediction")
    seg_fill(ax[2], pm1, col1, asp); seg_edge(ax[2], pm1, col1, asp)
    mri(ax[3], g1, asp, "Error")
    for m, k, al in ((gm1 & pm1, "TP", 0.42), (gm1 & ~pm1, "FN", 0.68),
                     (~gm1 & pm1, "FP", 0.68)):
        seg_fill(ax[3], m, ERROR_COLOUR[k][0], asp, al)
        seg_edge(ax[3], m, ERROR_COLOUR[k][0], asp, 0.55)
    scalebar(ax[0], mmx); letter(ax[0], "c")
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERROR_COLOUR.values()],
               loc="lower center", ncol=3, frameon=False, fontsize=S["small"],
               bbox_to_anchor=(0.5, -0.02), handlelength=1.1,
               columnspacing=1.6)
    paths["C"] = save(fig, "fig3_result_candC")

    rule = ("axial slice of maximum area of the union of reference and "
            "prediction; the same slice AND the same crop are used for every "
            "column, and in candidate B for every region row as well, so the "
            "rows are comparable and the grid needs no padding")
    for cand in paths:
        record(figure="3", candidate=cand, panel="c", case=case,
               region=rows if cand == "B" else "WT",
               selection_rule=FROZEN["main"][1],
               quantitative_value=FROZEN["main"][2], slice_rule=rule,
               slice_index=(zs if cand == "B" else z0),
               plane="axial", array_axis=a, aspect=round(asp, 4),
               mm_per_px=round(mmx, 4),
               source_files=[d["ref_path"], d["seq_paths"]["t1c"],
                             f"probabilities via ns.load_mean_prob, folds "
                             f"{tuple(ns.FOLDS)}"],
               spatial_validation=d["checks"].get("prediction", "PASS"),
               label_validation=d["checks"]["labels"], output=paths[cand])
    log("", f"  fig 3  regions {rows} · 3 candidates")


# ===========================================================================
#  FIGURE 4 · FAILURE AND BEHAVIOUR REGIMES
#
#  Four predefined regimes, not four arbitrary examples. Panel titles are
#  neutral: "small-lesion case", not "challenging case".
# ===========================================================================

def resolve_4c():
    """
    The false-positive component for panel (c), from the existing validated
    fp_components.csv. No new nearest-distance analysis is introduced merely
    to select a picture.

    The rule: the largest false-positive component on which the six rules do
    NOT unanimously agree. A component every rule removes shows nothing about
    disagreement, which is what the panel is for. If the component cannot be
    identified reproducibly the panel STOPS.
    """
    f = f"{W2}/fp_components.csv"
    if not os.path.exists(f):
        return None, "fp_components.csv not found"
    d = pd.read_csv(f)
    # is_fp may be a string too, and `== 1` on a string column silently
    # yields an EMPTY frame rather than raising -- which would report "no
    # false-positive rows" on a file full of them. Coerce, and say so if
    # nothing survives.
    is_fp = pd.to_numeric(d["is_fp"], errors="coerce")
    if is_fp.isna().all():
        is_fp = d["is_fp"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "t", "yes", "y"}).astype(int)
    fp = d[is_fp == 1].copy()
    if not len(fp):
        return None, (f"no rows with is_fp set, out of {len(d)}; check the "
                      f"column's encoding (values seen: "
                      f"{sorted(set(d['is_fp'].astype(str)))[:4]})")
    rules = [c for c in ("size@20", "size@50", "entropy@0.45",
                         "entropy@0.50", "entropy@0.56", "confidence@0.45")
             if c in fp.columns]
    if not len(fp) or not rules:
        return None, "no false-positive rows or no rule columns"
    # The rule columns are not necessarily numeric: in the real file they
    # arrive as strings. Summing those concatenates rather than counts, and
    # the comparison then raises. Coerce explicitly, accepting the several
    # encodings a boolean column can have, and REFUSE to guess: a value that
    # is none of these stops the panel rather than being treated as False,
    # because silently reading an unknown code as "retained" would put the
    # wrong component in the figure.
    TRUE = {"1", "true", "t", "yes", "y", "removed", "1.0"}
    FALSE = {"0", "false", "f", "no", "n", "retained", "0.0", "", "nan"}
    dec = pd.DataFrame(index=fp.index)
    for r in rules:
        col = fp[r]
        if pd.api.types.is_numeric_dtype(col):
            dec[r] = col.fillna(0).astype(int).clip(0, 1)
            continue
        norm = col.astype(str).str.strip().str.lower()
        unknown = sorted(set(norm) - TRUE - FALSE)
        if unknown:
            return None, (f"column {r} holds values this script will not "
                          f"interpret: {unknown[:4]}. Fix the encoding or "
                          f"state it; guessing would risk drawing the wrong "
                          f"component.")
        dec[r] = norm.isin(TRUE).astype(int)
    fp["n_removed"] = dec.sum(axis=1)
    for r in rules:
        fp[r] = dec[r]
    split = fp[(fp.n_removed > 0) & (fp.n_removed < len(rules))]
    if not len(split):
        return None, "no component on which the rules disagree"
    vol = pd.to_numeric(split["volume_mm3"], errors="coerce")
    if vol.isna().all():
        return None, "volume_mm3 is not numeric in any row"
    pick = split.loc[vol.idxmax()]
    return (pick, rules), None


def regime_panel(ax, case, target_mask_fn, title, note, asp_out):
    """One regime panel, validated then drawn. Returns its manifest entry."""
    try:
        d = load_validated(case, need_pred=True)
    except Exception as e:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\nSTOPPED\n{type(e).__name__}",
                ha="center", va="center", fontsize=S["small"], color="#b00")
        return {"status": f"STOPPED: {type(e).__name__}: {e}"}
    a = d["planes"]["axial"]
    mmx, mmy, asp = geometry(a, d["spacing"])
    g, p, z, extra = target_mask_fn(d, a)
    if z is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\nSTOPPED\nno target on any slice",
                ha="center", va="center", fontsize=S["small"], color="#b00")
        return {"status": "STOPPED: empty target"}
    bx = bbox(np.take(g | p, z, axis=a), pad=extra)
    base = np.take(window(d["vols"]["t1c"], g), z, axis=a)[bx]
    gm, pm = np.take(g, z, axis=a)[bx], np.take(p, z, axis=a)[bx]
    mri(ax, base, asp, title)
    for m, k, al in ((gm & pm, "TP", 0.42), (gm & ~pm, "FN", 0.70),
                     (~gm & pm, "FP", 0.70)):
        seg_fill(ax, m, ERROR_COLOUR[k][0], asp, al)
        seg_edge(ax, m, ERROR_COLOUR[k][0], asp, 0.55)
    ax.set_xlabel(note, fontsize=S["tiny"], labelpad=2)
    scalebar(ax, mmx, mm=10)
    asp_out.append((mmx, asp))
    return {"status": "PASS", "slice_index": z, "aspect": round(asp, 4),
            "mm_per_px": round(mmx, 4), "checks": d["checks"]}


def figure4():
    head("FIGURE 4 · REGIMES")
    entries = {}

    def wt_masks(d, a):
        g = np.isin(d["gt"], REGION_DEF["WT"][0])
        p = np.isin(d["pred"], REGION_DEF["WT"][0])
        return g, p, slice_max_area(g | p, a), 16

    def small_masks(d, a):
        """Centre on the smallest scored lesion, not the largest structure."""
        g = np.isin(d["gt"], REGION_DEF["WT"][0])
        p = np.isin(d["pred"], REGION_DEF["WT"][0])
        lab, n = ndimage.label(g)
        if n:
            sizes = ndimage.sum(g, lab, range(1, n + 1))
            small = int(np.argmin(sizes)) + 1
            tgt = lab == small
            return g, p, slice_at_centroid(tgt, a), 22
        return g, p, slice_max_area(g | p, a), 22

    def missed_masks(d, a):
        """Centre on the largest reference component the prediction misses."""
        g = np.isin(d["gt"], REGION_DEF["WT"][0])
        p = np.isin(d["pred"], REGION_DEF["WT"][0])
        lab, n = ndimage.label(g & ~p)
        if n:
            sizes = ndimage.sum(g & ~p, lab, range(1, n + 1))
            tgt = lab == int(np.argmax(sizes)) + 1
            return g, p, slice_at_centroid(tgt, a), 18
        return g, p, slice_max_area(g | p, a), 18

    spec = [
        ("a", "Small-lesion case", FROZEN["small"], small_masks,
         "smallest scored lesion, 4.125 mm3"),
        ("b", "High lesion multiplicity", FROZEN["multi"], wt_masks,
         "18 lesions, 75th percentile of count"),
        ("d", "Large missed lesion", FROZEN["missed"], missed_masks,
         "1209 mm3, missed at baseline"),
    ]
    asp_out = []

    # ---- candidate A · 2x2, conservative -------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(COL2 * 0.66, 3.9))
    fig.subplots_adjust(wspace=0.05, hspace=0.24, bottom=0.085)
    flat = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
    for k, (pn, title, frozen, fn, note) in enumerate(spec):
        slot = flat[0] if pn == "a" else (flat[1] if pn == "b" else flat[3])
        e = regime_panel(slot, frozen[0], fn, f"({pn}) {title}", note, asp_out)
        entries[pn] = dict(case=frozen[0], selection_rule=frozen[1],
                           quantitative_value=frozen[2], **e)
    # panel (c) from the component table
    got, why = resolve_4c()
    if got is None:
        flat[2].axis("off")
        flat[2].text(0.5, 0.5, f"(c) False-positive component\nSTOPPED\n{why}",
                     ha="center", va="center", fontsize=S["small"],
                     color="#b00")
        entries["c"] = {"status": f"STOPPED: {why}"}
        log("", f"  panel c STOPPED: {why}")
    else:
        pick, rules = got
        try:
            dd = load_validated(pick.case, need_pred=True)
            a = dd["planes"]["axial"]
            mmx, mmy, asp = geometry(a, dd["spacing"])
            pmask = np.isin(dd["pred"], REGION_DEF[pick.region][0])
            lab, n = ndimage.label(pmask)      # suppression connectivity
            sel = lab == int(pick.supp_component_id)
            if not sel.any():
                raise ValueError("component id not present in the labelling")
            z = slice_at_centroid(sel, a)
            bx = bbox(np.take(sel, z, axis=a), pad=24)
            base = np.take(window(dd["vols"]["t1c"], sel), z, axis=a)[bx]
            s2 = np.take(sel, z, axis=a)[bx]
            mri(flat[2], base, asp, "(c) False-positive component")
            seg_fill(flat[2], s2, ERROR_COLOUR["FP"][0], asp, 0.55)
            seg_edge(flat[2], s2, ERROR_COLOUR["FP"][0], asp, 0.85)
            flat[2].set_xlabel(
                f"{pick.volume_mm3:.0f} mm3 · removed by "
                f"{int(pick.n_removed)}/{len(rules)} rules",
                fontsize=S["tiny"], labelpad=2)
            scalebar(flat[2], mmx, mm=10)
            entries["c"] = {"status": "PASS", "case": pick.case,
                            "region": pick.region,
                            "component": int(pick.supp_component_id),
                            "volume_mm3": float(pick.volume_mm3),
                            "rules_removing": int(pick.n_removed),
                            "selection_rule": "largest FP component on which "
                                              "the six rules disagree",
                            "slice_index": z, "aspect": round(asp, 4)}
            log("", f"  panel c {pick.case} {pick.region} comp "
                    f"{int(pick.supp_component_id)} · "
                    f"{pick.volume_mm3:.0f} mm3")
        except Exception as e:
            flat[2].axis("off")
            flat[2].text(0.5, 0.5, f"(c) False-positive component\nSTOPPED\n"
                                   f"{type(e).__name__}",
                         ha="center", va="center", fontsize=S["small"],
                         color="#b00")
            entries["c"] = {"status": f"STOPPED: {type(e).__name__}: {e}"}
            log("", f"  panel c STOPPED: {type(e).__name__}: {e}")

    letter(flat[0], "d")
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERROR_COLOUR.values()],
               loc="lower center", ncol=3, frameon=False, fontsize=S["small"],
               bbox_to_anchor=(0.5, -0.015), handlelength=1.1,
               columnspacing=1.6)
    pA = save(fig, "fig4_regimes_candA")

    # ---- candidate B · one row of four, for a wide slot ----------------
    fig, ax = plt.subplots(1, 4, figsize=(COL2, 2.1))
    fig.subplots_adjust(wspace=0.04, bottom=0.16)
    for k, (pn, title, frozen, fn, note) in enumerate(spec):
        slot = ax[0] if pn == "a" else (ax[1] if pn == "b" else ax[3])
        regime_panel(slot, frozen[0], fn, f"({pn}) {title}", note, asp_out)
    if got is not None:
        try:
            mri(ax[2], base, asp, "(c) False-positive component")
            seg_fill(ax[2], s2, ERROR_COLOUR["FP"][0], asp, 0.55)
            seg_edge(ax[2], s2, ERROR_COLOUR["FP"][0], asp, 0.85)
            ax[2].set_xlabel(f"{pick.volume_mm3:.0f} mm3 · removed by "
                             f"{int(pick.n_removed)}/{len(rules)} rules",
                             fontsize=S["tiny"], labelpad=2)
        except Exception:
            ax[2].axis("off")
    else:
        ax[2].axis("off")
    letter(ax[0], "d")
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERROR_COLOUR.values()],
               loc="lower center", ncol=3, frameon=False, fontsize=S["small"],
               bbox_to_anchor=(0.5, -0.02), handlelength=1.1,
               columnspacing=1.6)
    pB = save(fig, "fig4_regimes_candB")

    for cand, path in (("A", pA), ("B", pB)):
        for pn, e in entries.items():
            record(figure="4", candidate=cand, panel=pn,
                   slice_rule=("lesion-centroid slice for the target "
                               "structure; maximum-area slice where no single "
                               "target is defined"),
                   plane="axial", output=path, **e)
    log("", f"  fig 4  panels {sorted(entries)} · 2 candidates")
    return entries


# ===========================================================================
#  SUPPLEMENTARY S1 · ORTHOGONAL VIEWS
#
#  The same case and the same segmentation in three planes, to show spatial
#  extent. Supplementary rather than main text: three views of one finding is
#  presentation, and the axial panel in figure 3 already carries the point.
# ===========================================================================

def supplementary_views(d, case):
    gt, pred, sp = d["gt"], d["pred"], d["spacing"]
    if pred is None:
        log("", "  S1     STOPPED: no prediction")
        return
    ls, col = REGION_DEF["WT"]
    g, p = np.isin(gt, ls), np.isin(pred, ls)

    # A reproducible anatomical reference point: the centroid of the union.
    # All three planes pass through the SAME point, so they are three views of
    # one place rather than three unrelated slices.
    if not (g | p).any():
        log("", "  S1     STOPPED: empty region")
        return
    centre = np.argwhere(g | p).mean(0).round().astype(int)
    rule = ("all three planes pass through the centroid of the union of "
            f"reference and prediction, voxel {tuple(int(c) for c in centre)}")

    info = {}
    for plane in ("axial", "coronal", "sagittal"):
        a = d["planes"][plane]
        idx = int(centre[a])
        bx = bbox(np.take(g | p, idx, axis=a), pad=14)
        mmx, mmy, asp = geometry(a, sp)
        h = bx[0].stop - bx[0].start
        info[plane] = (a, idx, bx, mmx, asp, h * mmx)

    fig, ax = plt.subplots(
        1, 3, figsize=(COL2 * 0.86, 2.5),
        gridspec_kw={"width_ratios": [info[p][5] for p in
                                      ("axial", "coronal", "sagittal")]})
    fig.subplots_adjust(wspace=0.05, bottom=0.13)
    for k, plane in enumerate(("axial", "coronal", "sagittal")):
        a, idx, bx, mmx, asp, _ = info[plane]
        base = np.take(window(d["vols"]["t1c"], g), idx, axis=a)[bx]
        gm, pm = np.take(g, idx, axis=a)[bx], np.take(p, idx, axis=a)[bx]
        mri(ax[k], base, asp, plane)
        for m, key, al in ((gm & pm, "TP", 0.42), (gm & ~pm, "FN", 0.70),
                           (~gm & pm, "FP", 0.70)):
            seg_fill(ax[k], m, ERROR_COLOUR[key][0], asp, al)
            seg_edge(ax[k], m, ERROR_COLOUR[key][0], asp, 0.55)
        if k == 0:
            scalebar(ax[k], mmx); letter(ax[k], "S1")
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERROR_COLOUR.values()],
               loc="lower center", ncol=3, frameon=False, fontsize=S["small"],
               bbox_to_anchor=(0.5, -0.018), handlelength=1.1,
               columnspacing=1.6)
    path = save(fig, "figS1_orthogonal")
    record(figure="S1", candidate="single", panel="S1", case=case,
           region="WT", selection_rule=FROZEN["main"][1],
           quantitative_value=FROZEN["main"][2], slice_rule=rule,
           slice_index={p: info[p][1] for p in info},
           plane="axial, coronal, sagittal",
           aspect={p: round(info[p][4], 4) for p in info},
           source_files=[d["ref_path"], d["seq_paths"]["t1c"]],
           spatial_validation="PASS", label_validation=d["checks"]["labels"],
           output=path)
    log("", f"  S1     three planes through {tuple(int(c) for c in centre)} · "
            f"aspects {[round(info[p][4], 3) for p in info]}")


# ===========================================================================
#  CANDIDATE COMPARISON
#
#  Scored against the brief's own criteria. The scores are judgements, stated
#  as such, and the recommendation is not the prettiest candidate.
# ===========================================================================

COMPARISON = {
 "1": {
  "A": dict(clarity=9, anatomy=8, error=0, labels=9, hierarchy=9,
            density=6, repro=10, journal=9, misread=9, overall=9,
            note="one row, sequences then labels. Reads in a second and "
                 "matches how the Data section is written."),
  "B": dict(clarity=8, anatomy=7, error=0, labels=9, hierarchy=7,
            density=9, repro=10, journal=7, misread=7, overall=7,
            note="2x3 with labels shown alone and on T1c. More information, "
                 "but the labels-alone panel invites reading the colours as "
                 "a result rather than an annotation."),
  "C": dict(clarity=8, anatomy=10, error=0, labels=6, hierarchy=8,
            density=6, repro=10, journal=8, misread=7, overall=8,
            note="wide field, outlines only. Best anatomical context; the "
                 "small compartments are hard to see, which matters because "
                 "this figure exists to show them."),
  "recommended": "A",
  "why": "This is a Data-section figure whose only job is to show what a "
         "case contains. A wins on the criterion that matters most here -- "
         "label interpretability with no risk of being read as a result -- "
         "and C's better anatomy comes at the cost of the compartments the "
         "figure is for. B is the strongest supplementary version.",
 },
 "2": {
  "A": dict(clarity=9, anatomy=7, error=0, labels=10, hierarchy=9,
            density=8, repro=10, journal=9, misread=9, overall=9,
            note="labels row above regions row, with the composition under "
                 "each panel. The two colour systems are separated by the "
                 "two row headings."),
  "B": dict(clarity=8, anatomy=8, error=0, labels=8, hierarchy=7,
            density=9, repro=10, journal=8, misread=6, overall=7,
            note="nesting on one panel with the compositions beside it. "
                 "Compact and makes the set relation explicit, but three "
                 "overlapping contours on one slice can be misread as three "
                 "separate findings."),
  "C": dict(clarity=8, anatomy=7, error=0, labels=9, hierarchy=8,
            density=5, repro=10, journal=8, misread=9, overall=7,
            note="single-column stack. Survives journal scaling best; uses "
                 "four times the vertical space for the same content."),
  "recommended": "A",
  "why": "The figure's whole purpose is that WT, TC and ET are unions of "
         "labels, and A states each union in words beneath the panel that "
         "shows it. B is more elegant and more easily misread. C is the one "
         "to use if the journal is strictly single-column.",
 },
 "3": {
  "A": dict(clarity=9, anatomy=8, error=9, labels=8, hierarchy=9,
            density=6, repro=10, journal=9, misread=9, overall=9,
            note="WT only, four columns. The region the headline numbers "
                 "are reported on, at a size where the error classes are "
                 "unmistakable."),
  "B": dict(clarity=9, anatomy=7, error=9, labels=9, hierarchy=8,
            density=10, repro=10, journal=8, misread=8, overall=9,
            note="all three tumour regions as rows. Shows something A "
                 "cannot: that ET is a thin rim, which is why a voxel-count "
                 "filter damages it most. Three times the height."),
  "C": dict(clarity=9, anatomy=8, error=10, labels=8, hierarchy=9,
            density=6, repro=10, journal=9, misread=9, overall=8,
            note="error-emphasised: the same four columns with the error "
                 "panel given 35% more width. Emphasis without a second "
                 "thing to track, at the cost of showing only WT."),
  "recommended": "B",
  "why": "B earns its space. The ET row visually explains a number in the "
         "results -- size@50 costs 0.1421 in ET against 0.0764 in WT, and "
         "the rim morphology is why -- so the figure does work the tables "
         "cannot. A is the safer choice if space is tight and is equally "
         "defensible; C is the better single-row choice if the error panel needs the emphasis.",
 },
 "4": {
  "A": dict(clarity=9, anatomy=8, error=9, labels=8, hierarchy=9,
            density=8, repro=10, journal=9, misread=8, overall=9,
            note="2x2. Each regime gets enough area for its phenomenon to "
                 "be visible, and the block fits a single column."),
  "B": dict(clarity=8, anatomy=7, error=8, labels=8, hierarchy=8,
            density=9, repro=10, journal=8, misread=8, overall=8,
            note="one row of four. Fits a wide slot and reads left to "
                 "right, but each panel is small and the small-lesion panel "
                 "suffers most -- which is the panel that most needs area."),
  "recommended": "A",
  "why": "The small-lesion regime is the hardest to see and the most "
         "informative, and A gives it four times the area B does. The "
         "trade-off runs the wrong way in B for exactly the panel that "
         "carries the point.",
 },
}


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    print("=" * 78)
    print(" QUALITATIVE VISUALISATION PIPELINE · BraTS-METS 2026")
    print(f" {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 78)

    inv = audit()

    head("2 · SPATIAL AND LABEL VALIDATION")
    main_case = FROZEN["main"][0]
    try:
        d = load_validated(main_case, need_pred=True)
    except Exception as e:
        log("", f"  {main_case}: {type(e).__name__}: {e}")
        log("", """
  STOPPED. No case is substituted and nothing is transposed blindly. Fix the
  cause or accept that the affected figures are not produced.""")
        write_outputs(inv, status="FAIL")
        return
    for k, v in d["checks"].items():
        log("", f"  {k:12s} {v}")
    ip, th = min(d["spacing"]), max(d["spacing"])
    log("", f"""
  The voxel is anisotropic by {th/ip:.3f}x. Through-plane panels are drawn at
  aspect {ip/th:.3f}; at 1.0 they would be squashed by {100*(1-ip/th):.0f}%.
  Planes come from the affine: {d['planes']}""")

    head("3 · FIGURES")
    z, box, a, asp, mmx, rule = figure1(d, main_case)
    figure2(d, main_case, z, box, a, asp, mmx, rule)
    figure3(d, main_case, a, rule)
    figure4()
    supplementary_views(d, main_case)

    head("4 · CANDIDATE COMPARISON")
    crit = ["clarity", "anatomy", "error", "labels", "hierarchy",
            "density", "repro", "journal", "misread", "overall"]
    for fig_id, block in COMPARISON.items():
        log("", f"\n  FIGURE {fig_id}")
        log("", "    cand  " + " ".join(f"{c[:5]:>6s}" for c in crit))
        for cand in [k for k in block if len(k) == 1]:
            row = block[cand]
            log("", f"    {cand:<5s} " +
                    " ".join(f"{row[c]:>6d}" for c in crit))
        for cand in [k for k in block if len(k) == 1]:
            log("", f"      {cand}: {block[cand]['note']}")
        log("", f"\n    RECOMMENDED CANDIDATE: {block['recommended']}")
        log("", f"    WHY: {block['why']}")

    head("5 · FIGURE 5 · NOT PRODUCED")
    log("", """
  RECONSTRUCTION VALIDATION FAILED.

  The suppressed prediction masks were not retained: the arms were scored and
  their masks discarded. Nine reconstruction attempts were made against the
  retained counts under the official evaluator, and the closest reached 19 of
  24 case-region cells. The discrepancies, in the order they were diagnosed
  and fixed:

    component definition   evaluator 26-connectivity with dilation used
                           where the suppression rule uses scipy's default
                           6-connectivity                            13/24
    composition order      suppression applied after the nested label was
                           composed rather than per region before      19/24
    evaluator matching     matched predicted components counted instead of
                           matched reference lesions
    evaluator identity     a project diagnostic tool selected by filename
                           search instead of metrics_MET
    evaluator loading      metrics_MET imported directly, skipping the clone,
                           the dependency installs, the NumPy 2.x pass, the
                           sys.path entry and the chdir
    challenge_name         omitted; the evaluator's behaviour depends on it
    scratch path           a tag containing a dot truncated the evaluator's
                           temporary directory name

  The final attempt called ns.official_score, the arm's own path, and still
  stopped on the evaluator's scratch layout. At that point the remaining
  candidates were about the inputs -- the archived probability copy, and
  whether the retained counts came from an identical evaluator state -- not
  about the reconstruction logic.

  FIGURE 5 IS NOT DRAWN. An unvalidated reconstruction is an illustration of
  the reconstruction code, not evidence from the experiment, and it would
  look like evidence. The suppression results are reported numerically in
  full: four tables, the headroom comparison, and the eligibility analysis.

  This is the same judgement that declined a second model and a second
  dataset.""")

    write_outputs(inv, status="PASS")


def write_outputs(inv, status):
    man = {"generated": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
           "pipeline": os.path.basename(__file__) if "__file__" in dir() else
                       "qualitative_pipeline.py",
           "status": status,
           "cohort": {"reference_cases": inv["gt_count"],
                      "phase_b_cohort": inv["cohort"],
                      "clean_stratum": inv["clean"],
                      "note": "probability maps retained for the Phase B "
                              "cohort only; figures needing a prediction are "
                              "restricted to it"},
           "colour_systems": {
               "native_labels": {n: c for n, c in LABEL_COLOUR.values()},
               "regions": {r: c for r, (_, c) in REGION_DEF.items()},
               "errors": {l: c for c, l in ERROR_COLOUR.values()},
               "note": "two separate systems; no hue is shared between a "
                       "reference class and a prediction error"},
           "transpose": f"array = file.transpose{INV} (argsort of NPZ_TO_GT)",
           "display": {"mri_interpolation": "bicubic",
                       "segmentation_interpolation": "nearest",
                       "windowing": "percentile 0.5-99.5, fixed across all "
                                    "figures",
                       "dpi": S["dpi"], "pdf_fonttype": 42},
           "figure5": "NOT PRODUCED: reconstruction validation failed",
           "panels": MANIFEST}
    with open(f"{OUT}/figure_manifest.json", "w") as f:
        json.dump(man, f, indent=2, default=str)
    with open(f"{OUT}/validation_report.txt", "w") as f:
        f.write("\n".join(str(r) for r in REPORT))
    print(f"""
  written to {OUT}

    figure_manifest.json    every panel: case, rule, value, slice rule,
                            slice index, plane, aspect, source files,
                            spatial and label validation
    validation_report.txt   this transcript
    fig1_dataset_cand{{A,B,C}}     fig2_regions_cand{{A,B,C}}
    fig3_result_cand{{A,B,C}}      fig4_regimes_cand{{A,B}}
    figS1_orthogonal

  RECOMMENDED: fig 1 A · fig 2 A · fig 3 B · fig 4 A · S1 supplementary
  Figure 5 is not produced, and the reason is in section 5 above.""")


if __name__ == "__main__":
    main()
