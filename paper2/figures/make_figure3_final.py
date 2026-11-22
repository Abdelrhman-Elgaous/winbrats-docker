# -*- coding: utf-8 -*-
"""
==============================================================================
 FIGURE 3 · MAIN QUALITATIVE RESULT · final
==============================================================================
 Candidate B, refined. Three tumour regions as rows, four columns each:

     T1c  |  Reference  |  Prediction  |  Error

 WHY B RATHER THAN A OR C
 ------------------------
 B earns its space because the ET row explains a number the tables only
 state. The ET reference is a thin enhancing rim, not a mass, and a rim has a
 large surface relative to its volume -- which is why a voxel-count filter
 damages that region hardest:

     size@50:   ET  -0.1421      WT  -0.0764

 A is the same story on WT alone: elegant, smaller, and it explains nothing.
 C enlarged the error panel, which puts it at a DIFFERENT SCALE from the
 three beside it, so a reader compares sizes that are not comparable. That is
 a distortion in a results figure, not a question of space.

 REFINEMENTS APPLIED, AND THE LIMITS THEY RESPECT
 ------------------------------------------------
   error clarity   TP is drawn first, without a contour, at alpha 0.30; FN
                   and FP follow with contours at 0.72. TP covers by far the
                   largest area, and its contours were competing with the
                   very classes the panel exists to show. No panel is
                   enlarged, and no arrows, boxes or in-image labels are
                   added -- the only levers used are drawing order and alpha.
   row labels      exactly centred on each row, one size and weight, closer
                   to the panels
   vertical space  rows tightened; one shared crop so the three rows are the
                   same shape and the grid needs no padding
   panel label     "(c)" in the same position, size and weight as "(a)" in
                   figure 1 and "(b)" in figure 2

 WHAT IS HELD CONSTANT, DELIBERATELY
 -----------------------------------
   one windowing for every row, so the rows are visually comparable
   one colour per region for BOTH reference and prediction -- the column
   heading says which is which, and recolouring the prediction would imply
   it is a different kind of thing
   reference and prediction at identical alpha and line width, so neither
   side of the comparison is visually favoured
   one slice and one crop for all twelve panels

 CASE SELECTION IS FROZEN
 ------------------------
   BraTS-MET-00629-000, the median WT lesion-wise Dice of the clean stratum
   (0.6517, rank 78 of 155). Same case and slice as figures 1 and 2.

 Planes come from the affine and panel aspect from the header zooms: the
 voxel is 1.172 x 1.172 x 1.5 mm.
==============================================================================
"""

import os
import sys
import json
import glob

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
GT_DIR = ns.GT_DIR
OUT = f"{DRIVE}/paper2/figures/qualitative"

CASE = "BraTS-MET-00629-000"
RULE = "median WT lesion-wise Dice of the clean stratum"
VALUE = "WT Dice 0.6517, rank 78/155"

REGION_DEF = {"WT": ((1, 2, 3), "#00a8b5"), "TC": ((1, 3), "#b845c9"),
              "ET": ((3,), "#2f6fe0")}
ERROR_COLOUR = {"TP": ("#1f6fb2", "True positive"),
                "FN": ("#d1342c", "False negative"),
                "FP": ("#f0921f", "False positive")}
CROP_PAD = 14

S = {"dpi": 600, "title": 7.2, "small": 6.0, "tiny": 5.2,
     "hair": 0.45, "panel": 7.0}
SEG_ALPHA, SEG_LW = 0.40, 0.70
TP_ALPHA, ERR_ALPHA = 0.30, 0.72

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": S["title"], "savefig.dpi": S["dpi"], "figure.dpi": S["dpi"],
    "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.linewidth": S["hair"], "axes.edgecolor": "#333333",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INV = tuple(np.argsort(ns.NPZ_TO_GT))
os.makedirs(OUT, exist_ok=True)


def plane_axes(affine):
    codes = nib.aff2axcodes(affine)
    fa = {}
    for i, c in enumerate(codes):
        if c in "SI":
            fa["axial"] = i
        elif c in "AP":
            fa["coronal"] = i
        elif c in "RL":
            fa["sagittal"] = i
    to_array = {f: k for k, f in enumerate(INV)}
    return {p: to_array[a] for p, a in fa.items()}, codes


def load_and_validate(case):
    """Validate before drawing. A failure stops the figure: no case is
    substituted and nothing is transposed blindly."""
    ref_path = f"{GT_DIR}/{case}.nii.gz"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(ref_path)
    ref = nib.load(ref_path)
    gt_file = np.asanyarray(ref.dataobj)
    hit = (glob.glob(f"{DRIVE}/**/{case}-t1c.nii.gz", recursive=True)
           or glob.glob(f"{DRIVE}/**/{case}_t1c.nii.gz", recursive=True))
    if not hit:
        raise FileNotFoundError(f"{case}: t1c not found")
    img = nib.load(hit[0])
    if img.shape != ref.shape:
        raise ValueError(f"{case}: t1c shape {img.shape} != {ref.shape}")
    if not np.allclose(img.affine, ref.affine, atol=1e-3):
        raise ValueError(f"{case}: t1c affine differs from the reference")

    prob = ns.load_mean_prob(case)           # raises if the map is absent
    pred = ns.build_nested_label(prob, suppressor=None)
    gt = gt_file.transpose(INV)
    if pred.shape != gt.shape:
        raise ValueError(f"{case}: prediction {pred.shape} != {gt.shape}")

    planes, codes = plane_axes(ref.affine)
    zooms = tuple(float(v) for v in ref.header.get_zooms()[:3])
    spacing = tuple(zooms[INV[k]] for k in range(3))
    return {"gt": gt, "pred": pred,
            "t1c": np.asanyarray(img.dataobj).transpose(INV),
            "planes": planes, "codes": codes, "spacing": spacing,
            "t1c_path": hit[0], "ref_path": ref_path}


def window(vol, extra=None, lo=0.5, hi=99.5):
    """ONE windowing for every row. Percentiles, not absolute limits: the
    release is not intensity-normalised and mean non-zero intensity varies by
    up to 2,401-fold across cases."""
    ref = vol[vol > 0]
    if extra is not None and extra.any():
        ref = np.concatenate([ref, vol[extra]])
    if not ref.size:
        return vol
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((vol - a) / max(b - a, 1e-6), 0, 1)


def mri(ax, img2d, aspect, title=None):
    ax.imshow(np.rot90(img2d), cmap="gray", vmin=0, vmax=1,
              interpolation="bicubic", aspect=aspect, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(S["hair"]); sp.set_color("#333333")
    if title:
        ax.set_title(title, fontsize=S["title"], pad=2.0)


def fill(ax, mask2d, colour, aspect, alpha):
    """Nearest-neighbour always: interpolating a mask invents partial
    membership that does not exist."""
    if not mask2d.any():
        return
    rgba = np.zeros(mask2d.shape + (4,))
    rgba[mask2d.astype(bool)] = to_rgba(colour, alpha)
    ax.imshow(np.rot90(rgba), interpolation="nearest", aspect=aspect,
              rasterized=True)


def edge(ax, mask2d, colour, aspect, lw=SEG_LW):
    if not mask2d.any():
        return
    ax.contour(np.rot90(mask2d.astype(float)), levels=[0.5], colors=[colour],
               linewidths=lw, antialiased=True)
    ax.set_aspect(aspect)


def scalebar(ax, mm_per_px, avoid=None, mm=20):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w = mm / mm_per_px
    if w > 0.55 * abs(x1 - x0):
        mm = 10
        w = mm / mm_per_px
    side = "right"
    if avoid is not None and avoid.any():
        disp = np.rot90(avoid.astype(bool))
        band = disp[int(0.75 * disp.shape[0]):, :]
        if band.size:
            h = band.shape[1] // 2
            side = "left" if band[:, h:].sum() > band[:, :h].sum() else "right"
    xl = (min(x0, x1) + 0.045 * abs(x1 - x0)) if side == "left" else \
         (max(x0, x1) - 0.045 * abs(x1 - x0) - w)
    yb = max(y0, y1) - 0.055 * abs(y1 - y0)
    stroke = [pe.withStroke(linewidth=1.0, foreground="#0a0a0a")]
    bar = Rectangle((xl, yb), w, 0.014 * abs(y1 - y0), facecolor="white",
                    edgecolor="none", zorder=8)
    bar.set_path_effects(stroke)
    ax.add_patch(bar)
    t = ax.text(xl + w / 2, yb + 0.022 * abs(y1 - y0), f"{mm} mm",
                color="white", fontsize=S["tiny"], ha="center", va="bottom",
                zorder=8)
    t.set_path_effects(stroke)


def main():
    print("=" * 74)
    print(" FIGURE 3 · MAIN QUALITATIVE RESULT · candidate B, refined")
    print("=" * 74)
    d = load_and_validate(CASE)
    gt, pred, sp = d["gt"], d["pred"], d["spacing"]
    a = d["planes"]["axial"]

    rows = [r for r in ("WT", "TC", "ET")
            if np.isin(gt, REGION_DEF[r][0]).any()]
    if not rows:
        print("  STOPPED: no tumour region in the reference")
        return

    # ONE slice and ONE crop for all twelve panels. Per-region crops give each
    # row a different shape, the grid pads to the tallest -- which is where
    # the whitespace came from -- and the rows stop being comparable because
    # each is zoomed to its own extent.
    union = (np.isin(gt, REGION_DEF["WT"][0])
             | np.isin(pred, REGION_DEF["WT"][0]))
    other = tuple(i for i in range(3) if i != a)
    z = int(np.argmax(union.sum(axis=other)))
    sl = np.take(union, z, axis=a)
    idx = np.argwhere(sl)
    lo = np.maximum(idx.min(0) - CROP_PAD, 0)
    hi = np.minimum(idx.max(0) + CROP_PAD + 1, sl.shape)
    box = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))

    rem = [i for i in range(3) if i != a]
    mm_x = sp[rem[0]]
    aspect = sp[rem[1]] / sp[rem[0]]
    slice_rule = ("axial slice of maximum area of the union of the WT "
                  "reference and prediction; the same slice and the same "
                  "crop are used for every panel and every row")

    print(f"""
  case            {CASE}
  selection rule  {RULE} ({VALUE})
  rows            {rows}
  slice rule      {slice_rule}
  slice index     {z}   (array axis {a}, affine {d['codes']})
  panel aspect    {aspect:.4f}
  crop            shared, pad {CROP_PAD} voxels = {CROP_PAD * mm_x:.0f} mm
""")

    fig, ax = plt.subplots(len(rows), 4,
                           figsize=(7.0, 1.34 * len(rows) + 0.40),
                           squeeze=False)
    # bottom reserves the key strip. The single key was anchored at -0.008,
    # outside the axes area, and the tight bounding box then kept the empty
    # band between the panels and the key; reserving the strip in the layout
    # puts the keys directly beneath the panels instead.
    fig.subplots_adjust(wspace=0.022, hspace=0.012, bottom=0.105,
                        top=0.952, left=0.048, right=0.996)

    base = np.take(window(d["t1c"], union), z, axis=a)[box]
    for i, reg in enumerate(rows):
        labs, col = REGION_DEF[reg]
        g = np.take(np.isin(gt, labs), z, axis=a)[box]
        p = np.take(np.isin(pred, labs), z, axis=a)[box]

        mri(ax[i][0], base, aspect, "T1c" if i == 0 else None)
        mri(ax[i][1], base, aspect, "Reference" if i == 0 else None)
        fill(ax[i][1], g, col, aspect, SEG_ALPHA)
        edge(ax[i][1], g, col, aspect)
        mri(ax[i][2], base, aspect, "Prediction" if i == 0 else None)
        fill(ax[i][2], p, col, aspect, SEG_ALPHA)
        edge(ax[i][2], p, col, aspect)

        # Error panel. TP first and WITHOUT a contour: it covers by far the
        # largest area, and its contour competed with the two classes the
        # panel exists to show. FN and FP then draw over it with contours.
        mri(ax[i][3], base, aspect, "Error" if i == 0 else None)
        fill(ax[i][3], g & p, ERROR_COLOUR["TP"][0], aspect, TP_ALPHA)
        for m, k in ((g & ~p, "FN"), (~g & p, "FP")):
            fill(ax[i][3], m, ERROR_COLOUR[k][0], aspect, ERR_ALPHA)
            edge(ax[i][3], m, ERROR_COLOUR[k][0], aspect)

        ax[i][0].text(-0.030, 0.5, reg, transform=ax[i][0].transAxes,
                      fontsize=S["title"], fontweight="bold", ha="right",
                      va="center")
        # A shared slice keeps the rows comparable but can leave an inner
        # region with nothing on it. An unlabelled blank row reads as a
        # segmentation failure, so it is named for what it is.
        if not (g.any() or p.any()):
            ax[i][1].text(0.5, 0.035, "absent on this slice",
                          transform=ax[i][1].transAxes, ha="center",
                          va="bottom", fontsize=S["tiny"], color="white",
                          path_effects=[pe.withStroke(linewidth=1.0,
                                                      foreground="#0a0a0a")])
        if i == 0:
            scalebar(ax[i][0], mm_x, avoid=g | p)

    # TWO keys, because the figure uses TWO colour systems and only one was
    # explained. The region colours -- cyan, magenta, blue in the Reference
    # and Prediction columns -- had no key at all, so the reader had to infer
    # them. Worse, the ET region blue sits close to the true-positive blue,
    # and with only the error key present the two could be conflated.
    #
    # The region key is labelled by column, not just by name: the colours
    # apply to Reference and Prediction, and the error key to the Error
    # column. That is what tells a reader which system governs which panel.
    reg_handles = [Patch(facecolor=to_rgba(c, SEG_ALPHA), edgecolor=c,
                         linewidth=0.5, label=r)
                   for r, (_, c) in REGION_DEF.items() if r in rows]
    err_handles = [Patch(facecolor=to_rgba(c, 0.62), edgecolor=c,
                         linewidth=0.5, label=l)
                   for c, l in ERROR_COLOUR.values()]

    kw = dict(frameon=False, fontsize=S["small"], handlelength=1.0,
              handleheight=0.8, handletextpad=0.4, columnspacing=1.4,
              borderpad=0.0, borderaxespad=0.0, title_fontsize=S["tiny"],
              alignment="left")
    leg1 = fig.legend(handles=reg_handles, loc="upper left",
                      bbox_to_anchor=(0.048, 0.062), ncol=len(reg_handles),
                      title="Region, in Reference and Prediction", **kw)
    fig.add_artist(leg1)
    fig.legend(handles=err_handles, loc="upper left",
               bbox_to_anchor=(0.548, 0.062), ncol=3,
               title="Voxel class, in Error", **kw)
    fig.text(0.001, 0.998, "(c)", fontsize=S["panel"], fontweight="bold",
             ha="left", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig3_result_final.{ext}")
    plt.close(fig)

    man = {"figure": "3", "version": "candidate B, refined",
           "case": CASE, "selection_rule": RULE, "quantitative_value": VALUE,
           "rows": rows, "columns": ["T1c", "Reference", "Prediction",
                                      "Error"],
           "slice_rule": slice_rule, "slice_index": z, "plane": "axial",
           "array_axis": a, "affine_axcodes": d["codes"],
           "panel_aspect": round(aspect, 4), "mm_per_px": round(mm_x, 4),
           "crop": f"shared across all panels, pad {CROP_PAD} voxels",
           "region_colours": {r: c for r, (_, c) in REGION_DEF.items()},
           "error_colours": {l: c for c, l in ERROR_COLOUR.values()},
           "legends": "two keys: region colours for the Reference and "
                      "Prediction columns, voxel classes for the Error "
                      "column. The region key was missing in an earlier "
                      "version, which left cyan, magenta and blue "
                      "unexplained and risked confusing the ET region blue "
                      "with the true-positive blue.",
           "alpha": {"reference_and_prediction": SEG_ALPHA,
                     "true_positive": TP_ALPHA,
                     "false_negative_and_positive": ERR_ALPHA},
           "held_constant": [
               "one windowing for every row",
               "one colour per region for both reference and prediction",
               "identical alpha and line width for reference and prediction",
               "one slice and one crop for all panels"],
           "error_panel_note": "TP drawn first without a contour at lower "
                               "alpha; FN and FP drawn over it with "
                               "contours. No panel is enlarged and no "
                               "arrows, boxes or in-image labels are used.",
           "windowing": "percentile 0.5-99.5",
           "interpolation": {"mri": "bicubic", "segmentation": "nearest"},
           "source_files": [d["ref_path"], d["t1c_path"],
                            f"probabilities via ns.load_mean_prob, folds "
                            f"{tuple(ns.FOLDS)}"],
           "spatial_validation": "PASS (shape and affine, reference and "
                                 "prediction)",
           "why_this_candidate": "the ET row shows the reference is a thin "
                                 "rim, which explains why size@50 costs "
                                 "0.1421 in ET against 0.0764 in WT; "
                                 "candidate A shows WT alone and explains "
                                 "nothing; candidate C enlarged the error "
                                 "panel and so put it at a different scale "
                                 "from the three beside it",
           "output": f"{OUT}/fig3_result_final.{{pdf,png}}"}
    with open(f"{OUT}/fig3_manifest.json", "w") as f:
        json.dump(man, f, indent=2, default=str)

    print(f"""  written
    {OUT}/fig3_result_final.pdf
    {OUT}/fig3_result_final.png
    {OUT}/fig3_manifest.json""")


if __name__ == "__main__":
    main()
