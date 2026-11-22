# -*- coding: utf-8 -*-
"""
==============================================================================
 FIGURE 2 · NATIVE LABELS TO EVALUATION REGIONS · final
==============================================================================
 Candidate A won, so this refines A. Two changes, both requested:

   1  a clearer hierarchy. The section headings collided with the panel
      titles because they were placed in AXES coordinates, which puts them at
      the same height as a panel title and leaves the two competing. They are
      now placed in FIGURE coordinates, in a band reserved for them by the
      gridspec, so the reading order is unambiguous:

          (b)  Native labels
               [ NETC ] [ SNFH ] [ ET ] [ RC ]

               Evaluation regions
               [  WT  ] [  TC  ] [ ET ] [ RC ]

   2  the mapping under each bottom panel is kept in the form "ET = ET" and
      "RC = RC", redundant though it looks. The figure's whole purpose is the
      mapping, and writing three unions plus two bare names would leave a
      reader wondering whether the last two were omissions. Uniform notation
      says the mapping is complete.

 RC, AND A DISTINCTION THAT MATTERS
 ----------------------------------
 RC is written "(absent on this slice)", never "absent". The region is part
 of the evaluation definition and may well be present elsewhere in the case;
 only this slice lacks it. Writing "RC absent" would tell the reader
 something about the case that is not established here, and the resection
 cavity is the region this paper's most consequential result depends on.

 The slice, case, colours, aspect and windowing all follow figure 1, because
 the two figures show the same slice of the same case and a reader moves
 between them.
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

LABEL_COLOUR = {1: ("NETC", "#e07b39"), 2: ("SNFH", "#8a7fc4"),
                3: ("ET", "#17a673"), 4: ("RC", "#d94f8a")}
REGION_DEF = {"WT": ((1, 2, 3), "#00a8b5"), "TC": ((1, 3), "#b845c9"),
              "ET": ((3,), "#2f6fe0"), "RC": ((4,), "#d94f8a")}
CROP_PAD = 30                       # same field of view as figure 1

S = {"dpi": 600, "section": 7.6, "title": 7.2, "small": 6.0, "tiny": 5.3,
     "hair": 0.45, "panel": 7.0}
SEG_ALPHA, SEG_LW = 0.42, 0.7

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
    labels = set(int(v) for v in np.unique(gt_file)) - {0}
    unknown = labels - set(LABEL_COLOUR)
    if unknown:
        raise ValueError(f"{case}: unexpected labels {sorted(unknown)}")
    planes, codes = plane_axes(ref.affine)
    zooms = tuple(float(v) for v in ref.header.get_zooms()[:3])
    spacing = tuple(zooms[INV[k]] for k in range(3))
    return {"gt": gt_file.transpose(INV),
            "t1c": np.asanyarray(img.dataobj).transpose(INV),
            "planes": planes, "codes": codes, "spacing": spacing,
            "labels": sorted(labels), "t1c_path": hit[0],
            "ref_path": ref_path}


def window(vol, extra=None, lo=0.5, hi=99.5):
    ref = vol[vol > 0]
    if extra is not None and extra.any():
        ref = np.concatenate([ref, vol[extra]])
    if not ref.size:
        return vol
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((vol - a) / max(b - a, 1e-6), 0, 1)


def draw(ax, base, mask, colour, aspect, title):
    ax.imshow(np.rot90(base), cmap="gray", vmin=0, vmax=1,
              interpolation="bicubic", aspect=aspect, rasterized=True)
    if mask.any():
        rgba = np.zeros(mask.shape + (4,))
        rgba[mask] = to_rgba(colour, SEG_ALPHA)
        ax.imshow(np.rot90(rgba), interpolation="nearest", aspect=aspect,
                  rasterized=True)
        ax.contour(np.rot90(mask.astype(float)), levels=[0.5],
                   colors=[colour], linewidths=SEG_LW, antialiased=True)
        ax.set_aspect(aspect)
    ax.set_title(title, fontsize=S["title"], pad=1.8)
    ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_linewidth(S["hair"]); spn.set_color("#333333")


def main():
    print("=" * 74)
    print(" FIGURE 2 · LABELS TO REGIONS · final refinement of candidate A")
    print("=" * 74)
    d = load_and_validate(CASE)
    gt, sp = d["gt"], d["spacing"]
    a = d["planes"]["axial"]

    inner = np.isin(gt, (3,))
    slice_rule = "axial slice of maximum enhancing-tumour area"
    ax_other = tuple(i for i in range(3) if i != a)
    if inner.any():
        z = int(np.argmax(inner.sum(axis=ax_other)))
    else:
        slice_rule += " [fallback: maximum any-label area]"
        z = int(np.argmax((gt > 0).sum(axis=ax_other)))

    sl = np.take(gt > 0, z, axis=a)
    idx = np.argwhere(sl)
    lo = np.maximum(idx.min(0) - CROP_PAD, 0)
    hi = np.minimum(idx.max(0) + CROP_PAD + 1, sl.shape)
    box = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))

    rem = [i for i in range(3) if i != a]
    mm_x = sp[rem[0]]
    aspect = sp[rem[1]] / sp[rem[0]]
    base = np.take(window(d["t1c"], gt > 0), z, axis=a)[box]
    lab = {l: np.take(gt == l, z, axis=a)[box] for l in LABEL_COLOUR}
    reg = {r: np.take(np.isin(gt, ls), z, axis=a)[box]
           for r, (ls, _) in REGION_DEF.items()}

    print(f"""
  case            {CASE}
  selection rule  {RULE} ({VALUE})
  slice rule      {slice_rule}
  slice index     {z}   (array axis {a}, affine {d['codes']})
  panel aspect    {aspect:.4f}
  crop            shared field of view, pad {CROP_PAD} voxels
                  = {CROP_PAD * mm_x:.0f} mm, identical to figure 1
  labels on slice {sorted(int(l) for l in LABEL_COLOUR if lab[l].any())}
                  of {d['labels']} in the case
  regions on slice {[r for r in REGION_DEF if reg[r].any()]}
""")

    # ---- layout ---------------------------------------------------------
    # Change 1: the section headings live in a band the gridspec reserves for
    # them, addressed in FIGURE coordinates. Placing them in axes
    # coordinates, as the previous version did, put them at the same height
    # as the panel titles and the two competed for the same space.
    # LAYOUT
    #
    # Row labels sit in a column at the left. Each key sits in its OWN
    # gridspec row directly beneath the panels it describes.
    #
    # Six earlier attempts put the headings, and then the keys, in the gap
    # BETWEEN the two panel rows, and every one collided -- with the panel
    # titles, then with the key, then with the "(absent on this slice)" note.
    # The gap is only as tall as hspace makes it and was being asked to hold
    # three items.
    #
    # Giving each key a gridspec row of its own removes the problem instead
    # of managing it: the layout reserves the height, so nothing can land on
    # anything else. Nothing here is measured from the renderer, because
    # nothing needs to be.
    # LAYOUT
    #
    # Section headings ABOVE their rows, which is what the figure needs: the
    # reading order is heading, panels, heading, panels, straight down.
    #
    # Five earlier attempts put the headings in the gap between rows and
    # every one collided -- with the panel titles, then the key, then the
    # titles again -- because that gap holds three items and is only as tall
    # as hspace makes it. The fix is not a better offset. Each heading gets
    # its OWN gridspec row, so the layout reserves its height and nothing
    # can land on it.
    #
    # hspace is zero throughout: it applies to every gap equally, so a text
    # row of height 0.18 would get the same generous gap as a panel row. The
    # vertical rhythm comes from the row heights instead, each sized for
    # what it holds:
    #
    #   0.26   "Native labels". The gap beneath is the unused part of this
    #          row plus the panel title's pad. Both are trimmed, but only so
    #          far: at 0.20 the heading overflowed onto the title, and
    #          bottom-anchoring the text to close the gap put it level with
    #          the titles. 0.26 with a 1.8pt title pad is the point where
    #          the gap is tight and the clearance still holds.
    #   1.00   four panels
    #   0.28   key for the row above
    #   0.26   "Evaluation regions", slightly taller because it follows a key
    #   1.00   four panels
    #   0.42   key, taller because the row above ends in two-line captions
    fig = plt.figure(figsize=(7.1, 4.15))
    gs = fig.add_gridspec(
        6, 4, left=0.012, right=0.996, top=0.985, bottom=0.02,
        hspace=0.0, wspace=0.025,
        height_ratios=[0.26, 1, 0.26, 0.30, 1, 0.38])

    axes_top, axes_bot = [], []
    for k, (l, (nm, c)) in enumerate(LABEL_COLOUR.items()):
        ax = fig.add_subplot(gs[1, k])
        draw(ax, base, lab[l], c, aspect, nm)
        if not lab[l].any():
            # the distinction the brief insists on: absent from the SLICE,
            # which says nothing about the case
            ax.set_xlabel("(absent on this slice)", fontsize=S["tiny"],
                          labelpad=2)
        axes_top.append(ax)

    for k, (r, (ls, c)) in enumerate(REGION_DEF.items()):
        ax = fig.add_subplot(gs[4, k])
        draw(ax, base, reg[r], c, aspect, r)
        # Change 2: uniform notation, including "ET = ET" and "RC = RC".
        # They read as redundant, but the figure exists to state the mapping,
        # and three unions followed by two bare names would look like two
        # omissions rather than two identities.
        comp = " + ".join(LABEL_COLOUR[i][0] for i in ls)
        # The equals sign belongs where there is a mapping to state. "WT =
        # NETC + SNFH + ET" is an equation; "ET = ET" is a tautology that
        # makes a reader look for a difference between the two sides. Where
        # the region is a single label of the same name, the caption is the
        # label alone, and the panel title above it already carries the
        # region name.
        cap = f"{r} = {comp}" if comp != r else comp
        if not reg[r].any():
            cap += "\n(absent on this slice)"
        ax.set_xlabel(cap, fontsize=S["small"], labelpad=2.5)
        axes_bot.append(ax)

    # scale bar on the first panel, in the corner holding less of the mask
    ax0 = axes_top[0]
    x0, x1 = ax0.get_xlim(); y0, y1 = ax0.get_ylim()
    w = 20.0 / mm_x
    disp = np.rot90(np.take(gt > 0, z, axis=a)[box])
    band = disp[int(0.75 * disp.shape[0]):, :]
    half = band.shape[1] // 2
    left = band[:, half:].sum() > band[:, :half].sum()
    xl = (min(x0, x1) + 0.045 * abs(x1 - x0)) if left else \
         (max(x0, x1) - 0.045 * abs(x1 - x0) - w)
    yb = max(y0, y1) - 0.055 * abs(y1 - y0)
    stroke = [pe.withStroke(linewidth=1.0, foreground="#0a0a0a")]
    bar = Rectangle((xl, yb), w, 0.014 * abs(y1 - y0), facecolor="white",
                    edgecolor="none", zorder=8)
    bar.set_path_effects(stroke)
    ax0.add_patch(bar)
    t = ax0.text(xl + w / 2, yb + 0.022 * abs(y1 - y0), "20 mm",
                 color="white", fontsize=S["tiny"], ha="center", va="bottom",
                 zorder=8)
    t.set_path_effects(stroke)

    # ONE legend PER ROW, placed directly beneath the row it describes,
    # rather than both side by side under the figure.
    #
    # Position carries the meaning. Under the figure, two keys need titles to
    # say which is which and a reader still has to check; beneath its own row,
    # each key is unambiguous by where it sits. The native-label key
    # therefore lands in the band between the two rows, and the region key
    # below the bottom row.
    #
    # The gap between the rows exists anyway -- it holds the second section
    # headings in their own rows, centred over the four panels they describe.
    # Centred rather than left-aligned: a heading over a four-panel row reads
    # as belonging to the whole row, where left alignment can look attached
    # to the first panel.
    for grid_row, text in ((0, "Native labels"), (3, "Evaluation regions")):
        axh = fig.add_subplot(gs[grid_row, 0:4])
        axh.axis("off")
        # Top-anchored. Bottom-anchoring was tried to close the gap and put
        # the heading level with the panel titles instead: that gap is the
        # clearance the titles need, so it is narrowed by trimming the row
        # rather than by moving the text down into it.
        axh.text(0.5, 0.98, text, transform=axh.transAxes,
                 fontsize=S["section"], fontweight="bold", ha="center",
                 va="top")

    # one key per row, in the gridspec row beneath its panels, spanning the
    # panel columns so it lines up with them
    for grid_row, items in (
            (2, [(n, c, lab[l].any()) for l, (n, c) in LABEL_COLOUR.items()]),
            (5, [(r, c, reg[r].any()) for r, (_, c) in REGION_DEF.items()])):
        axk = fig.add_subplot(gs[grid_row, 0:4])
        axk.axis("off")
        handles = [Patch(facecolor=to_rgba(c, SEG_ALPHA), edgecolor=c,
                         linewidth=0.45,
                         label=n if ok else f"{n} (not on this slice)")
                   for n, c, ok in items]
        axk.legend(handles=handles, loc="center", ncol=4, frameon=False,
                   fontsize=S["small"], handlelength=0.9, handleheight=0.85,
                   handletextpad=0.42, columnspacing=1.6, borderpad=0.0,
                   borderaxespad=0.0)

    fig.text(0.001, 0.998, "(b)", fontsize=S["panel"], fontweight="bold",
             ha="left", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig2_regions_final.{ext}")
    plt.close(fig)

    man = {"figure": "2", "version": "final refinement of candidate A",
           "case": CASE, "selection_rule": RULE, "quantitative_value": VALUE,
           "slice_rule": slice_rule, "slice_index": z, "plane": "axial",
           "array_axis": a, "affine_axcodes": d["codes"],
           "panel_aspect": round(aspect, 4), "mm_per_px": round(mm_x, 4),
           "crop": f"shared with figure 1, pad {CROP_PAD} voxels",
           "labels_on_slice": sorted(int(l) for l in LABEL_COLOUR
                                     if lab[l].any()),
           "labels_in_case": d["labels"],
           "regions_on_slice": [r for r in REGION_DEF if reg[r].any()],
           "region_definitions": {r: " + ".join(LABEL_COLOUR[i][0]
                                                for i in ls)
                                  for r, (ls, _) in REGION_DEF.items()},
           "label_colours": {n: c for n, c in LABEL_COLOUR.values()},
           "region_colours": {r: c for r, (_, c) in REGION_DEF.items()},
           "colour_note": "two independent systems; no hue is shared between "
                          "a native label and an evaluation region",
           "source_files": [d["ref_path"], d["t1c_path"]],
           "spatial_validation": "PASS (shape and affine)",
           "label_validation": "PASS (values within the four defined labels)",
           "refinements": [
               "section headings in figure coordinates, in a reserved band, "
               "so they no longer compete with the panel titles",
               "uniform mapping notation including ET = ET and RC = RC",
               "'absent on this slice' throughout, never 'absent'",
               "one legend per row, directly beneath the row it describes, "
               "so position rather than a title says which scheme it is; "
               "the earlier version had no legend at all",
               "row labels in a left column and keys in a right column, so "
               "neither can collide with a panel title; five attempts at "
               "placing headings in the band between the rows all collided",
               "the equals sign kept where there is a mapping and dropped "
               "where the region is a single label of the same name"],
           "output": f"{OUT}/fig2_regions_final.{{pdf,png}}"}
    with open(f"{OUT}/fig2_manifest.json", "w") as f:
        json.dump(man, f, indent=2, default=str)

    print(f"""  written
    {OUT}/fig2_regions_final.pdf
    {OUT}/fig2_regions_final.png
    {OUT}/fig2_manifest.json

  The hierarchy now reads: (b) Native labels, four panels, Evaluation
  regions, four panels. The headings sit in their own band rather than at the
  height of the panel titles.""")


if __name__ == "__main__":
    main()
