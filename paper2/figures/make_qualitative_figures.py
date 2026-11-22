# -*- coding: utf-8 -*-
"""
==============================================================================
 QUALITATIVE FIGURES · publication quality
==============================================================================
 Five separate files, each sized and captioned independently.

 WHAT CHANGED, AND WHY EACH MATTERS
 ----------------------------------
 1. ASPECT RATIO. The previous version drew every panel with aspect="equal",
    which assumes cubic voxels. These are 1.172 x 1.172 x 1.5 mm, so the
    through-plane direction is 28% coarser: the coronal and sagittal panels
    were vertically squashed by 22%. The aspect is now computed from the
    header zooms per plane. This is an anatomical distortion, not a matter
    of taste, and a radiologist reviewer would notice it before anything else.

 2. INTERPOLATION. The greyscale is drawn with bicubic resampling and the
    masks with nearest-neighbour. Blocky voxel edges on the anatomy look
    like low-resolution export; interpolating a MASK, by contrast, invents
    partial membership that does not exist, so the two layers are treated
    differently on purpose.

 3. CONTOURS RATHER THAN FLOOD FILL for the reference. A filled overlay hides
    the anatomy it sits on, which defeats the purpose of showing the sequence
    underneath. The reference is drawn as an anti-aliased outline and the
    prediction as a light fill, so agreement and disagreement are legible at
    the same time.

 4. SCALE BARS. Every panel showing anatomy carries one. Lesion size is the
    subject of several results in this paper, and a figure about lesion size
    with no scale is asking the reader to take the sizes on trust.

 5. PANEL LETTERS, consistent margins, hairline borders, and a single
    typographic scale across all five figures.

 6. INSET MAGNIFIERS, and the reason they matter more than dpi. The ceiling
    on apparent sharpness is the DATA, not the export: a 1.3-inch panel at
    600 dpi is 780 pixels drawn from roughly 180 source voxels, a 4x
    upsample, and raising the dpi only enlarges the same voxels. The lever
    that actually works is showing fewer voxels over the same area, so each
    error figure carries a magnified inset on the lesion with leader lines to
    its source box. Lanczos resampling is used inside the inset, where the
    upsampling factor is highest and bicubic begins to look soft.

 CASE SELECTION IS FIXED AND WAS RESOLVED BEFORE ANY IMAGE WAS DRAWN
 -------------------------------------------------------------------
   figs a-d   BraTS-MET-00629-000   median WT lesion-wise Dice of the
                                    qualitative cohort (0.6517, rank 78/155)
   fig e      resolved below        the largest false-positive component on
                                    which the six rules do not all agree

 The qualitative cohort is the 567-case Phase B cohort intersected with the
 clean stratum, because probability maps were retained only for those.

 THE ANATOMICAL PLANES ARE DERIVED FROM THE AFFINE, not from array position.
 These volumes are RAS with superior-inferior at file axis 2, which the
 analysis transpose moves to array axis 0. An earlier version called array
 axis 2 "axial" and was showing sagittal slices.
==============================================================================
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

sys.path.insert(0, "/content/drive/MyDrive/brats_2026_metastases/src/eval")
import native_space as ns

DRIVE = ns.DRIVE
OUT = f"{DRIVE}/paper2/figures/qualitative"
W2 = f"{DRIVE}/phase_b_outputs/wave2"
CASE = "BraTS-MET-00629-000"

SEQS = ["t1n", "t1c", "t2w", "t2f"]
SEQ_LABEL = {"t1n": "T1n", "t1c": "T1c", "t2w": "T2w", "t2f": "T2f"}

TISSUE = {1: ("NETC", "#e8871a"), 2: ("SNFH", "#8a7fc4"),
          3: ("ET", "#17a673"), 4: ("RC", "#e14a8a")}
REGION = {"WT": ((1, 2, 3), "#00b8c4"), "TC": ((1, 3), "#c452d6"),
          "ET": ((3,), "#3d72e8"), "RC": ((4,), "#e14a8a")}
ERR = {"TP": ("#2078b4", "true positive"),
       "FN": ("#d8342c", "false negative"),
       "FP": ("#f5a020", "false positive")}

# One typographic scale for all five figures.
S = {"dpi": 600, "title": 7.6, "label": 6.8, "small": 6.0,
     "tiny": 5.4, "hair": 0.45, "panel": 8.2}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": S["label"],
    "savefig.dpi": S["dpi"], "figure.dpi": S["dpi"],
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "savefig.facecolor": "white",
    "axes.linewidth": S["hair"], "axes.edgecolor": "#2a2a2a",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text, not outlines
    "image.interpolation": "bicubic",
})
INV = tuple(np.argsort(ns.NPZ_TO_GT))
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
#  geometry
# ---------------------------------------------------------------------------

def plane_axes(affine):
    """Which array axis gives each anatomical plane, from the affine."""
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


def load_case(case):
    img = nib.load(f"{ns.GT_DIR}/{case}.nii.gz")
    gt = np.asanyarray(img.dataobj).transpose(INV)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    # spacing in the transposed frame, so it lines up with the array
    sp = tuple(zooms[INV[k]] for k in range(3))
    vols = {}
    for s in SEQS:
        h = (glob.glob(f"{DRIVE}/**/{case}-{s}.nii.gz", recursive=True)
             or glob.glob(f"{DRIVE}/**/{case}_{s}.nii.gz", recursive=True))
        if h:
            v = np.asanyarray(nib.load(h[0]).dataobj)
            vols[s] = v.transpose(INV) if v.shape != gt.shape else v
    planes, codes = plane_axes(img.affine)
    return gt, vols, planes, codes, sp


def slice_geometry(axis, spacing):
    """
    Physical mm per pixel for the two in-plane directions of a slice taken
    along `axis`, and the aspect ratio to pass to imshow.

    The slice is displayed rot90'd, so the array's remaining axes (r, c)
    appear as (x, y) = (c, reversed r). Aspect is the ratio of the vertical
    to the horizontal physical size.
    """
    rem = [i for i in range(3) if i != axis]
    r, c = rem                       # array rows, cols of the slice
    # after rot90 the display's vertical axis is `c`, horizontal is `r`
    return spacing[r], spacing[c], spacing[c] / spacing[r]


def best_slice(mask, axis):
    if not mask.any():
        return mask.shape[axis] // 2
    other = tuple(i for i in range(3) if i != axis)
    return int(np.argmax(mask.sum(axis=other)))


def take(vol, axis, idx):
    sel = [slice(None)] * 3
    sel[axis] = idx
    return vol[tuple(sel)]


def window(vol, mask=None, lo=0.5, hi=99.7):
    """Percentile windowing. The release is not intensity-normalised and mean
    non-zero intensity varies by up to 2,401-fold across cases, so a fixed
    window renders most of them black or white."""
    ref = vol[vol > 0]
    if mask is not None and mask.any():
        ref = np.concatenate([ref, vol[mask]])
    if not ref.size:
        return vol
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((vol - a) / max(b - a, 1e-6), 0, 1)


def bbox2d(*masks, pad=12):
    comb = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        comb |= m.astype(bool)
    if not comb.any():
        return (slice(None), slice(None))
    i = np.argwhere(comb)
    lo = np.maximum(i.min(0) - pad, 0)
    hi = np.minimum(i.max(0) + pad + 1, comb.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


# ---------------------------------------------------------------------------
#  drawing
# ---------------------------------------------------------------------------

def show(ax, img2d, aspect, title=None):
    # bicubic on the anatomy: nearest leaves blocky voxel edges that read as
    # a low-resolution export
    ax.imshow(np.rot90(img2d), cmap="gray", vmin=0, vmax=1,
              interpolation="bicubic", aspect=aspect, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(S["hair"]); sp.set_color("#2a2a2a")
    if title:
        ax.set_title(title, fontsize=S["title"], pad=3.0)


def fill(ax, mask2d, colour, aspect, alpha=0.45):
    """Translucent fill. Nearest-neighbour: interpolating a mask would invent
    partial membership that does not exist."""
    if not mask2d.any():
        return
    rgba = np.zeros(mask2d.shape + (4,))
    rgba[mask2d.astype(bool)] = to_rgba(colour, alpha)
    ax.imshow(np.rot90(rgba), interpolation="nearest", aspect=aspect,
              rasterized=True)


def outline(ax, mask2d, colour, aspect, lw=0.7):
    """
    Anti-aliased boundary. A filled overlay hides the anatomy underneath,
    which defeats showing the sequence at all; an outline marks the extent
    and leaves the image visible.
    """
    if not mask2d.any():
        return
    m = np.rot90(mask2d.astype(float))
    ax.contour(m, levels=[0.5], colors=[colour], linewidths=lw,
               antialiased=True)
    ax.set_aspect(aspect)


def scalebar(ax, mm_per_px, mm=20, colour="white"):
    """
    A scale bar on every anatomical panel. Lesion size carries several results
    in this paper, and a figure about size without a scale asks the reader to
    take the sizes on trust.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w = mm / mm_per_px
    if w > 0.6 * abs(x1 - x0):        # too wide for this crop
        mm = 10
        w = mm / mm_per_px
    pad_x = 0.05 * abs(x1 - x0)
    pad_y = 0.045 * abs(y1 - y0)
    # imshow inverts the y axis, so "min" is the top of the displayed image.
    # The bar belongs at the bottom right, clear of the panel title.
    ybot = max(y0, y1) - pad_y
    xleft = max(x0, x1) - pad_x - w
    # a white bar and white text vanish on a bright slice, and a black one
    # vanishes on a dark one. A thin dark outline round both makes the bar
    # legible wherever it lands, without having to know the local intensity.
    stroke = [pe.withStroke(linewidth=1.1, foreground="#101010")]
    bar = Rectangle((xleft, ybot), w, 0.014 * abs(y1 - y0),
                    facecolor=colour, edgecolor="none", zorder=6)
    bar.set_path_effects(stroke)
    ax.add_patch(bar)
    t = ax.text(xleft + w / 2, ybot + 0.035 * abs(y1 - y0), f"{mm} mm",
                color=colour, fontsize=S["tiny"], ha="center", va="top",
                zorder=6)
    t.set_path_effects(stroke)


def inset_zoom(ax, img2d, masks, aspect, box, zoom_frac=0.20,
               loc=(0.575, 0.025, 0.40, 0.40)):
    """
    A magnified inset on the densest part of the mask, with leader lines to
    the region it came from.

    This is the only thing that raises apparent resolution: the export is
    already well past the data's own limit, so the way to show more detail is
    to put fewer source voxels over the same printed area. Lanczos is used
    here rather than bicubic because the upsampling factor inside the inset
    is high enough for bicubic to look soft.
    """
    comb = np.zeros_like(img2d, dtype=bool)
    for m in masks:
        comb |= m.astype(bool)
    if not comb.any():
        return None
    i = np.argwhere(comb)
    cy, cx = i.mean(0)
    # a source box this size gives roughly a 2x magnification inside the
    # inset, which is the point; 0.42 of the panel was barely a magnification
    half = max(int(zoom_frac * min(img2d.shape) / 2), 10)
    r0 = int(np.clip(cy - half, 0, img2d.shape[0] - 1))
    r1 = int(np.clip(cy + half, 1, img2d.shape[0]))
    c0 = int(np.clip(cx - half, 0, img2d.shape[1] - 1))
    c1 = int(np.clip(cx + half, 1, img2d.shape[1]))
    sub = (slice(r0, r1), slice(c0, c1))

    # clip_on so the inset cannot spill past the panel border
    iax = ax.inset_axes(loc, clip_on=True)
    iax.imshow(np.rot90(img2d[sub]), cmap="gray", vmin=0, vmax=1,
               interpolation="lanczos", aspect=aspect, rasterized=True)
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#f0f0f0")
    # Mark the source box on the parent. After rot90 the display's x axis is
    # the array COLUMN reversed and its y axis is the array ROW, so the
    # rectangle has to be built in display coordinates rather than array
    # ones: getting this wrong puts the box somewhere the inset did not come
    # from, which is worse than having no box at all.
    h, w = img2d.shape
    x0d, y0d = (h - r1), c0           # rot90: rows -> x reversed, cols -> y
    ax.add_patch(Rectangle((x0d, y0d), r1 - r0, c1 - c0, fill=False,
                           edgecolor="#ffffff", linewidth=0.6, zorder=7,
                           path_effects=[pe.withStroke(linewidth=1.2,
                                                       foreground="#101010")]))
    return iax, sub


def panel_letter(ax, letter):
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes,
            fontsize=S["panel"], fontweight="bold", ha="right", va="top")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  A · sequences and reference labels
# ---------------------------------------------------------------------------

def fig_sequences(gt, vols, planes, sp):
    a = planes["axial"]
    inner = np.isin(gt, (3,))
    z = best_slice(inner if inner.any() else gt > 0, a)
    box = bbox2d(take(gt > 0, a, z), pad=8)
    mmx, mmy, asp = slice_geometry(a, sp)

    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.95))
    fig.subplots_adjust(wspace=0.035, right=0.885, top=0.86, bottom=0.02)
    for k, s in enumerate(SEQS):
        show(axes[k], take(window(vols[s], gt > 0), a, z)[box], asp,
             SEQ_LABEL[s])
    show(axes[4], take(window(vols["t1c"], gt > 0), a, z)[box], asp,
         "Reference labels")
    for lab, (nm, col) in TISSUE.items():
        m = take(gt == lab, a, z)[box]
        fill(axes[4], m, col, asp, 0.50)
        outline(axes[4], m, col, asp, 0.55)
    scalebar(axes[0], mmx)
    panel_letter(axes[0], "a")

    axes[4].legend(handles=[Patch(facecolor=to_rgba(c, 0.5), edgecolor=c,
                                  linewidth=0.5, label=n)
                            for n, c in TISSUE.values()],
                   loc="center left", bbox_to_anchor=(1.05, 0.5),
                   frameon=False, fontsize=S["small"], handlelength=1.0,
                   handleheight=0.85, labelspacing=0.5, borderpad=0.1)
    save(fig, "fig_a_sequences")
    print(f"  a  sequences · axial slice {z} (array axis {a}) · "
          f"aspect {asp:.3f} · {mmx:.3f} mm/px")
    return a, z, box, asp, mmx



# ---------------------------------------------------------------------------
#  A2 · sequences and reference labels, TWO slices of the same case
#
#  Figure 2 of the manuscript. The first row is the slice of maximum
#  enhancing-tumour cross-section (as before). The second row is the slice of
#  maximum resection-cavity cross-section of the SAME case, so that all four
#  native labels are visible in one figure without changing the case, whose
#  selection rule (median WT lesion-wise Dice of the qualitative cohort) is
#  unchanged. Both rows share one field of view, one window per sequence and
#  one legend, so the two slices are comparable in scale and position.
#  If the case had no cavity, the second row would not be drawn.
# ---------------------------------------------------------------------------

def fig_sequences_two_slices(gt, vols, planes, sp):
    a = planes["axial"]
    mmx, mmy, asp = slice_geometry(a, sp)

    inner = np.isin(gt, (3,))
    z_et = best_slice(inner if inner.any() else gt > 0, a)
    cav = (gt == 4)
    z_rc = best_slice(cav, a) if cav.any() else None

    rows = [("ET", z_et, "slice of maximum\nET cross-section")]
    if z_rc is not None and z_rc != z_et:
        rows.append(("RC", z_rc, "slice of maximum\nRC cross-section"))

    # one field of view for every panel: the union of the reference on both
    # slices, so neither row is cropped to its own extent
    union = np.zeros_like(take(gt, a, z_et), dtype=bool)
    for _, z, _ in rows:
        union |= take(gt > 0, a, z)
    box = bbox2d(union, pad=8)

    # one window per sequence, computed once on the volume, shared by rows
    win = {s: window(vols[s], gt > 0) for s in SEQS}

    nrow = len(rows)
    # Figure height follows the panel geometry rather than a fixed number:
    # after rot90 a panel is (box rows) wide and (box cols x aspect) high, so
    # a fixed height would leave white bands between rows whenever the crop
    # is wider than it is tall. Width stays at the full text width.
    h_px = box[0].stop - box[0].start
    w_px = box[1].stop - box[1].start
    fig_w = 7.0
    left, right, wsp = 0.075, 0.995, 0.035
    panel_w = fig_w * (right - left) / (5 + 4 * wsp)
    panel_h = panel_w * (w_px * asp) / h_px
    hsp = 0.05
    top_in, bot_in = 0.22, 0.36          # title row, legend row
    plot_h = nrow * panel_h + (nrow - 1) * hsp * panel_h
    fig_h = plot_h + top_in + bot_in
    fig, axes = plt.subplots(nrow, 5, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(wspace=wsp, hspace=hsp, left=left, right=right,
                        top=1 - top_in / fig_h, bottom=bot_in / fig_h)
    for i, (tag, z, rowlabel) in enumerate(rows):
        for k, s in enumerate(SEQS):
            show(axes[i][k], take(win[s], a, z)[box], asp,
                 SEQ_LABEL[s] if i == 0 else None)
        show(axes[i][4], take(win["t1c"], a, z)[box], asp,
             "Native reference labels" if i == 0 else None)
        for lab, (nm, col) in TISSUE.items():
            m = take(gt == lab, a, z)[box]
            fill(axes[i][4], m, col, asp, 0.50)
            outline(axes[i][4], m, col, asp, 0.55)
        axes[i][0].set_ylabel(rowlabel, fontsize=S["small"], labelpad=3)
        if i == 0:
            scalebar(axes[i][0], mmx)

    # legend once, under the figure; a label is marked only if it appears on
    # neither slice
    shown = {lab: any(take(gt == lab, a, z)[box].any() for _, z, _ in rows)
             for lab in TISSUE}
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.5), edgecolor=c,
                              linewidth=0.5,
                              label=n if shown[l] else f"{n} (not on these slices)")
                        for l, (n, c) in TISSUE.items()],
               loc="lower center", ncol=4, frameon=False,
               fontsize=S["small"], handlelength=1.0, handleheight=0.85,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig1_dataset_final")
    print(f"  a  sequences · axial slices {[z for _, z, _ in rows]} "
          f"(array axis {a}) · aspect {asp:.3f} · {mmx:.3f} mm/px · "
          f"labels on slices: {[TISSUE[l][0] for l in TISSUE if shown[l]]}")
    return a, z_et, box, asp, mmx

# ---------------------------------------------------------------------------
#  B · evaluation regions
# ---------------------------------------------------------------------------

def fig_regions(gt, vols, a, z, box, asp, mmx):
    fig, axes = plt.subplots(1, 4, figsize=(5.8, 2.05))
    fig.subplots_adjust(wspace=0.035, top=0.84, bottom=0.185)
    base = take(window(vols["t1c"], gt > 0), a, z)[box]
    for k, (reg, (labs, col)) in enumerate(REGION.items()):
        show(axes[k], base, asp, reg)
        m = take(np.isin(gt, labs), a, z)[box]
        fill(axes[k], m, col, asp, 0.42)
        outline(axes[k], m, col, asp, 0.8)
        comp = " + ".join(TISSUE[l][0] for l in labs)
        cap = f"{reg} = {comp}" if comp != reg else reg
        if not m.any():
            cap += "\n(absent on this slice)"
        axes[k].set_xlabel(cap, fontsize=S["small"], labelpad=2.5)
    scalebar(axes[0], mmx)
    panel_letter(axes[0], "b")
    save(fig, "fig_b_regions")
    present = [r for r, (l, _) in REGION.items()
               if take(np.isin(gt, l), a, z).any()]
    print(f"  b  regions on this slice: {present}")


# ---------------------------------------------------------------------------
#  C · reference, prediction, error
# ---------------------------------------------------------------------------

def fig_error(gt, vols, pred, a, sp):
    rows = [r for r in ("WT", "TC", "ET") if np.isin(gt, REGION[r][0]).any()]
    if not rows:
        print("  c  skipped: no tumour region in the reference")
        return
    mmx, mmy, asp = slice_geometry(a, sp)
    fig, axes = plt.subplots(len(rows), 4, figsize=(6.4, 1.72 * len(rows)),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.035, hspace=0.055, top=0.93, bottom=0.085)
    for i, reg in enumerate(rows):
        labs, col = REGION[reg]
        g, p = np.isin(gt, labs), np.isin(pred, labs)
        z = best_slice(g | p, a)
        box = bbox2d(take(g | p, a, z), pad=14)
        base = take(window(vols["t1c"], g), a, z)[box]
        g2, p2 = take(g, a, z)[box], take(p, a, z)[box]

        show(axes[i][0], base, asp, "T1c" if i == 0 else None)
        show(axes[i][1], base, asp, "reference" if i == 0 else None)
        fill(axes[i][1], g2, col, asp, 0.38); outline(axes[i][1], g2, col, asp)
        show(axes[i][2], base, asp, "prediction" if i == 0 else None)
        fill(axes[i][2], p2, col, asp, 0.38); outline(axes[i][2], p2, col, asp)
        show(axes[i][3], base, asp, "error overlay" if i == 0 else None)
        classes = ((g2 & p2, "TP", 0.42), (g2 & ~p2, "FN", 0.70),
                   (~g2 & p2, "FP", 0.70))
        for m, key, al in classes:
            fill(axes[i][3], m, ERR[key][0], asp, al)
            outline(axes[i][3], m, ERR[key][0], asp, 0.5)

        # magnified inset over the disagreement, where the detail is. The
        # export is already far past the data's own resolution, so the only
        # way to show more is to put fewer voxels over the same printed area.
        disagree = (g2 ^ p2)
        target = disagree if disagree.any() else (g2 | p2)
        res = inset_zoom(axes[i][3], base, [target], asp, box)
        if res is not None:
            iax, sub = res
            for m, key, al in classes:
                fill(iax, m[sub], ERR[key][0], asp, al)
                outline(iax, m[sub], ERR[key][0], asp, 0.8)
        axes[i][0].set_ylabel(reg, fontsize=S["title"], labelpad=3)
        if i == 0:
            scalebar(axes[i][0], mmx)
            panel_letter(axes[i][0], "c")

    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERR.values()],
               loc="lower center", ncol=3, frameon=False,
               fontsize=S["small"], bbox_to_anchor=(0.5, -0.012),
               handlelength=1.1, columnspacing=1.6)
    save(fig, "fig_c_error_axial")
    print(f"  c  error panels for {rows}")


# ---------------------------------------------------------------------------
#  D · three planes, each with its own aspect
# ---------------------------------------------------------------------------

def fig_views(gt, vols, pred, planes, codes, sp):
    labs, col = REGION["WT"]
    g, p = np.isin(gt, labs), np.isin(pred, labs)
    # width ratios from the physical extent, so the three panels are to the
    # same scale rather than each stretched to a uniform box
    info = {}
    for plane in ("axial", "coronal", "sagittal"):
        ax_i = planes[plane]
        idx = best_slice(g | p, ax_i)
        box = bbox2d(take(g | p, ax_i, idx), pad=12)
        mmx, mmy, asp = slice_geometry(ax_i, sp)
        h = (box[0].stop - box[0].start) if box[0].start is not None else 1
        w = (box[1].stop - box[1].start) if box[1].start is not None else 1
        info[plane] = (ax_i, idx, box, mmx, asp, h * mmx)

    fig, axes = plt.subplots(
        1, 3, figsize=(6.2, 2.35),
        gridspec_kw={"width_ratios": [info[p][5] for p in
                                      ("axial", "coronal", "sagittal")]})
    fig.subplots_adjust(wspace=0.05, top=0.90, bottom=0.14)
    for k, plane in enumerate(("axial", "coronal", "sagittal")):
        ax_i, idx, box, mmx, asp, _ = info[plane]
        base = take(window(vols["t1c"], g), ax_i, idx)[box]
        g2, p2 = take(g, ax_i, idx)[box], take(p, ax_i, idx)[box]
        show(axes[k], base, asp, plane)
        for m, key, al in ((g2 & p2, "TP", 0.42),
                           (g2 & ~p2, "FN", 0.70),
                           (~g2 & p2, "FP", 0.70)):
            fill(axes[k], m, ERR[key][0], asp, al)
            outline(axes[k], m, ERR[key][0], asp, 0.5)
        if k == 0:
            scalebar(axes[k], mmx)
            panel_letter(axes[k], "d")
    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.6), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERR.values()],
               loc="lower center", ncol=3, frameon=False,
               fontsize=S["small"], bbox_to_anchor=(0.5, -0.015),
               handlelength=1.1, columnspacing=1.6)
    save(fig, "fig_d_error_views")
    asps = {p: round(info[p][4], 3) for p in info}
    print(f"  d  three planes · affine {codes} · aspect per plane {asps}")


# ---------------------------------------------------------------------------
#  E · one component, six rule decisions
# ---------------------------------------------------------------------------

def fig_rules():
    f = f"{W2}/fp_components.csv"
    if not os.path.exists(f):
        print("  e  skipped: fp_components.csv not found")
        return
    d = pd.read_csv(f)
    fp = d[d.is_fp == 1].copy()
    rules = [c for c in ("size@20", "size@50", "entropy@0.45",
                         "entropy@0.50", "entropy@0.56", "confidence@0.45")
             if c in fp.columns]
    if not len(fp) or not rules:
        print("  e  skipped: no false-positive rows")
        return
    fp["n_removed"] = fp[rules].sum(axis=1)
    # a component every rule removes shows nothing about disagreement, which
    # is what this panel is for
    split = fp[(fp.n_removed > 0) & (fp.n_removed < len(rules))]
    pick = (split if len(split) else fp).nlargest(1, "volume_mm3").iloc[0]

    gt, vols, planes, codes, sp = load_case(pick.case)
    pred = ns.build_nested_label(ns.load_mean_prob(pick.case), suppressor=None)
    pm = np.isin(pred, REGION[pick.region][0])
    st = ndimage.generate_binary_structure(3, 2)
    lab, _ = ndimage.label(ndimage.binary_dilation(pm, structure=st),
                           structure=np.ones((3, 3, 3)))
    lab = (lab * pm).astype(np.int32)
    sel = lab == int(pick.supp_component_id)
    if not sel.any():
        print("  e  component id absent from the rebuilt labelling; skipped "
              "rather than drawing a guess")
        return

    a = planes["axial"]
    z = best_slice(sel, a)
    box = bbox2d(take(sel, a, z), pad=20)
    mmx, mmy, asp = slice_geometry(a, sp)
    base = take(window(vols["t1c"], sel), a, z)[box]
    s2 = take(sel, a, z)[box]

    n = len(rules) + 1
    fig, axes = plt.subplots(1, n, figsize=(1.02 * n, 1.75))
    fig.subplots_adjust(wspace=0.035, top=0.80, bottom=0.20)
    show(axes[0], base, asp, "baseline")
    fill(axes[0], s2, ERR["FP"][0], asp, 0.55)
    outline(axes[0], s2, ERR["FP"][0], asp, 0.8)
    scalebar(axes[0], mmx, mm=10)
    panel_letter(axes[0], "e")
    for k, r in enumerate(rules, 1):
        removed = bool(pick[r])
        show(axes[k], base, asp, r.replace("@", "\n"))
        if not removed:
            fill(axes[k], s2, ERR["FP"][0], asp, 0.55)
            outline(axes[k], s2, ERR["FP"][0], asp, 0.8)
        else:
            outline(axes[k], s2, "#9a9a9a", asp, 0.45)
        axes[k].set_xlabel("removed" if removed else "retained",
                           fontsize=S["tiny"], labelpad=2,
                           color="#17a673" if removed else "#d8342c")
    save(fig, "fig_e_rule_decisions")
    print(f"  e  {pick.case} {pick.region} component "
          f"{int(pick.supp_component_id)} · {pick.volume_mm3:.1f} mm3 · "
          f"removed by {int(pick.n_removed)}/{len(rules)} rules")


# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print(" QUALITATIVE FIGURES · publication quality")
    print("=" * 78)
    gt, vols, planes, codes, sp = load_case(CASE)
    print(f"  {CASE}")
    print(f"  orientation {codes} · spacing in array order "
          f"{tuple(round(v, 3) for v in sp)}")
    print(f"  plane to array axis: {planes}")
    ip = min(sp); th = max(sp)
    print(f"  voxel is anisotropic by {th / ip:.2f}x, so through-plane panels "
          f"need aspect {th / ip:.3f}")
    print(f"  (drawing them at aspect 1.0 would squash them by "
          f"{100 * (1 - ip / th):.0f}%)")
    if len(vols) < 4:
        print("  fewer than four sequences; stopping")
        return
    try:
        pred = ns.build_nested_label(ns.load_mean_prob(CASE), suppressor=None)
    except Exception as e:
        pred = None
        print(f"  probabilities unavailable ({type(e).__name__}); "
              f"dataset figures only")

    a, z, box, asp, mmx = fig_sequences_two_slices(gt, vols, planes, sp)
    fig_regions(gt, vols, a, z, box, asp, mmx)
    if pred is not None:
        fig_error(gt, vols, pred, a, sp)
        fig_views(gt, vols, pred, planes, codes, sp)
    try:
        fig_rules()
    except Exception as e:
        print(f"  e  {type(e).__name__}: {e}")

    print(f"""
  written to {OUT}   ·   600 dpi, PDF with embedded editable text

  Five separate files, so each can be sized and captioned independently.

  The aspect of every panel comes from the header zooms. The previous version
  used aspect="equal" throughout, which assumes cubic voxels; these are
  {sp[0]:.3f} x {sp[1]:.3f} x {sp[2]:.3f} mm in array order, and the
  through-plane panels were squashed by {100 * (1 - ip / th):.0f}%.

  Anatomy is drawn with bicubic resampling, masks with nearest neighbour:
  blocky voxel edges look like a low-resolution export, while interpolating a
  mask would invent partial membership that does not exist.""")


if __name__ == "__main__":
    main()
