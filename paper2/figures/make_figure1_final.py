# -*- coding: utf-8 -*-
"""
==============================================================================
 FIGURE 1 · DATASET OVERVIEW · final
==============================================================================
 Candidate A won the comparison, so this is a refinement of A rather than a
 new design. Four changes, all requested:

   1  the panel label is smaller and closer to the figure edge
   2  "Reference labels" becomes "Native reference labels", which is more
      precise and further from reading as a fifth modality
   3  the legend is more compact and takes less space below the figure
   4  the four modalities stay equal in size, on one row, with the reference
      panel after them. No two-row layout.

 ON POINT 4 AND THE FIELD OF VIEW
 --------------------------------
 Candidate C showed the whole brain and so showed WHERE the lesion sits,
 which candidate A's tighter crop loses. But C used contours alone and the
 NETC compartment vanished entirely, so it failed at the one thing a dataset
 figure has to do.

 The context is recovered here without breaking the equality: the crop is
 widened for ALL FIVE panels together, not for one. Every panel keeps the
 same field of view, the same size and the same scale, and the lesion is seen
 in its anatomical setting. Fills are kept, so the compartments remain
 visible.

 CASE SELECTION IS FROZEN AND WAS RESOLVED QUANTITATIVELY
 --------------------------------------------------------
   BraTS-MET-00629-000, the median WT lesion-wise Dice of the clean stratum
   (0.6517, rank 78 of 155). Not selected visually, and not replaced because
   another case looks better.

 SLICE RULE, DETERMINISTIC AND RECORDED
 --------------------------------------
   the axial slice of maximum enhancing-tumour area, falling back to maximum
   any-label area if ET is absent from the case. The same slice is used for
   all five panels.

 The anatomical plane comes from the affine, and the panel aspect from the
 header zooms: the voxel is 1.172 x 1.172 x 1.5 mm, so a through-plane panel
 drawn at aspect 1.0 would be squashed by 22%.
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

SEQS = ["t1n", "t1c", "t2w", "t2f"]
SEQ_LABEL = {"t1n": "T1n", "t1c": "T1c", "t2w": "T2w", "t2f": "T2f"}
LABEL_COLOUR = {1: ("NETC", "#e07b39"), 2: ("SNFH", "#8a7fc4"),
                3: ("ET", "#17a673"), 4: ("RC", "#d94f8a")}

# Point 4: one field of view for all five panels, wide enough to place the
# lesion anatomically. 30 voxels of padding is about 35 mm at this spacing.
CROP_PAD = 30

S = {"dpi": 600, "title": 7.4, "small": 5.9, "tiny": 5.2,
     "hair": 0.45, "panel": 7.0}        # point 1: panel label 8.4 -> 7.0
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


def load_and_validate(case):
    """Load and check before drawing. A failure stops the figure; no case is
    substituted and nothing is transposed blindly."""
    ref_path = f"{GT_DIR}/{case}.nii.gz"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(ref_path)
    ref = nib.load(ref_path)
    gt_file = np.asanyarray(ref.dataobj)

    paths, vols = {}, {}
    for s in SEQS:
        hit = (glob.glob(f"{DRIVE}/**/{case}-{s}.nii.gz", recursive=True)
               or glob.glob(f"{DRIVE}/**/{case}_{s}.nii.gz", recursive=True))
        if not hit:
            raise FileNotFoundError(f"{case}: sequence {s} not found")
        img = nib.load(hit[0])
        if img.shape != ref.shape:
            raise ValueError(f"{case} {s}: shape {img.shape} != {ref.shape}")
        if not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ValueError(f"{case} {s}: affine differs from the reference")
        paths[s] = hit[0]
        vols[s] = np.asanyarray(img.dataobj).transpose(INV)

    labels = set(int(v) for v in np.unique(gt_file)) - {0}
    unknown = labels - set(LABEL_COLOUR)
    if unknown:
        raise ValueError(f"{case}: unexpected label values {sorted(unknown)}")

    planes, codes = plane_axes(ref.affine)
    zooms = tuple(float(v) for v in ref.header.get_zooms()[:3])
    spacing = tuple(zooms[INV[k]] for k in range(3))
    return {"gt": gt_file.transpose(INV), "vols": vols, "planes": planes,
            "codes": codes, "spacing": spacing, "labels": sorted(labels),
            "paths": paths, "ref_path": ref_path}


def window(vol, extra=None, lo=0.5, hi=99.5):
    """Fixed percentile window, identical for every sequence. The release is
    not intensity-normalised, so a fixed absolute window would render most
    cases black or white; the percentiles are not tuned per image."""
    ref = vol[vol > 0]
    if extra is not None and extra.any():
        ref = np.concatenate([ref, vol[extra]])
    if not ref.size:
        return vol
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((vol - a) / max(b - a, 1e-6), 0, 1)


def main():
    print("=" * 74)
    print(" FIGURE 1 · DATASET OVERVIEW · final refinement of candidate A")
    print("=" * 74)
    d = load_and_validate(CASE)
    gt, vols, sp = d["gt"], d["vols"], d["spacing"]
    a = d["planes"]["axial"]

    inner = np.isin(gt, (3,))
    slice_rule = "axial slice of maximum enhancing-tumour area"
    z = None
    if inner.any():
        z = int(np.argmax(inner.sum(axis=tuple(i for i in range(3) if i != a))))
    else:
        slice_rule += " [fallback: maximum any-label area, ET absent]"
        z = int(np.argmax((gt > 0).sum(axis=tuple(i for i in range(3)
                                                  if i != a))))

    # Point 4: ONE crop, applied to all five panels, so the four modalities
    # and the reference panel are the same size and the same field of view.
    sl = np.take(gt > 0, z, axis=a)
    idx = np.argwhere(sl)
    lo = np.maximum(idx.min(0) - CROP_PAD, 0)
    hi = np.minimum(idx.max(0) + CROP_PAD + 1, sl.shape)
    box = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))

    rem = [i for i in range(3) if i != a]
    mm_x, mm_y = sp[rem[0]], sp[rem[1]]
    aspect = mm_y / mm_x

    base = {s: np.take(window(vols[s], gt > 0), z, axis=a)[box] for s in SEQS}
    lab = {l: np.take(gt == l, z, axis=a)[box] for l in LABEL_COLOUR}
    present = {l for l in LABEL_COLOUR if lab[l].any()}

    print(f"""
  case            {CASE}
  selection rule  {RULE}
                  {VALUE}
  slice rule      {slice_rule}
  slice index     {z}   (array axis {a}, plane from affine {d['codes']})
  spacing         {tuple(round(v, 3) for v in sp)} mm, array order
  panel aspect    {aspect:.4f}   (1.0 would squash this plane by
                  {100 * (1 - min(sp[rem[0]], sp[rem[1]]) / max(sp[rem[0]], sp[rem[1]])):.0f}%)
  crop            one field of view for all five panels, pad {CROP_PAD} voxels
                  = {CROP_PAD * mm_x:.0f} mm, so the lesion is placed
                  anatomically without breaking panel equality
  labels present  {sorted(int(l) for l in present)} of {d['labels']}
""")

    # ---- the figure -----------------------------------------------------
    fig, ax = plt.subplots(1, 5, figsize=(7.0, 1.62))
    # Point 3: a tighter block. The legend row is reserved in the layout
    # rather than pushed outside it, which is what left a band of empty page
    # under the earlier version.
    fig.subplots_adjust(wspace=0.022, left=0.012, right=0.995,
                        top=0.885, bottom=0.135)

    for k, s in enumerate(SEQS):
        ax[k].imshow(np.rot90(base[s]), cmap="gray", vmin=0, vmax=1,
                     interpolation="bicubic", aspect=aspect, rasterized=True)
        ax[k].set_title(SEQ_LABEL[s], fontsize=S["title"], pad=2.5)

    # Point 2: "Native reference labels". More precise than "Reference
    # labels", and it says annotation rather than acquisition -- the panel
    # must not read as a fifth modality.
    ax[4].imshow(np.rot90(base["t1c"]), cmap="gray", vmin=0, vmax=1,
                 interpolation="bicubic", aspect=aspect, rasterized=True)
    ax[4].set_title("Native reference labels", fontsize=S["title"], pad=2.5)
    for l, (nm, c) in LABEL_COLOUR.items():
        m = lab[l]
        if not m.any():
            continue
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = to_rgba(c, SEG_ALPHA)
        ax[4].imshow(np.rot90(rgba), interpolation="nearest", aspect=aspect,
                     rasterized=True)
        ax[4].contour(np.rot90(m.astype(float)), levels=[0.5], colors=[c],
                      linewidths=SEG_LW, antialiased=True)
        ax[4].set_aspect(aspect)

    for axis in ax:
        axis.set_xticks([]); axis.set_yticks([])
        for spn in axis.spines.values():
            spn.set_linewidth(S["hair"]); spn.set_color("#333333")

    # scale bar, on the bottom corner holding less of the mask
    x0, x1 = ax[0].get_xlim(); y0, y1 = ax[0].get_ylim()
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
    ax[0].add_patch(bar)
    t = ax[0].text(xl + w / 2, yb + 0.022 * abs(y1 - y0), "20 mm",
                   color="white", fontsize=S["tiny"], ha="center",
                   va="bottom", zorder=8)
    t.set_path_effects(stroke)

    # Point 1: smaller, lighter, and close to the figure edge rather than
    # hanging off the first panel.
    fig.text(0.004, 0.972, "(a)", fontsize=S["panel"], fontweight="bold",
             ha="left", va="top")

    # Point 3: one compact row, hairline swatches, tight spacing. A label
    # absent from this slice is still listed -- it belongs to the scheme --
    # but marked, so nobody hunts for a colour that is not there.
    handles = []
    for l, (nm, c) in LABEL_COLOUR.items():
        txt = nm if l in present else f"{nm} (not on this slice)"
        handles.append(Patch(facecolor=to_rgba(c, SEG_ALPHA), edgecolor=c,
                             linewidth=0.45, label=txt))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=S["small"], bbox_to_anchor=(0.5, -0.005),
               handlelength=0.85, handleheight=0.8, handletextpad=0.4,
               columnspacing=1.0, borderpad=0.0, borderaxespad=0.0)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1_dataset_final.{ext}")
    plt.close(fig)

    man = {"figure": "1", "version": "final refinement of candidate A",
           "case": CASE, "selection_rule": RULE, "quantitative_value": VALUE,
           "slice_rule": slice_rule, "slice_index": z,
           "plane": "axial", "array_axis": a, "affine_axcodes": d["codes"],
           "spacing_mm_array_order": [round(v, 4) for v in sp],
           "panel_aspect": round(aspect, 4), "mm_per_px": round(mm_x, 4),
           "crop": f"one field of view for all five panels, pad {CROP_PAD} "
                   f"voxels ({CROP_PAD * mm_x:.0f} mm)",
           "labels_present_on_slice": sorted(int(l) for l in present),
           "labels_in_case": d["labels"],
           "label_colours": {n: c for n, c in LABEL_COLOUR.values()},
           "windowing": "percentile 0.5-99.5, identical for every sequence",
           "interpolation": {"mri": "bicubic", "segmentation": "nearest"},
           "source_files": [d["ref_path"]] + [d["paths"][s] for s in SEQS],
           "spatial_validation": "PASS (shape and affine per sequence)",
           "label_validation": "PASS (values within the four defined labels)",
           "refinements": [
               "panel label smaller and at the figure edge",
               "'Native reference labels' rather than 'Reference labels'",
               "compact single-row legend",
               "four modalities equal, one row, shared field of view"],
           "output": f"{OUT}/fig1_dataset_final.{{pdf,png}}"}
    with open(f"{OUT}/fig1_manifest.json", "w") as f:
        json.dump(man, f, indent=2, default=str)

    print(f"""  written
    {OUT}/fig1_dataset_final.pdf
    {OUT}/fig1_dataset_final.png
    {OUT}/fig1_manifest.json

  The four refinements are applied. The field of view is now wide enough to
  place the lesion anatomically, which is what candidate C had and A lacked,
  without giving the reference panel a different size from the modalities.""")


if __name__ == "__main__":
    main()
