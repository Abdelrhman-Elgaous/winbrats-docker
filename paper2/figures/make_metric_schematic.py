# -*- coding: utf-8 -*-
"""
Schematic: why voxel-overlap Dice and lesion-wise Dice disagree.
One synthetic slice: a reference with one large and two small lesions, and a
prediction that finds the large one well, misses both small ones, and adds one
spurious component. Voxel Dice is high; lesion-wise Dice is low. The numbers
are computed from the drawn masks, not typed in.
Output: figures/fig_metric_schematic.pdf (and .png).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
S = {"title": 7.4, "label": 6.6, "small": 6.0, "tiny": 5.4}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": S["label"], "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42})
INK, GREY = "#2a2a2a", "#8c8c8c"
REF, PRED = "#1f78b4", "#f5a020"
TP, FN, FP = "#2078b4", "#d8342c", "#f5a020"

N = 120
yy, xx = np.mgrid[:N, :N]


def disc(cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


ref = disc(62, 50, 22) | disc(24, 92, 5) | disc(96, 86, 4)
pred = disc(64, 52, 21) | disc(30, 28, 4)      # large lesion found (offset), one spurious blob

# voxel-overlap Dice
inter = (ref & pred).sum()
dice_vox = 2 * inter / (ref.sum() + pred.sum())

# lesion-wise: match components by any overlap (the schematic's rule)
rl, nr = ndimage.label(ref)
pl, npred = ndimage.label(pred)
S_sum, matched_pred = 0.0, set()
for k in range(1, nr + 1):
    g = rl == k
    hits = set(np.unique(pl[g])) - {0}
    if hits:
        p = np.isin(pl, list(hits))
        S_sum += 2 * (g & p).sum() / (g.sum() + p.sum())
        matched_pred |= hits
n_fp = npred - len(matched_pred)
dice_lw = S_sum / (nr + n_fp)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
fig.subplots_adjust(wspace=0.03, left=0.005, right=0.995, top=0.91, bottom=0.11)
for ax in axes:
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#2a2a2a")
    ax.imshow(np.full((N, N), 0.93), cmap="gray", vmin=0, vmax=1)


def fill(ax, m, c, a=0.55):
    rgba = np.zeros(m.shape + (4,)); rgba[m] = matplotlib.colors.to_rgba(c, a)
    ax.imshow(rgba, interpolation="nearest")
    ax.contour(m.astype(float), levels=[0.5], colors=[c], linewidths=0.7)


fill(axes[0], ref, REF)
axes[0].set_title("Reference: 3 lesions", fontsize=S["title"])
fill(axes[1], pred, PRED)
axes[1].set_title("Prediction: 2 components", fontsize=S["title"])
fill(axes[2], ref & pred, TP, 0.6)
fill(axes[2], ref & ~pred, FN, 0.75)
fill(axes[2], ~ref & pred, FP, 0.75)
axes[2].set_title("Overlay", fontsize=S["title"])

# annotations on the overlay
axes[2].annotate("missed", (92, 24), (104, 10), fontsize=S["tiny"], color=FN,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=FN), ha="center")
axes[2].annotate("missed", (86, 96), (104, 110), fontsize=S["tiny"], color=FN,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=FN), ha="center")
axes[2].annotate("false\npositive", (28, 30), (14, 12), fontsize=S["tiny"],
                 color="#b36b00", arrowprops=dict(arrowstyle="-", lw=0.5,
                 color="#b36b00"), ha="center", linespacing=1.1)

fig.legend(handles=[Patch(facecolor=TP, alpha=0.6, label="detected"),
                    Patch(facecolor=FN, alpha=0.75, label="missed"),
                    Patch(facecolor=FP, alpha=0.75, label="false positive")],
           loc="lower center", ncol=3, frameon=False, fontsize=S["small"],
           bbox_to_anchor=(0.5, 0.0), handlelength=1.0, columnspacing=1.6)

fig.savefig(f"{OUT}/fig_metric_schematic.pdf")
fig.savefig(f"{OUT}/fig_metric_schematic.png")
print(f"voxel Dice {dice_vox:.3f} · lesion-wise {dice_lw:.3f} · S={S_sum:.3f} N_gt={nr} N_fp={n_fp}")
