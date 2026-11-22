# -*- coding: utf-8 -*-
"""
Pipeline schematic of the segmentation system used in this study (Methods
figure). Horizontal flow: training data -> dataset fingerprint -> rule-based
configuration -> 3D residual-encoder U-Net trained by five-fold
cross-validation -> frozen ensemble -> thresholds -> (suppression rule acts
here) -> nested label -> lesion-wise evaluation. Original block diagram.
Output: figures/fig_nnunet_schematic.pdf (and .png).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Ellipse, Arc, Circle, Polygon)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
S = {"title": 5.6, "label": 5.0, "small": 4.6, "tiny": 4.2}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": S["label"], "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42})

INK, GREY, LIGHT = "#2a2a2a", "#7a7a7a", "#bdbdbd"
PANEL = "#f0f0f0"
ACC = "#2e8b57"
STUDY = "#d98c3a"
ENC, DEC = "#dbe7f3", "#e9f0e3"
REGC = {"WT": "#00b8c4", "TC": "#c452d6", "ET": "#3d72e8", "RC": "#e14a8a"}

W, H = 170, 46
fig, ax = plt.subplots(figsize=(7.2, 7.2 * H / W))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
MID = 24.0   # the main flow line


def txt(x, y, s, fs=S["small"], color=INK, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, linespacing=1.2,
            **kw)


def panel(x, y, w, h, fc=PANEL, ec="none", lw=0.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=1))


def arrow(x0, y0, x1, y1, lw=0.9, color=INK, ms=6, z=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


def database(cx, cy, w=7.5, h=9.0, n=4, color=GREY):
    ry = 1.0
    y0, y1 = cy - h / 2 + ry, cy + h / 2 - ry
    ax.add_patch(Rectangle((cx - w / 2, y0), w, y1 - y0, facecolor="white",
                           edgecolor="none", zorder=3))
    ax.plot([cx - w / 2] * 2, [y0, y1], color=color, lw=0.7, zorder=4)
    ax.plot([cx + w / 2] * 2, [y0, y1], color=color, lw=0.7, zorder=4)
    for i in range(n):
        y = y0 + i * (y1 - y0) / (n - 1)
        if i < n - 1:
            ax.add_patch(Arc((cx, y), w, 2 * ry, theta1=180, theta2=360,
                             edgecolor=color, linewidth=0.7, zorder=4))
    ax.add_patch(Ellipse((cx, y1), w, 2 * ry, facecolor="white",
                         edgecolor=color, linewidth=0.7, zorder=5))


def gear(cx, cy, r=1.5, color=ACC, teeth=8):
    th = np.linspace(0, 2 * np.pi, teeth * 2, endpoint=False)
    rr = np.where(np.arange(teeth * 2) % 2 == 0, r, r * 0.72)
    ax.add_patch(Polygon(np.c_[cx + rr * np.cos(th), cy + rr * np.sin(th)],
                         closed=True, facecolor=color, edgecolor="none",
                         zorder=6))
    ax.add_patch(Circle((cx, cy), r * 0.32, facecolor="white",
                        edgecolor="none", zorder=7))


def bars(cx, cy, w=7.0, h=6.0):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="white",
                           edgecolor=LIGHT, linewidth=0.5, zorder=3))
    hs = [0.35, 0.8, 0.55, 1.0, 0.45, 0.7]
    cols = ["#f28e2b", "#4e79a7", "#e15759", "#59a14f", "#76b7b2", "#b07aa1"]
    bw = (w - 1.4) / len(hs)
    for i, (hh, c) in enumerate(zip(hs, cols)):
        ax.add_patch(Rectangle((cx - w / 2 + 0.7 + i * bw, cy - h / 2 + 0.6),
                               bw * 0.78, hh * (h - 1.4), facecolor=c,
                               edgecolor="none", zorder=4))


def unet(cx, cy, w=15, h=9):
    L = 4
    ys, xes, xds, bws, bhs = [], [], [], [], []
    for i in range(L):
        s = 1.0 - 0.16 * i
        bw, bh = 2.4 * s, 2.0 * s
        y = cy + h / 2 - 0.4 - i * (h - 2.4) / (L - 1) - bh
        xe = cx - w / 2 + 0.5 + i * 1.5
        xd = cx + w / 2 - 0.5 - i * 1.5 - bw
        ys.append(y); xes.append(xe); xds.append(xd); bws.append(bw); bhs.append(bh)
        for x, fc in ((xe, ENC), (xd, DEC)):
            ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                                        boxstyle="round,pad=0,rounding_size=0.25",
                                        facecolor=fc, edgecolor=INK, linewidth=0.4,
                                        zorder=4))
        ax.plot([xe + 0.3 * bw, xe + 0.7 * bw], [y + 0.5 * bh] * 2, color=INK,
                lw=0.35, zorder=5)
        arrow(xe + bw, y + bh / 2, xd, y + bh / 2, lw=0.35, color=LIGHT, ms=2.5)
    for i in range(L - 1):
        arrow(xes[i] + bws[i] / 2, ys[i], xes[i + 1] + bws[i + 1] / 2,
              ys[i + 1] + bhs[i + 1], lw=0.4, ms=3)
        arrow(xds[i + 1] + bws[i + 1] / 2, ys[i + 1] + bhs[i + 1],
              xds[i] + bws[i] / 2, ys[i], lw=0.4, ms=3)


def cube(cx, cy, s=3.4, fill=None):
    d = s * 0.36
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, facecolor="white",
                           edgecolor=INK, linewidth=0.45, zorder=4))
    ax.add_patch(Polygon([(cx - s / 2, cy + s / 2), (cx - s / 2 + d, cy + s / 2 + d),
                          (cx + s / 2 + d, cy + s / 2 + d), (cx + s / 2, cy + s / 2)],
                         facecolor="#e8e8e8", edgecolor=INK, linewidth=0.45, zorder=4))
    ax.add_patch(Polygon([(cx + s / 2, cy - s / 2), (cx + s / 2 + d, cy - s / 2 + d),
                          (cx + s / 2 + d, cy + s / 2 + d), (cx + s / 2, cy + s / 2)],
                         facecolor="#d4d4d4", edgecolor=INK, linewidth=0.45, zorder=4))
    if fill:
        ax.add_patch(Circle((cx + 0.1 * s, cy - 0.05 * s), s * 0.17,
                            facecolor=fill, edgecolor="none", zorder=5))
        ax.add_patch(Circle((cx - 0.25 * s, cy + 0.22 * s), s * 0.08,
                            facecolor=fill, edgecolor="none", zorder=5))


# ============================================================== 1 · data ---
database(6.5, MID)
txt(6.5, MID + 8.0, "Training release\n1,296 cases\n4 sequences", fs=S["tiny"])
arrow(10.8, MID, 14.0, MID)

# ====================================================== 2 · fingerprint ---
gear(16.0, MID)
arrow(18.0, MID, 20.5, MID)
bars(24.5, MID)
txt(24.5, MID - 5.0, "Dataset fingerprint\nspacing · intensity\nimage and target size",
    fs=S["tiny"], va="top")
gear(30.3, MID)
txt(23.0, MID + 8.0, "fixed rules", fs=S["tiny"], color=GREY)
arrow(32.3, MID, 35.5, MID)

# ================================================== 3 · configuration ----
panel(35.0, 2.0, 36.5, 42.0)
txt(53.25, 40.8, "AUTOMATIC CONFIGURATION", fs=S["title"] - 0.9, fontweight="bold")
txt(37.0, 35, "Inferred from the fingerprint", fs=S["label"] - 0.4, ha="left",
    fontweight="bold")
txt(37.0, 32.2, "• target spacing and resampling\n• intensity normalisation\n"
    "• patch size 160 × 224 × 192\n• batch size 2\n• encoder depth and widths",
    fs=S["tiny"], ha="left", va="top")
txt(37.0, 17.5, "Fixed by the framework", fs=S["label"] - 0.4, ha="left",
    fontweight="bold")
txt(37.0, 14.7, "• 3D full-resolution U-Net\n   with residual encoder\n"
    "• Dice + cross-entropy loss\n• SGD, 1,000 epochs\n• data augmentation",
    fs=S["tiny"], ha="left", va="top")
arrow(71.6, MID, 75.0, MID)

# ===================================================== 4 · training -----
panel(75.5, 1.0, 28.0, 43.0, fc="white", ec=INK)
txt(89.5, 40.8, "NETWORK TRAINING", fs=S["title"] - 0.5, fontweight="bold")
txt(89.5, 36.6, "five-fold cross-validation", fs=S["tiny"], color=GREY)
unet(89.5, 27.5, w=16, h=10)
txt(89.5, 20.8, "ResEnc 3D U-Net", fs=S["tiny"], color=GREY)
txt(89.5, 7.3, "fold", fs=S["tiny"], color=GREY)
xs = [79.5, 84.5, 89.5, 94.5, 99.5]
for k, x in enumerate(xs):
    sel = k in (1, 2, 4)
    cube(x, 13.5, s=2.9, fill=REGC["ET"] if sel else None)
    txt(x + 0.5, 9.3, f"{k}", fs=S["tiny"], color=INK if sel else LIGHT)
# bracket under the selected folds
txt(89.5, 4, "folds 1, 2, 4 selected\n(companion paper)", fs=S["tiny"],
    color=GREY)
arrow(104.1, MID, 107.5, MID)

# ============================================== 5 · frozen inference -----
panel(108.0, 2.0, 40.0, 42.0, fc="white", ec=STUDY, lw=0.8, ls=(0, (2.5, 1.5)))
txt(128.0, 40.8, "FROZEN INFERENCE PATH", fs=S["title"], fontweight="bold")
txt(128.0, 36.6, "identical in every arm", fs=S["tiny"], color=STUDY)
txt(128.0, 33.2, "ensemble mean of folds 1, 2, 4\nno test-time augmentation", fs=S["tiny"])
for i, (r, c) in enumerate(REGC.items()):
    cube(116.0 + i * 8.0, 25.5, s=2.9, fill=c)
    txt(116.5 + i * 8.0, 21.6, f"p({r})", fs=S["tiny"])
txt(128.0, 18.6, "threshold  WT 0.40 · TC 0.35 · ET 0.45 · RC 0.60",
    fs=S["tiny"] - 0.6)
ax.add_patch(FancyBboxPatch((111.5, 11.4), 33.0, 4.4,
                            boxstyle="round,pad=0,rounding_size=0.6",
                            facecolor="#fbe9d6", edgecolor=STUDY, linewidth=0.7,
                            zorder=3))
txt(128.0, 13.6, "suppression rule acts here", fs=S["tiny"])
txt(128.0, 6.6, "nested label   WT → TC → ET → RC", fs=S["tiny"])
arrow(128.0, 17.6, 128.0, 16, lw=0.6, ms=4)
arrow(128.0, 11.2, 128.0, 8.4, lw=0.6, ms=4)
arrow(148.5, MID, 155.0, MID)

# ================================================== 6 · evaluation -------
gear(158.0, MID, r=1.7, color=INK)
txt(158.0, MID + 5.5, "Lesion-wise\nevaluator", fs=S["tiny"])
txt(158.0, MID - 4.5, "challenge\nimplementation", fs=S["tiny"], color=GREY,
    va="top")
arrow(160.0, MID, 162.5, MID, lw=0.6, ms=4)
txt(163.5, MID, "score", fs=S["tiny"], ha="left")

fig.savefig(f"{OUT}/fig_nnunet_schematic.pdf")
fig.savefig(f"{OUT}/fig_nnunet_schematic.png")
print("written", f"{OUT}/fig_nnunet_schematic.pdf")
