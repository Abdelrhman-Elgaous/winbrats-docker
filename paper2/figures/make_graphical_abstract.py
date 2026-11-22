# -*- coding: utf-8 -*-
"""
GRAPHICAL ABSTRACT, three-panel version (Figure 2 of the manuscript and the
separate Elsevier upload). Rebuilt in matplotlib with the corrected numbers:
exchange rate 0.76 (not 0.75) and THREE published mechanisms (not four).

Outputs, next to this script or in OUT:
  graphical_abstract.pdf   vector, for \includegraphics in the manuscript
  graphical_abstract.png   3984 x 1594 px (300 dpi), >= 1328 px wide
  graphical_abstract.tif   same, for the Elsevier upload
Font: Times-compatible serif (Elsevier permits Times, Arial, Courier, Symbol).
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

INK, GREY, LIGHT = "#1a1a1a", "#6e6e6e", "#9a9a9a"
BLUE, PALEBLUE = "#2b5797", "#cfdcec"
OCHRE, PALEOCHRE = "#c0701a", "#f3e2c8"
PANEL, PANELEDGE = "#f3f3f3", "#bdbdbd"
HATCH = "#555555"

W_IN, H_IN, DPI = 13.28, 5.31, 300          # Elsevier 500:200 ratio
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

def T(x, y, s, size=9.5, weight="normal", color=INK, ha="left", va="center",
      style="normal", ls=1.3, wrap=None):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            fontstyle=style, linespacing=ls, zorder=5)

def panel(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc=PANEL, ec=PANELEDGE, lw=1.0, zorder=1))

def card(x, y, w, h, fc="white", ec="#c9c9c9"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8, zorder=2))

def chevron(x, y, s=1.6):
    ax.add_patch(Polygon([(x, y + s), (x + s * 0.9, y), (x, y - s)], closed=True,
                         fc="#8a8a8a", ec="none", zorder=4))

# ---- strap line ---------------------------------------------------------
T(1.5, 38.2, "Lesion-wise brain metastasis segmentation   ·   BraTS-METS 2026   ·   "
             "nnU-Net baseline, 155 cases, four evaluation regions", 9.2, color=LIGHT)

PX = [1.5, 34.5, 67.5]; PW = 30.5; PY, PH = 1.2, 35.0
for x in PX: panel(x, PY, PW, PH)
chevron(32.4, 19.0); chevron(65.4, 19.0)

# ============ panel 1: headroom is arithmetic ==============================
x = PX[0]
T(x + 2, 33.7, "1  Headroom is arithmetic", 12.5, "bold", BLUE)
T(x + 2, 30.1, "The best any removal or recovery rule can do follows\n"
                "from the baseline error counts alone, before a method\nexists.", 10.0)
# mini chart
cx, cy, cw, ch = x + 2, 7.2, 26.5, 17.5
card(cx, cy, cw, ch)
ax0, ax1 = cx + 1.8, cx + cw - 1.2       # 0..12 scale
def sx(v): return ax0 + (ax1 - ax0) * v / 12
base = cy + 2.6
ax.plot([ax0, ax1], [base, base], color=LIGHT, lw=0.8, zorder=3)
for v in (0, 4, 8, 12):
    ax.plot([sx(v), sx(v)], [base, base + 0.5], color=LIGHT, lw=0.8, zorder=3)
    T(sx(v), base - 1.0, str(v), 8.5, color=LIGHT, ha="center")
ax.plot([sx(1), sx(1)], [base, cy + ch - 1.6], color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=3)
T(sx(1) + 0.4, cy + ch - 1.9, "MDE", 8.5, color=OCHRE)
# bars
ax.add_patch(Rectangle((sx(0), cy + 10.6), sx(11) - sx(0), 3.6, fc=PALEBLUE, ec=BLUE, lw=0.8, zorder=3))
T(sx(0.5), cy + 12.4, "recovering missed lesions", 9.0)
T(sx(10.6), cy + 12.4, "10–11×", 9.8, "bold", BLUE, ha="right")
ax.add_patch(Rectangle((sx(0), cy + 5.3), sx(2.1) - sx(0), 3.6, fc=PALEBLUE, ec=BLUE, lw=0.8, zorder=3))
T(sx(2.6), cy + 7.1, "removing false positives", 9.0)
T(sx(11.4), cy + 7.1, "1.3–2.1×", 9.8, "bold", BLUE, ha="right")
T(x + PW / 2, 4.9, "oracle improvement, in units of the minimum detectable effect\n"
                   "tumour regions; the resection cavity is separate", 8.3, color=LIGHT, ha="center")

# ============ panel 2: availability is not ==================================
x = PX[1]
T(x + 2, 33.7, "2  Availability is not", 12.5, "bold", BLUE)
T(x + 2, 30.4, "A ceiling counts errors. It does not ask whether the\n"
                "model produces anything a rule could act on.", 10.0)
cx, cy, cw, ch = x + 2, 19.3, 26.5, 8.2
card(cx, cy, cw, ch)
T(cx + 1.2, cy + ch - 1.3, "all missed lesions", 8.6, color=LIGHT)
bx0, bx1, by, bh = cx + 1.2, cx + cw - 1.2, cy + 2.9, 3.0
f = 0.218
ax.add_patch(Rectangle((bx0, by), (bx1 - bx0) * f, bh, fc=PALEBLUE, ec=BLUE, lw=0.8, zorder=3))
ax.add_patch(Rectangle((bx0 + (bx1 - bx0) * f, by), (bx1 - bx0) * (1 - f), bh, fc="white",
                       ec=HATCH, lw=0.8, hatch="////", zorder=3))
T(bx0, cy + 1.2, "21.8% surfaced", 8.8, "bold", BLUE)
T(bx1, cy + 1.2, "76–79% never surface", 8.8, "bold", INK, ha="right")
T(x + 2, 13.6, "Even a fifteen-fold lower detection threshold surfaces\n"
                "only a fifth of the missed lesions. The rest yield no\n"
                "overlapping candidate at any threshold tested, and the\n"
                "median probability inside a missed lesion is zero.", 10.0)
ax.plot([x + 2, x + PW - 2], [7.6, 7.6], color=GREY, lw=0.6, zorder=3)
T(x + 2, 5.2, "The wide ceiling is nominal, not actionable.", 10.5, "bold", OCHRE)

# ============ panel 3: the assessment decides ===============================
x = PX[2]
T(x + 2, 33.7, "3  The assessment decides", 12.5, "bold", BLUE)
card(x + 1.6, 24.4, PW - 3.2, 6.8)
T(x + 2.6, 29.3, "Six post-processing rules.", 10.0, "bold")
T(x + 16.0, 29.3, "None realised the room.", 10.0)
xx, yy = x + PW - 3.4, 29.3
ax.plot([xx-0.45, xx+0.45], [yy-0.45, yy+0.45], color=OCHRE, lw=1.6, zorder=6)
ax.plot([xx-0.45, xx+0.45], [yy+0.45, yy-0.45], color=OCHRE, lw=1.6, zorder=6)
T(x + 2.6, 26.4, "Size-based suppression harmed the tumour regions,\n"
                  "losing recall rather than gaining precision.", 10.0)
card(x + 1.6, 11.0, PW - 3.2, 12.4)
T(x + 2.6, 21.5, "One pre-registered training contrast.", 10.0, "bold")
T(x + 2.6, 17.9, "Inverse-size loss weighting raised candidate\n"
                  "availability by +4.07 points (95% CI 1.92–6.67),\n"
                  "with 90% of the gain inside the targeted size class.", 10.0)
T(x + 2.6, 13.3, "False positives rose to 1.44× baseline\n"
                  "against a bound of 1.25 fixed in advance.", 10.0, color=OCHRE)
ax.add_patch(Rectangle((x + PW - 9.4, 11.8), 7.0, 3.2, fc=PALEOCHRE, ec=OCHRE, lw=0.9, zorder=3))
T(x + PW - 5.9, 13.4, "STOP", 10.5, "bold", OCHRE, ha="center")
T(x + 2, 5.9, "The exchange rate of 0.76 false positives per lesion\n"
               "recovered is compatible with three independently\n"
               "published mechanisms, which locates the limit in the\n"
               "problem rather than in any one method.", 10.0)

for ext in ("pdf", "png", "tif"):
    fig.savefig(f"{OUT}/graphical_abstract.{ext}", dpi=DPI, facecolor="white",
                bbox_inches=None, pad_inches=0)
print("written to", OUT, "| PNG size:", int(W_IN * DPI), "x", int(H_IN * DPI), "px")
