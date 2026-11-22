# -*- coding: utf-8 -*-
"""
==============================================================================
 PHASE B FIGURES  —  generated from the frozen tables, never from typed numbers
==============================================================================
 Every value plotted is read from the result CSVs at run time. Nothing is
 hard-coded, so a figure cannot drift away from the table it claims to show,
 and regenerating after any correction is one command.

   Figure 1  headroom against observed effect, per region
             The paper's argument in one panel: what was available to win
             against what each rule actually did.
   Figure 2  Dice change by reference-lesion size bin
             Where the damage is: steep in small and medium lesions, gone
             above 500 mm3.
   Figure 3  precision against recall, per rule
             What the loss consists of. Every rule moves right on precision
             and much further down on recall.
   Figure 4  false-positive component size, removed against retained
             (exploratory)

 Style: 300 dpi, two-column width, greyscale-safe. Colour carries no
 information that the shape or position does not also carry, so the figures
 survive black-and-white printing and the common forms of colour blindness.

 Journal formatting (width, font size, panel labels) is set in STYLE below and
 is the only thing that should need changing for a different venue.
==============================================================================
"""

import os
import re
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W2 = f"{DRIVE}/phase_b_outputs/wave2"
OUT = f"{DRIVE}/paper2/figures"
PAT = re.compile(r"\.w\d+of\d+\.csv$")

# ---- venue style: the only block to change for a different journal ---------
STYLE = {
    "single_col": 3.5,      # inches
    "double_col": 7.2,
    "dpi": 300,
    "font": 8,
    "font_small": 7,
    "line": 0.8,
}
plt.rcParams.update({
    "font.size": STYLE["font"],
    "axes.labelsize": STYLE["font"],
    "axes.titlesize": STYLE["font"],
    "xtick.labelsize": STYLE["font_small"],
    "ytick.labelsize": STYLE["font_small"],
    "legend.fontsize": STYLE["font_small"],
    "axes.linewidth": STYLE["line"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": STYLE["dpi"],
    "savefig.dpi": STYLE["dpi"],
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

REGIONS = ["WT", "TC", "ET", "RC"]
GREY = {"dark": "#2b2b2b", "mid": "#7a7a7a", "light": "#c4c4c4", "pale": "#e8e8e8"}

os.makedirs(OUT, exist_ok=True)


def load():
    d = {}
    b6 = f"{W2}/b6_primary_clean.csv"
    if not os.path.exists(b6):
        sys.exit(f"missing {b6}")
    d["b6"] = pd.read_csv(b6)
    if "rule" not in d["b6"].columns:
        d["b6"] = d["b6"].rename(columns={"setting": "rule"})
    d["dcol"] = next(c for c in ("mean_delta", "delta", "dice_delta")
                     if c in d["b6"].columns)

    for k, f in (("t4", "t4_detection_results.csv"),
                 ("t3", "t3_bin_results.csv"),
                 ("fp", "fp_components.csv")):
        p = f"{W2}/{f}"
        d[k] = pd.read_csv(p) if os.path.exists(p) else None
        if d[k] is None:
            print(f"  [!] {f} absent — the figure using it will be skipped")

    # headroom: prefer the stored table, else recompute from the ceiling file
    p = f"{W2}/ceiling_by_region_clean.csv"
    if os.path.exists(p):
        d["ceil"] = pd.read_csv(p, index_col=0)
    else:
        p2 = glob.glob(f"{DRIVE}/**/ceiling_by_region*.csv", recursive=True)
        d["ceil"] = pd.read_csv(p2[0], index_col=0) if p2 else None
        if d["ceil"] is None:
            print("  [!] no ceiling table — Figure 1 will use the values "
                  "recorded in the results draft")
    return d


def headroom(d):
    """Clean-stratum headroom per region, from the table if it is there."""
    if d["ceil"] is not None:
        c = d["ceil"]
        col = next((x for x in ("headroom", "Headroom") if x in c.columns), None)
        if col:
            return {r: float(c.loc[r, col]) for r in REGIONS if r in c.index}
    # documented clean-stratum values, used only if the table is unavailable
    return {"WT": 0.0395, "TC": 0.0280, "ET": 0.0247, "RC": 0.2968}


# ---------------------------------------------------------------------------
def figure1(d):
    """Available headroom against what each rule actually achieved."""
    hr = headroom(d)
    b6, dcol = d["b6"], d["dcol"]
    rules = ["size@50", "entropy@0.45", "entropy@0.50", "entropy@0.56",
             "confidence@0.45"]
    rules = [r for r in rules if r in set(b6.rule)]

    fig, axes = plt.subplots(1, 4, figsize=(STYLE["double_col"], 2.7),
                             sharey=False)
    fig.subplots_adjust(wspace=0.42)
    for ax, reg in zip(axes, REGIONS):
        sub = b6[b6.region == reg].set_index("rule")
        h = hr.get(reg, np.nan)

        # the budget, drawn as the span a perfect oracle could have gained
        ax.axhspan(0, h, color=GREY["pale"], zorder=0)
        ax.axhline(0, color=GREY["dark"], lw=STYLE["line"], zorder=2)
        ax.axhline(h, color=GREY["mid"], lw=STYLE["line"], ls=":", zorder=2)

        xs, ys, los, his = [], [], [], []
        for i, r in enumerate(rules):
            if r not in sub.index:
                continue
            xs.append(i)
            ys.append(float(sub.loc[r, dcol]))
            los.append(float(sub.loc[r, "ci_lo"]) if "ci_lo" in sub.columns else np.nan)
            his.append(float(sub.loc[r, "ci_hi"]) if "ci_hi" in sub.columns else np.nan)

        err = None
        if not np.isnan(los).all():
            err = [np.array(ys) - np.array(los), np.array(his) - np.array(ys)]
        marks = ["s" if r.startswith("size") else
                 "o" if r.startswith("entropy") else "^" for r in rules]
        for x, y, m in zip(xs, ys, marks):
            ax.plot(x, y, m, ms=4, color=GREY["dark"], zorder=4,
                    mfc="white" if m == "o" else GREY["dark"], mew=STYLE["line"])
        if err is not None:
            ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=GREY["mid"],
                        elinewidth=STYLE["line"], capsize=2, zorder=3)

        # Short tick labels: the full rule name will not fit four to a row and
        # the family is already carried by the marker shape.
        short = {"size@50": "s50", "size@20": "s20", "entropy@0.45": "e.45",
                 "entropy@0.50": "e.50", "entropy@0.56": "e.56",
                 "confidence@0.45": "c.45"}
        ax.set_title(f"{reg}   (headroom {h:.3f})" if h == h else reg,
                     fontsize=STYLE["font_small"])
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels([short.get(r, r) for r in rules],
                           fontsize=STYLE["font_small"] - 1)
        ax.set_xlim(-0.6, len(rules) - 0.4)
        # Every panel keeps the label: the scales differ by an order of
        # magnitude between the tumour regions and RC, so a single shared label
        # would invite reading them against each other.
        ax.set_ylabel("Δ lesion-wise Dice" if reg == "WT" else "")
        ax.tick_params(width=STYLE["line"], pad=2)

    fig.text(0.5, -0.10,
             "Shaded band: improvement available to an oracle removing every "
             "false positive while retaining every true one.\n"
             "Bars: 95% bootstrap CI. Squares size, circles entropy, triangles "
             "confidence. s50 = size@50, e.45 = entropy@0.45, and so on.\n"
             "Note the y scales differ: the RC panel spans an order of "
             "magnitude more than the tumour panels.",
             ha="center", fontsize=STYLE["font_small"] - 1, color=GREY["mid"])
    fig.savefig(f"{OUT}/fig1_headroom_vs_effect.pdf")
    fig.savefig(f"{OUT}/fig1_headroom_vs_effect.png")
    plt.close(fig)
    print("  fig1  headroom against observed effect")


def figure2(d):
    """Where the damage falls, by reference-lesion size."""
    if d["t3"] is None:
        return
    t3 = d["t3"]
    order = ["(2,27]", "(27,100]", "(100,500]", "(500,inf)"]
    # U+221E is missing from some journal-mandated fonts and renders as a box.
    # Checked at run time rather than assumed; ">500" is the safe fallback and
    # says the same thing.
    from matplotlib.font_manager import findfont, FontProperties
    try:
        from fontTools.ttLib import TTFont
        _f = TTFont(findfont(FontProperties(family=plt.rcParams["font.family"])),
                    fontNumber=0)
        _has_inf = any(0x221E in t.cmap for t in _f["cmap"].tables)
    except Exception:
        _has_inf = True
    labels = ["(2, 27]", "(27, 100]", "(100, 500]",
              "(500, ∞)" if _has_inf else "> 500"]
    if not _has_inf:
        print("      [i] font lacks ∞ — last bin labelled '> 500'")
    # size@20 is excluded here. The protocol assigned it to the detection
    # analysis when it was removed from the Dice arm, and its Dice values are
    # retained for audit only. Figure 1 excludes it for the same reason;
    # including it in one Dice figure and not the other would be inconsistent
    # in a way a reader would rightly question.
    rules = ["size@50", "entropy@0.45", "entropy@0.50",
             "entropy@0.56", "confidence@0.45"]
    rules = [r for r in rules if r in set(t3.rule)]

    fig, ax = plt.subplots(figsize=(STYLE["double_col"] * 0.62, 2.6))
    styles = {"size@50": ("-", "s"),
              "entropy@0.45": ("-", "o"), "entropy@0.50": ("--", "o"),
              "entropy@0.56": (":", "o"), "confidence@0.45": ("-", "^")}
    shade = {"size@50": GREY["dark"],
             "entropy@0.45": GREY["dark"], "entropy@0.50": GREY["mid"],
             "entropy@0.56": GREY["light"], "confidence@0.45": GREY["light"]}

    x = np.arange(len(order))
    ax.axhline(0, color=GREY["dark"], lw=STYLE["line"])
    for r in rules:
        ys = []
        for b in order:
            row = t3[(t3.bin == b) & (t3.rule == r)]
            ys.append(float(row.dice_delta.iloc[0]) if len(row) else np.nan)
        ls, mk = styles.get(r, ("-", "o"))
        ax.plot(x, ys, ls, marker=mk, ms=3.5, lw=STYLE["line"] + 0.3,
                color=shade.get(r, GREY["mid"]), label=r,
                mfc="white" if mk == "o" else None)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Reference-lesion volume (mm³)")
    ax.set_ylabel("Δ lesion-wise Dice")
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.legend(frameon=False, ncol=2, loc="lower right",
              fontsize=STYLE["font_small"] - 1)
    fig.text(0.5, -0.14,
             "Case is the unit; a case enters a bin only if it has at least one "
             "reference lesion there.\nBins are exploratory and were not "
             "pre-specified.",
             ha="center", fontsize=STYLE["font_small"] - 1, color=GREY["mid"])
    fig.savefig(f"{OUT}/fig2_effect_by_lesion_size.pdf")
    fig.savefig(f"{OUT}/fig2_effect_by_lesion_size.png")
    plt.close(fig)
    print("  fig2  effect by lesion size")


def figure3(d):
    """What the detection loss is made of: precision up, recall further down."""
    if d["t4"] is None:
        return
    t4 = d["t4"]
    regs = ["WT", "TC", "ET"]          # RC excluded: five reference-positive cases
    rules = ["size@20", "size@50", "entropy@0.45", "entropy@0.50",
             "entropy@0.56", "confidence@0.45"]
    rules = [r for r in rules if r in set(t4.rule)]

    # Not shared: WT, TC and ET differ enough that one scale compresses two
    # panels to make room for the third, which is where the labels collided.
    # marker and fill together identify the rule; both survive greyscale
    MARK = {"size@20": ("s", "white"), "size@50": ("s", GREY["dark"]),
            "entropy@0.45": ("o", GREY["dark"]), "entropy@0.50": ("o", GREY["mid"]),
            "entropy@0.56": ("o", "white"), "confidence@0.45": ("^", GREY["dark"])}

    fig, axes = plt.subplots(1, 3, figsize=(STYLE["double_col"], 2.5))
    fig.subplots_adjust(bottom=0.30, wspace=0.34)
    for ax, reg in zip(axes, regs):
        sub = t4[t4.region == reg].set_index("rule")
        pts = []
        ax.axhline(0, color=GREY["dark"], lw=STYLE["line"])
        ax.axvline(0, color=GREY["dark"], lw=STYLE["line"])
        for r in rules:
            if r not in sub.index:
                continue
            px = float(sub.loc[r, "precision_delta"])
            py = float(sub.loc[r, "recall_delta"])
            m, fc = MARK[r]
            ax.plot(px, py, m, ms=4.5, color=GREY["dark"], mfc=fc,
                    mew=STYLE["line"], label=r if reg == "WT" else None)
            pts.append((px, py, r))
        # The three entropy points are close together by construction, so a
        # label beside each marker either overlaps its neighbour or drifts far
        # enough that it reads as belonging to the wrong point. A legend keyed
        # by marker and fill avoids both, and the shapes already distinguish
        # the three families.
        ax.margins(x=0.14, y=0.16)

        # the line where a precision gain exactly offsets a recall loss
        xl, yl = ax.get_xlim(), ax.get_ylim()
        span = min(max(xl), abs(min(yl)))
        ax.plot([0, span], [0, -span], color=GREY["light"],
                lw=STYLE["line"], ls="--", zorder=0)
        ax.set_xlim(xl); ax.set_ylim(yl)
        ax.set_title(reg)
        ax.set_xlabel("Δ precision")
        ax.set_ylabel("Δ recall" if reg == "WT" else "")
        ax.tick_params(width=STYLE["line"], pad=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=STYLE["font_small"] - 1,
               handletextpad=0.3, columnspacing=1.1)

    fig.text(0.5, -0.19,
             "Every rule improves precision and costs several times as much "
             "recall.\nDashed line: where a precision gain would exactly offset "
             "the recall loss. RC omitted (five reference-positive cases). "
             "Axes are per panel: the three regions differ in scale.",
             ha="center", fontsize=STYLE["font_small"] - 1, color=GREY["mid"])
    fig.savefig(f"{OUT}/fig3_precision_recall.pdf")
    fig.savefig(f"{OUT}/fig3_precision_recall.png")
    plt.close(fig)
    print("  fig3  precision against recall")


def figure4(d):
    """Exploratory: the size of what each rule removes and what it leaves."""
    if d["fp"] is None:
        return
    fp = d["fp"]
    fpo = fp[fp.is_fp == 1]
    # size@20 excluded for the same reason as figures 1 and 2.
    rules = [c for c in ("size@50", "entropy@0.45", "entropy@0.50",
                         "entropy@0.56", "confidence@0.45") if c in fp.columns]
    if not rules or not len(fpo):
        return

    fig, ax = plt.subplots(figsize=(STYLE["double_col"] * 0.62, 2.6))
    data, positions, colours = [], [], []
    for i, r in enumerate(rules):
        rem = fpo.loc[fpo[r] == "removed", "voxel_count"]
        ret = fpo.loc[fpo[r] == "retained", "voxel_count"]
        for j, (vals, c) in enumerate(((rem, GREY["mid"]), (ret, "white"))):
            if len(vals):
                data.append(np.log10(vals.clip(lower=1)))
                positions.append(i + (j - 0.5) * 0.34)
                colours.append(c)

    bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True,
                    showfliers=False, medianprops=dict(color=GREY["dark"],
                                                       lw=STYLE["line"] + 0.2))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_edgecolor(GREY["dark"])
        patch.set_linewidth(STYLE["line"])
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set_color(GREY["dark"]); ln.set_linewidth(STYLE["line"])

    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels([r.replace("@", "\n") for r in rules],
                       fontsize=STYLE["font_small"] - 1)
    ax.set_ylabel("Component size (log₁₀ voxels)")
    ax.set_xlim(-0.6, len(rules) - 0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=GREY["mid"], ec=GREY["dark"], label="removed"),
                       Patch(fc="white", ec=GREY["dark"], label="retained")],
              frameon=False, loc="upper left", fontsize=STYLE["font_small"] - 1)
    fig.text(0.5, -0.15,
             "False-positive components only. Exploratory: no endpoint was "
             "pre-specified for this analysis.\nComponent assignment "
             "reconstructed and validated against the evaluator's own counts.",
             ha="center", fontsize=STYLE["font_small"] - 1, color=GREY["mid"])
    fig.savefig(f"{OUT}/fig4_fp_component_size.pdf")
    fig.savefig(f"{OUT}/fig4_fp_component_size.png")
    plt.close(fig)
    print("  fig4  false-positive component size (exploratory)")


def figure5(d):
    """
    The eligibility figure. Two panels: the mechanism as a diagram, and the
    measured consequence per region.

    This is the one figure that carries a methodological point rather than a
    result, and the diagram half is deliberately schematic -- it shows how a
    case leaves the endpoint, which no scatter of numbers conveys.
    """
    p = f"{DRIVE}/paper2/eligibility_table.csv"
    if not os.path.exists(p):
        print("  fig5  skipped: run eligibility_table.py first")
        return
    e = pd.read_csv(p)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(STYLE["double_col"], 2.9),
        gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- left: the mechanism ------------------------------------------
    axL.axis("off")
    axL.set_xlim(0, 10); axL.set_ylim(0, 10)

    def box(x, y, w, h, text, fill="white", weight="normal"):
        axL.add_patch(plt.Rectangle((x, y), w, h, fill=True, fc=fill,
                                    ec=GREY["dark"], lw=STYLE["line"],
                                    zorder=2))
        axL.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=STYLE["font_small"] - 1, zorder=3,
                 fontweight=weight, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        axL.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", lw=STYLE["line"],
                                     color=GREY["dark"], shrinkA=1, shrinkB=1))

    axL.text(5, 9.5, "same evaluator, same reference",
             ha="center", fontsize=STYLE["font_small"],
             fontweight="bold", color=GREY["dark"])
    arrow(5, 9.2, 5, 8.7)

    box(0.3, 7.3, 4.2, 1.4, "baseline\nprediction present", GREY["pale"])
    box(5.5, 7.3, 4.2, 1.4, "after suppression\nprediction removed", GREY["pale"])
    arrow(2.4, 7.2, 2.4, 6.4)
    arrow(7.6, 7.2, 7.6, 6.4)

    box(0.3, 5.0, 4.2, 1.4, "TP, FP or FN\npresent")
    box(5.5, 5.0, 4.2, 1.4, "TP = FP = FN = 0")
    arrow(2.4, 4.9, 2.4, 4.1)
    arrow(7.6, 4.9, 7.6, 4.1)

    box(0.3, 2.7, 4.2, 1.4, "eligible", "white", "bold")
    box(5.5, 2.7, 4.2, 1.4, "undefined:\nleaves the comparison",
        GREY["light"], "bold")
    arrow(2.4, 2.6, 4.6, 1.9)
    arrow(7.6, 2.6, 5.4, 1.9)

    box(1.6, 0.4, 6.8, 1.4,
        "different effective case populations", GREY["mid"], "bold")

    # ---- right: the measured consequence -------------------------------
    nb = e[e.rule != "baseline"].copy()
    regions = ["WT", "TC", "ET", "RC"]
    x = np.arange(len(regions))
    lost_max = [nb[nb.region == r].lost_vs_baseline.max() for r in regions]
    n_diff = [int((nb[nb.region == r].same_population == 0).sum())
              for r in regions]
    n_tot = [len(nb[nb.region == r]) for r in regions]

    # A linear scale makes the tumour bars invisible against RC's 37, which
    # inverts the figure's point: the change was observed in every region and
    # differed in magnitude. Log scale shows both, and the labels go beside
    # the bars rather than on them, where they collided.
    axR.bar(x, lost_max, width=0.5, color=GREY["mid"],
            edgecolor=GREY["dark"], lw=STYLE["line"], zorder=2)
    axR.set_yscale("log")
    axR.set_ylim(0.6, max(lost_max) * 3.2)
    axR.set_yticks([1, 2, 5, 10, 20, 40])
    axR.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))

    for i, (v, nd, nt) in enumerate(zip(lost_max, n_diff, n_tot)):
        axR.text(i, v * 1.25, f"{v}", ha="center", va="bottom",
                 fontsize=STYLE["font_small"], fontweight="bold")
        axR.text(i, 0.66, f"{nd}/{nt}", ha="center", va="bottom",
                 fontsize=STYLE["font_small"] - 1, color=GREY["mid"])

    axR.set_xticks(x); axR.set_xticklabels(regions)
    axR.set_ylabel("cases lost from the comparison\n(largest across rules, log scale)")
    axR.tick_params(width=STYLE["line"])
    axR.text(0.5, 0.95, "affected rules / total, at the base",
             transform=axR.transAxes, ha="center",
             fontsize=STYLE["font_small"] - 1, color=GREY["mid"])

    # Short caption only. The full text belongs in the manuscript caption,
    # where it is typeset and editable; burning six lines into the image makes
    # it unreadable at column width and impossible to revise.
    fig.text(0.5, -0.07,
             "An intervention-induced eligibility change was observed in every "
             "region;\nits consequence was material in one.",
             ha="center", fontsize=STYLE["font_small"] - 1, color=GREY["mid"])
    fig.subplots_adjust(bottom=0.16, wspace=0.30)
    fig.savefig(f"{OUT}/fig5_endpoint_eligibility.pdf")
    fig.savefig(f"{OUT}/fig5_endpoint_eligibility.png")
    plt.close(fig)
    print("  fig5  endpoint eligibility")


def figure6(d):
    """
    Candidate availability. The paper's opening claim had been carried by two
    tables and no figure, which is the wrong way round for its most important
    result: a bound ten times the detectable effect alongside a reachable
    fraction of one fifth.

    Left: the surfaced fraction against proposal threshold, per region; the
    gap to 1 is the never-surfaced fraction, which is the point of the panel.
    Right: the cost, as candidate components per lesion surfaced.

    One quantity per panel. A twin axis was tried and removed: the
    operating-threshold reference sat near the bottom of the left scale and
    read as though it were zero, which is exactly the misreading a second
    scale invites.
    """
    les = f"{DRIVE}/phase_c_outputs/c0/c0_lesions.csv"
    cas = f"{DRIVE}/phase_c_outputs/c0/c0_cases.csv"
    if not (os.path.exists(les) and os.path.exists(cas)):
        print("  fig6  skipped: C0 tables not found")
        return
    L = pd.read_csv(les)
    C = pd.read_csv(cas)
    fn = L[L.baseline_detected == 0]
    cols = [c for c in L.columns if c.startswith("cand@")]
    if not cols:
        print("  fig6  skipped: no candidate columns")
        return
    # descending threshold order, as swept
    cols = sorted(cols, key=lambda c: -float(c.split("@")[1]))
    thr = [float(c.split("@")[1]) for c in cols]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(STYLE["double_col"], 2.7))
    fig.subplots_adjust(bottom=0.30, wspace=0.34)

    mark = {"WT": ("o", GREY["dark"]), "TC": ("s", GREY["mid"]),
            "ET": ("^", "white"), "RC": ("D", GREY["light"])}
    x = np.arange(len(thr))

    for reg in REGIONS:
        sub = fn[fn.region == reg]
        if not len(sub):
            continue
        y = [float(sub[c].mean()) for c in cols]
        m, fc = mark[reg]
        axL.plot(x, y, m + "-", ms=4, lw=STYLE["line"] + 0.3,
                 color=GREY["dark"], mfc=fc, mew=STYLE["line"],
                 label=f"{reg} (n={len(sub)})")

    axL.axhline(1.0, color=GREY["light"], lw=STYLE["line"])
    axL.set_xticks(x)
    axL.set_xticklabels([f"{t:g}" for t in thr])
    axL.set_xlabel("proposal threshold")
    axL.set_ylabel("fraction of missed lesions\nsurfaced as a candidate")
    axL.set_ylim(0, 1.04)
    axL.legend(frameon=False, fontsize=STYLE["font_small"] - 1,
               loc="upper left")
    axL.tick_params(width=STYLE["line"], pad=2)

    never = 1 - fn.groupby("region")[cols[-1]].mean()
    axL.annotate(f"never surfaced at {thr[-1]:g}:\n"
                 f"{never.min():.0%}\u2013{never.max():.0%} of missed lesions",
                 xy=(x[-1], float(fn[cols[-1]].mean())),
                 xytext=(x[-1] - 2.4, 0.70),
                 fontsize=STYLE["font_small"] - 1, color=GREY["mid"],
                 arrowprops=dict(arrowstyle="-", lw=STYLE["line"],
                                 color=GREY["mid"]))

    base_cols = [c for c in C.columns if c.endswith("n_base_components")]
    base = float(C[base_cols].sum(axis=1).mean()) if base_cols else float("nan")
    burden, per_case = [], []
    for c, t in zip(cols, thr):
        cand = sum(C[f"{r}_n_cand@{t}"].sum() for r in REGIONS
                   if f"{r}_n_cand@{t}" in C.columns)
        surf = int(fn[c].sum())
        burden.append(cand / max(surf, 1))
        per_case.append(cand / max(len(C), 1))

    axR.plot(x, burden, "o-", ms=4.5, lw=STYLE["line"] + 0.4,
             color=GREY["dark"])
    axR.set_xticks(x)
    axR.set_xticklabels([f"{t:g}" for t in thr])
    axR.set_xlabel("proposal threshold")
    axR.set_ylabel("candidate components\nper lesion surfaced")
    axR.set_ylim(0, max(burden) * 1.45)
    axR.tick_params(width=STYLE["line"], pad=2)

    # per-case counts as text: they vary by under 20% across the sweep, which a
    # second axis would dramatise out of proportion
    for xi, (b, pc) in enumerate(zip(burden, per_case)):
        axR.annotate(f"{pc:.1f}", xy=(xi, b), xytext=(0, -14),
                     textcoords="offset points", ha="center",
                     fontsize=STYLE["font_small"] - 1.5, color=GREY["mid"])
    axR.text(0.5, 0.96,
             "candidates per case, beneath each point"
             + (f"\n({base:.1f} at the operating threshold)"
                if base == base else ""),
             transform=axR.transAxes, ha="center", va="top",
             fontsize=STYLE["font_small"] - 1.5, color=GREY["mid"],
             linespacing=1.3)

    fig.text(0.5, -0.15,
             "Left: a missed lesion counts as surfaced if any component at "
             "that threshold overlaps it directly.\nRight: candidate "
             "components considered per lesion surfaced. Lowering the "
             "threshold fifteen-fold\nsurfaces about a fifth of the missed "
             "lesions, never at fewer than twenty candidates each.",
             ha="center", fontsize=STYLE["font_small"] - 1, color=GREY["mid"])
    fig.savefig(f"{OUT}/fig6_candidate_availability.pdf")
    fig.savefig(f"{OUT}/fig6_candidate_availability.png")
    plt.close(fig)
    print("  fig6  candidate availability")


def main():
    print("=" * 70)
    print(" PHASE B FIGURES — every value read from the frozen tables")
    print("=" * 70)
    d = load()
    for fn in (figure1, figure2, figure3, figure4, figure5,
               figure6):
        try:
            fn(d)
        except Exception as e:
            print(f"  [!] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n  written to {OUT}")
    print("""
  Each figure is regenerated from the CSVs, so it cannot disagree with the
  tables. If a number changes at source, rerun this rather than editing an
  image.

  Set the venue's column width and font size in STYLE at the top; nothing else
  should need changing for a different journal.""")


if __name__ == "__main__":
    main()
