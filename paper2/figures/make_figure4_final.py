# -*- coding: utf-8 -*-
"""
==============================================================================
 FIGURE 4 · FAILURE AND BEHAVIOUR REGIMES · final
==============================================================================
 Candidate A, refined. Four predefined regimes, one panel each:

   (a) Smallest scored lesion      4.125 mm3
   (b) High lesion multiplicity    18 lesions, 75th percentile of count
   (c) False-positive component    3,948 mm3, removed by 3 of 6 rules
   (d) Large missed lesion         1,209 mm3, missed at baseline

 THREE PRESENTATION CHANGES. NO CASE, RULE OR RATIONALE IS TOUCHED.
 -----------------------------------------------------------------
   1  The outer "(d)" is removed. It collided with the panel label "(d)",
      so one letter meant two things -- the figure and its fourth panel. A
      figure whose panels are already lettered needs no outer label.

   2  Panel (a) is cropped around the 4.125 mm3 lesion it is about. Before,
      the largest object in the panel was a different, bigger missed lesion
      in red, so a reader would take THAT for the 4.125 mm3 one. The other
      lesion is not removed from the data or masked out -- only the field of
      view changes, and enough anatomy is kept for context.

   3  Panels (a) and (b) are cropped tighter. Their fields of view extended
      well past the brain, so black background consumed area that the lesions
      needed, which matters most in print.

 SUPPLEMENTARY S1 IS NOT PRODUCED.
 ---------------------------------
 The orthogonal-views figure required all three planes to pass through one
 anatomical point, and on this case the sagittal plane through that point
 carries almost no lesion. Changing the rule to "largest lesion-bearing
 slice per orientation" would produce three good-looking panels that are no
 longer views of the same target, which is the figure's entire purpose.

 Adjusting a selection rule because the image it produced looks poor is
 selection on appearance. It is refused here for the same reason it was
 refused everywhere else in this project, and a supplementary qualitative
 panel is not worth an exception. If orthogonal validation is wanted later,
 it should be designed with a rule meant for it rather than retrofitted to
 one that failed visually.

 CASE SELECTIONS ARE FROZEN AND WERE RESOLVED QUANTITATIVELY
 -----------------------------------------------------------
   (a)  BraTS-MET-01350-004   smallest detected reference lesion above
                              4.0 mm3, twice the evaluator's 2.0 mm3 floor.
                              The 2.0029 mm3 case is deliberately NOT used:
                              it clears the floor by 0.15%, and a panel about
                              small scored lesions should not turn on
                              rounding.
   (b)  BraTS-MET-00015-000   lesion count nearest the 75th percentile, 18.
                              The cohort maximum is 180, rejected as
                              unreadable rather than as unrepresentative.
   (c)  from fp_components.csv, the largest false-positive component on which
                              the six rules do NOT agree. A component every
                              rule removes shows nothing about disagreement,
                              which is what the panel is for.
   (d)  BraTS-MET-00138-000   largest confidently missed lesion, 1,209 mm3,
                              already named in the manuscript appendix.
==============================================================================
"""

import os
import sys
import json
import glob

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

DRIVE = ns.DRIVE
GT_DIR = ns.GT_DIR
W2 = f"{DRIVE}/phase_b_outputs/wave2"
OUT = f"{DRIVE}/paper2/figures/qualitative"

WT = (1, 2, 3)
ERROR_COLOUR = {"TP": ("#1f6fb2", "True positive"),
                "FN": ("#d1342c", "False negative"),
                "FP": ("#f0921f", "False positive")}

FROZEN = {
    "a": ("BraTS-MET-01350-004",
          "smallest detected reference lesion above 4.0 mm3",
          "4.125 mm3; evaluator floor 2.0 mm3"),
    "b": ("BraTS-MET-00015-000",
          "lesion count nearest the 75th percentile",
          "18 lesions; cohort maximum 180"),
    "d": ("BraTS-MET-00138-000",
          "largest confidently missed lesion, named in the appendix",
          "1209 mm3, WT lesion 1"),
}

S = {"dpi": 600, "title": 7.4, "small": 6.0, "tiny": 5.4,
     "hair": 0.45, "panel": 7.4}
TP_ALPHA, ERR_ALPHA, SEG_LW = 0.34, 0.70, 0.70
# Tight crops, and deliberately close to one another: a wide spread of pads
# gives the panels noticeably different aspects, and a row with one short and
# one tall panel leaves the difference as vertical whitespace. (a) and (b)
# also had fields of view reaching well past the brain, and (a) must centre
# on its own lesion rather than the largest object in the slice.
PAD = {"a": 24, "b": 20, "c": 20, "d": 22}

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
    return {p: {f: k for k, f in enumerate(INV)}[a] for p, a in fa.items()}, codes


def load_case(case):
    ref_path = f"{GT_DIR}/{case}.nii.gz"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(ref_path)
    ref = nib.load(ref_path)
    hit = (glob.glob(f"{DRIVE}/**/{case}-t1c.nii.gz", recursive=True)
           or glob.glob(f"{DRIVE}/**/{case}_t1c.nii.gz", recursive=True))
    if not hit:
        raise FileNotFoundError(f"{case}: t1c not found")
    img = nib.load(hit[0])
    if img.shape != ref.shape:
        raise ValueError(f"{case}: t1c shape {img.shape} != {ref.shape}")
    if not np.allclose(img.affine, ref.affine, atol=1e-3):
        raise ValueError(f"{case}: t1c affine differs from the reference")
    prob = ns.load_mean_prob(case)
    pred = ns.build_nested_label(prob, suppressor=None)
    gt = np.asanyarray(ref.dataobj).transpose(INV)
    if pred.shape != gt.shape:
        raise ValueError(f"{case}: prediction {pred.shape} != {gt.shape}")
    planes, codes = plane_axes(ref.affine)
    zooms = tuple(float(v) for v in ref.header.get_zooms()[:3])
    return {"gt": gt, "pred": pred,
            "t1c": np.asanyarray(img.dataobj).transpose(INV),
            "planes": planes, "codes": codes,
            "spacing": tuple(zooms[INV[k]] for k in range(3)),
            "t1c_path": hit[0], "ref_path": ref_path}


def window(vol, extra=None, lo=0.5, hi=99.5):
    ref = vol[vol > 0]
    if extra is not None and extra.any():
        ref = np.concatenate([ref, vol[extra]])
    if not ref.size:
        return vol
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((vol - a) / max(b - a, 1e-6), 0, 1)


def crop_around(mask2d, pad, shape):
    """
    A square field of view centred on THIS mask.

    Centred on the mask because panel (a) exists to show one small lesion,
    and centring on the slice's largest object put a different, bigger lesion
    at the centre instead.

    SQUARE because the crop otherwise follows each target's own extent: a
    compact lesion and a sprawling one produce panels of different aspect,
    and a row holding one short and one tall panel leaves the difference as
    vertical whitespace. Four square panels share one shape, so the rows need
    no spare height.
    """
    idx = np.argwhere(mask2d)
    if not len(idx):
        return (slice(None), slice(None))
    lo, hi = idx.min(0), idx.max(0)
    centre = (lo + hi) / 2.0
    half = float(max((hi - lo).max() / 2.0 + pad, pad))
    # keep the box inside the slice while staying square
    half = min(half, min(shape) / 2.0)
    out = []
    for k in range(2):
        c = float(np.clip(centre[k], half, shape[k] - half))
        out.append(slice(int(round(c - half)), int(round(c + half))))
    return tuple(out)


def draw_panel(ax, d, target, pad, title, note):
    """One regime panel. `target` selects the slice AND the crop, so the
    panel is centred on the structure it is about."""
    a = d["planes"]["axial"]
    rem = [i for i in range(3) if i != a]
    mm_x = d["spacing"][rem[0]]
    aspect = d["spacing"][rem[1]] / d["spacing"][rem[0]]

    g = np.isin(d["gt"], WT)
    p = np.isin(d["pred"], WT)
    if not target.any():
        raise ValueError("empty target")
    z = int(round(float(np.argwhere(target)[:, a].mean())))
    tgt2 = np.take(target, z, axis=a)
    box = crop_around(tgt2, pad, tgt2.shape)

    base = np.take(window(d["t1c"], g), z, axis=a)[box]
    g2, p2 = np.take(g, z, axis=a)[box], np.take(p, z, axis=a)[box]

    ax.imshow(np.rot90(base), cmap="gray", vmin=0, vmax=1,
              interpolation="bicubic", aspect=aspect, rasterized=True)
    for m, k, al in ((g2 & p2, "TP", TP_ALPHA), (g2 & ~p2, "FN", ERR_ALPHA),
                     (~g2 & p2, "FP", ERR_ALPHA)):
        if not m.any():
            continue
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = to_rgba(ERROR_COLOUR[k][0], al)
        ax.imshow(np.rot90(rgba), interpolation="nearest", aspect=aspect,
                  rasterized=True)
        if k != "TP":
            ax.contour(np.rot90(m.astype(float)), levels=[0.5],
                       colors=[ERROR_COLOUR[k][0]], linewidths=SEG_LW,
                       antialiased=True)
            ax.set_aspect(aspect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(S["hair"]); sp.set_color("#333333")
    ax.set_title(title, fontsize=S["title"], pad=2.2)
    ax.set_xlabel(note, fontsize=S["tiny"], labelpad=2.2)

    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w = 10.0 / mm_x
    disp = np.rot90((g2 | p2).astype(bool))
    band = disp[int(0.75 * disp.shape[0]):, :]
    h = band.shape[1] // 2
    left = band.size and band[:, h:].sum() > band[:, :h].sum()
    xl = (min(x0, x1) + 0.05 * abs(x1 - x0)) if left else \
         (max(x0, x1) - 0.05 * abs(x1 - x0) - w)
    yb = max(y0, y1) - 0.06 * abs(y1 - y0)
    stroke = [pe.withStroke(linewidth=1.0, foreground="#0a0a0a")]
    bar = Rectangle((xl, yb), w, 0.016 * abs(y1 - y0), facecolor="white",
                    edgecolor="none", zorder=8)
    bar.set_path_effects(stroke)
    ax.add_patch(bar)
    t = ax.text(xl + w / 2, yb + 0.024 * abs(y1 - y0), "10 mm",
                color="white", fontsize=S["tiny"], ha="center", va="bottom",
                zorder=8)
    t.set_path_effects(stroke)
    return {"slice_index": z, "aspect": round(aspect, 4),
            "mm_per_px": round(mm_x, 4), "crop_pad_voxels": pad}


def resolve_c():
    """Panel (c): the largest false-positive component on which the six rules
    disagree, from the existing validated table. The rule columns may be
    strings, so they are coerced explicitly and an unrecognised value stops
    the panel rather than being read as False."""
    f = f"{W2}/fp_components.csv"
    if not os.path.exists(f):
        return None, "fp_components.csv not found"
    d = pd.read_csv(f)
    is_fp = pd.to_numeric(d["is_fp"], errors="coerce")
    if is_fp.isna().all():
        is_fp = d["is_fp"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "t", "yes", "y"}).astype(int)
    fp = d[is_fp == 1].copy()
    rules = [c for c in ("size@20", "size@50", "entropy@0.45",
                         "entropy@0.50", "entropy@0.56", "confidence@0.45")
             if c in fp.columns]
    if not len(fp) or not rules:
        return None, "no false-positive rows or rule columns"
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
                          f"interpret: {unknown[:3]}")
        dec[r] = norm.isin(TRUE).astype(int)
    fp["n_removed"] = dec.sum(axis=1)
    split = fp[(fp.n_removed > 0) & (fp.n_removed < len(rules))]
    if not len(split):
        return None, "no component on which the rules disagree"
    vol = pd.to_numeric(split["volume_mm3"], errors="coerce")
    if vol.isna().all():
        return None, "volume_mm3 is not numeric"
    return (split.loc[vol.idxmax()], rules), None


def main():
    print("=" * 74)
    print(" FIGURE 4 · REGIMES · candidate A, refined")
    print("=" * 74)

    # (a) centre on the smallest SCORED lesion, not on the slice's largest
    # object -- the panel is about that lesion
    def target_a(d):
        g = np.isin(d["gt"], WT)
        lab, n = ndimage.label(g)
        if not n:
            raise ValueError("no reference component")
        sizes = ndimage.sum(g, lab, range(1, n + 1))
        vox = float(np.prod(d["spacing"]))
        scored = [(s, i + 1) for i, s in enumerate(sizes) if s * vox > 4.0]
        if not scored:
            raise ValueError("no reference lesion above 4 mm3")
        return lab == min(scored)[1]

    def target_b(d):
        return np.isin(d["gt"], WT) | np.isin(d["pred"], WT)

    def target_d(d):
        g, p = np.isin(d["gt"], WT), np.isin(d["pred"], WT)
        lab, n = ndimage.label(g & ~p)
        if not n:
            raise ValueError("nothing missed in this case")
        sizes = ndimage.sum(g & ~p, lab, range(1, n + 1))
        return lab == int(np.argmax(sizes)) + 1

    spec = [("a", "(a) Smallest scored lesion", target_a,
             "4.125 mm$^3$; evaluator floor 2.0 mm$^3$"),
            ("b", "(b) High lesion multiplicity", target_b,
             "18 lesions; 75th percentile of count"),
            ("d", "(d) Large missed lesion", target_d,
             "1209 mm$^3$; missed at baseline")]

    # THE MIDDLE GAP had two causes, and only one was hspace.
    #
    # The larger cause was unequal panel heights: each crop followed its own
    # target's extent, so a row with a short panel and a tall one left the
    # difference as whitespace. Squaring the crops removes that.
    #
    # hspace itself cannot go very low, because the band between the rows
    # carries the top row's caption AND the bottom row's title. At 0.12 the
    # two touched. 0.22 fits both with the panels now equal in height, and
    # the vertical spacing is settled at that.
    #
    # THE HORIZONTAL GAP is a different problem. The panels are square, so a
    # figure wider than two square panels plus their gap leaves the surplus
    # between the columns -- and no wspace setting can remove width the
    # figure does not need. The figure is narrowed from 5.4in to 4.15in, and
    # wspace then controls only the real gap.
    #
    # The floor on width is the longest caption, "4.125 mm3; evaluator floor
    # 2.0 mm3", which has to sit under its panel without wrapping. Below
    # roughly two caption widths it would, and a wrapped caption costs more
    # than the gap it saves.
    #
    # The captions are given in full. They were cut while the figure was
    # being narrowed to 3.72 and then 3.35in, where they would have wrapped,
    # but at 3.95in they fit -- and each carries something the bare number
    # does not: why 4.125 mm3 counts as the smallest SCORED lesion (the
    # evaluator's floor is 2.0), why 18 lesions is the right example (the
    # 75th percentile, not the maximum), and that the large lesion was missed
    # AT BASELINE rather than under some suppression rule.
    #
    # 3.95in, arrived at by trying both edges. 5.4in left an obvious gap
    # between the columns; 3.35in closed it completely but pressed the panel
    # titles to the full width of their panels, and a title as wide as its
    # panel reads as crowded even when nothing overlaps.
    #
    # 3.95in keeps a small, deliberate gap: enough that each column reads as
    # its own panel, little enough that the pair does not look padded.
    # Everything else -- titles, captions, colours, cases, crops, vertical
    # spacing -- is unchanged.
    fig, axes = plt.subplots(2, 2, figsize=(3.95, 4.30))
    fig.subplots_adjust(wspace=0.020, hspace=0.22, top=0.952, bottom=0.085,
                        left=0.012, right=0.995)
    slot = {"a": axes[0][0], "b": axes[0][1], "c": axes[1][0],
            "d": axes[1][1]}
    man = {}

    for key, title, fn, note in spec:
        case, rule, value = FROZEN[key]
        try:
            d = load_case(case)
            info = draw_panel(slot[key], d, fn(d), PAD[key], title, note)
            man[key] = dict(case=case, selection_rule=rule,
                            quantitative_value=value, status="PASS",
                            source_files=[d["ref_path"], d["t1c_path"]],
                            affine_axcodes=d["codes"], **info)
            print(f"  {key}  {case} · slice {info['slice_index']} · "
                  f"aspect {info['aspect']}")
        except Exception as e:
            slot[key].axis("off")
            slot[key].text(0.5, 0.5, f"{title}\nSTOPPED\n{type(e).__name__}",
                           ha="center", va="center", fontsize=S["small"],
                           color="#b00")
            man[key] = {"case": case, "status": f"STOPPED: "
                                                f"{type(e).__name__}: {e}"}
            print(f"  {key}  STOPPED: {type(e).__name__}: {e}")

    got, why = resolve_c()
    if got is None:
        slot["c"].axis("off")
        slot["c"].text(0.5, 0.5, f"(c) False-positive component\nSTOPPED\n"
                                 f"{why}", ha="center", va="center",
                       fontsize=S["small"], color="#b00")
        man["c"] = {"status": f"STOPPED: {why}"}
        print(f"  c  STOPPED: {why}")
    else:
        pick, rules = got
        try:
            d = load_case(pick.case)
            pmask = np.isin(d["pred"], WT)
            lab, n = ndimage.label(pmask)     # suppression connectivity
            sel = lab == int(pick.supp_component_id)
            if not sel.any():
                raise ValueError("component id absent from the labelling")
            info = draw_panel(
                slot["c"], d, sel, PAD["c"], "(c) False-positive component",
                f"{pick.volume_mm3:.0f} mm$^3$; removed by "
                f"{int(pick.n_removed)}/{len(rules)} rules")
            man["c"] = dict(case=pick.case, region=pick.region,
                            component=int(pick.supp_component_id),
                            volume_mm3=float(pick.volume_mm3),
                            rules_removing=int(pick.n_removed),
                            selection_rule="largest false-positive component "
                                           "on which the six rules disagree",
                            status="PASS", **info)
            print(f"  c  {pick.case} comp {int(pick.supp_component_id)} · "
                  f"{pick.volume_mm3:.0f} mm3")
        except Exception as e:
            slot["c"].axis("off")
            slot["c"].text(0.5, 0.5, f"(c) False-positive component\n"
                                     f"STOPPED\n{type(e).__name__}",
                           ha="center", va="center", fontsize=S["small"],
                           color="#b00")
            man["c"] = {"status": f"STOPPED: {type(e).__name__}: {e}"}
            print(f"  c  STOPPED: {type(e).__name__}: {e}")

    fig.legend(handles=[Patch(facecolor=to_rgba(c, 0.62), edgecolor=c,
                              linewidth=0.5, label=l)
                        for c, l in ERROR_COLOUR.values()],
               loc="lower center", ncol=3, frameon=False,
               fontsize=S["small"], bbox_to_anchor=(0.5, 0.004),
               handlelength=1.0, handleheight=0.8, handletextpad=0.4,
               columnspacing=1.5, borderpad=0.0, borderaxespad=0.0)
    # No outer figure letter. It collided with panel (d) -- one letter for
    # two things -- and a figure with lettered panels does not need one.

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig4_regimes_final.{ext}")
    plt.close(fig)

    out = {"figure": "4", "version": "candidate A, refined",
           "panels": man,
           "error_colours": {l: c for c, l in ERROR_COLOUR.values()},
           "alpha": {"true_positive": TP_ALPHA,
                     "false_negative_and_positive": ERR_ALPHA},
           "slice_rule": "centroid slice of the panel's own target "
                         "structure; the crop is centred on that target too, "
                         "so the panel shows what it is about rather than "
                         "the largest object in the slice",
           "crop_pads": PAD,
           "windowing": "percentile 0.5-99.5",
           "interpolation": {"mri": "bicubic", "segmentation": "nearest"},
           "changes_from_candidate_A": [
               "outer figure letter removed: it duplicated panel (d)",
               "panel (a) cropped around its own 4.125 mm3 lesion, which a "
               "larger missed lesion had been dominating; the other lesion "
               "is not removed from the data, only the field of view changed",
               "panels (a) and (b) cropped tighter, since their fields of "
               "view extended well past the brain"],
           "supplementary_S1": "NOT PRODUCED. All three planes had to pass "
                               "through one anatomical point, and the "
                               "sagittal plane through it carries almost no "
                               "lesion. Changing the rule to the largest "
                               "lesion-bearing slice per orientation would "
                               "give three good-looking panels that are no "
                               "longer views of the same target, which is "
                               "the figure's purpose. Adjusting a selection "
                               "rule because its image looks poor is "
                               "selection on appearance.",
           "unchanged": "no case, selection rule or rationale was altered; "
                        "the changes are presentation only",
           "output": f"{OUT}/fig4_regimes_final.{{pdf,png}}"}
    with open(f"{OUT}/fig4_manifest.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    ok = sum(1 for v in man.values() if v.get("status") == "PASS")
    print(f"""
  {ok} of 4 panels drawn

  written
    {OUT}/fig4_regimes_final.pdf
    {OUT}/fig4_regimes_final.png
    {OUT}/fig4_manifest.json

  S1 is not produced, and the reason is recorded in the manifest.""")


if __name__ == "__main__":
    main()
