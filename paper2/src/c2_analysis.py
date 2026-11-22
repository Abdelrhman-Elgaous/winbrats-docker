# -*- coding: utf-8 -*-
"""
==============================================================================
 C2 ANALYSIS · does contrast separate the groups once volume is held constant?
==============================================================================
 Read-only with respect to the measurement. Seconds.

 Applies the two models fixed in C2_DESIGN_frozen.md before any contrast value
 existed:

   PRIMARY    detected ~ contrast_t1c + log10(volume) + stratum
              Stratum enters as a fixed effect, absorbing case and region. The
              reported quantity is the coefficient on contrast and whether it
              survives the volume term.

   SECONDARY  within each stratum, each missed lesion is matched to the
              detected lesion closest in log10(volume); unmatched lesions are
              dropped; the paired contrast difference is tested with Wilcoxon
              over strata.

 Both were specified in advance precisely so that neither can be chosen after
 the fact. They rest on different assumptions -- the first on a log-linear
 volume relationship, the second on none -- so a disagreement between them is
 informative and is reported rather than resolved.

 t1c carries the inference. The other three sequences are reported
 descriptively, with no test and no correction, as fixed in the design.

 WHY THE RAW GROUP DIFFERENCE IS NOT THE ANSWER
 ----------------------------------------------
 C1 established that volume separates the groups in 96% of strata, and volume
 and contrast are related: a larger lesion has more interior voxels far from
 the boundary, so its median sits further from the surrounding tissue. A raw
 contrast difference would therefore partly restate C1. The question is what
 remains after volume is accounted for.

 WHAT A NULL RESULT MEANS
 ------------------------
 That appearance, as measured here, does not account for the rejections beyond
 what size already separates. That is a result: it narrows the remaining
 explanations to anatomical location, lesion subtype and annotation validity.
 It is not a failed analysis.
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
C2 = f"{DRIVE}/phase_c_outputs/c2"
SRC = f"{C2}/c2_contrast.csv"
SEQS = ["t1c", "t1n", "t2w", "t2f"]
ALPHA = 0.05


def load():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}")
    d = pd.read_csv(SRC)
    d["stratum"] = d.case + "|" + d.region
    d["logvol"] = np.log10(d.gt_volume_mm3.clip(lower=0.1))
    d["is_detected"] = (d.group == "detected").astype(int)
    return d


def primary(d):
    """
    Logistic regression with the stratum as a fixed effect. statsmodels if
    available; otherwise a conditional-logistic equivalent by within-stratum
    centring, which estimates the same within-stratum association.
    """
    print("\n" + "=" * 78)
    print(" PRIMARY · detected ~ contrast_t1c + log10(volume) + stratum")
    print("=" * 78)

    m = d.dropna(subset=["contrast_t1c", "logvol"]).copy()
    print(f"\n  lesions : {len(m)}   strata : {m.stratum.nunique()}")

    try:
        import statsmodels.formula.api as smf
        # The stratum term is what makes this a within-case comparison; without
        # it the model would pool across scans of very different appearance.
        fit = smf.logit("is_detected ~ contrast_t1c + logvol + C(stratum)",
                        data=m).fit(disp=0, maxiter=200)
        for term in ("contrast_t1c", "logvol"):
            b = fit.params[term]
            lo, hi = fit.conf_int().loc[term]
            p = fit.pvalues[term]
            print(f"\n  {term}")
            print(f"    coefficient : {b:+.4f}")
            print(f"    odds ratio  : {np.exp(b):.3f}   "
                  f"95% CI [{np.exp(lo):.3f}, {np.exp(hi):.3f}]")
            print(f"    p           : "
                  f"{'< 1e-300' if p == 0 else format(p, '.3g')}")
        return {"contrast_beta": float(fit.params["contrast_t1c"]),
                "contrast_p": float(fit.pvalues["contrast_t1c"]),
                "vol_beta": float(fit.params["logvol"]),
                "vol_p": float(fit.pvalues["logvol"])}
    except ImportError:
        print("""
  statsmodels unavailable. Falling back to within-stratum centring, which
  removes the stratum mean from both variables and estimates the same
  within-stratum association without the fixed-effect machinery.""")
        for c in ("contrast_t1c", "logvol", "is_detected"):
            m[c + "_c"] = m[c] - m.groupby("stratum")[c].transform("mean")
        # correlation of the centred detection indicator with centred contrast,
        # partialling out centred volume
        def partial(x, y, z):
            rxy = np.corrcoef(x, y)[0, 1]
            rxz = np.corrcoef(x, z)[0, 1]
            ryz = np.corrcoef(y, z)[0, 1]
            return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
        r = partial(m.is_detected_c, m.contrast_t1c_c, m.logvol_c)
        n = len(m)
        t = r * np.sqrt((n - 3) / (1 - r**2))
        p = 2 * (1 - stats.t.cdf(abs(t), n - 3))
        print(f"\n  partial correlation, detection with contrast, "
              f"volume held constant")
        print(f"    r = {r:+.4f}   p = {p:.3g}   (n = {n})")
        return {"contrast_partial_r": float(r), "contrast_p": float(p)}


def secondary(d):
    """Volume-matched pairs within stratum, Wilcoxon over strata."""
    print("\n" + "=" * 78)
    print(" SECONDARY · volume-matched pairs, Wilcoxon over strata")
    print("=" * 78)

    diffs, used, dropped = [], [], 0
    for st, g in d.dropna(subset=["contrast_t1c"]).groupby("stratum"):
        det = g[g.group == "detected"]
        mis = g[g.group == "missed"]
        if not len(det) or not len(mis):
            continue
        pairs = []
        avail = det.copy()
        for _, ml in mis.iterrows():
            if not len(avail):
                dropped += 1
                continue
            j = (avail.logvol - ml.logvol).abs().idxmin()
            pairs.append((ml.contrast_t1c, avail.loc[j, "contrast_t1c"],
                          abs(avail.loc[j, "logvol"] - ml.logvol)))
            avail = avail.drop(j)          # each detected lesion used once
        if pairs:
            mi = np.mean([p[0] for p in pairs])
            de = np.mean([p[1] for p in pairs])
            diffs.append(de - mi)
            used.append({"stratum": st, "n_pairs": len(pairs),
                         "mean_logvol_gap": float(np.mean([p[2] for p in pairs])),
                         "det_contrast": de, "mis_contrast": mi,
                         "diff": de - mi})

    u = pd.DataFrame(used)
    print(f"\n  strata with at least one matched pair : {len(u)}")
    print(f"  missed lesions left unmatched         : {dropped}")
    print(f"  pairs per stratum                     : "
          f"median {u.n_pairs.median():.0f}")
    print(f"  log10 volume gap within a pair        : "
          f"median {u.mean_logvol_gap.median():.3f}   "
          f"(ratio {10**u.mean_logvol_gap.median():.2f}x)")

    a = np.array(diffs)
    if len(a) < 2 or np.allclose(a, 0):
        print("\n  too few or degenerate differences to test")
        return None
    w = stats.wilcoxon(a)
    rng = np.random.default_rng(20260819)
    boot = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
    print(f"""
  paired difference in contrast (detected minus missed):
    median  {np.median(a):+.4f}
    mean    {a.mean():+.4f}
    95% CI  [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]
    Wilcoxon p = {w.pvalue:.3g}   (n = {len(a)} strata)
    strata favouring detected : {int((a > 0).sum())}/{len(a)} "
        f"({(a > 0).mean():.1%})""")

    if u.mean_logvol_gap.median() > 0.3:
        print(f"""
  Note: the median within-pair volume gap is {10**u.mean_logvol_gap.median():.1f}x, so
  matching is imperfect and residual volume differences remain. The regression
  is the better-controlled estimate; this check tests whether the conclusion
  depends on its functional form.""")
    return {"median_diff": float(np.median(a)), "p": float(w.pvalue),
            "n_strata": int(len(a))}


def other_sequences(d):
    print("\n" + "=" * 78)
    print(" OTHER SEQUENCES · descriptive only, no test, no correction")
    print("=" * 78)
    rows = []
    for s in SEQS[1:]:
        col = f"contrast_{s}"
        if col not in d.columns:
            continue
        g = d.dropna(subset=[col]).groupby("group")[col].median()
        rows.append({"sequence": s,
                     "detected": round(float(g.get("detected", np.nan)), 3),
                     "missed": round(float(g.get("missed", np.nan)), 3),
                     "n": int(d[col].notna().sum())})
    if rows:
        print("\n  " + pd.DataFrame(rows).to_string(index=False)
              .replace("\n", "\n  "))
    print("""
  These are medians, not tests. The design fixed t1c as the sole inferential
  endpoint; testing four sequences would add comparisons the question never
  asked for.""")


def main():
    print("=" * 78)
    print(" C2 ANALYSIS · pre-specified models")
    print("=" * 78)
    d = load()
    print(f"  lesions  : {len(d)}")
    print(f"  detected : {int((d.group=='detected').sum())}")
    print(f"  missed   : {int((d.group=='missed').sum())}")
    print(f"  strata   : {d.stratum.nunique()}")

    p = primary(d)
    s = secondary(d)
    other_sequences(d)

    print("\n" + "=" * 78)
    print(" READING THE RESULT")
    print("=" * 78)

    pv = p.get("contrast_p", np.nan)
    sv = s["p"] if s else np.nan
    agree = (pv < ALPHA) == (sv < ALPHA) if s else None

    # A p-value that underflows prints as 0, which reads as an exact zero
    # rather than "below the smallest representable value". Say which it is.
    def fmt(v):
        if v != v:
            return "n/a"
        return "< 1e-300" if v == 0 else f"{v:.3g}"

    print(f"""
  primary   contrast p = {fmt(pv)}
  secondary matched  p = {fmt(sv)}
  agreement : {'yes' if agree else 'NO' if agree is False else 'n/a'}""")

    if agree is False:
        print("""
  The two disagree. They assume different things about how volume relates to
  detection, and the disagreement is the finding rather than something to
  resolve by preferring one. Report both, and treat the contrast association
  as unsettled under this design.""")
    elif pv < ALPHA:
        print("""
  Contrast is associated with detection after volume is accounted for, on both
  specifications.

  This does NOT establish that the model rejects lesions because of contrast.
  Contrast covaries with lesion type, enhancement pattern and acquisition, none
  of which is controlled here. The design holds the scan, region, preprocessing
  and model output constant, and volume by adjustment; nothing else.""")
    else:
        print("""
  Contrast does not separate the groups once volume is accounted for, on either
  specification.

  This is a result. Appearance as measured here does not account for the
  rejections beyond what size already separates, which narrows the remaining
  candidate explanations to anatomical location, lesion subtype and treatment
  history, and reference-annotation validity.

  It does not show that appearance is irrelevant -- only that this measure, on
  this sequence, within these strata, does not carry it.""")


if __name__ == "__main__":
    main()
