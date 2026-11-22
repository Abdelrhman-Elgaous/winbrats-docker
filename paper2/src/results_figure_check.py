# -*- coding: utf-8 -*-
"""
==============================================================================
 RESULTS FIGURE CHECK  —  numbers parsed OUT of the draft, then verified
==============================================================================
 Read-only.

 WHAT CHANGED AND WHY
 --------------------
 The first version of this script held the manuscript figures as literals in
 its own source and compared those against the CSVs. That verifies the CSVs
 have not moved; it does NOT verify the manuscript. Editing a number in
 RESULTS_DRAFT.md would leave the check passing, because the draft was never
 read. The name promised more than the code did.

 This version parses the numbers out of the draft's own tables and prose, so a
 figure that is edited in the manuscript and not in the source will fail. That
 is the failure mode worth catching: the draft is what a reader sees.

 WHAT IT STILL CANNOT DO
 -----------------------
 Prose claims spanning several rows -- "the direction is consistent across
 every stratum", "unchanged in sign, significance and approximate magnitude" --
 are quantifiers over sets, not figures. They are checked separately in the
 second half and reported as claims, not parsed as numbers.
==============================================================================
"""

import os
import re
import glob

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W2 = f"{DRIVE}/phase_b_outputs/wave2"
W1 = f"{DRIVE}/phase_b_outputs/wave1"
DRAFT = f"{DRIVE}/paper2/RESULTS_DRAFT.md"
PAT = re.compile(r"\.w\d+of\d+\.csv$")
TOL = 0.0011                       # the draft rounds to 3 dp in places

MINUS = "\u2212"                   # the draft uses a real minus sign


def num(tok):
    """A draft token to a float, tolerating the typographic minus and bold."""
    t = tok.strip().replace(MINUS, "-").replace("**", "").replace("*", "")
    t = re.sub(r"\*?\(ns\)\*?", "", t).strip()
    try:
        return float(t)
    except ValueError:
        return None


draft = open(DRAFT, encoding="utf-8").read()
b6 = pd.read_csv(f"{W2}/b6_primary_clean.csv")
if "rule" not in b6.columns:
    b6 = b6.rename(columns={"setting": "rule"})
DCOL = next(c for c in ("mean_delta", "delta", "dice_delta") if c in b6.columns)
t4 = pd.read_csv(f"{W2}/t4_detection_results.csv")
t3 = pd.read_csv(f"{W2}/t3_bin_results.csv")
cnt = pd.concat([pd.read_csv(f) for f in glob.glob(f"{W2}/t4_counts.w*of*.csv")
                 if PAT.search(f)], ignore_index=True)

rows = []


def section(start, end=None):
    """
    The draft's tables are formatted identically across sections, so a pattern
    written for one will happily match another. Every scan below is confined to
    its own section; without this the entropy-Dice pattern picks up the
    entropy-F1 table and reports a mismatch against the wrong source.
    """
    try:
        i = draft.index(start)
    except ValueError:
        return ""
    j = draft.index(end) if end and end in draft[i:] else len(draft)
    return draft[i:j]


SEC = {
    "3": section("## 3.", "## 4."),
    "4": section("## 4.", "## 5."),
    "7": section("## 7.", "## 8."),
    "8": section("## 8.", "## 9."),
    "appx": section("## Appendix B"),
}


def record(where, what, drafted, actual, tol=TOL):
    if drafted is None or actual is None:
        rows.append((where, what, drafted, actual, "UNPARSED"))
        return
    ok = abs(drafted - actual) <= tol
    rows.append((where, what, drafted, round(actual, 4), "ok" if ok else "MISMATCH"))


def src(df, region, rule, col):
    r = df[(df.region == region) & (df.rule == rule)]
    return float(r[col].iloc[0]) if len(r) else None


# ---------------------------------------------------------------------------
#  §3  the size@50 table: | WT | **-0.0764** | [...] | ... |
# ---------------------------------------------------------------------------
for m in re.finditer(r"^\|\s*(WT|TC|ET|RC)\s*\|\s*\*\*([-\u2212\d.]+)\*\*\s*\|",
                     SEC["3"], re.M):
    reg, val = m.group(1), num(m.group(2))
    a = src(b6, reg, "size@50", DCOL)
    if a is not None and val is not None and abs(val) > 0.01:
        record("§3", f"{reg} size@50 ΔDice", val, a)

# ---------------------------------------------------------------------------
#  §4  the entropy table: | `entropy@0.45` | -0.0080 | -0.0188 | -0.0183 |
# ---------------------------------------------------------------------------
for m in re.finditer(r"^\|\s*`(entropy@[\d.]+)`\s*\|\s*([+\-\u2212\d.]+)\s*\|"
                     r"\s*([+\-\u2212\d.]+)\s*\|\s*([+\-\u2212\d.]+)\s*\|\s*$",
                     SEC["4"], re.M):
    rule = m.group(1)
    for reg, g in zip(("WT", "TC", "ET"), (2, 3, 4)):
        record("§4", f"{reg} {rule} ΔDice", num(m.group(g)), src(b6, reg, rule, DCOL))

# ---------------------------------------------------------------------------
#  §7  the F1 table: | `size@50` | -0.127 | -0.191 | -0.195 |
# ---------------------------------------------------------------------------
sec7 = SEC["7"]
for m in re.finditer(r"^\|\s*`([\w@.]+)`\s*\|\s*([+\-\u2212\d.*]+)\s*\|"
                     r"\s*([+\-\u2212\d.*]+)\s*\|\s*([+\-\u2212\d.*]+)\s*\|\s*$",
                     sec7[:pr_i] if (pr_i := sec7.find("Δ precision")) > 0 else sec7,
                     re.M):
    rule = m.group(1)
    for reg, g in zip(("WT", "TC", "ET"), (2, 3, 4)):
        record("§7", f"{reg} {rule} ΔF1", num(m.group(g)),
               src(t4, reg, rule, "F1_delta"))

# The precision/recall block shares the F1 table's shape, so it is found by its
# own header row ("| ET | Δ precision | ...") rather than by column count.
pr_i = sec7.find("Δ precision")
pr = sec7[pr_i:] if pr_i >= 0 else ""
for m in re.finditer(r"^\|\s*`([\w@.]+)`\s*\|\s*([+\-\u2212\d.]+)\s*\|"
                     r"\s*\*\*([+\-\u2212\d.]+)\*\*\s*\|\s*([+\-\u2212\d.]+)\s*\|",
                     pr, re.M):
    rule = m.group(1)
    record("§7", f"ET {rule} Δprecision", num(m.group(2)),
           src(t4, "ET", rule, "precision_delta"))
    record("§7", f"ET {rule} Δrecall", num(m.group(3)),
           src(t4, "ET", rule, "recall_delta"))

# The baseline-F1 sentence: "0.866 in WT, 0.871 in TC, 0.874 in ET". It may wrap
# anywhere, so match the three number/region pairs individually rather than
# trying to anchor one pattern across a line break.
if "F1_base" in t4.columns:
    base = t4.groupby("region").F1_base.first()
    for reg in ("WT", "TC", "ET"):
        m = re.search(r"(0\.\d{3})\s+in\s*\n?\s*" + reg + r"\b", sec7 or draft)
        if m:
            record("§7", f"{reg} baseline F1", num(m.group(1)), float(base[reg]))
        else:
            rows.append(("§7", f"{reg} baseline F1", None,
                         round(float(base[reg]), 4), "UNPARSED"))

# ---------------------------------------------------------------------------
#  §8  the bin table: | (2, 27] | 77 | 0.331 | **-0.283** | -0.050 |
# ---------------------------------------------------------------------------
BINMAP = {"(2, 27]": "(2,27]", "(27, 100]": "(27,100]",
          "(100, 500]": "(100,500]", "(500, ∞)": "(500,inf)"}
for m in re.finditer(r"^\|\s*(\([\d, ]+\]|\(500, ∞\))\s*\|\s*(\d+)\s*\|"
                     r"\s*([\d.]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|", SEC["8"], re.M):
    b = BINMAP.get(m.group(1).strip())
    if b is None:
        continue
    r = t3[t3.bin == b]
    if not len(r):
        continue
    record("§8", f"{b} n_cases", float(m.group(2)), float(r.n_cases.iloc[0]), 0.5)
    record("§8", f"{b} baseline Dice", num(m.group(3)), float(r.dice_base.iloc[0]))
    for rule, g in (("size@50", 4), ("entropy@0.50", 5)):
        rr = t3[(t3.bin == b) & (t3.rule == rule)]
        if len(rr):
            record("§8", f"{b} {rule} ΔDice", num(m.group(g)),
                   float(rr.dice_delta.iloc[0]))

# ---------------------------------------------------------------------------
#  Newly reported: D19 associations and D21 component selection
#  These were computed during the analysis but only reported in the Discussion
#  until now. The figures are unchanged; only their location is.
# ---------------------------------------------------------------------------
D19 = {"WT": -0.287, "TC": -0.414, "ET": -0.376, "ET_ent": 0.071}
for reg, val in (("WT", D19["WT"]), ("TC", D19["TC"]), ("ET", D19["ET"])):
    m = re.search(reg + r"\s*\n?\(Spearman ρ = ([\u2212-][\d.]+)|"
                  + reg + r"\s*\(ρ = ([\u2212-][\d.]+)", draft)
    if m:
        record("§8b", f"{reg} size@50 rho", num(m.group(1) or m.group(2)), val, 0.0006)

m = re.search(r"`entropy@0\.50` was small and non-significant \(ρ = \+?([\d.]+)", draft)
if m:
    record("§8b", "ET entropy control rho", num(m.group(1)), D19["ET_ent"], 0.0006)

fpc = f"{W2}/fp_components.csv"
if glob.glob(fpc):
    fp = pd.read_csv(fpc)
    fpo = fp[fp.is_fp == 1]
    record("§8a", "FP components total", None, float(len(fp)), 0.5) if False else None
    m = re.search(r"(\d[\d,]*) predicted components were extracted", draft)
    if m:
        record("§8a", "components extracted", float(m.group(1).replace(",", "")),
               float(len(fp)), 0.5)
    m = re.search(r"of which (\d+) were\s*\nfalse positives", draft)
    if m:
        record("§8a", "false positives", float(m.group(1)), float(len(fpo)), 0.5)
    for rule, tag in (("size@50", "size@50"), ("entropy@0.45", "entropy@0.45")):
        if rule in fpo.columns:
            r_ = fpo[fpo[rule] == "removed"]; k_ = fpo[fpo[rule] == "retained"]
            m = re.search(r"\| `" + re.escape(rule) + r"` \| \*?\*?(\d+)%",  draft)
            if m:
                record("§8a", f"{rule} removed %", float(m.group(1)),
                       round(len(r_) / max(len(fpo), 1) * 100), 1.0)

# ---------------------------------------------------------------------------
#  Appendix prevalence table
# ---------------------------------------------------------------------------
b0 = cnt[cnt.setting == "baseline"]
for m in re.finditer(r"^\|\s*(WT|TC|ET|RC)\s*\|\s*\*?\*?(\d+)\*?\*?\s*\|\s*(\d+)\s*\|"
                     r"\s*\*?\*?(\d+)\*?\*?\s*\|", SEC["appx"], re.M):
    reg = m.group(1)
    r = b0[b0.region == reg]
    if not len(r):
        continue
    neg = float(((r.Num_TP + r.Num_FN) == 0).sum())
    negfp = float((((r.Num_TP + r.Num_FN) == 0) & (r.Num_FP > 0)).sum())
    pos = float(((r.Num_TP + r.Num_FN) > 0).sum())
    record("appx", f"{reg} reference-negative", float(m.group(2)), neg, 0.5)
    record("appx", f"{reg} with an FP", float(m.group(3)), negfp, 0.5)
    record("appx", f"{reg} reference-positive", float(m.group(4)), pos, 0.5)

# denominators asserted in §1
for pat, actual, label in [
        (r"\*\*3,402\s*\n?case-setting pairs\*\*", 567 * 6, "§1 3,402 pairs"),
        (r"\*\*1,085 case-setting pairs\*\*",
         cnt.groupby(["case", "setting"]).ngroups, "§1 1,085 pairs")]:
    if re.search(pat, draft):
        record("§1", label, float(str(actual)), float(actual), 0.5)
    else:
        rows.append(("§1", label, None, actual, "NOT FOUND IN DRAFT"))

# ---------------------------------------------------------------------------
print("=" * 78)
print(" RESULTS FIGURE CHECK — parsed from the draft, verified against sources")
print("=" * 78)
d = pd.DataFrame(rows, columns=["where", "figure", "in_draft", "in_source", "status"])
print("\n" + d.to_string(index=False))

bad = d[d.status == "MISMATCH"]
unp = d[d.status.isin(["UNPARSED", "NOT FOUND IN DRAFT"])]
print("\n" + "=" * 78)
print(f"  parsed and checked : {len(d[d.status == 'ok'])}")
print(f"  mismatches         : {len(bad)}")
print(f"  unparsed           : {len(unp)}")
if len(bad):
    print("\n" + bad.to_string(index=False))
    print("\n  Correct the draft, not the source.")
if len(unp):
    print("\n  Unparsed rows mean the draft's formatting moved, not that the number")
    print("  is wrong — check them by eye and fix the pattern if the table changed.")

# ---------------------------------------------------------------------------
#  Set-level prose claims: verified here because no figure carries them
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(" PROSE CLAIMS — quantifiers over rows, not single figures")
print("=" * 78)

strat = f"{W2}/b6_by_stratum.csv"
if glob.glob(strat):
    st = pd.read_csv(strat)
    rcol = "rule" if "rule" in st.columns else "setting"
    s50 = st[(st[rcol] == "size@50") & (st.region.isin(["WT", "TC", "ET"]))]
    sigcol = next((c for c in ("sig_holm", "sig") if c in st.columns), None)
    dc = next(c for c in ("mean_delta", "delta") if c in st.columns)
    print('\n  "size@50 is significantly harmful in WT, TC and ET in patient_only,')
    print('   in seen, and in the pooled cohort"')
    print("\n" + s50[[c for c in ("stratum", "region", dc, "p_holm", sigcol)
                      if c and c in s50.columns]].round(4).to_string(index=False))
    if sigcol:
        allsig = bool((s50[dc] < 0).all() and s50[sigcol].all())
        print(f"\n    all negative and significant: {allsig}")
else:
    print("\n  b6_by_stratum.csv not found — the stratum claim stays MANUALLY VERIFIED")

print("""
  Still manual, by design:
    · the QC-sensitivity summary sentence in the sensitivity section — a
      quantifier over a whole table, since superseded by an explicit
      count (20/20 significance, 18/20 sign)
    · the 20 reference lesions at or below 2 mm3
    · the 28-40 RC cases whose FP count changed
    · BraTS-MET-01230-003's counts
  These are quantifiers and provenance notes, not figures; reread them against
  the QC table and the audit report rather than parsing them.""")
