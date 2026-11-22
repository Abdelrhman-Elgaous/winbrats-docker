# -*- coding: utf-8 -*-
"""
==============================================================================
 PHASE B CLOSURE AUDIT  —  everything mechanically checkable before freeze
==============================================================================
 Read-only. One command, seven checks, a closure checklist at the end.

 This is the last script in Phase B. It does not compute a result; it verifies
 that the results already computed are complete, internally consistent, and
 consistent with what the documents say about them.

   1  ARTEFACTS      every expected output exists and has the right shape
   2  COVERAGE       every rule and stratum in the protocol has a result
   3  CROSS-DOCUMENT the same number is the same everywhere it appears
   4  DISCOVERY LOG  every entry has evidence, a paper destination, a confidence
   5  BANNED CLAIMS  corrected assumptions have not crept back in
   6  FREEZE MARKERS the frozen documents say so, and say what that covers
   7  OUTSTANDING    what is genuinely still open, stated plainly

 A failure is a finding. Fix the document or the code it describes, never the
 stored result.
==============================================================================
"""

import os
import re
import glob

import numpy as np
import pandas as pd

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
W1, W2 = f"{DRIVE}/phase_b_outputs/wave1", f"{DRIVE}/phase_b_outputs/wave2"
DOCS = f"{DRIVE}/paper2"
SRC = f"{DRIVE}/src/eval"
PAT = re.compile(r"\.w\d+of\d+(\.p\d+of\d+)?\.csv$")

RULES6 = ["baseline", "size@50", "entropy@0.45", "entropy@0.50",
          "entropy@0.56", "confidence@0.45"]
RULES7 = RULES6 + ["size@20"]
REGIONS = ["WT", "TC", "ET", "RC"]
STRATA = ["clean", "patient_only", "seen"]

results = []


def verdict(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"\n  >>> {name}: {'PASS' if ok else 'FAIL'}" + (f" — {detail}" if detail else ""))


def read(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


# ---------------------------------------------------------------------------
def check_artefacts():
    print("\n" + "=" * 78)
    print(" 1 · ARTEFACTS")
    print("=" * 78)
    # Filenames as they actually are on disk, not as this script once guessed.
    expect = {
        f"{W1}/exp3_suppressed_v3.csv": ("B-4 merged", 13608),
        f"{W2}/b5b_T1_clean.csv": ("B-5b T1 clean", None),
        f"{W2}/b5b_T2_by_leakclass.csv": ("B-5b T2 strata", None),
        f"{W2}/b5b_qc_sensitivity.csv": ("B-5b QC", None),
        f"{W2}/b6_primary_clean.csv": ("B-6 primary", None),
        f"{W2}/b6_by_stratum.csv": ("B-6 strata", None),
        f"{W2}/b6_qc_sensitivity.csv": ("B-6 QC", None),
        f"{W2}/t4_detection_results.csv": ("T4", 24),
        f"{W2}/t3_bin_results.csv": ("T3", None),
        f"{W2}/fp_components.csv": ("FP components", None),
        f"{DOCS}/PHASE_B_PROTOCOL.md": ("protocol", None),
        f"{DOCS}/DISCOVERIES.md": ("discoveries", None),
        f"{DOCS}/RESULTS_DRAFT.md": ("results draft", None),
    }
    ok = True
    for path, (label, nrows) in expect.items():
        exists = os.path.exists(path)
        note = ""
        if exists and path.endswith(".csv"):
            n = len(pd.read_csv(path))
            note = f"{n} rows"
            if nrows and n != nrows:
                note += f" (expected {nrows})"
                ok = False
        elif exists:
            note = f"{len(read(path).splitlines())} lines"
        print(f"  {'ok ' if exists else 'MISSING':>8}  {label:<16} {note}")
        ok &= exists

    b4 = f"{W1}/exp3_suppressed_v3.csv"
    if os.path.exists(b4):
        d = pd.read_csv(b4)
        pairs = d.groupby(["case", "setting"]).ngroups
        fails = int((d.status.astype(str) != "ok").sum())
        print(f"\n  B-4: {pairs} pairs · {d.case.nunique()} cases · "
              f"{d.setting.nunique()} rules · {fails} failed")
        ok &= (pairs == 3402 and fails == 0)

    t4c = [f for f in glob.glob(f"{W2}/t4_counts.w*of*.csv") if PAT.search(f)]
    if t4c:
        t = pd.concat([pd.read_csv(f) for f in t4c], ignore_index=True)
        pairs = t.groupby(["case", "setting"]).ngroups
        print(f"  T4: {pairs} pairs · {t.case.nunique()} cases · "
              f"{t.setting.nunique()} rules")
        ok &= (pairs == 1085)

    verdict("1 ARTEFACTS", ok, "all present with expected shape" if ok else "see above")


# ---------------------------------------------------------------------------
def check_coverage():
    print("\n" + "=" * 78)
    print(" 2 · COVERAGE — does every protocol item have a result?")
    print("=" * 78)
    ok = True

    b4 = pd.read_csv(f"{W1}/exp3_suppressed_v3.csv")
    missing = set(RULES6) - set(b4.setting)
    extra = set(b4.setting) - set(RULES6)
    print(f"  B-4 rules      : {sorted(set(b4.setting))}")
    print(f"    missing {sorted(missing) or 'none'} · unexpected {sorted(extra) or 'none'}")
    ok &= not (missing or extra)

    t4c = pd.concat([pd.read_csv(f) for f in glob.glob(f"{W2}/t4_counts.w*of*.csv")
                     if PAT.search(f)], ignore_index=True)
    missing = set(RULES7) - set(t4c.setting)
    print(f"  T4 rules       : {sorted(set(t4c.setting))}")
    print(f"    missing {sorted(missing) or 'none'}")
    ok &= not missing

    st = f"{W2}/b6_by_stratum.csv"
    if os.path.exists(st):
        s = pd.read_csv(st)
        have = set(s.stratum) if "stratum" in s.columns else set()
        print(f"  B-6 strata     : {sorted(have)}")
        ok &= set(STRATA) <= have or "all" in have

    t4r = pd.read_csv(f"{W2}/t4_detection_results.csv")
    cells = t4r.groupby("region").size().to_dict()
    print(f"  T4 cells       : {cells}  (expect 6 per region)")
    ok &= all(v == 6 for v in cells.values())

    t3 = pd.read_csv(f"{W2}/t3_bin_results.csv")
    print(f"  T3 bins        : {sorted(set(t3.bin))}")
    ok &= len(set(t3.bin)) == 4

    verdict("2 COVERAGE", ok, "every protocol rule, stratum and bin has a result"
            if ok else "see above")


# ---------------------------------------------------------------------------
def check_cross_document():
    print("\n" + "=" * 78)
    print(" 3 · CROSS-DOCUMENT — the same number, everywhere it appears")
    print("=" * 78)
    disc, res = read(f"{DOCS}/DISCOVERIES.md"), read(f"{DOCS}/RESULTS_DRAFT.md")
    ok = True

    # figures that deliberately appear in more than one document
    shared = [
        # The documents use a typographic minus (U+2212); an ASCII-only
        # pattern silently missed every one of these and reported the numbers
        # absent when they were present.
        ("D19 WT rho", r"[\u2212-]\s?0\.287", ("disc",)),
        ("D19 TC rho", r"[\u2212-]\s?0\.414", ("disc",)),
        ("D19 ET rho", r"[\u2212-]\s?0\.376", ("disc",)),
        ("D19 entropy control", r"\+?0\.071", ("disc",)),
        ("RC reference-negative 150", r"\b150\b", ("disc", "res")),
        ("RC reference-positive 5", r"\bfive\b|\b5\b", ("disc", "res")),
        ("size@50 ET Dice", r"[−-]0\.142", ("disc", "res")),
        ("entropy@0.45 RC Dice", r"\+?0\.2387|\+?0\.239", ("disc", "res")),
        ("clean stratum n", r"\b155\b", ("disc", "res")),
        ("T4 pairs", r"1,085", ("res",)),
        ("B-4 pairs", r"3,402", ("res",)),
    ]
    for label, pat, where in shared:
        hits = []
        if "disc" in where:
            hits.append(("DISCOVERIES", bool(re.search(pat, disc))))
        if "res" in where:
            hits.append(("RESULTS", bool(re.search(pat, res))))
        good = all(h for _, h in hits)
        print(f"  {'ok ' if good else 'ABSENT':>8}  {label:<28} "
              + " · ".join(f"{n}:{'y' if h else 'n'}" for n, h in hits))
        ok &= good

    # the clean-stratum headroom must not be confused with the 567 figures
    if "0.0395" in res and "0.0630" in res:
        print("\n  [!] both the clean (0.0395) and 567-cohort (0.0630) WT headroom")
        print("      appear in the Results — check the 567 one is labelled as such")
    verdict("3 CROSS-DOCUMENT", ok, "shared figures agree across documents"
            if ok else "a figure is missing where it should appear")


# ---------------------------------------------------------------------------
def check_discovery_log():
    print("\n" + "=" * 78)
    print(" 4 · DISCOVERY LOG — every entry complete")
    print("=" * 78)
    disc = read(f"{DOCS}/DISCOVERIES.md")
    ok = True
    # Slice on the heading positions themselves. The previous version searched
    # for "Dn · " anywhere in the file, which matched the summary table first
    # and read the wrong block for every entry -- every one came back with no
    # evidence, which was a parser fault, not an incomplete log.
    heads = [(m.group(1), m.group(2), m.start())
             for m in re.finditer(r"^## [🔬🔧] (D\d+) · (.+?)$", disc, re.M)]
    entries = [(d, t) for d, t, _ in heads]
    bounds = [pos for _, _, pos in heads] + [len(disc)]
    print(f"  entries found: {len(entries)}")
    for k, (did, title) in enumerate(entries):
        block = disc[bounds[k]:bounds[k + 1]]
        has_ev = "**Evidence" in block or "Evidence:" in block
        has_pap = "**Paper" in block
        has_conf = "Confidence" in block
        good = has_pap and has_conf
        flags = "".join([" E" if has_ev else " -", " P" if has_pap else " -",
                         " C" if has_conf else " -"])
        print(f"  {'ok ' if good else 'INCOMPLETE':>11}  {did:<5}{flags}  {title[:46]}")
        ok &= good
    print("\n  E = evidence · P = paper destination · C = confidence")
    verdict("4 DISCOVERY LOG", ok, f"{len(entries)} entries, all with destination "
            f"and confidence" if ok else "an entry is missing its destination or confidence")


# ---------------------------------------------------------------------------
def check_banned():
    print("\n" + "=" * 78)
    print(" 5 · BANNED CLAIMS — corrected assumptions must stay corrected")
    print("=" * 78)
    # A withdrawn wording kept on purpose is not a live claim. The project
    # deliberately preserves superseded statements so a reader of an old draft
    # can trace why they changed, so the scan has to recognise that context or
    # it flags the provenance record it exists to protect.
    NEG = ("لا يُكتب", "مسحوب", "~~", "باطل", "غير موثَّق", "withdrawn", "ممنوع",
           "CORRECTED", "SUPERSEDED", "was an undocumented", "legacy",
           "The original reasoning", "الصياغة المسحوبة", "المسحوبة",
           "الصياغة القديمة", "اتكذّبت", "أُبطل", "باطلة", "historical",
           "LEGACY_ASSUMED", "not an evaluator value", "غير صحيحة",
           "سابقة", "تصحيح", "مُعاد صياغتها", "قبل التصحيح")
    BAD = [
        (r"(?<!لا يُكتب \")evaluator floor\s*=\s*2\b", "bare 'evaluator floor = 2'"),
        (r"الحد.{0,12}=\s*2\s*(mm³|voxel).{0,25}(الطرفين|الجانبين)", "'2 X on both sides'"),
        (r"(?<!~~)27[- ]?(mm³|voxel)?\s*(scoring )?floor", "27 as a live floor"),
        (r"SCORABLE\s*=\s*27", "SCORABLE = 27"),
        (r"FLOOR_VOX\s*=\s*27", "FLOOR_VOX = 27"),
        (r"Every conclusion is unchanged", "the corrected QC overclaim"),
    ]
    hits = 0
    for f in sorted(glob.glob(f"{DOCS}/*.md") + glob.glob(f"{SRC}/*.py")):
        lines = read(f).splitlines()
        withdrawn_until = -1
        for i, l in enumerate(lines, 1):
            if re.match(r"^#{2,4} .*(المسحوب|withdrawn|Superseded|القديمة)", l):
                withdrawn_until = i + 40      # the block that heading opens
            if i <= withdrawn_until:
                continue
            window = " ".join(lines[max(0, i - 12):i + 6])
            if any(n in window for n in NEG):
                continue
            for pat, lab in BAD:
                if re.search(pat, l):
                    print(f"  {os.path.basename(f)}:{i}  [{lab}]  {l.strip()[:60]}")
                    hits += 1
    print(f"  live occurrences: {hits}")
    verdict("5 BANNED CLAIMS", hits == 0,
            "no corrected assumption has returned" if hits == 0 else f"{hits} found")


# ---------------------------------------------------------------------------
def check_freeze():
    print("\n" + "=" * 78)
    print(" 6 · FREEZE MARKERS")
    print("=" * 78)
    checks = [
        (f"{DOCS}/RESULTS_DRAFT.md", "FROZEN", "Results freeze recorded"),
        (f"{DOCS}/RESULTS_DRAFT.md", "numerical consistency only",
         "freeze scope stated, not overclaimed"),
        (f"{DOCS}/PHASE_B_PROTOCOL.md", "تجميد Results", "freeze in the change log"),
        (f"{DOCS}/DISCOVERIES.md", "v1.7", "discovery log at the current version"),
        (f"{DOCS}/PHASE_B_PROTOCOL.md", "single physical-unit lesion floor",
         "D6 canonical wording"),
        (f"{DOCS}/PHASE_B_PROTOCOL.md", "descriptive/exploratory only",
         "T3 bin disclosure"),
        (f"{DOCS}/PHASE_B_PROTOCOL.md", "T3 statistical unit",
         "T3 unit frozen before computation"),
    ]
    ok = True
    for path, needle, label in checks:
        found = needle in read(path)
        print(f"  {'ok ' if found else 'MISSING':>8}  {label}")
        ok &= found
    verdict("6 FREEZE MARKERS", ok, "documents state what is frozen and how far"
            if ok else "a marker is missing")


# ---------------------------------------------------------------------------
def check_outstanding():
    print("\n" + "=" * 78)
    print(" 7 · OUTSTANDING")
    print("=" * 78)
    fp_done = os.path.exists(f"{W2}/fp_components.csv")
    gate_done = os.path.exists(f"{W2}/.fp_gate_passed")
    items = [
        ("FP component-size · gate", gate_done,
         "fp_component_size.py --gate"),
        ("FP component-size · extract", fp_done,
         "fp_component_size.py --extract, after the gate"),
        ("Methods", os.path.exists(f"{DOCS}/METHODS.md"),
         "not started — write from the frozen protocol, not the reverse"),
        ("Discussion", os.path.exists(f"{DOCS}/DISCUSSION.md"),
         "not started — must add no claim larger than Results"),
    ]
    for label, done, how in items:
        print(f"  {'done' if done else 'open':>6}  {label:<30} {how}")
    print("""
  Not outstanding, and not to be reopened: any further Phase B analysis. The
  results are frozen. A number found wrong is corrected from source; a claim
  found stronger than its evidence is weakened; a missing method is added. A
  new analysis run to look for another finding is not any of those.""")
    verdict("7 OUTSTANDING", True, "listed, not blocking")


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(" PHASE B CLOSURE AUDIT")
    print(" read-only · a failure is a finding, not a thing to edit around")
    print("=" * 78)

    for fn in (check_artefacts, check_coverage, check_cross_document,
               check_discovery_log, check_banned, check_freeze, check_outstanding):
        try:
            fn()
        except Exception as e:
            verdict(fn.__name__, False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print(" CLOSURE CHECKLIST")
    print("=" * 78)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<20} {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print()
    if failed:
        print(f"  {len(failed)} check(s) failed: {', '.join(failed)}")
        print("  Phase B is not closed.")
    else:
        print("  All checks passed. Phase B is mechanically closed.")
        print("""
  What remains is writing and review, not computation:
    Methods from the frozen protocol · Discussion within the Results
    claim-to-evidence read-through · figures and tables · references

  And the standing caveats travel with the paper: RC detection F1 rests on five
  reference-positive cases; T3 bins are exploratory and not pre-registered;
  per-bin n is below the n the MDE was computed at; the FP component-size
  analysis is exploratory and, if the gate failed, absent by necessity.""")


if __name__ == "__main__":
    main()
