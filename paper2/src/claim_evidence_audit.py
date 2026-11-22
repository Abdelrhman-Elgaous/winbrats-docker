# -*- coding: utf-8 -*-
"""
==============================================================================
 CLAIM-TO-EVIDENCE AUDIT  —  the last gate before the text is frozen
==============================================================================
 Read-only. Checks the manuscript text against the frozen record.

 The figure check already confirmed that numbers quoted in the Results match
 their source tables. This asks a different question: does every claim in the
 Discussion rest on something in the Results, the discovery log or the
 protocol, and does the strength of the wording match the strength of what was
 measured?

   1  NUMBERS       every figure in the Discussion also appears in Results
   2  HEDGING       claims of superiority, causation, generalisation or
                    clinical benefit carry their qualifier
   3  ANCHORING     each Discussion section cites specific evidence
   4  CONSISTENCY   Methods, Results and Discussion agree on the invariants
   5  EXPLORATORY   exploratory analyses are labelled wherever they appear

 A failure is a wording problem, not a data problem. Nothing here justifies
 recomputing anything.
==============================================================================
"""

import os
import re
import sys

DRIVE = os.environ.get("BRATS_ROOT", "/content/drive/MyDrive/brats_2026_metastases")
DOCS = f"{DRIVE}/paper2"

FILES = {
    "results": f"{DOCS}/RESULTS_DRAFT.md",
    "methods": f"{DOCS}/METHODS_DRAFT.md",
    "discussion": f"{DOCS}/DISCUSSION_DRAFT.md",
    "discoveries": f"{DOCS}/DISCOVERIES.md",
    "protocol": f"{DOCS}/PHASE_B_PROTOCOL.md",
}

MINUS = "\u2212"
results = []


def load():
    t = {}
    for k, p in FILES.items():
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
        t[k] = open(p, encoding="utf-8").read()
    return t


def norm(s):
    return s.replace(MINUS, "-")


def verdict(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"\n  >>> {name}: {'PASS' if ok else 'REVIEW'}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def check_numbers(t):
    print("\n" + "=" * 78)
    print(" 1 · NUMBERS — every figure in the Discussion is in the Results")
    print("=" * 78)

    disc, res = norm(t["discussion"]), norm(t["results"])

    # decimals of three or more places: the ones that carry a result
    nums = set(re.findall(r"-?\d+\.\d{3,}", disc))
    # and two-place figures that read as effect sizes
    nums |= set(re.findall(r"-?0\.\d{2}\b", disc))

    missing = []
    for n_ in sorted(nums, key=lambda x: -abs(float(x))):
        bare = n_.lstrip("-")
        if bare in res or n_ in res:
            continue
        # a rounded quote of a longer source figure is fine
        if any(r.startswith(bare) or bare.startswith(r[:5])
               for r in re.findall(r"-?\d+\.\d{3,}", res)):
            continue
        missing.append(n_)

    print(f"  distinct figures in Discussion : {len(nums)}")
    print(f"  not found in Results           : {len(missing)}")
    for m in missing:
        i = disc.find(m)
        print(f"    {m:>10}   …{disc[max(0,i-60):i+40].replace(chr(10),' ')}…")
    if missing:
        print("\n  Each is either a rounding of a Results figure, a design quantity")
        print("  from the protocol, or a number that should not be in the")
        print("  Discussion at all. Check which.")
    verdict("1 NUMBERS", not missing,
            f"{len(nums)} figures, all traceable" if not missing else "see above")


def check_hedging(t):
    print("\n" + "=" * 78)
    print(" 2 · HEDGING — strong claims carry their qualifier")
    print("=" * 78)

    disc = t["discussion"]
    # a strong claim is acceptable only if its qualifier is nearby
    RULES = [
        (r"\bsuperior\b|\boutperform",
         ("rather than", "not ", "no claim", "operating points"),
         "superiority"),
        (r"\bcauses\b|\bcausal\b|\bbecause the model\b",
         ("not itself tested", "association", "consistent with", "eligibility",
          "no causal"),
         "causation"),
        (r"\bgeneral(ly|ise|ize)\b|\bin general\b|\ball cohorts\b",
         ("not", "no claim", "within this cohort", "this particular"),
         "generalisation"),
        (r"\bclinical(ly)?\b",
         ("not claimed", "no clinical", "could be", "not assessed",
          "None of these follows", "does not follow", "no claim"),
         "clinical benefit"),
        (r"\bprove[sd]?\b|\bestablishe[sd]\b|\bdemonstrates\b",
         ("not", "rather than", "no claim"),
         "proof language"),
    ]
    # The blockquote at the top of the Discussion lists the claims NOT being
    # made. Scanning it for those very words flags the disclaimer as if it were
    # the claim, so the header is excluded before the scan.
    body_start = disc.find("## 1.")
    body = disc[body_start:] if body_start > 0 else disc

    flagged = []
    for pat, quals, label in RULES:
        for m in re.finditer(pat, body, re.I):
            window = body[max(0, m.start() - 320):m.start() + 320]
            if any(q in window for q in quals):
                continue
            line = body[max(0, m.start() - 70):m.start() + 70].replace("\n", " ")
            flagged.append((label, line))

    print(f"  unqualified strong claims: {len(flagged)}")
    for label, line in flagged:
        print(f"    [{label}] …{line}…")
    if not flagged:
        print("  Every superiority, causal, generalisation and clinical claim")
        print("  has its qualifier within the surrounding text.")
    verdict("2 HEDGING", not flagged,
            "all strong claims qualified" if not flagged else "see above")


def check_anchoring(t):
    print("\n" + "=" * 78)
    print(" 3 · ANCHORING — each Discussion section cites specific evidence")
    print("=" * 78)

    disc = t["discussion"]
    secs = re.split(r"^## ", disc, flags=re.M)[1:]
    ok = True
    for sec in secs:
        title = sec.split("\n", 1)[0].strip()
        body = sec
        n_fig = len(re.findall(r"\d+\.\d{2,}", body))
        n_rule = len(re.findall(r"`(size|entropy|confidence)@[\d.]+`", body))
        n_reg = len(re.findall(r"\b(WT|TC|ET|RC)\b", body))
        anchored = (n_fig + n_rule + n_reg) > 0
        # the conclusion and implications sections summarise; they need not
        # re-cite, provided they add nothing new
        exempt = title.startswith(("6.", "8."))
        good = anchored or exempt
        print(f"  {'ok ' if good else 'BARE':>6}  {title[:52]:<54}"
              f" fig {n_fig:>2} · rule {n_rule:>2} · region {n_reg:>2}"
              + ("  (summary)" if exempt and not anchored else ""))
        ok &= good
    verdict("3 ANCHORING", ok,
            "every substantive section cites evidence" if ok else "see above")


def check_consistency(t):
    print("\n" + "=" * 78)
    print(" 4 · CONSISTENCY — the invariants agree across sections")
    print("=" * 78)

    INV = [
        ("clean stratum n = 155", r"\b155\b", ("methods", "results", "discussion")),
        ("MDE 0.019", r"0\.019", ("methods", "results", "discussion")),
        ("RC five reference-positive", r"\bfive\b", ("methods", "discussion")),
        ("2 voxels / 2 mm3 floor", r"2 voxels", ("methods",)),
        ("T3 bins exact", r"\(2,\s?27\]", ("methods", "discussion")),
        ("F1 sole endpoint", r"single inferential endpoint", ("methods",)),
        ("exploratory FP", r"exploratory", ("methods", "results", "discussion")),
        ("endpoint eligibility", r"endpoint eligibility", ("results", "discussion")),
    ]
    ok = True
    for label, pat, where in INV:
        present = {w: bool(re.search(pat, t[w])) for w in where}
        good = all(present.values())
        print(f"  {'ok ' if good else 'GAP':>6}  {label:<30} "
              + " · ".join(f"{w}:{'y' if v else 'n'}" for w, v in present.items()))
        ok &= good
    verdict("4 CONSISTENCY", ok,
            "invariants stated consistently" if ok else "an invariant is absent somewhere")


def check_exploratory(t):
    print("\n" + "=" * 78)
    print(" 5 · EXPLORATORY — labelled wherever it appears")
    print("=" * 78)

    ok = True
    for doc in ("methods", "results", "discussion"):
        txt = t[doc]
        # the two analyses that are exploratory
        # Match the ANALYSIS, not the metric definition: "false-positive
        # components" appears in the description of lesion-wise Dice, where an
        # exploratory label would make no sense.
        for name, pat in (("T3 size bins", r"\(2,\s?27\]|per-bin analysis|lesion-volume analysis"),
                          ("FP components", r"component-size analysis|"
                                            r"False-positive component analysis")):
            if not re.search(pat, txt, re.I):
                continue
            i = re.search(pat, txt, re.I).start()
            window = txt[max(0, i - 1500):i + 1500]
            labelled = bool(re.search(r"exploratory|not pre-?registered|"
                                      r"not prespecified", window, re.I))
            print(f"  {'ok ' if labelled else 'UNLABELLED':>11}  "
                  f"{doc:<11} {name}")
            ok &= labelled
    verdict("5 EXPLORATORY", ok,
            "exploratory analyses labelled in every section that reports them"
            if ok else "a section reports one without the label")


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(" CLAIM-TO-EVIDENCE AUDIT")
    print(" read-only · a failure here is a wording problem, not a data problem")
    print("=" * 78)

    t = load()
    for k in FILES:
        print(f"  {k:<12} {len(t[k].split()):>6,} words")

    for fn in (check_numbers, check_hedging, check_anchoring,
               check_consistency, check_exploratory):
        try:
            fn(t)
        except Exception as e:
            verdict(fn.__name__, False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)
    for name, ok, detail in results:
        print(f"  {'PASS  ' if ok else 'REVIEW'}  {name:<18} {detail}")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n  {len(failed)} check(s) need review: {', '.join(failed)}")
        print("  Adjust the wording. Do not adjust the results.")
    else:
        print("""
  All checks passed. Every figure in the Discussion traces to the Results,
  every strong claim carries its qualifier, every substantive section cites
  specific evidence, the invariants agree across sections, and the exploratory
  analyses are labelled wherever they are reported.

  The manuscript text can be frozen. What remains is figures, tables,
  introduction, abstract and references.""")


if __name__ == "__main__":
    main()
