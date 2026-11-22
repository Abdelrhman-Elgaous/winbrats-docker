# -*- coding: utf-8 -*-
"""
==============================================================================
 B-2b — REGRESSION TEST AGAINST PAPER 1        ★G1b
==============================================================================
 The strongest available verification of the geometry fix, and it costs nothing
 extra: Paper 1 published per-case lesion-wise scores for the 567 cohort
 (3-fold mean = 0.6523). If native_space.py reproduces those numbers, the fix
 is validated against an INDEPENDENT, ALREADY-PUBLISHED result rather than
 against an internal sanity check.

 PER-CASE COMPARISON IS THE POINT
 --------------------------------
 Comparing only the aggregate mean is weak: offsetting errors cancel. This test
 compares case by case and reports the worst disagreement, so it works on a
 subset and still catches a single broken case.

 If the published per-case file is unavailable, the test falls back to the
 aggregate mean over whatever subset is scored -- clearly labelled as the
 weaker check, because a subset mean is not expected to equal 0.6523 exactly.

 GATE ★G1b
 ---------
   PASS  max per-case |delta| <= TOL        -> proceed to B-4 / B-5
   FAIL  anything else                      -> STOP, do not start B-4

 A failure here means the module does not reproduce a published result. That is
 not a tolerance to loosen; it is a bug to find.
==============================================================================
"""

import os, sys

# Make native_space importable regardless of cwd or how this is launched.
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
for _p in (_HERE, "/content/drive/MyDrive/brats_2026_metastases/src/eval"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import os, json, glob, argparse
from datetime import datetime

import numpy as np
import pandas as pd

import native_space as ns

OUT_DIR = f"{ns.DRIVE}/phase_b_outputs/wave1"
# The Phase A per-case file. phase_b_0_audit.py line 1054 states plainly:
#   "baseline is NOT recomputed -- it is read from per_case_dice_3fold.csv"
# so this is the published reference the regression test must reproduce.
# ORDER MATTERS. An earlier version listed per_case_dice_3fold.csv first and
# matched a 130-case statistical-test artifact, stopping before it reached the
# complete file. The full reference is:
#   results/metrics/lesionwise_validation_250/per_case_lesionwise.csv
#   567 cases x 4 regions x 2 configs = 4,536 rows, with lesionwise_dice,
#   legacy_dice and official num_tp / num_fp / num_fn.
# Broadest and most complete file first; partial artifacts last.
PUBLISHED_GLOB = [
    f"{ns.DRIVE}/results/metrics/lesionwise_validation_250/per_case_lesionwise.csv",
    f"{ns.DRIVE}/**/per_case_lesionwise.csv",
    f"{ns.DRIVE}/**/per_case*lesion*.csv",
    f"{ns.DRIVE}/**/per_case_dice_3fold.csv",
]

PUBLISHED_MEAN_3FOLD = 0.6523        # Paper 1, full 567 cohort
TOL_PER_CASE = 1e-3                  # identical pipeline -> should be ~1e-6
TOL_AGGREGATE = 5e-3

os.makedirs(OUT_DIR, exist_ok=True)


def find_published():
    for pat in PUBLISHED_GLOB:
        for p in sorted(glob.glob(pat, recursive=True)):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            cols = {c.lower() for c in df.columns}
            if "case" in cols and any("dice" in c for c in cols):
                print(f"  found published per-case file: {os.path.relpath(p, ns.DRIVE)}")
                return p, df
    print("  no published per-case file found — falling back to the aggregate check")
    return None, None


def normalise(df):
    """Reduce a published table to case -> mean lesion-wise Dice across regions."""
    cols = {c.lower(): c for c in df.columns}
    case_col = cols.get("case") or cols.get("case_id")
    dice_col = (cols.get("lesionwise_score_dice") or cols.get("lesionwise_dice")
                or cols.get("dice"))
    if not case_col or not dice_col:
        return None

    # Files carrying several configurations must be filtered to the deployed
    # 3-fold arm, or the mean silently blends 3-fold and 5-fold rows.
    if "config" in cols:
        ccol = cols["config"]
        vals = df[ccol].astype(str).unique().tolist()
        pick = next((v for v in vals if "3" in v), None)
        if pick is not None:
            print(f"  config column present {vals} -> using '{pick}'")
            df = df[df[ccol].astype(str) == pick]

    g = df.groupby(df[case_col])[dice_col].mean()
    return g.to_dict()


def main(n_cases, seed):
    print("=" * 74); print(" B-2b — REGRESSION TEST vs PAPER 1  (★G1b)"); print("=" * 74)

    pub_path, pub_df = find_published()
    published = normalise(pub_df) if pub_df is not None else None

    per_fold = {f: {os.path.basename(p)[:-4]
                    for p in glob.glob(f"{ns.ENSEMBLE_DIR}/fold_{f}/*.npz")}
                for f in ns.FOLDS}
    cohort = sorted(set.intersection(*per_fold.values()))
    print(f"  cohort: {len(cohort)} cases")

    if published:
        cohort = [c for c in cohort if c in published]
        print(f"  overlapping with published file: {len(cohort)}")

    rng = np.random.default_rng(seed)
    sel = sorted(rng.choice(cohort, size=min(n_cases, len(cohort)), replace=False).tolist())
    print(f"  scoring {len(sel)} case(s), seed={seed}\n")

    rows, deltas, errors = [], [], []
    for i, c in enumerate(sel, 1):
        try:
            r = ns.official_score(c, suppressor=None, tag="b2b")
        except ns.GeometryError as e:
            print(f"  [{i}/{len(sel)}] {c}  GUARD FAILED: {e}")
            errors.append({"case": c, "error": str(e)}); continue
        except Exception as e:
            print(f"  [{i}/{len(sel)}] {c}  ERROR: {e}")
            errors.append({"case": c, "error": str(e)}); continue

        rows += r
        mine = float(np.mean([x["lesionwise_dice"] for x in r]))
        if published and c in published:
            d = mine - published[c]
            deltas.append({"case": c, "mine": mine, "published": published[c], "delta": d})
            flag = "" if abs(d) <= TOL_PER_CASE else "   <-- MISMATCH"
            print(f"  [{i}/{len(sel)}] {c}  mine={mine:.4f}  pub={published[c]:.4f}  "
                  f"delta={d:+.6f}{flag}")
        else:
            print(f"  [{i}/{len(sel)}] {c}  mine={mine:.4f}")

    if not rows:
        raise SystemExit("no case scored successfully — fix the errors above first")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/b2b_scores.csv", index=False)

    print("\n" + "=" * 74); print(" VERDICT"); print("=" * 74)

    if deltas:
        dd = pd.DataFrame(deltas)
        worst = dd.iloc[dd["delta"].abs().idxmax()]
        max_abs = float(dd["delta"].abs().max())
        mode = "per-case"
        ok = max_abs <= TOL_PER_CASE
        print(f"  compared     : {len(dd)} case(s) against the published file")
        print(f"  max |delta|  : {max_abs:.6f}  (tolerance {TOL_PER_CASE})")
        print(f"  worst case   : {worst['case']}  mine={worst['mine']:.4f} "
              f"pub={worst['published']:.4f}")
        dd.to_csv(f"{OUT_DIR}/b2b_deltas.csv", index=False)
        detail = (f"max per-case |delta| = {max_abs:.6f} over {len(dd)} case(s)")
    else:
        mine_mean = float(df.groupby("case")["lesionwise_dice"].mean().mean())
        diff = mine_mean - PUBLISHED_MEAN_3FOLD
        max_abs, mode = abs(diff), "aggregate (WEAK)"
        ok = max_abs <= TOL_AGGREGATE
        print(f"  subset mean  : {mine_mean:.4f}")
        print(f"  published    : {PUBLISHED_MEAN_3FOLD:.4f}  (full 567 cohort)")
        print(f"  difference   : {diff:+.4f}")
        print("  NOTE: a subset mean is not expected to equal the full-cohort mean. "
              "This check is weak — locate the published per-case file for a real test.")
        detail = f"aggregate difference = {diff:+.4f} (weak check)"

    if errors:
        ok = False
        print(f"\n  {len(errors)} case(s) failed outright:")
        for e in errors[:5]:
            print(f"    {e['case']}: {e['error'][:120]}")

    status = "PASS" if ok else "FAIL"
    print(f"\n  ★G1b: {status}")
    if ok:
        print("  The geometry fix reproduces a published result. Proceed to B-4 / B-5.")
    else:
        print("  STOP. The module does not reproduce Paper 1. Do not start B-4.")
        print("  This is a bug to find, not a tolerance to loosen.")

    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "status": status, "mode": mode, "detail": detail,
               "n_scored": len(sel), "n_errors": len(errors), "seed": seed,
               "published_file": os.path.relpath(pub_path, ns.DRIVE) if pub_path else None},
              open(f"{OUT_DIR}/b2b_verdict.json", "w"), indent=2)
    print(f"\n  written: {OUT_DIR}/b2b_verdict.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-cases", type=int, default=20)
    ap.add_argument("-s", "--seed", type=int, default=137)
    a = ap.parse_args()
    main(a.n_cases, a.seed)
