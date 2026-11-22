# -*- coding: utf-8 -*-
"""
==============================================================================
 B-4 — SUPPRESSED ARM over the 567-case cohort
==============================================================================
 Computes BOTH arms. The baseline is recomputed here rather than read from
 per_case_dice_3fold.csv, so both arms pass through identical label
 construction, thresholds and precedence -- any difference between them then
 comes from the suppression rule alone. At six rules the baseline is 567 of
 3,402 evaluations (~17%), and B-2b already reproduced the published per-case
 scores exactly (max |delta| = 0.000000, n=20). The published file is kept only
 as a non-load-bearing cross-check at the end.

 PROTOCOL: PHASE_B_PROTOCOL.md v1.0 (Frozen 13 Aug 2026).

 DESIGN CONSTRAINTS MEASURED IN WAVE 0
 -------------------------------------
 Peak RSS was ~6.1 GiB and did NOT grow between a mid-sized case and the
 largest volume in the cohort (4, 420, 512, 512). Two consequences:

   1. Sequential execution is the safe default. No evidence of cumulative
      growth was observed across the tested cases, but that is not a proof of
      absence, so the runner checkpoints continuously and resumes cleanly.

   2. If workers are used, they must be scheduled SIZE-AWARE. Naive contiguous
      chunking can put several of the largest volumes in flight at the same
      moment and exhaust RAM mid-run. Cases are therefore sorted by file size
      and dealt round-robin, which spreads the heavy ones across workers.

 WHAT IT WRITES PER CASE
 -----------------------
   case · setting · region · lesionwise_dice · legacy_dice · leak_class
   seconds · status

 legacy_dice is carried alongside lesionwise_dice deliberately: it makes the
 voxel-vs-lesion gap computable over the full cohort at zero extra cost, which
 is otherwise a separate experiment.

 GUARDS  --  guards=False, protocol section 5 (frozen 13 Aug 2026)
 ------
 Guard 3 fires when suppression empties WT. Those are precisely the cases where
 suppression removed the most, i.e. Dice = 0. Excluding them biases the mean IN
 FAVOUR of suppression -- the quantity under test. The 20-case pilot proved it:
 the two cases size_pilot.py had skipped came back at WT = 0.0000 under both
 size rules, and their absence had inflated the 18-case mean.

 So guards are OFF here and every failure is written to the CSV as four rows
 carrying a `status` string. A row that is ABSENT means "not yet reached"; a row
 with status="FAILED: ..." means "reached and failed". Those two states were
 indistinguishable before, which is what invalidated Exp3.

 Metric columns stay EMPTY on failure -- never 0.0. Zero is a real and
 meaningful score here (suppression emptied the region), so writing zero for
 "did not run" would corrupt the exact comparison this arm exists for.

 Guards remain fully active in guard3_diagnose.py, which is where pipeline
 faults are classified. The guard's job is to catch pipeline faults, not hard
 cases; on a frozen pipeline the correct behaviour here is to record.
==============================================================================
"""


# Make native_space importable regardless of cwd or how this is launched.
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ""
for _p in (_HERE, "/content/drive/MyDrive/brats_2026_metastases/src/eval"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import os, sys, json, glob, time, argparse, traceback
from datetime import datetime

import numpy as np
import pandas as pd

import native_space as ns

OUT_DIR   = f"{ns.DRIVE}/phase_b_outputs/wave1"
CKPT_CSV  = f"{OUT_DIR}/exp3_suppressed_v3.csv"   # v3: adds `status`
FAIL_LOG  = f"{OUT_DIR}/exp3_failures.json"
LEAK_CSV  = f"{ns.DRIVE}/phase_b_outputs/wave0/leak_class_567.csv"
BASELINE  = f"{ns.DRIVE}/phase_b_outputs/per_case_dice_3fold.csv"

CKPT_EVERY = 1

# Suppression rules to evaluate. A sweep rather than a single operating point,
# because B-6 needs the threshold-stability curve: a result that only holds at
# one threshold is not a result.
# "baseline" is computed HERE rather than read from a published CSV. Two reasons:
#
#   1. COST. With a 9-rule sweep the baseline is 567 of 5,670 evaluations --
#      about 11%, not the "half the compute" an earlier plan version assumed
#      (that framing dated from when B-4 tested a single candidate rule).
#
#   2. IDENTICAL PATH. Both arms then go through exactly the same label
#      construction, thresholds and precedence, so any difference between them
#      comes from the suppression rule alone. Reading one arm from a file
#      produced by an older script leaves that guarantee resting on a sampled
#      comparison instead of on construction.
#
# B-2b reproduced the published per-case scores exactly (max |delta| = 0.000000,
# n=20), so recomputation is expected to agree -- it converts empirical
# agreement into a structural one at ~17% extra cost (567 of 3,402 evaluations).
#
# FINAL RULE SET -- protocol section 4, FROZEN 13 Aug 2026. Six rules, each tied
# to a question in the experiment map:
#
#   baseline         reference arm, computed here rather than read from a CSV so
#                    both arms pass through identical label construction
#   size@50          Exp2b -- the morphology control. Small components are the
#                    most likely false positives, and filtering them is free and
#                    fully interpretable. Any claim that uncertainty helps must
#                    beat this, or the added complexity is unjustified. 50 sits
#                    far above the evaluator's scoring floor, so the arm is
#                    answerable on the ranked metric.
#                    [CORRECTED 16 Aug: an earlier version of this note said the
#                    floor was "27 mm3". It is not. BraTS-MET sets
#                    lesion_volume_thresh = 2 VOXELS (metrics_MET.py:681) and
#                    applies it to predicted components (line 319) as well as to
#                    reference lesions (743, 747, 761). The figure 27 was an
#                    undocumented assumption of ours, never an evaluator value.
#                    See protocol section 13. No score in this file is affected:
#                    every number came from the evaluator using its own floor.]
#   entropy@*        Exp3 -- three points bracketing the Layer 1 operating point
#   confidence@0.45  Exp3 -- the dominant signal from Layer 1, as a control
#
# TWO SIZE RULES WERE REMOVED. Both removals are recorded in protocol section 4.
#
#   size@20   removed 4 Aug on an argument that was later shown to be wrong.
#             The original reasoning -- "20 is below the evaluator's 27-voxel
#             floor, so removing those components cannot move Dice" -- rests on
#             a floor value that does not exist. The real floor is 2 voxels, so
#             size@20 operates ten times above it.
#             The supporting test was misread too: on BraTS-MET-01340-004 the
#             rule deleted 3 WT voxels and one 1-voxel RC component and Dice
#             moved by 0.0001 -- but those components are BELOW 2, so the
#             evaluator had already discarded them. The test measured
#             pre-excluded components, not the rule.
#             fp_floor_probe.py later moved Dice in 9 of 9 cells where sub-20
#             components existed. The rule is a valid candidate again; it is NOT
#             reinstated here, and this run is unaffected either way.
#
#   size@100  removed 13 Aug, from the completed 20-case pilot. Mean effect
#             -0.0925 versus -0.0591 for size@50, and worse in 20 of the 24
#             cells where the two differed (better in at most 4). Worse in every
#             region except RC, where the two are identical. Harm is monotone in
#             the threshold, so H3 is answered by size@50 and a second, larger
#             size point buys nothing but compute.
#
# An earlier draft swept nine rules (~529 h projected). Layer 1 already located
# the region of interest, so points outside it generate numbers rather than
# information. A combined scorer was also dropped: it multiplies the threshold
# space and worsens multiple-comparison correction. If the individual results
# show complementarity, it can be added on evidence.
#
# For context on where 50 sits: published thresholds span zero (brain.exe, no
# filtering at all) through 5 mm3 (Newman-Norlund) to 20 mm3 (Rudzki), so 20 is
# the upper end of current practice and 50 probes past it.
RULES = {
    "baseline":        (None, None),
    "size@50":         ("size", 50),
    "entropy@0.45":    ("entropy", 0.45),
    "entropy@0.50":    ("entropy", 0.50),
    "entropy@0.56":    ("entropy", 0.56),
    "confidence@0.45": ("confidence", 0.45),
}

PROTOCOL_RULES = {"baseline", "size@50",
                  "entropy@0.45", "entropy@0.50", "entropy@0.56",
                  "confidence@0.45"}
assert set(RULES) == PROTOCOL_RULES, f"RULES drift vs §4: {set(RULES) ^ PROTOCOL_RULES}"

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  PER-WORKER OUTPUT PATHS
# ---------------------------------------------------------------------------
#  Every checkpoint write below is a FULL to_csv, not an append, and each worker
#  holds its own `rows` list in memory. Pointing several workers at one path
#  therefore makes the last writer ERASE the others, and the damage is silent:
#  the file stays valid CSV, it is just missing rows at random, and you find out
#  by counting at the end of a 300-hour run. Shards are combined afterwards by
#  merge_shards.py, which verifies the union is exactly 567 x 6 with no pair
#  appearing twice.

def _suffix(worker, n_workers, part, n_parts):
    """
    Two independent splits, each with its own suffix segment:

      .w{k}of{n}   cases  -- round-robin over the cost order
      .p{k}of{n}   rules  -- round-robin over the selected rules

    Both are needed because they carve the work along different axes and a
    process is identified by the pair. The case split alone caps parallelism
    at n_workers; adding the rule split multiplies it without redoing the case
    partition, so shards already on disk stay valid.
    """
    out = ""
    if n_workers > 1:
        out += f".w{worker}of{n_workers}"
    if n_parts > 1:
        out += f".p{part}of{n_parts}"
    return out


def _shard_path(worker, n_workers, part=0, n_parts=1):
    sfx = _suffix(worker, n_workers, part, n_parts)
    return CKPT_CSV if not sfx else CKPT_CSV.replace(".csv", f"{sfx}.csv")


def _fail_path(worker, n_workers, part=0, n_parts=1):
    sfx = _suffix(worker, n_workers, part, n_parts)
    return FAIL_LOG if not sfx else FAIL_LOG.replace(".json", f"{sfx}.json")


def _sibling_shards():
    """Every shard file on disk, whatever its worker/part split."""
    import re as _re
    pat = _re.compile(r"\.w\d+of\d+(\.p\d+of\d+)?\.csv$")
    return sorted(f for f in glob.glob(CKPT_CSV.replace(".csv", ".w*of*.csv"))
                  if pat.search(f))


def safe_tag(rule):
    """
    Filesystem-safe tag. The official evaluator builds intermediate paths from
    the prediction filename and truncates at the first dot, so a tag like
    "entropy@0.50" becomes "entropy@0" and the run fails with a missing-file
    error. Dots and '@' are therefore removed entirely.
    """
    return rule.replace("@", "_at_").replace(".", "p")


def make_suppressor(kind, thr):
    if kind is None:
        return None                      # baseline: no components removed
    if kind == "size":
        return ns.size_suppressor(thr)
    if kind == "entropy":
        return ns.entropy_suppressor(thr)
    return ns.confidence_suppressor(thr)


def load_leak_class():
    """
    Three-way leakage class per case: clean / patient_only / seen.

    Two independent mechanisms are folded in. Ensemble leakage (the deployed
    folds saw the case itself) was reported in Paper 1 as 320/247. Patient
    leakage (another timepoint of the same patient was in training) was found
    later: splits_final.json is grouped per case, not per patient, and 118
    patients in this cohort contribute more than one session. Of the 320 cases
    previously counted as leakage-free, 165 carry patient leakage -- so the
    earlier binary split misclassified about half of that stratum.

    clean = 155 cases. That is the stratum the headline result is computed on;
    the others are reported as sensitivity analyses.
    """
    df = pd.read_csv(LEAK_CSV)
    counts = df["leak_class"].value_counts().to_dict()
    print(f"  leak_class        : " +
          " · ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return dict(zip(df["case"], df["leak_class"]))


def load_cohort_ids():
    """The pre-registered 567. Fails loud rather than falling back."""
    if not os.path.exists(LEAK_CSV):
        raise FileNotFoundError(
            f"cohort file missing: {LEAK_CSV}\n"
            "  B-4 will not fall back to a directory listing — the cohort is "
            "pre-registered and must come from this file. Run leak_class.py first.")
    ids = pd.read_csv(LEAK_CSV)["case"].tolist()
    if len(ids) != 567:
        print(f"  [!] cohort file has {len(ids)} rows, protocol says 567 — verify")
    return ids


def ordered_cases():
    """
    Order by expected COST, not by file size.

    The smoke test showed runtime is driven by LESION COUNT, not volume: a
    0.23 GiB case took 14,218 s while a larger one took 489 s. The evaluator
    dilates and runs connected components per lesion, so cost scales with the
    number of components. File size still bounds peak RAM, so both matter, but
    ordering by size alone tells you nothing about how long a case will take.

    num_fp + num_tp + num_fn from the published per-case file is the best
    available proxy. Falls back to file size when that file is unavailable.
    """
    cost = {}
    pub = f"{ns.DRIVE}/results/metrics/lesionwise_validation_250/per_case_lesionwise.csv"
    if os.path.exists(pub):
        try:
            pd_pub = pd.read_csv(pub)
            if "config" in pd_pub.columns:
                pick = next((v for v in pd_pub["config"].astype(str).unique()
                             if "3" in v), None)
                if pick:
                    pd_pub = pd_pub[pd_pub["config"].astype(str) == pick]
            g = pd_pub.groupby("case")[["num_tp", "num_fp", "num_fn"]].sum().sum(axis=1)
            cost = g.to_dict()
            print(f"  lesion-count cost model loaded for {len(cost)} cases")
        except Exception as e:
            print(f"  [note] cost model unavailable ({e}) — falling back to file size")

    # The COHORT is defined by leak_class_567.csv, not by what happens to be on
    # disk. Intersecting fold directories makes the denominator depend on file
    # availability: one missing npz and the cohort silently becomes 566, which
    # no downstream script would notice. Missing predictions are reported and
    # then recorded as failure rows, not dropped.
    cohort = sorted(load_cohort_ids())
    per_fold = {f: {os.path.basename(p)[:-4]
                    for p in glob.glob(f"{ns.ENSEMBLE_DIR}/fold_{f}/*.npz")}
                for f in ns.FOLDS}
    available = set.intersection(*per_fold.values()) if per_fold else set()

    missing = [c for c in cohort if c not in available]
    if missing:
        print(f"  [!] {len(missing)} cohort case(s) have no prediction in all folds:")
        for c in missing[:10]:
            print(f"        {c}")
        print("      these will be written as FAILED rows, not dropped")

    extra = len(available - set(cohort))
    if extra:
        print(f"  [note] {extra} prediction(s) on disk are outside the cohort — ignored")

    cases = [c for c in cohort if c in available]
    sized = []
    for c in cases:
        try:
            sz = os.path.getsize(f"{ns.ENSEMBLE_DIR}/fold_{ns.FOLDS[0]}/{c}.npz")
        except OSError:
            sz = 0
        sized.append((cost.get(c, sz / 1e6), sz, c))
    sized.sort(reverse=True)          # most expensive first: fail fast, not late
    return ([c for _, _, c in sized],
            {c: sz for _, sz, c in sized},
            {c: w for w, _, c in sized},
            missing)


def _check_schema(df, path):
    if "leak_free" in df.columns:
        raise RuntimeError(
            f"{path} carries the pre-30-Jul 'leak_free' column. That schema was "
            "replaced by the three-level 'leak_class'; mixing them in one file is "
            "what invalidated v2. Archive the file and restart.")
    if "status" not in df.columns:
        raise RuntimeError(
            f"{path} has no 'status' column -- it predates the 13 Aug guard "
            "policy and may contain silently skipped failures. Archive it.")
    stale = set(df["setting"].astype(str)) - PROTOCOL_RULES
    if stale:
        raise RuntimeError(
            f"{path} contains rows for rules not in the frozen protocol: {stale}. "
            "Archive the file and restart.")


def done_keys(ckpt):
    """
    Resume state, with three schema guards. exp3_suppressed_v2.csv was found to
    mix rows written under the pre-30-Jul binary `leak_free` column with rows
    written under the three-level `leak_class`, plus rows for the removed
    size@20 rule. Silently concatenating that into a new run is how a table
    ends up carrying numbers from a superseded protocol.
    """
    own_rows, seen = [], set()

    # OWN file -> both the seen set and the in-memory buffer this process
    # rewrites on every checkpoint.
    if os.path.exists(ckpt):
        df = pd.read_csv(ckpt)
        _check_schema(df, ckpt)
        own_rows = df.to_dict("records")
        seen |= set(zip(df["case"], df["setting"]))
        n_fail = int((df["status"].astype(str) != "ok").sum()) // len(REGIONS)
        print(f"  resume   : {len(df)} rows in {os.path.basename(ckpt)} "
              f"({n_fail} failed pair(s))")

    # SIBLING shards -> the seen set ONLY. Their rows must never enter this
    # process's buffer: every checkpoint write is a full to_csv, so copying a
    # sibling's rows here would duplicate every one of them at merge time.
    # Reading them matters when the rule split is introduced mid-run: pairs
    # already finished by the un-split process must not be recomputed.
    extra = 0
    for f in _sibling_shards():
        if os.path.abspath(f) == os.path.abspath(ckpt):
            continue
        d = pd.read_csv(f)
        _check_schema(d, f)
        before = len(seen)
        seen |= set(zip(d["case"], d["setting"]))
        extra += len(seen) - before
    if extra:
        print(f"  siblings : {extra} pair(s) already done elsewhere — skipped, "
              f"not copied")

    return seen, own_rows


REGIONS = ("WT", "TC", "ET", "RC")


def fail_rows(case, rule, leak, seconds, detail):
    """
    Four rows -- one per region -- with the metric columns EMPTY and the reason
    in `status`. Empty, never 0.0: zero is a real score (suppression emptied the
    region) and writing it for "did not run" would corrupt the comparison.
    """
    return [{"case": case, "setting": rule, "region": reg,
             "lesionwise_dice": None, "legacy_dice": None,
             "leak_class": leak, "seconds": round(seconds, 2),
             "status": detail} for reg in REGIONS]


def main(rules, limit, worker, n_workers, cheap_first,
         only_cases=None, part=0, n_parts=1):
    print("=" * 74); print(" B-4 — SUPPRESSED ARM")
    print(" protocol: PHASE_B_PROTOCOL.md v1.0 (Frozen 13 Aug 2026)")
    print("=" * 74)

    # ---- fast metric ------------------------------------------------------
    # native_space.official_score only calls set_fast_metric() when this
    # variable is present; with it unset the module keeps whatever state it has,
    # which in a fresh subprocess is the ORIGINAL implementation. A notebook
    # cell that ran ns.set_fast_metric(True) does NOT carry into a nohup child.
    # Protocol section 3 lists the fast path as part of the frozen setup and B0
    # verified it bit-identical (delta = 0), so it is enabled here explicitly
    # rather than left to the environment.
    if "BRATS_FAST_METRIC" not in os.environ:
        os.environ["BRATS_FAST_METRIC"] = "1"
    print(f"  fast     : BRATS_FAST_METRIC={os.environ['BRATS_FAST_METRIC']} "
          f"(2.73x, equivalence verified in B0)")

    cases, sizes, costs, missing = ordered_cases()
    if cheap_first:
        cases = cases[::-1]
        print("  order    : CHEAPEST FIRST (--cheap-first) — valid for production; "
              "reduces loss per Colab disconnect")
    if only_cases:
        # Explicit case selection. Needed because --limit slices the COST order
        # and cost is lesion count, so the cheapest cases are the empty ones --
        # there is no combination of --limit/--cheap-first that reaches a case
        # with lesions in it. Smoke tests only; the full run takes the cohort.
        bad = [c for c in only_cases if c not in cases]
        if bad:
            raise SystemExit(f"not in cohort or no prediction available: {bad}")
        cases = [c for c in cases if c in only_cases]
        print(f"  cases    : EXPLICIT — {cases}")

    # ---- rule split -------------------------------------------------------
    if n_parts > 1:
        if not 0 <= part < n_parts:
            raise SystemExit(f"--part must be in [0,{n_parts-1}], got {part}")
        rules = list(rules)[part::n_parts]
        if not rules:
            raise SystemExit(f"part {part} of {n_parts} selects no rule from "
                             f"{len(RULES)} — reduce --n-parts")
        print(f"  part     : {part} of {n_parts}  ->  rules {rules}")

    ckpt = _shard_path(worker, n_workers, part, n_parts)
    faillog = _fail_path(worker, n_workers, part, n_parts)

    # ---- per-process scratch ---------------------------------------------
    # The rule split puts several processes on the SAME case at the same time,
    # and _purge_case_temp() deletes tmp_gt/<case> before scoring. Sharing one
    # scratch directory would have one process delete files another is reading
    # -- silent corruption, not a crash. The case split alone never had this
    # problem because no two workers ever touch the same case.
    if n_parts > 1 or n_workers > 1:
        ns.WORK_DIR = os.path.join(ns.WORK_DIR, f"proc_w{worker}p{part}")
        os.makedirs(ns.WORK_DIR, exist_ok=True)
        print(f"  scratch  : {ns.WORK_DIR}")

    if n_workers > 1 or n_parts > 1:
        if not 0 <= worker < max(n_workers, 1):
            raise SystemExit(f"--worker must be in [0,{n_workers-1}], got {worker}")
        # A single-worker file and shard files must never be mixed: the merge
        # would double-count every pair the single run already covered.
        if os.path.exists(CKPT_CSV):
            raise SystemExit(
                f"{os.path.basename(CKPT_CSV)} exists (a single-worker run wrote it). "
                "Archive it before starting a sharded run, or the merge will "
                "double-count.")
        if n_workers > 1:
            cases = cases[worker::n_workers]      # round-robin over size order
        print(f"  worker   : {worker} of {n_workers}  ->  {os.path.basename(ckpt)}")
    else:
        others = glob.glob(CKPT_CSV.replace(".csv", ".w*of*.csv"))
        if others:
            raise SystemExit(
                f"{len(others)} shard file(s) exist from a previous sharded run. "
                "Merge or archive them before running single-worker.")
    if limit:
        cases = cases[:limit]

    leak = load_leak_class()
    seen, rows = done_keys(ckpt)

    # Cohort cases with no prediction: record once, as failures, so the
    # denominator stays 567 and the gap is visible in the table itself.
    for c in missing:
        for r in rules:
            if (c, r) not in seen:
                rows += fail_rows(c, r, leak.get(c), 0.0,
                                  "FAILED: no prediction npz in all folds")
                seen.add((c, r))
    if missing:
        pd.DataFrame(rows).to_csv(ckpt, index=False)

    todo = [(c, r) for c in cases for r in rules if (c, r) not in seen]
    print(f"  cases    : {len(cases)}")
    print(f"  rules    : {list(rules)}")
    print(f"  already  : {len(seen)} case-rule pairs")
    print(f"  to do    : {len(todo)}")
    if sizes:
        big = max(sizes.values()) / 1024**3
        print(f"  largest  : {big:.2f} GiB per fold  (peak RSS measured ~6.1 GiB)")
    if not todo:
        print("\n  nothing to do — already complete"); return

    failures = json.load(open(faillog)) if os.path.exists(faillog) else []
    t_start = time.time()

    for i, (case, rule) in enumerate(todo, 1):
        kind, thr = rules[rule]
        t0 = time.time()
        try:
            # guards=False — protocol section 5. A case that empties WT under
            # suppression is a RESULT (Dice 0), not a pipeline fault, and must
            # enter the sample.
            res = ns.official_score(case, suppressor=make_suppressor(kind, thr),
                                    tag=f"b4_{safe_tag(rule)}", guards=False)
            dt = time.time() - t0
            got = {r["region"] for r in res}
            if got != set(REGIONS):
                raise RuntimeError(f"evaluator returned regions {sorted(got)}")
            for r in res:
                r["leak_class"] = leak.get(case)
                r["seconds"] = round(dt, 2)
                r["setting"] = rule
                r["status"] = "ok"
            rows += res
            wt = next((r["lesionwise_dice"] for r in res if r["region"] == "WT"), float("nan"))
            print(f"  [{i}/{len(todo)}] {case} {rule:>18s}  WT={wt:.4f}  {dt:.1f}s")

        except Exception as e:
            dt = time.time() - t0
            detail = f"FAILED: {type(e).__name__}: {e}"
            rows += fail_rows(case, rule, leak.get(case), dt, detail)
            failures.append({"case": case, "rule": rule,
                             "type": "guard" if isinstance(e, ns.GeometryError) else "error",
                             "error": str(e), "trace": traceback.format_exc()[-500:],
                             "when": datetime.now().isoformat(timespec="seconds")})
            print(f"  [{i}/{len(todo)}] {case} {rule:>18s}  RECORDED AS FAILED — {str(e)[:80]}")

        if i % CKPT_EVERY == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            json.dump(failures, open(faillog, "w"), indent=2)
            # Cost-weighted ETA. A flat mean rate is badly biased under EITHER
            # ordering -- expensive-first starts optimistic, cheap-first starts
            # catastrophic -- because per-case time spans ~190x (75 s to
            # 14,218 s). Normalising elapsed time by the lesion-count cost
            # already consumed gives a rate that is comparable across the run.
            #
            # The floor on cost matters: empty cases have a lesion count of 0,
            # and dividing by a near-zero denominator is what produced the
            # spurious "ETA 2101 h" on 14 Aug. Each case contributes at least
            # 1.0 unit so the denominator can never collapse.
            _cost = lambda c: max(float(costs.get(c, 1.0) or 1.0), 1.0)
            done_cost = sum(_cost(c) for c, _ in todo[:i])
            left_cost = sum(_cost(c) for c, _ in todo[i:])
            elapsed = time.time() - t_start
            per_unit = elapsed / max(done_cost, 1e-9)
            eta = per_unit * left_cost / 3600
            print(f"      checkpoint: {len(rows)} rows · {elapsed/i:.1f}s/eval "
                  f"(flat) · ETA {eta:.1f}h (cost-weighted)")

    pd.DataFrame(rows).to_csv(ckpt, index=False)
    json.dump(failures, open(faillog, "w"), indent=2)

    print("\n" + "=" * 74); print(" SUMMARY"); print("=" * 74)
    df = pd.DataFrame(rows)
    if "status" not in df.columns:
        df["status"] = "ok"
    ok = df[df["status"].astype(str) == "ok"]
    print(f"  rows        : {len(df)}  ({len(ok)} ok, {len(df)-len(ok)} failed)")
    print(f"  failed pairs: {(len(df)-len(ok))//4}")

    # Optional cross-check against any published per-case file. This is no
    # longer load-bearing -- the baseline arm is computed above -- but agreement
    # on the overlapping cases is free extra evidence.
    if os.path.exists(BASELINE) and len(df):
        try:
            pub = pd.read_csv(BASELINE)
            ccol = next(c for c in pub.columns if "case" in c.lower())
            dcol = next(c for c in pub.columns if "dice" in c.lower())
            pub_mean = pub.groupby(ccol)[dcol].mean()
            mine = (ok[ok.setting == "baseline"]
                    .groupby("case")["lesionwise_dice"].mean())
            common = mine.index.intersection(pub_mean.index)
            if len(common):
                d = (mine[common] - pub_mean[common]).abs().max()
                print(f"\n  cross-check vs {os.path.relpath(BASELINE, ns.DRIVE)}")
                print(f"    overlapping cases : {len(common)}")
                print(f"    max |delta|       : {d:.6f}"
                      + ("   <-- INVESTIGATE" if d > 1e-3 else "   (agrees)"))
        except Exception as e:
            print(f"\n  [note] cross-check skipped: {e}")

    if len(ok):
        piv = ok.groupby("setting")["lesionwise_dice"].mean().sort_values(ascending=False)
        print("\n  mean lesion-wise Dice by rule (all regions, all cases):")
        for k, v in piv.items():
            print(f"    {k:>18s}  {v:.4f}")
        print("\n  NOTE: these are means over whatever has been scored so far and are")
        print("  NOT the analysis. B-5b stratifies by leak_class and B-6 runs the")
        print("  paired tests; the headline number is the 155-case CLEAN stratum.")
        print("  Pre-registered MDE at n=155 (SD of paired differences 0.082,")
        print("  alpha 0.05, power 0.80) is about 0.019 Dice — a smaller true")
        print("  effect is below detection, which is not the same as absent.")

        if "leak_class" in ok.columns and ok["leak_class"].notna().any():
            print("\n  by leak_class (scored so far):")
            for k, g in ok.groupby("leak_class"):
                print(f"    {str(k):>14s}  n_rows={len(g):>5d}  "
                      f"mean={g['lesionwise_dice'].mean():.4f}")

    if len(df) > len(ok):
        print(f"\n  >> {(len(df)-len(ok))//4} failed pair(s). They ARE in the CSV with")
        print("     status set — part of the sample, not missing from it. Diagnose")
        print("     them with guard3_diagnose.py before the statistical analysis:")
        print("     a Class-2 failure is a real Dice-0 result and stays; a Class-1")
        print("     or Class-3 failure is a pipeline fault and must be fixed and")
        print("     re-run. Do NOT analyse before that split is known.")
        bad = df[df["status"].astype(str) != "ok"]
        for (c, st), _ in list(bad.groupby(["case", "setting"]))[:15]:
            print(f"       {c} · {st}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="*", default=None, help="subset of rule names")
    ap.add_argument("--limit", type=int, default=0, help="first N cases (smoke test)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--part", type=int, default=0,
                    help="which rule-slice this process runs (0-based)")
    ap.add_argument("--n-parts", type=int, default=1,
                    help="split the rules into N slices; combine with --worker "
                         "to run N x n_workers processes")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="explicit case IDs — smoke tests only, not for the full run")
    ap.add_argument("--cheap-first", action="store_true",
                    help="reverse the cost order — for smoke-testing resume only")
    a = ap.parse_args([x for x in sys.argv[1:] if not x.startswith("-f")])

    sel = {k: v for k, v in RULES.items() if (not a.rules or k in a.rules)}
    main(sel, a.limit, a.worker, a.n_workers, a.cheap_first, a.cases,
         a.part, a.n_parts)
