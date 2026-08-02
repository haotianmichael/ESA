"""
Step 2c evaluator — score SquiggleSeek vs RawHash2 with `uncalled pafstats`.

Runs pafstats (the standard RawHash-family metric) against the squigulator-truth
PAF for each tool, parses TP/FP/FN/precision/recall/F1, prints a side-by-side
table and appends to head2head_pafstats.csv. Also self-checks the truth PAF
against itself (must be ~100%).

This script only orchestrates external tools you install (uncalled). It does not
compute any custom criterion — pafstats is the judge.

Example:
    python evaluate/rawhash_compare.py \
        --truth pafout/ground_truth.paf \
        --paf SquiggleSeek=pafout/squiggleseek.paf --paf RawHash2=pafout/rawhash2.paf \
        --amp_noise default --dwell_std 4 --ref_bp 1000000 --preset sensitive
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_pafstats(uncalled_bin, truth_paf, tool_paf):
    """Return UNCALLED pafstats' confusion-matrix text for ``tool_paf`` vs truth.

    Two paths, both produce the identical summary format:
      * ``uncalled_bin`` ending in ``.py`` -> the pure-python ``uncalled/pafstats.py``
        run IN-PROCESS. UNCALLED bundles HDF5 and often fails to ``pip install``
        (its HDF5 install-examples target breaks), but ``pafstats.py`` imports only
        ``sys/numpy/re/argparse`` — no C-extension — so we run just that one file:
            wget https://raw.githubusercontent.com/skovaka/UNCALLED/master/uncalled/pafstats.py
            python evaluate/rawhash_compare.py ... --scorer pafstats --uncalled pafstats.py
      * otherwise the installed ``uncalled`` binary via ``uncalled pafstats``.
    """
    padded = _pad_unmapped(truth_paf, tool_paf)
    if str(uncalled_bin).endswith(".py"):
        return _run_pafstats_py(uncalled_bin, truth_paf, padded)
    cmd = [uncalled_bin, "pafstats", "-r", str(truth_paf), "--annotate", str(padded)]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return proc.stderr


def _pad_unmapped(truth_paf, tool_paf):
    """pafstats keys on the *infile* read set, so reads a tool never emitted a
    line for are invisible to it (not counted FN) — different tools drop reads
    differently, which would bias recall. Fix the denominator at the truth set:
    write a temp PAF = tool records + an unmapped ``*`` record for every truth
    read the tool is missing. Then pafstats' FN column matches the truth set,
    the same denominator the builtin scorer uses. Returns the temp path."""
    import tempfile

    truth = _read_paf(truth_paf)
    have = set()
    lines = []
    with open(tool_paf) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c and c[0]:
                have.add(c[0])
            lines.append(line.rstrip("\n"))
    for q, (_tn, ts, te, _st, _s) in truth.items():
        if q not in have:
            qlen = max(1, te - ts)
            lines.append("\t".join(str(x) for x in
                         (q, qlen, 0, 0, "*", "*", 0, 0, 0, 0, 0, 0)))
    fd, path = tempfile.mkstemp(suffix=".paf", prefix="padded_")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _run_pafstats_py(pafstats_py, truth_paf, tool_paf):
    """Import the vendored/downloaded pure-python pafstats.py and call run() with
    a minimal args namespace, capturing the summary it writes to stdout."""
    import importlib.util
    import io
    from contextlib import redirect_stdout
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("uncalled_pafstats", str(pafstats_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns = SimpleNamespace(infile=str(tool_paf), ref_paf=str(truth_paf),
                         max_reads=None, annotate=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.run(ns)  # annotate=False -> summary goes to stdout
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Built-in pafstats-equivalent scorer (no external tool needed).
# Rule: match reads by qname; a tool mapping is a true positive if it is on the
# same contig+strand as the truth and its reference interval OVERLAPS the truth
# interval (the standard "mapped to the right locus"). Reads absent from the tool
# PAF are false negatives (unmapped). This mirrors UNCALLED pafstats' locus
# criterion so we are not inventing a window-based measure.
# --------------------------------------------------------------------------- #
def _read_paf(path):
    """qname -> (tname, tstart, tend, strand, score). score comes from a
    ``cs:f:`` tag if present (SquiggleSeek), else +inf (always mapped)."""
    m = {}
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            q, strand, tname = c[0], c[4], c[5]
            if tname == "*" or strand == "*":
                continue  # PAF unmapped record -> treat read as unmapped (FN)
            try:
                ts, te = int(c[7]), int(c[8])
            except ValueError:
                continue
            score = float("inf")
            for tag in c[12:]:
                if tag.startswith("cs:f:"):
                    score = float(tag[5:])
                    break
            if q not in m:  # keep the first (primary) record per read
                m[q] = (tname, ts, te, strand, score)
    return m


def _score_truth_tool(truth, tool, require_strand=True, score_threshold=None):
    """Score parsed truth/tool dicts. score_threshold: tool reads below it are
    treated as unmapped (work-point matching)."""
    tp = fp = fn = 0
    for q, (tn, ts, te, st, _s) in truth.items():
        if q not in tool:
            fn += 1
            continue
        tn2, ts2, te2, st2, sc2 = tool[q]
        if score_threshold is not None and sc2 < score_threshold:
            fn += 1  # withheld -> unmapped
            continue
        overlap = tn2 == tn and max(ts, ts2) < min(te, te2)
        if overlap and (not require_strand or st2 == st):
            tp += 1
        else:
            fp += 1
    extra = sum(1 for q in tool if q not in truth)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "extra": extra,
            "precision": p, "recall": r, "f1": f}


def builtin_pafstats(truth_paf, tool_paf, require_strand=True):
    # Reads the tool mapped that are NOT in the truth set are OUTSIDE the
    # evaluation universe (ignored as 'extra', not FP). The truth PAF defines it.
    return _score_truth_tool(_read_paf(truth_paf), _read_paf(tool_paf), require_strand)


def _paf_all_qnames(path):
    """Every query name (column 0) in a PAF, mapped or unmapped ('*' rows)."""
    q = set()
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c and c[0]:
                q.add(c[0])
    return q


def qname_overlap(truth_paf, tool_paf):
    """Fraction of TRUTH read-ids present in ``tool_paf`` (and raw counts).

    pafstats matches reads by qname, so if the tool and truth id formats differ
    the intersection is empty and recall silently reads 0. Report the ratio so an
    id mismatch is caught before the P/R/F1 table is trusted (PROMPT §六)."""
    tq = _paf_all_qnames(truth_paf)
    pq = _paf_all_qnames(tool_paf)
    inter = tq & pq
    frac = (len(inter) / len(tq)) if tq else 0.0
    return {"truth": len(tq), "tool": len(pq), "matched": len(inter), "frac": frac}


def pr_sweep(truth_paf, tool_paf, require_strand=True, n_points=15):
    """Sweep the confidence score threshold on a scored tool PAF -> PR curve.
    Returns list of (threshold, tp, fp, fn, precision, recall, f1)."""
    truth = _read_paf(truth_paf)
    tool = _read_paf(tool_paf)
    scores = sorted(s for (*_, s) in tool.values() if s != float("inf"))
    if not scores:
        m = _score_truth_tool(truth, tool, require_strand)
        return [(float("-inf"), m["tp"], m["fp"], m["fn"], m["precision"], m["recall"], m["f1"])]
    import numpy as np
    thresholds = [float("-inf")] + [float(np.quantile(scores, q)) for q in np.linspace(0, 1, n_points)]
    rows = []
    for t in sorted(set(thresholds)):
        m = _score_truth_tool(truth, tool, require_strand, score_threshold=t)
        rows.append((t, m["tp"], m["fp"], m["fn"], m["precision"], m["recall"], m["f1"]))
    return rows


def recall_at_precision(sweep_rows, target_p):
    """Among sweep rows meeting precision >= target_p, return the one with the
    highest recall (i.e. the loosest threshold that still hits the precision)."""
    ok = [r for r in sweep_rows if r[4] >= target_p]
    if not ok:
        return None
    return max(ok, key=lambda r: r[5])


def _find(pattern, text, cast=float):
    m = re.search(pattern, text, re.IGNORECASE)
    return cast(m.group(1)) if m else None


def _prf_from_counts(tp, fp, fn):
    d = {"tp": tp, "fp": fp, "fn": fn, "extra": None,
         "precision": None, "recall": None, "f1": None}
    if None in (tp, fp, fn):
        return d
    d["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    d["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    p, r = d["precision"], d["recall"]
    d["f1"] = 2 * p * r / (p + r) if (p + r) else 0.0
    return d


def parse_pafstats(text):
    """Parse UNCALLED pafstats' 2x2 confusion matrix (percentages of total reads):
        Summary: N reads, ...
                P      N
           T  <TP%>  <TN%>
           F  <FP%>  <FN%>
    """
    total = _find(r"Summary:\s*(\d+)\s*reads", text, int)

    def row(letter):
        m = re.search(rf"^\s*{letter}\s+([\d.]+)\s+([\d.]+)", text, re.MULTILINE)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    tp_pct, _tn_pct = row("T")
    fp_pct, fn_pct = row("F")
    if total is None or None in (tp_pct, fp_pct, fn_pct):
        return {"tp": None, "fp": None, "fn": None, "extra": None,
                "precision": None, "recall": None, "f1": None}
    return _prf_from_counts(round(tp_pct / 100 * total),
                            round(fp_pct / 100 * total),
                            round(fn_pct / 100 * total))


def run_mapeval(k8_bin, paftools_js, tool_paf, extra=""):
    """minimap2 paftools.js mapeval — truth is encoded in the squigulator read
    names, so no separate truth PAF is needed. Returns stdout text."""
    cmd = f"{k8_bin} {paftools_js} mapeval {extra} {tool_paf}"
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return proc.stdout


def parse_mapeval(text):
    """paftools mapeval prints cumulative rows; the final row's columns give
    total mapped, wrong (error), and the error fraction. Tolerant: last data row."""
    rows = []
    for line in text.splitlines():
        c = line.split()
        if len(c) >= 4 and c[0] in ("Q", "U") and c[1].replace(".", "").isdigit():
            rows.append(c)
    # This format varies by version; echo raw and let the user confirm.
    return {"tp": None, "fp": None, "fn": None, "extra": None,
            "precision": None, "recall": None, "f1": None, "raw_rows": len(rows)}


def parse_args():
    ap = argparse.ArgumentParser(description="SquiggleSeek vs RawHash2 head-to-head")
    ap.add_argument("--truth", required=True, help="ground_truth.paf")
    ap.add_argument("--paf", action="append", default=[], metavar="NAME=path.paf",
                    help="tool PAF as NAME=path (repeatable), e.g. SquiggleSeek=ss.paf")
    ap.add_argument("--uncalled", default="uncalled",
                    help="the `uncalled` binary, OR a path to the pure-python "
                         "uncalled/pafstats.py (ending .py) run in-process — no "
                         "HDF5 build needed. Download: wget https://raw."
                         "githubusercontent.com/skovaka/UNCALLED/master/uncalled/pafstats.py")
    ap.add_argument("--scorer", default="builtin", choices=["builtin", "pafstats", "mapeval"],
                    help="'builtin' = internal locus scorer; 'pafstats' = uncalled pafstats; "
                         "'mapeval' = paftools.js mapeval.")
    ap.add_argument("--k8", default="k8")
    ap.add_argument("--paftools", default="paftools.js")
    ap.add_argument("--ignore_strand", action="store_true",
                    help="builtin scorer: do not require the strand to match.")
    # work-point matching (Step 1)
    ap.add_argument("--sweep", default=None, metavar="NAME",
                    help="Tool name (must be a --paf whose PAF has cs:f scores) to PR-sweep.")
    ap.add_argument("--match_to", default=None, metavar="NAME",
                    help="Report --sweep tool's recall at the precision of this tool.")
    ap.add_argument("--csv", default=None, help="default: evaluate/head2head_pafstats.csv")
    ap.add_argument("--amp_noise", default="default")
    ap.add_argument("--dwell_std", default="")
    ap.add_argument("--ref_bp", default="")
    ap.add_argument("--preset", default="")
    return ap.parse_args()


def score_one(args, truth, tool):
    if args.scorer == "builtin":
        return builtin_pafstats(truth, tool, require_strand=not args.ignore_strand)
    if args.scorer == "pafstats":
        raw = run_pafstats(args.uncalled, truth, tool)
        print(raw)
        return parse_pafstats(raw)
    raw = run_mapeval(args.k8, args.paftools, tool)  # mapeval (truth in read names)
    print(raw)
    return parse_mapeval(raw)


def main():
    args = parse_args()
    print(f"[scorer] {args.scorer}", flush=True)

    # gate 1: truth vs truth must be ~100% (builtin only; standard tools use read-name truth)
    print("=== self-check: builtin(truth, truth) — expect P=R=F1=100%, FP=FN=0 ===", flush=True)
    print("self-check:", builtin_pafstats(args.truth, args.truth,
                                          require_strand=not args.ignore_strand), flush=True)

    pafs = {}
    rows = []
    for spec in args.paf:
        if "=" not in spec:
            sys.exit(f"--paf must be NAME=path, got: {spec}")
        name, path = spec.split("=", 1)
        pafs[name] = path
        print(f"\n=== score: {name}  ({path}) ===", flush=True)
        ov = qname_overlap(args.truth, path)
        print(f"[qname-check] {name}: {ov['matched']}/{ov['truth']} truth read-ids "
              f"matched ({ov['frac'] * 100:.1f}%); tool PAF has {ov['tool']} reads",
              flush=True)
        if ov["truth"] and ov["frac"] < 0.5:
            print(f"[qname-check][WARN] {name} matches <50% of truth read-ids — tool "
                  f"and truth may be different read sets or use different id formats; "
                  f"recall is under-counted until fixed.", flush=True)
        m = score_one(args, args.truth, path)
        print("parsed:", m, flush=True)
        rows.append((name, m))

    print("\n================ Step 2c head-to-head (locus criterion) ================")
    print(f"  {'method':<16s} {'TP':>7} {'FP':>7} {'FN':>7} {'extra':>7} {'P':>7} {'R':>7} {'F1':>7}")
    for name, m in rows:
        def pct(x):
            return f"{x*100:6.1f}" if isinstance(x, float) else "   n/a"
        print(f"  {name:<16s} {str(m['tp']):>7} {str(m['fp']):>7} {str(m['fn']):>7} "
              f"{str(m.get('extra', '-')):>7} {pct(m['precision'])} {pct(m['recall'])} {pct(m['f1'])}")
    print("=======================================================================")
    print("(extra = tool mappings outside the truth set; ignored, not counted as FP.)")

    # --- Step 1: work-point matching (PR sweep on the SquiggleSeek PAF) ---
    if args.sweep and args.sweep in pafs:
        req = not args.ignore_strand
        sweep_rows = pr_sweep(args.truth, pafs[args.sweep], require_strand=req)
        print(f"\n======== {args.sweep} PR sweep (score=cosine) ========")
        print(f"  {'threshold':>10} {'TP':>7} {'FP':>7} {'FN':>7} {'P':>7} {'R':>7} {'F1':>7}")
        for t, tp, fp, fn, p, r, f1 in sweep_rows:
            ts = "-inf" if t == float("-inf") else f"{t:.4f}"
            print(f"  {ts:>10} {tp:>7} {fp:>7} {fn:>7} {p*100:6.1f} {r*100:6.1f} {f1*100:6.1f}")

        all_row = sweep_rows[0]  # threshold -inf = map all
        wp_rows = [(f"{args.sweep}@all", all_row[4], all_row[5], all_row[6])]
        target = dict(rows).get(args.match_to) if args.match_to else None
        if target and target.get("precision") is not None:
            tp_p = target["precision"]
            best = recall_at_precision(sweep_rows, tp_p)
            if best:
                wp_rows.append((f"{args.sweep}@P>={tp_p*100:.1f}%",
                                best[4], best[5], best[6]))
            else:
                wp_rows.append((f"{args.sweep}@P>={tp_p*100:.1f}%", None, None, None))
            wp_rows.append((args.match_to, target["precision"], target["recall"], target["f1"]))

        print(f"\n======== work-point matched table ========")
        print(f"  {'setting':<26s} {'P':>7} {'R':>7} {'F1':>7}")
        for name, p, r, f1 in wp_rows:
            cells = "   n/a    n/a    n/a" if p is None else f"{p*100:6.1f} {r*100:6.1f} {f1*100:6.1f}"
            print(f"  {name:<26s} {cells}")
        print("==========================================")

    csv_path = Path(args.csv) if args.csv else Path(__file__).resolve().parent / "head2head_pafstats.csv"
    header = ["timestamp", "method", "tp", "fp", "fn", "precision", "recall", "f1",
              "n_reads", "amp_noise", "dwell_std", "ref_bp", "preset"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for name, m in rows:
            n_reads = (m["tp"] or 0) + (m["fn"] or 0) if m["tp"] is not None else ""
            w.writerow([ts, name, m["tp"], m["fp"], m["fn"], m["precision"], m["recall"],
                        m["f1"], n_reads, args.amp_noise, args.dwell_std, args.ref_bp, args.preset])
    print(f"[info] appended {len(rows)} rows -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
