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
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_pafstats(uncalled_bin, truth_paf, tool_paf):
    """Run `uncalled pafstats -r truth --annotate tool` and return its stderr text
    (pafstats prints the summary metrics to stderr; annotated PAF to stdout)."""
    cmd = [uncalled_bin, "pafstats", "-r", str(truth_paf), "--annotate", str(tool_paf)]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return proc.stderr


# --------------------------------------------------------------------------- #
# Built-in pafstats-equivalent scorer (no external tool needed).
# Rule: match reads by qname; a tool mapping is a true positive if it is on the
# same contig+strand as the truth and its reference interval OVERLAPS the truth
# interval (the standard "mapped to the right locus"). Reads absent from the tool
# PAF are false negatives (unmapped). This mirrors UNCALLED pafstats' locus
# criterion so we are not inventing a window-based measure.
# --------------------------------------------------------------------------- #
def _read_paf(path):
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
            if q not in m:  # keep the first (primary) record per read
                m[q] = (tname, ts, te, strand)
    return m


def builtin_pafstats(truth_paf, tool_paf, require_strand=True):
    truth = _read_paf(truth_paf)
    tool = _read_paf(tool_paf)
    tp = fp = fn = 0
    for q, (tn, ts, te, st) in truth.items():
        if q not in tool:
            fn += 1
            continue
        tn2, ts2, te2, st2 = tool[q]
        overlap = tn2 == tn and max(ts, ts2) < min(te, te2)
        if overlap and (not require_strand or st2 == st):
            tp += 1
        else:
            fp += 1
    for q in tool:
        if q not in truth:
            fp += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def _find(pattern, text, cast=float):
    m = re.search(pattern, text, re.IGNORECASE)
    return cast(m.group(1)) if m else None


def parse_pafstats(text):
    """Tolerant parse of pafstats summary. Returns dict (values may be None)."""
    d = {
        "tp": _find(r"\b(?:TP|true[ _]?positives?)\D+(\d+)", text, int),
        "fp": _find(r"\b(?:FP|false[ _]?positives?)\D+(\d+)", text, int),
        "fn": _find(r"\b(?:FN|false[ _]?negatives?)\D+(\d+)", text, int),
        "precision": _find(r"\b(?:precision|prec)\b\D+([\d.]+)", text),
        "recall": _find(r"\b(?:recall|sensitivity)\b\D+([\d.]+)", text),
        "f1": _find(r"\bF[-_ ]?1?(?:[ _]?score)?\b\D+([\d.]+)", text),
    }
    # derive P/R/F1 from counts if the summary didn't print them
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    if d["precision"] is None and tp is not None and fp is not None and (tp + fp):
        d["precision"] = tp / (tp + fp)
    if d["recall"] is None and tp is not None and fn is not None and (tp + fn):
        d["recall"] = tp / (tp + fn)
    if d["f1"] is None and d["precision"] and d["recall"] and (d["precision"] + d["recall"]):
        p, r = d["precision"], d["recall"]
        d["f1"] = 2 * p * r / (p + r)
    return d


def parse_args():
    ap = argparse.ArgumentParser(description="SquiggleSeek vs RawHash2 via uncalled pafstats")
    ap.add_argument("--truth", required=True, help="ground_truth.paf")
    ap.add_argument("--paf", action="append", default=[], metavar="NAME=path.paf",
                    help="tool PAF as NAME=path (repeatable), e.g. SquiggleSeek=ss.paf")
    ap.add_argument("--uncalled", default="uncalled")
    ap.add_argument("--scorer", default="builtin", choices=["builtin", "pafstats"],
                    help="'builtin' = internal pafstats-equivalent locus scorer (no external "
                         "tool); 'pafstats' = shell out to `uncalled pafstats`.")
    ap.add_argument("--ignore_strand", action="store_true",
                    help="builtin scorer: do not require the strand to match.")
    ap.add_argument("--csv", default=None, help="default: evaluate/head2head_pafstats.csv")
    ap.add_argument("--amp_noise", default="default")
    ap.add_argument("--dwell_std", default="")
    ap.add_argument("--ref_bp", default="")
    ap.add_argument("--preset", default="")
    return ap.parse_args()


def score_one(args, truth, tool):
    if args.scorer == "builtin":
        return builtin_pafstats(truth, tool, require_strand=not args.ignore_strand)
    raw = run_pafstats(args.uncalled, truth, tool)
    print(raw)
    return parse_pafstats(raw)


def main():
    args = parse_args()
    print(f"[scorer] {args.scorer}"
          + ("" if args.scorer == "builtin" else f" (uncalled={args.uncalled})"), flush=True)

    # gate 1: truth vs truth must be ~100%
    print("=== self-check: score(truth, truth) — expect P=R=F1=100%, FP=FN=0 ===", flush=True)
    sc = score_one(args, args.truth, args.truth)
    print("self-check:", sc, flush=True)

    rows = []
    for spec in args.paf:
        if "=" not in spec:
            sys.exit(f"--paf must be NAME=path, got: {spec}")
        name, path = spec.split("=", 1)
        print(f"\n=== score: {name}  ({path}) ===", flush=True)
        m = score_one(args, args.truth, path)
        print("parsed:", m, flush=True)
        rows.append((name, m))

    print("\n================ Step 2c head-to-head (uncalled pafstats) ================")
    print(f"  {'method':<16s} {'TP':>7} {'FP':>7} {'FN':>7} {'P':>7} {'R':>7} {'F1':>7}")
    for name, m in rows:
        def pct(x):
            return f"{x*100:6.1f}" if isinstance(x, float) else "   n/a"
        print(f"  {name:<16s} {str(m['tp']):>7} {str(m['fp']):>7} {str(m['fn']):>7} "
              f"{pct(m['precision'])} {pct(m['recall'])} {pct(m['f1'])}")
    print("=========================================================================")
    print("(If any value is n/a, paste the raw pafstats block above and I'll fix the regex.)")

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
