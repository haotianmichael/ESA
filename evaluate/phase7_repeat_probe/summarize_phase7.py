#!/usr/bin/env python3
"""
Phase 7 summarizer -- collate the metrics.json from several probe sub-dirs into one
paper-ready comparison table (rows = dataset x graph_label x label).

Reads each --dirs/<subdir>/metrics.json produced by repeat_probe.py and emits
summary.tsv + summary.md (markdown) under --out_dir, and prints to stdout.

Example:
  python evaluate/phase7_repeat_probe/summarize_phase7.py \
      --dirs .../neurosamble_ecoli .../rawsamble_ecoli .../neurosamble_yeast \
      --out_dir .../phase7_repeat_probe
"""
from __future__ import annotations

import argparse
import json
import os
import sys


COLS = ["dataset", "graph", "label", "n_pos", "pos%",
        "disp_ROC", "disp_PR", "deg_ROC", "deg_PR"]


def die(msg, code=2):
    print(f"[error] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def fmt(x):
    return "nan" if (isinstance(x, float) and x != x) else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(description="Summarize Phase-7 probe metrics.json across dirs")
    ap.add_argument("--dirs", nargs="+", required=True, help="probe sub-dirs (each has metrics.json)")
    ap.add_argument("--out_dir", required=True, help="phase7 root to write summary.tsv/.md")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for d in args.dirs:
        mp = os.path.join(d, "metrics.json")
        if not os.path.exists(mp):
            print(f"[warn] no metrics.json in {d}; skipping", file=sys.stderr)
            continue
        with open(mp) as f:
            m = json.load(f)
        dataset = m.get("dataset", "?")
        graph = m.get("graph_label", "?")
        for lname, e in m.get("labels", {}).items():
            rows.append({
                "dataset": dataset, "graph": graph,
                "label": e.get("desc", lname),
                "n_pos": e.get("n_positive", 0),
                "pos%": round(100 * e.get("positive_frac", 0.0), 2),
                "disp_ROC": e.get("dispersion_roc_auc", float("nan")),
                "disp_PR": e.get("dispersion_pr_auc", float("nan")),
                "deg_ROC": e.get("degree_roc_auc", float("nan")),
                "deg_PR": e.get("degree_pr_auc", float("nan")),
            })
    if not rows:
        die("no metrics.json found in any --dirs")

    # deterministic ordering: dataset, graph, label
    rows.sort(key=lambda r: (r["dataset"], r["graph"], r["label"]))

    # TSV
    tsv = os.path.join(args.out_dir, "summary.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r[c]) for c in COLS) + "\n")

    # Markdown
    md = os.path.join(args.out_dir, "summary.md")
    with open(md, "w") as f:
        f.write("# Phase 7 -- repeat/multi-locus probe summary\n\n")
        f.write("Predictor = partner-locus dispersion (embedding-graph feature); "
                "degree = confounded baseline. Labels are independent repeat definitions "
                "from reads->REF minimap2 (with secondaries).\n\n")
        f.write("| " + " | ".join(COLS) + " |\n")
        f.write("|" + "|".join(["---"] * len(COLS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[c]) for c in COLS) + " |\n")
        f.write("\nLabels: `n_loci>=2` = interspersed/multi-locus; `n_aln>=2` = ANY multi-copy "
                "incl. tandem (rDNA array); `mapq<thr` = ambiguous primary placement.\n")

    # stdout
    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in COLS}
    line = "  ".join(c.ljust(widths[c]) for c in COLS)
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in COLS))
    print(f"\n[summary] wrote {tsv} and {md}")


if __name__ == "__main__":
    main()
