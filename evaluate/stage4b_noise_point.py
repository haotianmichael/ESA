"""STAGE4b Step 4b — record one noise point.

Parses a run's SUMMARY_real.txt (the 'Step 2c head-to-head' table written by
run_real_data.sh / rawhash_compare) and appends one row per tool
(method, k, tp, fp, fn, precision, recall, f1) to evaluate/stage4b_noise.csv.

Parsing the SUMMARY (rather than re-scoring) guarantees the CSV matches the
head-to-head numbers exactly. Only the head-to-head rows are matched (they have
7 numeric columns: TP FP FN extra P R F1), so the PR-sweep / work-point rows are
ignored.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import datetime

_ROW = re.compile(
    r"^\s+(SquiggleSeek|RawHash2)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_summary(path):
    rows = {}
    with open(path) as f:
        for line in f:
            m = _ROW.match(line.rstrip("\n"))
            if m:
                name = m.group(1)
                rows[name] = {
                    "tp": int(m.group(2)), "fp": int(m.group(3)), "fn": int(m.group(4)),
                    "precision": float(m.group(6)), "recall": float(m.group(7)),
                    "f1": float(m.group(8)),
                }
    return rows


def main():
    ap = argparse.ArgumentParser(description="append one STAGE4b noise point to the CSV")
    ap.add_argument("--summary", required=True, help="k{K}/SUMMARY_real.txt")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--csv", required=True, help="evaluate/stage4b_noise.csv")
    args = ap.parse_args()

    rows = parse_summary(args.summary)
    if "SquiggleSeek" not in rows or "RawHash2" not in rows:
        raise SystemExit(f"[noise-point][FATAL] could not parse head-to-head rows from {args.summary} "
                         f"(found: {sorted(rows)})")

    header = ["timestamp", "method", "k_noise", "tp", "fp", "fn", "precision", "recall", "f1"]
    write_header = not os.path.exists(args.csv)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for method in ("SquiggleSeek", "RawHash2"):
            m = rows[method]
            w.writerow([ts, method, args.k, m["tp"], m["fp"], m["fn"],
                        f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"])
    print(f"[noise-point] k={args.k}  SS F1={rows['SquiggleSeek']['f1']}  "
          f"RawHash2 F1={rows['RawHash2']['f1']}  -> {args.csv}", flush=True)


if __name__ == "__main__":
    main()
