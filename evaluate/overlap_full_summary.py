"""
Neurosamble Phase 4 -- write phase4_summary.csv from a completed full-scale run.

Reuses the Phase-3 parsers (pafstats / GFA contiguity / chained%) and folds in the
throughput-gate numbers: encode / index / query wall-clock, peak RSS, peak GPU mem.
Neurosamble-only timings are attached to the neurosamble row.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from neurosamble_noise_aggregate import (  # noqa: E402
    aun,
    gfa_lengths,
    n50,
    parse_chained,
    parse_pafstats,
)

TOOLS = ["neurosamble", "rawsamble", "mm2"]


def _load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 summary CSV writer")
    p.add_argument("--run_dir", required=True, help="OUTDIR of the full-scale run")
    p.add_argument("--encode_sec", type=float, default=float("nan"))
    p.add_argument("--index_sec", type=float, default=float("nan"))
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run = args.run_dir
    qstats = _load_json(os.path.join(run, "index", "query_stats.json"))
    manifest = _load_json(os.path.join(run, "encode", "encode_manifest.json"))
    istats = _load_json(os.path.join(run, "index", "index_stats.json"))

    query_sec = qstats.get("query_sec", "")
    peak_rss = qstats.get("peak_rss_gb", "")
    peak_gpu = manifest.get("peak_gpu_mem_gb", "")

    rows = []
    for tool in TOOLS:
        pstats = parse_pafstats(os.path.join(run, f"pafstats_{tool}.err")) if tool != "mm2" else \
            {"precision": "", "recall": "", "f1": ""}
        lengths = gfa_lengths(os.path.join(run, f"{tool}.gfa"))
        row = {
            "tool": tool,
            "precision": pstats["precision"],
            "recall": pstats["recall"],
            "f1": pstats["f1"],
            "n50": n50(lengths),
            "aun": round(aun(lengths), 2),
            "longest_unitig": max(lengths) if lengths else 0,
            "unitig_count": len(lengths),
            "chained_pct": parse_chained(os.path.join(run, "chained_reads.out"), tool),
            "encode_sec": "",
            "index_sec": "",
            "query_sec": "",
            "peak_rss_gb": "",
            "peak_gpu_mem_gb": "",
        }
        if tool == "neurosamble":
            row["encode_sec"] = "" if args.encode_sec != args.encode_sec else round(args.encode_sec, 1)
            row["index_sec"] = "" if args.index_sec != args.index_sec else round(args.index_sec, 1)
            row["query_sec"] = query_sec
            row["peak_rss_gb"] = peak_rss
            row["peak_gpu_mem_gb"] = peak_gpu
        rows.append(row)

    out_csv = args.out_csv or os.path.join(run, "phase4_summary.csv")
    fields = ["tool", "precision", "recall", "f1", "n50", "aun", "longest_unitig",
              "unitig_count", "chained_pct", "encode_sec", "index_sec", "query_sec",
              "peak_rss_gb", "peak_gpu_mem_gb"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] wrote {out_csv}", flush=True)

    # Provenance / throughput-gate echo.
    print(f"[summary] n_reads={qstats.get('n_reads','?')} "
          f"n_windows={qstats.get('n_windows','?')} "
          f"nlist={istats.get('nlist','?')} nprobe={qstats.get('nprobe','?')} "
          f"encode_sec={args.encode_sec} index_sec={args.index_sec} "
          f"query_sec={query_sec} peak_rss_gb={peak_rss} peak_gpu_mem_gb={peak_gpu}",
          flush=True)
    for tool in TOOLS:
        r = next(x for x in rows if x["tool"] == tool)
        print(f"[summary] {tool:>11}: P={r['precision']} R={r['recall']} F1={r['f1']} "
              f"N50={r['n50']} auN={r['aun']} longest={r['longest_unitig']} "
              f"chained%={r['chained_pct']}", flush=True)


if __name__ == "__main__":
    main()
