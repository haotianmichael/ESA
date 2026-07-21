"""Score one noise-sweep point and append a row to noise_sweep.csv.

Reuses the head-to-head scorers (builtin locus criterion, PR sweep). For the
given point it records three rows — SquiggleSeek@all, SquiggleSeek@P>=99.9%,
RawHash2@all — plus a one-line trend summary and the truth-invariance evidence
(distinct coords + first 5) that verification gate 1 asks for.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rawhash_compare import (  # noqa: E402
    _read_paf, builtin_pafstats, pr_sweep, recall_at_precision,
)

HEADER = ["timestamp", "method", "work_point", "amp_noise", "dwell_std",
          "tp", "fp", "fn", "precision", "recall", "f1", "n_reads", "ref_bp", "checkpoint"]


def _pct(x):
    return f"{x*100:.1f}" if isinstance(x, (int, float)) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--ss", required=True, help="squiggleseek.paf")
    ap.add_argument("--rh", default=None, help="rawhash2.paf (optional)")
    ap.add_argument("--amp", required=True)
    ap.add_argument("--dwell", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref_bp", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--target_p", type=float, default=0.999)
    a = ap.parse_args()

    # --- gate 1 evidence: truth coords must be identical across noise points ---
    truth = _read_paf(a.truth)
    coords = sorted(int(v[1]) for v in truth.values())
    n_reads = len(coords)
    print(f"[gate1] amp={a.amp} dwell={a.dwell}: distinct truth coords = "
          f"{len(set(coords))} / {n_reads}; first5 = {coords[:5]}", flush=True)

    # --- SquiggleSeek: @all and the P>=target work point ---
    ss_all = builtin_pafstats(a.truth, a.ss)
    sweep = pr_sweep(a.truth, a.ss, n_points=50)   # finer grid for an accurate P>=99.9 point
    best = recall_at_precision(sweep, a.target_p)
    if best:
        ss_wp = {"tp": best[1], "fp": best[2], "fn": best[3],
                 "precision": best[4], "recall": best[5], "f1": best[6]}
    else:
        ss_wp = {k: None for k in ("tp", "fp", "fn", "precision", "recall", "f1")}

    rows = [("SquiggleSeek", "@all", ss_all),
            ("SquiggleSeek", f"@P>={a.target_p*100:.1f}%", ss_wp)]

    rh = None
    if a.rh and os.path.exists(a.rh):
        rh = builtin_pafstats(a.truth, a.rh)
        rows.append(("RawHash2", "@all", rh))

    # --- append CSV ---
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for method, wp, m in rows:
            w.writerow([ts, method, wp, a.amp, a.dwell, m["tp"], m["fp"], m["fn"],
                        m["precision"], m["recall"], m["f1"], n_reads, a.ref_bp, a.checkpoint])

    # --- trend line ---
    ss_f1 = ss_all["f1"]
    rh_f1 = rh["f1"] if rh else None
    d = f"{(ss_f1 - rh_f1)*100:+.1f}" if rh_f1 is not None else "n/a"
    print(f"[trend] amp={str(a.amp):>7} dwell={str(a.dwell):>2} | "
          f"SS_F1={_pct(ss_f1)} SS_R@P{a.target_p*100:.1f}={_pct(ss_wp['recall'])} | "
          f"RH_F1={_pct(rh_f1)} RH_R={_pct(rh['recall']) if rh else 'n/a'} | dF1={d}", flush=True)


if __name__ == "__main__":
    main()
