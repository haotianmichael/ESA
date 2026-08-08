"""STAGE4b Step 4c — plot F1-vs-noise for SquiggleSeek vs RawHash2 on REAL reads.

Reads evaluate/stage4b_noise.csv (written by stage4b_noise_point.py) and draws the
two F1-vs-k curves. The x-axis is HONESTLY labelled 'additive Gaussian noise on
real reads (x read MAD)' — this is NOT squigulator amp_noise, so this figure is a
separate panel from the STAGE2 (simulated) noise sweep.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--metric", default="f1", choices=["f1", "precision", "recall"])
    args = ap.parse_args()

    series = defaultdict(list)  # method -> [(k, value), ...]
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            series[row["method"]].append((float(row["k_noise"]), float(row[args.metric])))

    plt.figure(figsize=(6.4, 4.4))
    styles = {"SquiggleSeek": dict(marker="o", color="#1f77b4"),
              "RawHash2": dict(marker="s", color="#d62728")}
    for method, pts in series.items():
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        plt.plot(xs, ys, label=method, linewidth=2, **styles.get(method, {}))

    plt.xlabel("additive Gaussian noise on real reads  (k x read MAD)")
    plt.ylabel(f"{args.metric.upper()} (%)")
    plt.title("Real-read noise robustness: SquiggleSeek vs RawHash2")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
