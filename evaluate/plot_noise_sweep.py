"""Plot the noise-sweep figure (paper figure E1) from noise_sweep.csv.

F1 vs amplitude noise, one line per method (SquiggleSeek@all, RawHash2@all), for a
chosen dwell_std row. Run on the server where matplotlib is installed:

    python evaluate/plot_noise_sweep.py --csv evaluate/noise_sweep.csv --dwell 4 \
        --out evaluate/noise_sweep_E1.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def load(csv_path, dwell):
    # (method, work_point, amp) -> f1 ; keep the last timestamp for each key
    f1 = {}
    rec = {}
    amps = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if str(r["dwell_std"]) != str(dwell):
                continue
            key = (r["method"], r["work_point"], r["amp_noise"])
            f1[key] = float(r["f1"]) if r["f1"] not in ("", "None") else None
            rec.setdefault(r["method"], {})
            if r["amp_noise"] not in amps:
                amps.append(r["amp_noise"])
    return f1, amps


def amp_order(amps):
    # 'default' first, then numeric ascending
    def k(a):
        return (0, 0.0) if a == "default" else (1, float(a))
    return sorted(set(amps), key=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dwell", default="4")
    ap.add_argument("--out", default="evaluate/noise_sweep_E1.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1, amps = load(a.csv, a.dwell)
    xs = amp_order(amps)
    xi = list(range(len(xs)))
    ss = [f1.get(("SquiggleSeek", "@all", a_)) for a_ in xs]
    rh = [f1.get(("RawHash2", "@all", a_)) for a_ in xs]
    ss = [None if v is None else v * 100 for v in ss]
    rh = [None if v is None else v * 100 for v in rh]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xi, ss, "-o", color="#2563eb", lw=2.2, ms=6, label="SquiggleSeek (learned)")
    ax.plot(xi, rh, "-s", color="#dc2626", lw=2.2, ms=6, label="RawHash2 (hash)")
    ax.set_xticks(xi)
    ax.set_xticklabels([("default" if x == "default" else f"{float(x):g}×") for x in xs])
    ax.set_xlabel("squigulator amplitude noise")
    ax.set_ylabel("mapping F1 (%)  @all")
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Noise robustness (dwell_std={a.dwell}, 1 Mb ref, 5000 reads)")
    ax.legend(frameon=False)
    for x, y in zip(xi, ss):
        if y is not None:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color="#2563eb")
    for x, y in zip(xi, rh):
        if y is not None:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=8, color="#dc2626")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
