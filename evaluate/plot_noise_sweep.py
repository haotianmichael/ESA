"""Plot a noise-sweep figure from noise_sweep.csv.

Two figures for the paper (choose the axis with --vary):
  E1:  F1 vs amplitude noise at a fixed dwell_std   (--vary amp   --dwell 4)
  E1b: F1 vs dwell_std at a fixed amp_noise          (--vary dwell --amp default)
One line per method (SquiggleSeek@all, RawHash2@all). Run on the server where
matplotlib is installed:

    python evaluate/plot_noise_sweep.py --csv evaluate/noise_sweep.csv \
        --vary amp --dwell 4 --out evaluate/noise_sweep_E1.png
"""
from __future__ import annotations

import argparse
import csv


def load(csv_path, vary, fixed_col, fixed_val):
    # (method, work_point, x) -> f1 ; keep the last row for each key
    f1 = {}
    xs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if str(r[fixed_col]) != str(fixed_val):
                continue
            x = r[vary]
            key = (r["method"], r["work_point"], x)
            f1[key] = float(r["f1"]) if r["f1"] not in ("", "None") else None
            if x not in xs:
                xs.append(x)
    return f1, xs


def order(vals):
    # 'default' first, then numeric ascending
    def k(a):
        return (0, 0.0) if a == "default" else (1, float(a))
    return sorted(set(vals), key=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--vary", choices=["amp", "dwell"], default="amp")
    ap.add_argument("--dwell", default="4", help="fixed dwell_std when --vary amp")
    ap.add_argument("--amp", default="default", help="fixed amp_noise when --vary dwell")
    ap.add_argument("--out", default="evaluate/noise_sweep_E1.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.vary == "amp":
        vary_col, fixed_col, fixed_val, xlabel = "amp_noise", "dwell_std", a.dwell, "amplitude noise"
        fixed_desc = f"dwell_std={a.dwell}"
    else:
        vary_col, fixed_col, fixed_val, xlabel = "dwell_std", "amp_noise", a.amp, "dwell_std"
        fixed_desc = f"amp_noise={a.amp}"

    f1, raw = load(a.csv, vary_col, fixed_col, fixed_val)
    xs = order(raw)
    xi = list(range(len(xs)))
    ss = [f1.get(("SquiggleSeek", "@all", a_)) for a_ in xs]
    rh = [f1.get(("RawHash2", "@all", a_)) for a_ in xs]
    ss = [None if v is None else v * 100 for v in ss]
    rh = [None if v is None else v * 100 for v in rh]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xi, ss, "-o", color="#2563eb", lw=2.2, ms=6, label="SquiggleSeek (learned)")
    ax.plot(xi, rh, "-s", color="#dc2626", lw=2.2, ms=6, label="RawHash2 (hash)")
    ax.set_xticks(xi)
    if a.vary == "amp":
        ax.set_xticklabels([("default" if x == "default" else f"{float(x):g}×") for x in xs])
    else:
        ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(f"squigulator {xlabel}")
    ax.set_ylabel("mapping F1 (%)  @all")
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Noise robustness ({fixed_desc}, 5000 reads)")
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
