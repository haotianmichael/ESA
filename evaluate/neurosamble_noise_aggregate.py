"""
Phase 3 Step 3-4 -- aggregate the event-noise sweep into a CSV + two curves.

For each k in the sweep, the head-to-head wrote an isolated run dir under
``OUTDIR/k{k}/run_*/``. This script reads, per k and per tool (Neurosamble /
Rawsamble):
  * overlap precision / recall / F1  (from pafstats_{tool}.err),
  * N50 and auN                      (recomputed from {tool}.gfa LN:i:, one basis),
  * chained read %                   (TOTAL R from chained_reads.out).

It writes ``OUTDIR/noise_sweep_L{block}.csv`` (metadata line first) and two PNGs
(``noise_curve_recall_L{block}.png`` / ``..._f1_L{block}.png``) with the
intrinsic-device-noise band (``k <= r_real``) shaded and ``r_real`` marked.

Pure parsing/curve math is importable without matplotlib (imported inside plot()).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

TOOLS = ["neurosamble", "rawsamble"]


# --------------------------------------------------------------------------- #
# GFA contiguity (same LN:i: basis as compute_aun.py, so N50 & auN are aligned)
# --------------------------------------------------------------------------- #
def gfa_lengths(path: str) -> List[int]:
    lengths: List[int] = []
    if not path or not os.path.exists(path):
        return lengths
    with open(path) as f:
        for line in f:
            if not line.startswith("S"):
                continue
            m = re.search(r"LN:i:(\d+)", line)
            if m:
                lengths.append(int(m.group(1)))
            else:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[2] and parts[2] != "*":
                    lengths.append(len(parts[2]))
    return lengths


def n50(lengths: List[int]) -> int:
    if not lengths:
        return 0
    s = sorted(lengths, reverse=True)
    total = sum(s)
    acc = 0
    for L in s:
        acc += L
        if acc * 2 >= total:
            return L
    return s[-1]


def aun(lengths: List[int]) -> float:
    total = sum(lengths)
    if total <= 0:
        return 0.0
    return sum(L * L for L in lengths) / total


# --------------------------------------------------------------------------- #
# pafstats + chained% parsing
# --------------------------------------------------------------------------- #
def parse_pafstats(path: str) -> Dict[str, float]:
    """Pull precision/recall/f1 (the lines printed BEFORE the mt:f:0.0 throughput
    ZeroDivision) from a pafstats stderr dump."""
    out = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*Precision:\s*([0-9.]+)", line)
            if m:
                out["precision"] = float(m.group(1))
            m = re.match(r"\s*Recall:\s*([0-9.]+)", line)
            if m:
                out["recall"] = float(m.group(1))
            m = re.match(r"\s*F1 Score:\s*([0-9.]+)", line)
            if m:
                out["f1"] = float(m.group(1))
    return out


def parse_chained(path: str, tool: str) -> float:
    """Return the TOTAL R (chained fraction) for ``tool`` from chained_reads.out.

    The file interleaves '---- <tool> ----' sections each ending in a
    'TOTAL ... <R>' line; we return the last column of that tool's TOTAL line.
    """
    if not path or not os.path.exists(path):
        return float("nan")
    cur = None
    with open(path) as f:
        for line in f:
            m = re.match(r"----\s*(\S+)\s*----", line)
            if m:
                cur = m.group(1)
                continue
            if cur == tool and line.startswith("TOTAL"):
                parts = line.split()
                try:
                    return float(parts[-1])
                except ValueError:
                    return float("nan")
    return float("nan")


def latest_run_dir(kdir: str) -> Optional[str]:
    runs = sorted(glob.glob(os.path.join(kdir, "run_*")))
    return runs[-1] if runs else None


def collect_row(run_dir: str, tool: str) -> Dict[str, float]:
    pstats = parse_pafstats(os.path.join(run_dir, f"pafstats_{tool}.err"))
    lengths = gfa_lengths(os.path.join(run_dir, f"{tool}.gfa"))
    return {
        "precision": pstats["precision"],
        "recall": pstats["recall"],
        "f1": pstats["f1"],
        "n50": n50(lengths),
        "aun": round(aun(lengths), 2),
        "chained_pct": parse_chained(os.path.join(run_dir, "chained_reads.out"), tool),
    }


# --------------------------------------------------------------------------- #
# Plotting (matplotlib imported lazily)
# --------------------------------------------------------------------------- #
def plot(rows, r_real, block, metric, ylabel, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted({r["k"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"neurosamble": "#1f77b4", "rawsamble": "#d62728"}
    labels = {"neurosamble": "Neurosamble", "rawsamble": "Rawsamble"}
    for tool in TOOLS:
        xs, ys = [], []
        for k in ks:
            match = [r for r in rows if r["k"] == k and r["tool"] == tool]
            if match:
                xs.append(k)
                ys.append(match[0][metric])
        ax.plot(xs, ys, "o-", color=colors[tool], label=labels[tool])

    xmax = max(ks) if ks else 1.0
    # Shade the physically-plausible intrinsic-device-noise band.
    if r_real is not None and r_real > 0:
        ax.axvspan(0, min(r_real, xmax), color="0.85", alpha=0.6, zorder=0,
                   label="within intrinsic device noise")
        ax.axvline(r_real, ls="--", color="0.4", lw=1.2)
        ax.text(r_real, ax.get_ylim()[1], f"  r_real={r_real:.3f}",
                va="top", ha="left", fontsize=9, color="0.3")
        if xmax > r_real:
            ax.text(min(r_real + (xmax - r_real) * 0.5, xmax), 0.02,
                    "stress test", ha="center", va="bottom",
                    fontsize=9, color="0.3", transform=ax.get_xaxis_transform())

    ax.set_xlabel("added event-noise level  k  (sigma = k · MAD(raw))")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Event-noise robustness (block L={block})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[plot] wrote {out_png}", flush=True)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Aggregate the Phase 3 event-noise sweep")
    p.add_argument("--outdir", required=True)
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--r_real", type=float, required=True)
    p.add_argument("--k_list", required=True, help="space-separated k values (same as the sweep)")
    return p.parse_args()


def main():
    args = parse_args()
    ks = [k for k in args.k_list.split() if k != ""]

    rows = []
    for k in ks:
        kdir = os.path.join(args.outdir, f"k{k}")
        run_dir = latest_run_dir(kdir)
        if run_dir is None:
            print(f"[agg][WARN] no run dir under {kdir}; skipping k={k}", flush=True)
            continue
        for tool in TOOLS:
            vals = collect_row(run_dir, tool)
            rows.append({
                "k": float(k),
                "tool": tool,
                "within_band": int(float(k) <= args.r_real),
                **vals,
            })

    csv_path = os.path.join(args.outdir, f"noise_sweep_L{args.block}.csv")
    with open(csv_path, "w", newline="") as f:
        f.write(f"# r_real={args.r_real:.6f},noise_mode=event,block={args.block},seed={args.seed}\n")
        w = csv.writer(f)
        w.writerow(["k", "tool", "precision", "recall", "f1", "n50", "aun",
                    "chained_pct", "within_band"])
        for r in sorted(rows, key=lambda x: (x["k"], x["tool"])):
            w.writerow([r["k"], r["tool"], r["precision"], r["recall"], r["f1"],
                        r["n50"], r["aun"], r["chained_pct"], r["within_band"]])
    print(f"[agg] wrote {csv_path}", flush=True)

    # Console echo so the headline numbers land in the log too.
    print("[agg] recall by k (Neurosamble | Rawsamble):", flush=True)
    for k in ks:
        n = [r for r in rows if r["k"] == float(k) and r["tool"] == "neurosamble"]
        rw = [r for r in rows if r["k"] == float(k) and r["tool"] == "rawsamble"]
        nv = n[0]["recall"] if n else float("nan")
        rv = rw[0]["recall"] if rw else float("nan")
        band = "in-band" if float(k) <= args.r_real else "stress"
        print(f"  k={k:>5}: {nv:.4f} | {rv:.4f}   ({band})", flush=True)

    plot(rows, args.r_real, args.block, "recall", "overlap recall",
         os.path.join(args.outdir, f"noise_curve_recall_L{args.block}.png"))
    plot(rows, args.r_real, args.block, "f1", "overlap F1",
         os.path.join(args.outdir, f"noise_curve_f1_L{args.block}.png"))


if __name__ == "__main__":
    main()
