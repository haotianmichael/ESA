"""
Phase 3 Step 0 -- intrinsic event-level noise calibration (measure only).

The event-mode sweep adds block-constant noise of magnitude ``sigma = k * MAD(raw)``
on top of REAL R9.4 reads, which already carry the full physical sequencer noise.
To keep the headline figure defensible we must anchor -- from the reads themselves --
what counts as "within the intrinsic device-noise range", instead of guessing.

For each sampled read we estimate its intrinsic within-event residual noise with a
block-mean baseline (block == the sweep's NOISE_BLOCK, the event-mode dwell proxy):

    baseline_i = mean of the length-`block` block containing sample i
    residual   = raw - baseline
    ratio      = MAD(residual) / MAD(raw)

``r_real = median over reads of ratio`` -- the fraction of each read's MAD that is
its own intrinsic event-level noise. The physically-plausible band is then
``k <= r_real`` (added noise <= intrinsic noise, i.e. total variance at most
doubles); ``k > r_real`` is the amplification / stress region.

Changes NO data. Deterministic given ``--seed`` (reservoir sample of reads).
The residual-ratio math (``event_residual_ratio``) is pure numpy so it is unit
testable without pyslow5.
"""
from __future__ import annotations

import argparse
import random

import numpy as np


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def event_residual_ratio(raw, block: int) -> float:
    """MAD(within-block residual) / MAD(raw) for one read (block-mean baseline).

    Returns ``nan`` when ``MAD(raw) == 0`` (flat/degenerate read) so the caller
    can skip it.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n = int(raw.shape[0])
    L = max(1, int(block))
    if n == 0:
        return float("nan")
    mad_raw = _mad(raw)
    if mad_raw <= 0:
        return float("nan")
    n_blocks = (n + L - 1) // L
    pad = n_blocks * L - n
    if pad:
        padded = np.concatenate([raw, np.full(pad, raw[-1], dtype=np.float64)])
    else:
        padded = raw
    blocks = padded.reshape(n_blocks, L)
    baseline = np.repeat(blocks.mean(axis=1), L)[:n]
    residual = raw - baseline
    return _mad(residual) / mad_raw


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 intrinsic event-noise calibration")
    p.add_argument("--blow5", required=True, help="clean subset blow5 (k=0)")
    p.add_argument("--block", type=int, default=9, help="event-mode block length (dwell proxy)")
    p.add_argument("--n", type=int, default=200, help="number of reads to sample")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True, help="write r_real (one float) here")
    return p.parse_args()


def main():
    args = parse_args()
    import pyslow5

    rng = random.Random(args.seed)
    # Reservoir sample of raw signals so we do not hold the whole file in RAM.
    reservoir = []
    s = pyslow5.Open(args.blow5, "r")
    seen = 0
    for rec in s.seq_reads(pA=False):   # RAW int16, matches the noise model
        seen += 1
        sig = np.asarray(rec["signal"], dtype=np.float64)
        if len(reservoir) < args.n:
            reservoir.append(sig)
        else:
            j = rng.randint(0, seen - 1)
            if j < args.n:
                reservoir[j] = sig
    s.close()

    ratios = []
    for sig in reservoir:
        r = event_residual_ratio(sig, args.block)
        if np.isfinite(r):
            ratios.append(r)

    if not ratios:
        raise SystemExit("[calib] no usable reads (all flat?) -- cannot estimate r_real")

    r_real = float(np.median(ratios))
    with open(args.out, "w") as f:
        f.write(f"{r_real:.6f}\n")
    print(f"[calib] sampled_reads={len(ratios)} (of {seen} in file)  block={args.block}  "
          f"seed={args.seed}", flush=True)
    print(f"[calib] r_real={r_real:.6f}  -> physically-plausible band: k <= {r_real:.4f} "
          f"(k above that = stress/amplification)", flush=True)
    print(f"{r_real:.6f}")   # last line = machine-readable value


if __name__ == "__main__":
    main()
