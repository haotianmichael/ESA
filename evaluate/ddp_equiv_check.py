"""Standalone proof that the 2-GPU all-gathered contrastive loss equals the
single-GPU batch-``BATCH`` loss (the critical correctness claim of the DDP work).

WHY: ``train_encoder_hardneg`` builds InfoNCE from the GLOBAL batch's in-batch
negatives. Under DDP each GPU forwards ``BATCH // world_size`` samples and
all-gathers the embeddings into the SAME ``[BATCH, D]`` tensors the single-GPU
path builds, so the loss (and its negative pool) is identical. This script proves
it on a FIXED seed + FIXED tiny dataset by exercising the REAL functions used in
training (``_forward_local_embeddings`` + ``gathered_infonce_loss`` from
pilot_recall).

The check is about the gather+loss MATH, which is independent of
``input_signal_len``; a small isl is used so the check runs fast and never OOMs
(the isl=3000 memory behavior is proven separately by the plumbing smoke + real
STAGE 1, not here).

Run it (on the run server, conda env py310, 2 GPUs)::

    # full 2-GPU proof (this is the one that matters):
    torchrun --nproc_per_node=2 evaluate/ddp_equiv_check.py

    # single-process sanity (prints the batch-BATCH loss only, no comparison):
    python evaluate/ddp_equiv_check.py

WHAT "CORRECT" LOOKS LIKE (2-GPU run, printed on rank 0):
    * "loss(single-GPU batch=48) = X.XXXXXX"
    * "loss(2-GPU gathered,   48) = X.XXXXXX"
    * "|Δloss| = <something < 1e-4>"
    * "max|gathered_y1 - full_y1| = <something < 1e-4>"  (and same for y2)
    * final line: "[ddp-equiv] PASS"
A nonzero exit code / "FAIL" means the DDP loss does NOT match single-GPU and the
gathered embeddings are not the concatenation of the ranks' local embeddings —
do not trust any DDP training until this passes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make src/ + this dir importable exactly like pilot_recall does.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for p in (str(_HERE), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from dna2vec.config_schema import SignalModelConfigSchema  # noqa: E402
from dna2vec.signal_encoder import signal_encoder_from_config  # noqa: E402

from pilot_recall import (  # noqa: E402
    _setup_distributed,
    _forward_local_embeddings,
    gathered_infonce_loss,
)


def _make_fixed_batch(batch, H, L, T, seed=1234):
    """Deterministic (seed-fixed) x1/x2/xneg dicts for ``batch`` samples.

    Returns full-batch dicts (all ``batch`` rows). Slicing to a rank's local
    range happens in ``_slice``. Masks are all-valid at T resolution; the gather
    + loss math (what we test) does not depend on the signal content, only on its
    being identical between the single-GPU and per-rank views."""
    rng = np.random.default_rng(seed)
    x1 = {
        "signal": torch.from_numpy(rng.standard_normal((batch, L)).astype(np.float32)),
        "attention_mask": torch.ones(batch, T, dtype=torch.long),
    }
    x2 = {
        "signal": torch.from_numpy(rng.standard_normal((batch, L)).astype(np.float32)),
        "attention_mask": torch.ones(batch, T, dtype=torch.long),
    }
    xneg = None
    if H > 0:
        xneg = {
            "signal": torch.from_numpy(rng.standard_normal((batch, H, L)).astype(np.float32)),
            "attention_mask": torch.ones(batch, H, T, dtype=torch.long),
        }
    return x1, x2, xneg


def _slice(d, lo, hi):
    if d is None:
        return None
    return {k: v[lo:hi] for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description="DDP contrastive-loss equivalence check")
    ap.add_argument("--batch", type=int, default=48, help="global (effective) batch")
    ap.add_argument("--hard_negatives", type=int, default=8)
    ap.add_argument("--input_signal_len", type=int, default=500,
                    help="small on purpose: the gather/loss equivalence is isl-independent")
    ap.add_argument("--downsample_factor", type=int, default=5)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--encoder_type", type=str, default="mamba",
                    choices=["mamba", "transformer", "cnn_rnn"])
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    rank, world_size, local_rank = _setup_distributed()
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda:0"
    else:
        device = "cpu"

    if args.batch % world_size != 0:
        if rank == 0:
            print(f"[ddp-equiv] FAIL: batch {args.batch} not divisible by world_size {world_size}")
        sys.exit(2)
    local_bs = args.batch // world_size

    # Identical weights on every rank: deterministic init from a fixed seed.
    torch.manual_seed(0)
    cfg = SignalModelConfigSchema(
        encoder_type=args.encoder_type,
        downsample_factor=args.downsample_factor,
        n_mamba_blocks=args.n_blocks,
        input_signal_len=args.input_signal_len,
    )
    encoder, pooling = signal_encoder_from_config(cfg)
    encoder = encoder.to(device).eval()   # eval() -> no dropout -> deterministic
    pooling = pooling.to(device)

    L = args.input_signal_len
    T = L // args.downsample_factor
    x1, x2, xneg = _make_fixed_batch(args.batch, args.hard_negatives, L, T)

    if rank == 0:
        print(f"[ddp-equiv] world_size={world_size} local_bs={local_bs} "
              f"batch={args.batch} H={args.hard_negatives} isl={L} device={device}", flush=True)

    with torch.no_grad():
        # (b) DDP path: this rank forwards its local slice, all-gather -> loss.
        lo, hi = rank * local_bs, (rank + 1) * local_bs
        y1l, y2l, ynegl = _forward_local_embeddings(
            encoder, pooling, _slice(x1, lo, hi), _slice(x2, lo, hi),
            _slice(xneg, lo, hi), device)
        loss_ddp, y1_gathered, y2_gathered, _ = gathered_infonce_loss(
            y1l, y2l, ynegl, args.temperature, world_size, rank)

    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    # (a) single-GPU reference (rank 0): encode ALL `batch` samples in one shot and
    # compute the batch-`batch` loss with the SAME loss function (world_size=1 =>
    # gather is the identity). Same encoder weights, same inputs => this is exactly
    # what a single GPU with batch=`batch` would compute.
    with torch.no_grad():
        y1f, y2f, ynegf = _forward_local_embeddings(encoder, pooling, x1, x2, xneg, device)
        loss_single, y1_full, y2_full, _ = gathered_infonce_loss(
            y1f, y2f, ynegf, args.temperature, 1, 0)

    dloss = abs(float(loss_single) - float(loss_ddp))
    max_y1 = float((y1_gathered - y1_full).abs().max()) if world_size > 1 else 0.0
    max_y2 = float((y2_gathered - y2_full).abs().max()) if world_size > 1 else 0.0

    print(f"[ddp-equiv] loss(single-GPU batch={args.batch}) = {float(loss_single):.6f}", flush=True)
    print(f"[ddp-equiv] loss(2-GPU gathered,  {args.batch:>3d}) = {float(loss_ddp):.6f}", flush=True)
    print(f"[ddp-equiv] |Δloss| = {dloss:.3e}   (must be < {args.tol:g})", flush=True)
    if world_size > 1:
        print(f"[ddp-equiv] max|gathered_y1 - full_y1| = {max_y1:.3e}", flush=True)
        print(f"[ddp-equiv] max|gathered_y2 - full_y2| = {max_y2:.3e}", flush=True)
    else:
        print("[ddp-equiv] NOTE: world_size==1 — this only prints the single-GPU "
              "loss. Launch with `torchrun --nproc_per_node=2` for the real proof.",
              flush=True)

    ok = (dloss < args.tol) and (max_y1 < args.tol) and (max_y2 < args.tol)
    if world_size > 1:
        print(f"[ddp-equiv] {'PASS' if ok else 'FAIL'}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()
    if world_size > 1 and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
