"""STAGE4b Step 2 — warm-start fine-tune the encoder on REAL reads (anchor A).

Query side  = real train-read signal, trimmed `trim_fixed` samples (with training
              trim-JITTER +/-`jitter`), then the first `input_signal_len` samples.
Reference   = pore-model expected signal of ref[tstart:tstart+unit_length] ('+') or
              revcomp(ref[tend-unit_length:tend]) ('-')  == anchor A (chosen because
              Step 0's fixed-2500 already hit 45.8% vs 0.2% random, and because the
              head-to-head scores retrieved-coord vs truth tstart, so training must
              target tstart). DB/reference side is UNCHANGED (pore-model over CFT073).

Reuses the canonical DDP all-gathered InfoNCE (pilot_recall.train_encoder_hardneg)
so the effective batch / negative pool / loss are identical to STAGE1. Warm-starts
from canonical_encoder.pt, saves real_encoder_v1.pt (new path; never overwrites).

Reports (all on held-out REAL val reads, same store/tol for both):
  * BASELINE  = canonical encoder recall@1/@10  (the number to beat)
  * FINETUNED = real_encoder_v1 recall@1/@10
Plus the retained train set's qstart distribution (to size the jitter).

Launch under torchrun (2 GPU DDP), same as canonical training:
  torchrun --nproc_per_node=2 evaluate/stage4b_finetune.py ...

DEV-ONLY: not run here. WATCH the training loss — if it collapses to 0.0000 it is
a leakage/overfit signal; stop and report (Ctrl-C the job).
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for p in (str(_HERE), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from dna2vec.pore_model import PoreModel  # noqa: E402
from dna2vec.signal_dataset import SignalPairDataset, preprocess_window  # noqa: E402
from inference_signal import SignalEvalModel  # noqa: E402
from upsert_signal import read_single_fasta  # noqa: E402
from real_data_eval import read_fasta_name  # noqa: E402

from pilot_recall import (  # noqa: E402
    _setup_distributed, _dist_info, load_encoder, save_encoder,
    train_encoder_hardneg, build_store, evaluate_recall,
)


# --------------------------------------------------------------------------- #
class RealJitterPairDataset(SignalPairDataset):
    """SignalPairDataset whose QUERY window comes from a cached real-read slice with
    a random trim jitter each draw. query_signals holds per-read cached slices that
    cover [trim_fixed-jitter : trim_fixed+jitter+input_signal_len]; the reference /
    hard-negative construction (via inherited _anchor/_ref_window) is unchanged."""

    def __init__(self, *, jitter, **kw):
        super().__init__(**kw)   # query_signals = list of cached slices
        self.jitter = int(jitter)

    def _query_window(self, idx):
        sig = np.asarray(self.query_signals[idx], dtype=np.float32)
        j = int(np.random.randint(-self.jitter, self.jitter + 1)) if self.jitter > 0 else 0
        start = self.jitter + j   # cache is aligned so trim_fixed sits at offset `jitter`
        start = int(min(max(0, start), max(0, len(sig) - self.input_signal_len)))
        return preprocess_window(sig[start:], self.input_signal_len, self.downsample_factor)

    def _pair(self, idx):
        anchor, strand = self._anchor(idx)
        q_sig, q_mask = self._query_window(idx)
        r_sig, r_mask = self._ref_window(anchor, strand)
        x_1 = {"signal": q_sig, "attention_mask": q_mask}
        x_2 = {"signal": r_sig, "attention_mask": r_mask}
        if self.hard_negatives <= 0:
            return x_1, x_2
        neg_sigs, neg_masks = [], []
        for _ in range(self.hard_negatives):
            sign = 1 if np.random.rand() < 0.5 else -1
            offset = np.random.randint(self.hard_neg_min_bp, self.hard_neg_max_bp + 1)
            ns, nm = self._ref_window(anchor + sign * offset, strand)
            neg_sigs.append(ns); neg_masks.append(nm)
        x_neg = {"signal": np.stack(neg_sigs), "attention_mask": np.stack(neg_masks)}
        return x_1, x_2, x_neg


class _EvalRead:
    __slots__ = ("signal", "reference_start")
    def __init__(self, signal, reference_start):
        self.signal = signal; self.reference_start = reference_start


# --------------------------------------------------------------------------- #
def read_pairs_tsv(path):
    """-> list of (read_id, strand, tstart, tend, qstart)."""
    out = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 5:
                continue
            out.append((c[0], c[1], int(c[2]), int(c[3]), int(c[4])))
    return out


def extract_windows(blow5, want_ids, cache_lo, cache_hi):
    """Single pass; for each id in want_ids cache raw[cache_lo:cache_hi] as float16
    (all such reads passed the >= trim+isl+jitter length filter in Step 1)."""
    import pyslow5
    cache = {}
    s = pyslow5.Open(blow5, "r")
    for rec in s.seq_reads(pA=True):
        rid = rec["read_id"]
        if rid not in want_ids:
            continue
        sig = np.asarray(rec["signal"], dtype=np.float32)
        if len(sig) >= cache_hi:
            cache[rid] = sig[cache_lo:cache_hi].astype(np.float16)
        if len(cache) == len(want_ids):
            break
    return cache


def _anchor_low(strand, tstart, tend, unit_length):
    return (tend - unit_length) if strand == "-" else tstart


def build_val_reads(val_pairs, cache, args):
    """_EvalRead list: query = cached window from trim_fixed (NO jitter); coord = anchor A."""
    reads = []
    off = args.jitter  # trim_fixed sits at offset `jitter` inside the cache
    for rid, strand, ts, te, _qs in val_pairs:
        if rid not in cache:
            continue
        sig = np.asarray(cache[rid], dtype=np.float32)[off:]   # from global trim_fixed onward
        reads.append(_EvalRead(sig, _anchor_low(strand, ts, te, args.unit_length)))
    return reads


def _mk_store_args(args):
    return Namespace(unit_length=args.unit_length, overlap=args.overlap, index_stride=0,
                     both_strands=1, faiss_cpu=args.faiss_cpu)


def eval_val(tag, signal_model, reference_seq, pore, val_reads, device, args):
    store = build_store(signal_model, f"stage4b-{tag}", device, reference_seq, pore, _mk_store_args(args))
    topk = [1, 5, 10, 50]
    out = {}
    for tol in (args.tol_bp, args.tol_bp_loose):
        rec, mrr = evaluate_recall(store, val_reads, topk, tol, args.unit_length)
        out[tol] = (rec, mrr)
        print(f"[val:{tag}] tol=+/-{tol}bp  " +
              "  ".join(f"@{k}={rec[k]*100:5.1f}%" for k in topk) + f"  MRR={mrr:.3f}",
              flush=True)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="STAGE4b Step 2 — real-read warm-start fine-tune")
    p.add_argument("--data_dir", required=True, help="experiments/stage4b_data (Step 1 output)")
    p.add_argument("--blow5", required=True)
    p.add_argument("--ref", required=True, help="CFT073 ref.fa")
    p.add_argument("--canonical", required=True, help="canonical_encoder.pt (warm start + baseline)")
    p.add_argument("--out_dir", required=True, help="experiments/stage4b_finetune")
    p.add_argument("--save_encoder", required=True, help="real_encoder_v1.pt (new; never overwrite canonical)")
    p.add_argument("--pore_model", default=None)
    p.add_argument("--kmer_len", type=int, default=6)
    p.add_argument("--samples_per_kmer", type=int, default=9)
    p.add_argument("--dwell_mean", type=float, default=9.0, help="approx samples/base, for qstart->samples")
    p.add_argument("--trim_fixed", type=int, default=2500)
    p.add_argument("--jitter", type=int, default=500, help="training trim jitter (samples); must be <= Step-1 --jitter")
    p.add_argument("--unit_length", type=int, default=300)
    p.add_argument("--overlap", type=int, default=285)
    p.add_argument("--hard_negatives", type=int, default=8)
    p.add_argument("--hard_neg_min_bp", type=int, default=30)
    p.add_argument("--hard_neg_max_bp", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--train_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe", type=int, default=1)
    p.add_argument("--faiss_cpu", type=int, default=1)
    p.add_argument("--val_limit", type=int, default=3000, help="val reads used for recall (speed)")
    p.add_argument("--tol_bp", type=int, default=15)
    p.add_argument("--tol_bp_loose", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = _setup_distributed()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = (f"cuda:{local_rank}" if world_size > 1 else args.device) if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # input_signal_len comes from the checkpoint; resolve it before sizing the cache.
    _ck = torch.load(args.canonical, map_location="cpu")
    isl = int(_ck["signal_config"]["input_signal_len"])
    del _ck
    cache_lo = args.trim_fixed - args.jitter
    cache_hi = args.trim_fixed + args.jitter + isl
    if cache_lo < 0:
        raise ValueError(f"trim_fixed({args.trim_fixed}) - jitter({args.jitter}) < 0")

    train_pairs = read_pairs_tsv(os.path.join(args.data_dir, "train_pairs.tsv"))
    val_pairs = read_pairs_tsv(os.path.join(args.data_dir, "val_pairs.tsv"))[: args.val_limit]
    if rank == 0:
        qs = np.array([p[4] for p in train_pairs], dtype=np.float64)
        pct = {q: float(np.percentile(qs, q)) for q in (10, 50, 90, 99)}
        print(f"[stage4b-ft] train pairs={len(train_pairs)} val(eval)={len(val_pairs)} "
              f"isl={isl} trim={args.trim_fixed} jitter=+/-{args.jitter} "
              f"(cache [{cache_lo}:{cache_hi}])", flush=True)
        print(f"[stage4b-ft] retained-train qstart (bp): p10={pct[10]:.0f} p50={pct[50]:.0f} "
              f"p90={pct[90]:.0f} p99={pct[99]:.0f} max={qs.max():.0f}", flush=True)
        print(f"[stage4b-ft]   ~samples (x{args.dwell_mean:g}): p50={pct[50]*args.dwell_mean:.0f} "
              f"p90={pct[90]*args.dwell_mean:.0f}  (jitter covers +/-{args.jitter} samples) "
              f"-> if p90 samples >> {args.jitter}, val weak may need larger --jitter (re-run Step 1 too)",
              flush=True)

    reference_seq = read_single_fasta(args.ref)
    ref_contig = read_fasta_name(args.ref)
    pore = PoreModel(kmer_table_path=(args.pore_model or os.environ.get("PORE_MODEL_PATH")),
                     kmer_len=args.kmer_len, samples_per_kmer=args.samples_per_kmer)
    if rank == 0:
        print(f"[stage4b-ft] ref contig={ref_contig} ({len(reference_seq)} bp); "
              f"pore synthetic={pore.synthetic}", flush=True)

    # ---- cache real signals: train on ALL ranks; val only on rank 0 -------------
    train_ids = {p[0] for p in train_pairs}
    want = set(train_ids)
    if rank == 0:
        want |= {p[0] for p in val_pairs}
    if rank == 0:
        print(f"[stage4b-ft] extracting {len(want)} cached windows from blow5 "
              f"(single pass, float16)...", flush=True)
    cache = extract_windows(args.blow5, want, cache_lo, cache_hi)
    train_pairs = [p for p in train_pairs if p[0] in cache]
    if rank == 0:
        print(f"[stage4b-ft] cached train={len(train_pairs)} "
              f"(dropped {len(train_ids) - len(train_pairs)} missing/short in blow5)", flush=True)

    # ---- warm start from canonical (all ranks) ---------------------------------
    model, cfg = load_encoder(args.canonical, device, args)
    args.input_signal_len = cfg.input_signal_len
    args.downsample_factor = cfg.downsample_factor

    # ---- BASELINE: canonical on real val (rank 0), the number to beat ----------
    if rank == 0:
        val_reads = build_val_reads(val_pairs, cache, args)
        print(f"[stage4b-ft] === BASELINE (canonical, no fine-tune) on {len(val_reads)} real val reads ===",
              flush=True)
        eval_val("baseline", model, reference_seq, pore, val_reads, device, args)
    if world_size > 1:
        dist.barrier()

    # ---- fine-tune dataset (query = real train reads, anchor A) -----------------
    dataset = RealJitterPairDataset(
        jitter=args.jitter,
        query_signals=[cache[p[0]] for p in train_pairs],
        query_coords=[p[2] for p in train_pairs],           # tstart (PAF col 8)
        query_ends=[p[3] for p in train_pairs],             # tend   (PAF col 9)
        query_strands=[p[1] for p in train_pairs],          # strand (PAF col 5)
        reference_seq=reference_seq, pore_model=pore,
        unit_length=args.unit_length,
        input_signal_len=cfg.input_signal_len, downsample_factor=cfg.downsample_factor,
        samples_per_kmer=args.samples_per_kmer,
        hard_negatives=args.hard_negatives,
        hard_neg_min_bp=args.hard_neg_min_bp, hard_neg_max_bp=args.hard_neg_max_bp,
    )

    if rank == 0:
        print(f"[stage4b-ft] === fine-tuning (warm start) {args.train_steps} steps, "
              f"batch {args.batch_size}, H={args.hard_negatives} — WATCH loss; if it hits "
              f"0.0000 stop (leakage/overfit) ===", flush=True)
    encoder = train_encoder_hardneg(model.encoder, model.pooling, dataset, device, args)
    if world_size > 1:
        dist.barrier()

    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    # ---- rank 0: save + FINETUNED val recall ------------------------------------
    save_encoder(args.save_encoder, encoder, cfg)
    ft_model = SignalEvalModel(encoder=encoder, pooling=model.pooling, device=device,
                               input_signal_len=cfg.input_signal_len,
                               downsample_factor=cfg.downsample_factor,
                               embedding_dim=cfg.embedding_dim)
    val_reads = build_val_reads(val_pairs, cache, args)
    print(f"[stage4b-ft] === FINETUNED (real_encoder_v1) on {len(val_reads)} real val reads ===", flush=True)
    eval_val("finetuned", ft_model, reference_seq, pore, val_reads, device, args)
    print("[stage4b-ft] DONE — compare FINETUNED vs BASELINE recall@10 (must clearly exceed). "
          f"saved -> {args.save_encoder}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
