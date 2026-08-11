"""
Neurosamble Phase 1 -- read<->read anchor-density gate (build + probe only).

GATE QUESTION (and nothing else): when BOTH sides are noisy real reads, does the
frozen encoder produce dense, accurate anchors between reads that truly overlap?
This script builds an all-vs-all read-window FAISS index (à la Rawsamble ava),
queries every read's windows back against it, and reports the anchor-density
distribution split by true-overlap vs non-overlap read pairs. It does NOT chain,
run miniasm, or emit an assembly PAF -- that is deliberately out of scope until
this gate is passed.

Reuse (frozen, unmodified): ``load_encoder`` (pilot_recall), ``SignalEvalModel``
(inference_signal), ``SignalFaissStore`` (signal_faiss_store), ``read_blow5``
(real_data_eval), ``tile_read`` / ``build_readwindow_index`` (overlap_index).
The read->reference mapping path is untouched.

Truth (one of):
  --overlap_truth PAF : ready-made read-pair overlap truth (lines: qname ... tname).
  --ref_truth_paf PAF : read->reference PAF; two '+'-strand reads truly overlap iff
                        their reference intervals on the SAME contig & SAME strand
                        overlap by >= --min_overlap_bp. Same-strand only.

Heavy imports (torch / faiss / encoder) live inside ``main`` so the pure truth
logic and ``tile_read`` stay importable (e.g. by tests) with no torch present.

Example (self-check note: self-neighbors -- neighbor read_id == query read_id --
are always excluded, the read<->read analogue of a k=0 self-hit):

    python evaluate/overlap_probe.py \
        --real_reads reads.blow5 \
        --load_encoder real_encoder_v1.pt \
        --ref_truth_paf truth.paf \
        --out_dir out/neurosamble_phase1 --limit 4000 --faiss_cpu 1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Make sibling evaluate/ modules importable when run as a script, and expose
# ``dna2vec`` (src/) for the encoder side.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ``tile_read`` / ``build_readwindow_index`` are numpy-only (no torch), safe to
# import at module top so tests can pull ``tile_read`` cheaply.
from overlap_index import build_readwindow_index, tile_read  # noqa: E402


# --------------------------------------------------------------------------- #
# Truth derivation (pure: stdlib + numpy only, no torch)
# --------------------------------------------------------------------------- #
def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """Order a read pair so (A,B) and (B,A) collapse to the same key."""
    return (a, b) if a <= b else (b, a)


def parse_ref_truth_paf(path: str) -> Dict[str, Tuple[str, str, int, int]]:
    """Parse a read->reference PAF into ``read_id -> (contig, strand, tstart, tend)``.

    PAF columns: qname qlen qstart qend strand tname tlen tstart tend ... When a
    read has multiple alignments, the longest reference span (primary) wins.
    """
    info: Dict[str, Tuple[str, str, int, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                fields = line.split()
            if len(fields) < 9:
                continue
            rid, strand, contig = fields[0], fields[4], fields[5]
            try:
                tstart, tend = int(fields[7]), int(fields[8])
            except ValueError:
                continue
            if tend < tstart:
                tstart, tend = tend, tstart
            span = tend - tstart
            prev = info.get(rid)
            if prev is None or span > (prev[3] - prev[2]):
                info[rid] = (contig, strand, tstart, tend)
    return info


def derive_read_pair_truth(
    read_info: Dict[str, Tuple[str, str, int, int]],
    subsample_ids,
    min_overlap_bp: int = 100,
) -> Set[Tuple[str, str]]:
    """Read-pair overlap truth from reference intervals (same contig, same strand '+').

    Two reads truly overlap iff both are '+' strand, on the same contig, within
    the subsample, and their reference intervals overlap by ``>= min_overlap_bp``.
    Uses a per-contig start-sorted sweep so it stays efficient.
    """
    sub = set(subsample_ids)
    by_contig: Dict[str, List[Tuple[int, int, str]]] = {}
    for rid, (contig, strand, tstart, tend) in read_info.items():
        if strand != "+" or rid not in sub:
            continue
        by_contig.setdefault(contig, []).append((tstart, tend, rid))

    truth: Set[Tuple[str, str]] = set()
    for lst in by_contig.values():
        lst.sort()  # by tstart
        for i in range(len(lst)):
            ts_i, te_i, rid_i = lst[i]
            for j in range(i + 1, len(lst)):
                ts_j, te_j, rid_j = lst[j]
                if ts_j >= te_i:
                    break  # start-sorted: no later read can overlap read i
                overlap = min(te_i, te_j) - max(ts_i, ts_j)
                if overlap >= min_overlap_bp:
                    truth.add(canonical_pair(rid_i, rid_j))
    return truth


def parse_overlap_truth(path: str, subsample_ids) -> Set[Tuple[str, str]]:
    """Parse a ready-made read-pair overlap truth (lines: ``qname ... tname``).

    ``tname`` is column 6 for a PAF-shaped file, else column 2 for a bare
    two-column pair list. Only pairs with both reads in the subsample are kept.
    """
    sub = set(subsample_ids)
    truth: Set[Tuple[str, str]] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            q = fields[0]
            t = fields[5] if len(fields) >= 6 else fields[1]
            if q != t and q in sub and t in sub:
                truth.add(canonical_pair(q, t))
    return truth


# --------------------------------------------------------------------------- #
# Metric helpers (pure)
# --------------------------------------------------------------------------- #
def _percentile_with_zeros(nonzero_counts: List[int], zero_count: int, q: float) -> float:
    """Linear-interpolated percentile of a multiset = ``zero_count`` zeros + counts.

    Avoids materializing the (potentially millions of) implicit zero-anchor
    non-overlap pairs. Matches numpy's default 'linear' interpolation.
    """
    vals = sorted(nonzero_counts)
    n = zero_count + len(vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(vals[0]) if zero_count == 0 else 0.0

    def val_at(i: int) -> float:
        return 0.0 if i < zero_count else float(vals[i - zero_count])

    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return val_at(lo)
    frac = pos - lo
    return val_at(lo) * (1.0 - frac) + val_at(hi) * frac


def _hist_bucket(c: int) -> str:
    if c <= 0:
        return "0"
    if c == 1:
        return "1"
    if c == 2:
        return "2"
    if c <= 4:
        return "3-4"
    if c <= 9:
        return "5-9"
    return "10+"


_BUCKETS = ["0", "1", "2", "3-4", "5-9", "10+"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Neurosamble Phase 1: read<->read anchor-density gate (probe only)."
    )
    p.add_argument("--real_reads", required=True, help="reads blow5/slow5 (query = target set)")
    p.add_argument("--load_encoder", required=True, help="fine-tuned encoder checkpoint (.pt)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=4000,
                   help="subsample N reads (seeded); 0 = use all eligible reads")
    p.add_argument("--win", type=int, default=2000, help="tiling window length in samples")
    p.add_argument("--stride", type=int, default=1000, help="tiling stride in samples")
    p.add_argument("--topk", type=int, default=10, help="neighbors per query window")
    p.add_argument("--min_anchors", type=int, default=5,
                   help="chainable threshold (matches Rawsamble ava): >= this many "
                        "anchors makes a read-pair a chainable candidate")
    p.add_argument("--min_overlap_bp", type=int, default=100,
                   help="reference-interval overlap (bp) to call a true read-pair "
                        "overlap when deriving truth from --ref_truth_paf")
    p.add_argument("--pore_model", default=None,
                   help="ONT k-mer pore model (accepted for interface parity; the "
                        "read<->read probe encodes recorded signal, no expected signal)")
    p.add_argument("--samples_per_kmer", type=int, default=9,
                   help="accepted for interface parity with the mapping path")
    p.add_argument("--device", default="cuda:0", help="encoder device (GPU)")
    p.add_argument("--faiss_cpu", type=int, default=1,
                   help="1 = FAISS index on CPU (default; avoids GPU OOM); 0 = on --device")
    p.add_argument("--seed", type=int, default=1234, help="deterministic subsample seed")
    # Exactly one truth source.
    p.add_argument("--overlap_truth", default=None,
                   help="ready-made read-pair overlap truth PAF (lines: qname ... tname)")
    p.add_argument("--ref_truth_paf", default=None,
                   help="read->reference PAF; read-pair overlap truth is DERIVED from it")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    if bool(args.overlap_truth) == bool(args.ref_truth_paf):
        raise SystemExit(
            "provide exactly ONE of --overlap_truth or --ref_truth_paf"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # Heavy imports live here so the truth logic / tile_read stay torch-free.
    import torch  # noqa: F401
    from real_data_eval import read_blow5
    from pilot_recall import load_encoder
    from signal_faiss_store import SignalFaissStore

    print(f"[seed] subsample seed = {args.seed}", flush=True)

    # 1) Load reads. keep_min defaults to 2100 (>= win 2000 + margin) in read_blow5.
    all_reads = read_blow5(args.real_reads, limit=None)
    reads_by_id = {r.id: r for r in all_reads}
    print(f"[reads] loaded {len(all_reads)} reads from {args.real_reads}", flush=True)

    # Eligible read universe. For the derived-truth path we restrict to '+' strand
    # reads present in the PAF (same-strand ava); for a ready-made pair truth we
    # take every loaded read.
    read_info: Dict[str, Tuple[str, str, int, int]] = {}
    if args.ref_truth_paf:
        read_info = parse_ref_truth_paf(args.ref_truth_paf)
        eligible = [rid for rid in reads_by_id if read_info.get(rid, (None, None))[1] == "+"]
        print(
            f"[truth] ref PAF: {len(read_info)} aligned reads; "
            f"{len(eligible)} are '+'-strand and present in the blow5",
            flush=True,
        )
    else:
        eligible = list(reads_by_id.keys())

    # 2) Deterministic subsample.
    rng = random.Random(args.seed)
    eligible_sorted = sorted(eligible)
    if args.limit and len(eligible_sorted) > args.limit:
        chosen = set(rng.sample(eligible_sorted, args.limit))
    else:
        chosen = set(eligible_sorted)
    reads_sub = [reads_by_id[rid] for rid in eligible_sorted if rid in chosen]
    n_reads = len(reads_sub)
    print(f"[subsample] using {n_reads} reads (limit={args.limit}, seed={args.seed})", flush=True)
    if n_reads < 2:
        raise SystemExit("need >= 2 subsampled reads to probe overlaps")

    # 3) Encoder (GPU) + FAISS store (CPU by default).
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available; running encoder on CPU", flush=True)
    model, _cfg = load_encoder(args.load_encoder, device, args)

    store_device = "cpu" if int(args.faiss_cpu) else device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    index_name = f"neurosamble_rw_{os.getpid()}"
    store = SignalFaissStore(signal_model=model, index_name=index_name, device=store_device)
    store.drop_table()  # start clean; nothing is persisted afterwards
    store = SignalFaissStore(signal_model=model, index_name=index_name, device=store_device)

    # 4) Build the all-vs-all read-window index.
    n_windows, n_reads_idx = build_readwindow_index(
        reads_sub, model, store, win=args.win, stride=args.stride, namespace="reads"
    )

    # 5) Query each read's windows back against the index; form anchors.
    #    Canonicalize + dedup: an A->B anchor and its mirror B->A collapse to the
    #    same (q_off, t_off) tuple under canonical ordering, so store one score
    #    per (q_off, t_off) to avoid 2x inflation.
    pair_anchors: Dict[Tuple[str, str], Dict[Tuple[int, int], float]] = {}
    for r in reads_sub:
        tiles = tile_read(r.signal, win=args.win, stride=args.stride)
        if not tiles:
            continue
        offs = [int(off) for off, _ in tiles]
        wins = [w for _, w in tiles]
        results = store.query_batch(wins, [None] * len(wins), top_k=args.topk)
        for j, res in enumerate(results):
            q_off = offs[j]
            for m in res["matches"]:
                meta = m["metadata"]
                t_rid = meta["read_id"]
                if t_rid == r.id:
                    continue  # self-exclusion (k=0-style self-hit)
                t_off = int(meta["offset"])
                score = float(m["score"])
                pair = canonical_pair(r.id, t_rid)
                if pair[0] == r.id:
                    key = (q_off, t_off)
                else:
                    key = (t_off, q_off)
                bucket = pair_anchors.setdefault(pair, {})
                if key not in bucket or score > bucket[key]:
                    bucket[key] = score

    cand_counts: Dict[Tuple[str, str], int] = {p: len(a) for p, a in pair_anchors.items()}
    n_candidate_pairs = len(cand_counts)

    # 6) Truth set restricted to the subsample.
    if args.ref_truth_paf:
        truth_pairs = derive_read_pair_truth(read_info, chosen, args.min_overlap_bp)
    else:
        truth_pairs = parse_overlap_truth(args.overlap_truth, chosen)
    n_true_pairs = len(truth_pairs)

    # 7) Metrics.
    true_counts = np.array(
        [cand_counts.get(p, 0) for p in truth_pairs], dtype=float
    )
    if true_counts.size:
        med_true = float(np.median(true_counts))
        p10_true = float(np.percentile(true_counts, 10))
        p90_true = float(np.percentile(true_counts, 90))
        mean_true = float(true_counts.mean())
    else:
        med_true = p10_true = p90_true = mean_true = 0.0

    nontrue_nonzero = [c for p, c in cand_counts.items() if p not in truth_pairs]
    total_pairs = n_reads * (n_reads - 1) // 2
    total_non_overlap = max(0, total_pairs - n_true_pairs)
    zero_count = max(0, total_non_overlap - len(nontrue_nonzero))
    med_nontrue = _percentile_with_zeros(nontrue_nonzero, zero_count, 50.0)
    p90_nontrue = _percentile_with_zeros(nontrue_nonzero, zero_count, 90.0)

    # Chainable recall: true pairs reaching the chainable anchor threshold.
    n_true_ge = sum(1 for p in truth_pairs if cand_counts.get(p, 0) >= args.min_anchors)
    chainable_recall = (n_true_ge / n_true_pairs) if n_true_pairs else 0.0

    # Anchor/seed precision: of candidate pairs with >= min_anchors, fraction true.
    ge_pairs = [p for p, c in cand_counts.items() if c >= args.min_anchors]
    n_ge = len(ge_pairs)
    n_ge_true = sum(1 for p in ge_pairs if p in truth_pairs)
    seed_precision = (n_ge_true / n_ge) if n_ge else 0.0

    # Histograms (anchors/pair) split by true vs non-true.
    true_hist = {b: 0 for b in _BUCKETS}
    for p in truth_pairs:
        true_hist[_hist_bucket(cand_counts.get(p, 0))] += 1
    nontrue_hist = {b: 0 for b in _BUCKETS}
    for c in nontrue_nonzero:
        nontrue_hist[_hist_bucket(c)] += 1
    nontrue_hist["0"] += zero_count  # implicit zero-anchor non-overlap pairs

    # 8) Report to stdout.
    print("", flush=True)
    print("==================== NEUROSAMBLE PHASE-1 ANCHOR PROBE ====================")
    print(f"seed                 : {args.seed}")
    print(f"win / stride         : {args.win} / {args.stride}   topk={args.topk}   "
          f"min_anchors={args.min_anchors}")
    print(f"n_reads              : {n_reads}")
    print(f"n_windows            : {n_windows}")
    print(f"n_true_pairs         : {n_true_pairs}   (within subsample)")
    print(f"n_candidate_pairs    : {n_candidate_pairs}   (>=1 anchor)")
    print("-- anchors per TRUE-overlap pair --")
    print(f"  median={med_true:.2f}  p10={p10_true:.2f}  p90={p90_true:.2f}  mean={mean_true:.2f}")
    print("-- anchors per NON-overlap pair (expected ~0) --")
    print(f"  median={med_nontrue:.2f}  p90={p90_nontrue:.2f}")
    print(f"chainable_recall     : {chainable_recall:.4f}   "
          f"(true pairs with >= {args.min_anchors} anchors: {n_true_ge}/{n_true_pairs})")
    print(f"seed_precision       : {seed_precision:.4f}   "
          f"(true among >= {args.min_anchors}-anchor candidates: {n_ge_true}/{n_ge})")
    print("-- histogram (anchors/pair)  bucket : true | non-true --")
    for b in _BUCKETS:
        print(f"  {b:>4} : {true_hist[b]:>10} | {nontrue_hist[b]:>12}")
    print("==========================================================================")
    print(
        "VERDICT-INPUT: "
        f"n_reads={n_reads} n_windows={n_windows} n_true_pairs={n_true_pairs} "
        f"n_candidate_pairs={n_candidate_pairs} min_anchors={args.min_anchors} "
        f"chainable_recall={chainable_recall:.4f} seed_precision={seed_precision:.4f} "
        f"median_anchors_true={med_true:.2f} p10_anchors_true={p10_true:.2f} "
        f"p90_anchors_true={p90_true:.2f} mean_anchors_true={mean_true:.2f} "
        f"median_anchors_nontrue={med_nontrue:.2f} p90_anchors_nontrue={p90_nontrue:.2f}",
        flush=True,
    )

    # 9) Write CSV (isolation: never overwrite; timestamp-suffix if it exists).
    csv_path = os.path.join(args.out_dir, "anchor_probe.csv")
    if os.path.exists(csv_path):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.out_dir, f"anchor_probe_{ts}.csv")
    rows = [
        ("seed", args.seed),
        ("win", args.win),
        ("stride", args.stride),
        ("topk", args.topk),
        ("min_anchors", args.min_anchors),
        ("min_overlap_bp", args.min_overlap_bp),
        ("truth_source", "ref_truth_paf" if args.ref_truth_paf else "overlap_truth"),
        ("n_reads", n_reads),
        ("n_windows", n_windows),
        ("n_true_pairs", n_true_pairs),
        ("n_candidate_pairs", n_candidate_pairs),
        ("median_anchors_true", round(med_true, 4)),
        ("p10_anchors_true", round(p10_true, 4)),
        ("p90_anchors_true", round(p90_true, 4)),
        ("mean_anchors_true", round(mean_true, 4)),
        ("median_anchors_nontrue", round(med_nontrue, 4)),
        ("p90_anchors_nontrue", round(p90_nontrue, 4)),
        ("chainable_recall", round(chainable_recall, 6)),
        ("seed_precision", round(seed_precision, 6)),
    ]
    for b in _BUCKETS:
        rows.append((f"hist_true_{b}", true_hist[b]))
    for b in _BUCKETS:
        rows.append((f"hist_nontrue_{b}", nontrue_hist[b]))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    print(f"[csv] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
