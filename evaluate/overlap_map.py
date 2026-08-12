"""
Neurosamble Phase 2 -- read<->read overlapper entry (index -> anchors -> chains -> PAF).

This replaces the Phase-1 probe for real runs. On a FIXED read subset (given by
``--read_ids``) it:
  1. loads only those reads (keep_ids through ``read_blow5``),
  2. builds the all-vs-all read-window FAISS index (Phase-1 ``build_readwindow_index``),
  3. forms per-canonical-pair anchor dicts (Phase-1 anchor block, self-excluded),
  4. chains each pair's anchors (``overlap_chain.chain_anchors``),
  5. writes ``neurosamble.paf`` (12 std PAF cols, one line per surviving chain),
     with an ``mt:f:0.0`` tag on EVERY line (RawHash ``pafstats.py`` requires an
     ``mt:f:`` tag on its input PAF or it raises ValueError).

The read->reference mapping path and the Phase-1 probe are untouched. Frozen
stores (``signal_faiss_store.py`` / ``faiss_store.py``) are reused unchanged.
Heavy imports (torch / faiss / encoder) live inside ``main`` so the module stays
cheap to import.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Make sibling evaluate/ modules importable when run as a script, and expose src/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Numpy-only / torch-free reuse.
from overlap_index import build_readwindow_index, tile_read  # noqa: E402
from overlap_chain import chain_anchors  # noqa: E402
from overlap_probe import canonical_pair  # noqa: E402


def read_fasta_ids(path: str):
    """Return the set of sequence ids (first header token) in a FASTA."""
    ids = set()
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def read_id_list(path: str):
    """One read-id per line (blank/`#` lines ignored)."""
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def parse_args():
    p = argparse.ArgumentParser(
        description="Neurosamble Phase 2: read<->read overlapper -> PAF."
    )
    p.add_argument("--real_reads", required=True, help="reads blow5/slow5 (the subset signals)")
    p.add_argument("--load_encoder", required=True, help="fine-tuned encoder checkpoint (.pt)")
    p.add_argument("--reads_fasta", required=True,
                   help="basecalled reads FASTA (for qname parity + downstream truth)")
    p.add_argument("--read_ids", required=True,
                   help="REQUIRED file of read-ids to include (the fixed subset)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--win", type=int, default=2000)
    p.add_argument("--stride", type=int, default=1000)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--samples_per_kmer", type=int, default=9,
                   help="signal->bp conversion (offset_bp = offset_samples // this)")
    # Chaining thresholds (defaults mirror Rawsamble -x ava).
    p.add_argument("--min_chaining_score", type=float, default=40.0)
    p.add_argument("--min_num_anchors", type=int, default=5)
    p.add_argument("--max_gap_bp", type=int, default=2500)
    p.add_argument("--bw_bp", type=int, default=5000)
    p.add_argument("--device", default="cuda:0", help="encoder device (GPU)")
    p.add_argument("--faiss_cpu", type=int, default=1,
                   help="1 = FAISS index on CPU (default); 0 = on --device")
    p.add_argument("--seed", type=int, default=1234,
                   help="printed for provenance; the run is deterministic given --read_ids")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    spk = max(1, int(args.samples_per_kmer))

    # Heavy imports here so the module stays torch-free at import time.
    import torch
    from real_data_eval import read_blow5
    from pilot_recall import load_encoder
    from signal_faiss_store import SignalFaissStore
    from paf_io import mapq_from_scores

    print(f"[seed] {args.seed}   (deterministic given --read_ids)", flush=True)

    # 1) Load ONLY the subset reads.
    keep_ids = read_id_list(args.read_ids)
    print(f"[reads] read_ids file lists {len(keep_ids)} ids", flush=True)
    reads = read_blow5(args.real_reads, limit=None, keep_ids=keep_ids)
    reads_by_id = {r.id: r for r in reads}
    print(f"[reads] loaded {len(reads)} subset reads from {args.real_reads}", flush=True)
    if len(reads) < 2:
        raise SystemExit("need >= 2 reads in the subset")

    # 2) Encoder (GPU) + FAISS store (CPU by default) + read-window index.
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available; encoder on CPU", flush=True)
    model, _cfg = load_encoder(args.load_encoder, device, args)

    store_device = "cpu" if int(args.faiss_cpu) else device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    index_name = f"neurosamble_map_{os.getpid()}"
    store = SignalFaissStore(signal_model=model, index_name=index_name, device=store_device)
    store.drop_table()  # start clean; nothing is persisted afterwards
    store = SignalFaissStore(signal_model=model, index_name=index_name, device=store_device)

    n_windows, _n_reads_idx = build_readwindow_index(
        reads, model, store, win=args.win, stride=args.stride, namespace="reads"
    )

    # 3) Form per-canonical-pair anchor dicts. This mirrors the Phase-1 anchor
    #    block verbatim (self-exclude; canonicalize + dedup A->B / B->A).
    pair_anchors: Dict[Tuple[str, str], Dict[Tuple[int, int], float]] = {}
    for r in reads:
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
                    continue  # self-exclusion
                t_off = int(meta["offset"])
                score = float(m["score"])
                pair = canonical_pair(r.id, t_rid)
                key = (q_off, t_off) if pair[0] == r.id else (t_off, q_off)
                bucket = pair_anchors.setdefault(pair, {})
                if key not in bucket or score > bucket[key]:
                    bucket[key] = score
    print(f"[anchors] {len(pair_anchors)} candidate read-pairs with >=1 anchor", flush=True)

    # 4) Chain each pair; collect surviving chains as PAF records.
    #    record = (qname,qlen,qs,qe,strand,tname,tlen,ts,te,nmatch,alen,chain_score)
    records = []
    for pair, anchors in pair_anchors.items():
        chains = chain_anchors(
            anchors, spk, args.min_num_anchors, args.min_chaining_score,
            max_gap_bp=args.max_gap_bp, bw_bp=args.bw_bp,
        )
        if not chains:
            continue
        qname, tname = pair
        qlen = max(1, len(reads_by_id[qname].signal) // spk)
        tlen = max(1, len(reads_by_id[tname].signal) // spk)
        for ch in chains:
            qs = max(0, min(ch.q_start, qlen))
            qe = max(0, min(ch.q_end, qlen))
            ts = max(0, min(ch.t_start, tlen))
            te = max(0, min(ch.t_end, tlen))
            if qe <= qs or te <= ts:
                continue
            span = qe - qs
            records.append(
                (qname, qlen, qs, qe, "+", tname, tlen, ts, te, span, span, ch.score)
            )

    # 5) Write neurosamble.paf. mapq from min-max normalized chain scores; every
    #    line carries an mt:f:0.0 tag (required by pafstats.py).
    scores = [rec[-1] for rec in records]
    mapqs = mapq_from_scores(scores)
    paf_path = os.path.join(args.out_dir, "neurosamble.paf")
    if os.path.exists(paf_path):
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        paf_path = os.path.join(args.out_dir, f"neurosamble_{ts}.paf")
    reported_pairs = set()
    with open(paf_path, "w") as f:
        for rec, mapq in zip(records, mapqs):
            (qname, qlen, qs, qe, strand, tname, tlen, ts, te, nmatch, alen, _sc) = rec
            reported_pairs.add((qname, tname))
            cols = [qname, qlen, qs, qe, strand, tname, tlen, ts, te, nmatch, alen, mapq]
            f.write("\t".join(str(c) for c in cols) + "\tmt:f:0.0\n")

    n_pairs_reported = len(reported_pairs)
    n_chains = len(records)

    # 6) qname-parity check vs reads.fasta ids.
    fasta_ids = read_fasta_ids(args.reads_fasta)
    paf_ids = set()
    for (qname, _1, _2, _3, _4, tname, *_rest) in records:
        paf_ids.add(qname)
        paf_ids.add(tname)
    present = paf_ids & fasta_ids
    missing = paf_ids - fasta_ids

    print("", flush=True)
    print("==================== NEUROSAMBLE PHASE-2 OVERLAP MAP ====================")
    print(f"n_reads              : {len(reads)}")
    print(f"n_windows            : {n_windows}")
    print(f"n_candidate_pairs    : {len(pair_anchors)}   (>=1 anchor)")
    print(f"n_pairs_reported     : {n_pairs_reported}   (>=1 surviving chain)")
    print(f"n_chains             : {n_chains}")
    print(f"thresholds           : min_chaining_score={args.min_chaining_score} "
          f"min_num_anchors={args.min_num_anchors} max_gap_bp={args.max_gap_bp} bw_bp={args.bw_bp}")
    print("-- qname parity vs reads.fasta --")
    print(f"  paf read-ids={len(paf_ids)}  present_in_fasta={len(present)}  MISSING={len(missing)}")
    if missing:
        sample = list(sorted(missing))[:5]
        print(f"  [WARN] {len(missing)} PAF ids not in reads.fasta (qname mismatch!) "
              f"e.g. {sample}")
    else:
        print("  [OK] every PAF read-id is present in reads.fasta")
    print(f"[paf] wrote {paf_path}")
    print("========================================================================")


if __name__ == "__main__":
    main()
