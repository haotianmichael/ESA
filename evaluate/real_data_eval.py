"""Real-data head-to-head entry (no squigulator).

Reads an EXTERNAL blow5 + EXTERNAL reference, builds the SquiggleSeek index with a
trained encoder (``--load_encoder``), retrieves top-1, and writes
``squiggleseek_real.paf``. Real read ids carry no coordinates, so ground truth
follows RawHash's native convention — minimap2 of basecalled reads — or a
user-supplied PAF (``--truth_paf``). Everything lands in ``--out_dir`` so RawHash2
and ``rawhash_compare.py`` score the identical inputs.

    minimap2-of-basecalled truth is RawHash's own setup (basecalling defines
    perfection). It is UNFAVOURABLE to SquiggleSeek but fair — reported as-is.

The heavy deps (torch, pyslow5, the encoder) are imported inside ``main`` so the
PAF/minimap2 helpers below stay importable for smoke tests.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


# --------------------------------------------------------------------------- #
# torch-free helpers (truth from minimap2, PAF normalization) — unit-testable
# --------------------------------------------------------------------------- #
def keep_primary_paf(raw_paf: str, out_paf: str) -> int:
    """Keep the highest-MAPQ (primary) minimap2 alignment per read; write 12 cols.

    minimap2's default output IS PAF (cols: qname qlen qs qe strand tname tlen ts
    te nmatch alen mapq). We keep one line per query so the truth is unambiguous.
    """
    best = {}
    with open(raw_paf) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            q = c[0]
            try:
                mapq = int(c[11])
            except ValueError:
                mapq = 0
            if q not in best or mapq > best[q][0]:
                best[q] = (mapq, c[:12])
    with open(out_paf, "w") as f:
        for q, (_mq, c) in best.items():
            f.write("\t".join(c) + "\n")
    return len(best)


def minimap2_truth(reference_fasta: str, fastq: str, out_paf: str,
                   minimap2_bin: str = "minimap2", preset: str = "map-ont",
                   threads: int = 8) -> int:
    """Map basecalled reads with minimap2 (PAF out) and keep the primary per read."""
    raw = out_paf + ".raw"
    with open(raw, "w") as f:
        subprocess.run([minimap2_bin, "-x", preset, "-t", str(threads),
                        reference_fasta, fastq],
                       stdout=f, stderr=subprocess.DEVNULL, check=True)
    return keep_primary_paf(raw, out_paf)


# --------------------------------------------------------------------------- #
def read_blow5(path: str, limit=None):
    """Read (read_id, signal) records from a blow5/slow5 into lightweight reads."""
    import numpy as np
    import pyslow5

    class RealRead:
        __slots__ = ("id", "signal", "reference_name", "reference_start",
                     "reference_end", "strand")

        def __init__(self, rid, signal, ref_name):
            self.id = rid
            self.signal = signal
            self.reference_name = ref_name
            self.reference_start = None      # unknown for real reads
            self.reference_end = None
            self.strand = None

    s = pyslow5.Open(path, "r")
    reads = []
    for rec in s.seq_reads(pA=True):
        sig = np.asarray(rec["signal"], dtype=np.float32)
        reads.append(RealRead(rec["read_id"], sig, None))
        if limit and len(reads) >= limit:
            break
    return reads


def parse_args():
    p = argparse.ArgumentParser(description="SquiggleSeek real-data head-to-head entry")
    p.add_argument("--real_reads", required=True, help="external blow5/slow5")
    p.add_argument("--real_reference", required=True, help="external reference FASTA (single record)")
    p.add_argument("--load_encoder", required=True, help="trained encoder (encoder_final_mamba_v3.pt)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pore_model", default=None, help="ONT R9 6-mer table (else $PORE_MODEL_PATH)")
    p.add_argument("--kmer_len", type=int, default=6)
    p.add_argument("--samples_per_kmer", type=int, default=9)
    p.add_argument("--unit_length", type=int, default=300)
    p.add_argument("--overlap", type=int, default=285, help="index tiling overlap; stride = unit_length - overlap")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--faiss_cpu", type=int, default=1, help="1 = FAISS index on CPU (large genomes)")
    p.add_argument("--limit", type=int, default=0, help="only the first N reads (0 = all); use for smoke tests")
    # ground truth: one of the two
    p.add_argument("--truth_paf", default=None, help="ready-made truth PAF (skip basecalling)")
    p.add_argument("--basecall_cmd", default=None,
                   help="shell template with {blow5} and {fastq}, e.g. buttery-eel/dorado")
    p.add_argument("--minimap2_bin", default="minimap2")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.join(os.path.dirname(here), "src"))

    import torch
    from dna2vec.pore_model import PoreModel
    from signal_faiss_store import SignalFaissStore
    from upsert_signal import read_single_fasta, build_signal_reference_index
    from paf_io import write_mapping_paf, mapq_from_scores
    from pilot_recall import load_encoder

    device = args.device if torch.cuda.is_available() else "cpu"
    store_device = "cpu" if args.faiss_cpu else device

    # 1) real reads + real reference
    reads = read_blow5(args.real_reads, limit=(args.limit or None))
    reference_seq = read_single_fasta(args.real_reference)
    ref_name = "ref"
    for r in reads:
        r.reference_name = ref_name
    print(f"[real] reads          = {args.real_reads}  ({len(reads)} loaded)", flush=True)
    print(f"[real] reference      = {args.real_reference}  ({len(reference_seq)} bp)", flush=True)

    pore = PoreModel(kmer_table_path=args.pore_model, kmer_len=args.kmer_len,
                     samples_per_kmer=args.samples_per_kmer)
    if pore.synthetic:
        print("[real][WARN] synthetic pore model (set --pore_model or $PORE_MODEL_PATH for real runs)", flush=True)

    # 2) build the SquiggleSeek index with the trained encoder (no training)
    model, _cfg = load_encoder(args.load_encoder, device, args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    store = SignalFaissStore(signal_model=model, index_name="real-data", device=store_device)
    store.drop_table()
    store = SignalFaissStore(signal_model=model, index_name="real-data", device=store_device)
    stride = max(1, args.unit_length - args.overlap)
    build_signal_reference_index(reference_seq, pore, model, store,
                                 unit_length=args.unit_length, stride=stride, both_strands=True)

    # 3) retrieve top-1 -> squiggleseek_real.paf (reported coord + strand + cosine)
    results = store.query_batch([r.signal for r in reads], [None] * len(reads), top_k=1)
    quads = []
    for r, res in zip(reads, results):
        if not res["matches"]:
            continue
        m0 = res["matches"][0]
        quads.append((r, m0["metadata"]["coord"], m0["metadata"].get("strand", "+"),
                      float(m0["score"])))
    mapqs = mapq_from_scores([s for *_, s in quads])
    entries = [(r, pos, mq, strand, score)
               for (r, pos, strand, score), mq in zip(quads, mapqs)]
    ss_paf = os.path.join(args.out_dir, "squiggleseek_real.paf")
    write_mapping_paf(entries, reference_seq, args.samples_per_kmer, ss_paf)

    # 4) ground truth: user PAF, else minimap2-of-basecalled (RawHash's convention)
    truth_paf = os.path.join(args.out_dir, "ground_truth_real.paf")
    if args.truth_paf:
        shutil.copy(args.truth_paf, truth_paf)
        truth_src = f"user PAF ({args.truth_paf})"
    elif args.basecall_cmd:
        fastq = os.path.join(args.out_dir, "reads.basecalled.fastq")
        cmd = args.basecall_cmd.format(blow5=args.real_reads, fastq=fastq)
        print(f"[real] basecalling: {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=True)
        n = minimap2_truth(args.real_reference, fastq, truth_paf, args.minimap2_bin)
        truth_src = f"minimap2 -x map-ont of basecalled reads ({n} mapped) [RawHash convention]"
    else:
        sys.exit("need --truth_paf OR --basecall_cmd for ground truth")

    # 5) stage ref + blow5 so RawHash2 runs on the identical inputs
    ref_out = os.path.join(args.out_dir, "ref.fasta")
    shutil.copy(args.real_reference, ref_out)
    blow5_out = os.path.join(args.out_dir, "reads.blow5")
    if os.path.abspath(args.real_reads) != os.path.abspath(blow5_out):
        shutil.copy(args.real_reads, blow5_out)

    print("\n==== real-data inputs (archive these) ====", flush=True)
    print(f"  reads.blow5   = {blow5_out}", flush=True)
    print(f"  ref.fasta     = {ref_out}", flush=True)
    print(f"  truth source  = {truth_src}", flush=True)
    print(f"  squiggleseek  = {ss_paf}  ({len(entries)}/{len(reads)} mapped)", flush=True)
    print(f"  ground_truth  = {truth_paf}", flush=True)
    print("  next: run RawHash2 on reads.blow5, then rawhash_compare.py with ground_truth_real.paf",
          flush=True)


if __name__ == "__main__":
    main()
