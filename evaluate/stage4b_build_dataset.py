"""STAGE4b Step 1 — build the real-read fine-tuning dataset + train/val/test split.

Reads the real blow5 + minimap2-of-basecalled truth PAF (CFT073, AE014075.1) and
produces, for STAGE4b Step 2 fine-tuning:

  experiments/stage4b_data/
    train_pairs.tsv   read_id<TAB>strand<TAB>tstart<TAB>tend   (filtered positives)
    val_pairs.tsv     "                                          "
    test_reads.txt    read_id per line   (held-out, UNFILTERED except unmapped)
    split_summary.txt human-readable counts + filter breakdown + spot-check

POSITIVE-PAIR ALIGNMENT (confirmed with the user; mirrors signal_dataset._anchor):
  query side     = real read signal, trim fixed `trim_fixed` samples, then the
                   first `input_signal_len` samples (what SquiggleSeek encodes).
  reference side = pore-model expected signal of the reference window the read's
                   5' (query≈0) traverses:
                     '+' : ref[tstart : tstart+unit_length]
                     '-' : revcomp(ref[tend-unit_length : tend])
  -> Step 2 feeds query_coords[i]=tstart (PAF col 8), query_ends[i]=tend (col 9),
     query_strands[i]=strand (col 5) into SignalPairDataset, whose _anchor already
     does the '+ -> tstart' / '- -> tend-win_bp + revcomp' logic. (NOT the read's
     qstart/qend.)

FILTERS (train/val positives ONLY; test stays unfiltered except drop-unmapped):
  * primary alignment  (tp:A:P tag)
  * mapq >= --min_mapq (default 50)
  * 5' soft-clip small: qstart (PAF col 3) <= --max_qstart (default 50) for BOTH
    strands (signal sample 0 <-> read query base 0 regardless of strand, so the
    query window starts at read query≈0 and must line up with the reference window
    that begins at the alignment's query offset -> require small qstart).
  * enough signal for a full (jittered) training window:
    len_raw_signal >= trim_fixed + input_signal_len + --jitter

SPLIT: by READ-ID, 90/5/5, fixed --seed. Filtering is applied only when building
train/val positives; the test split keeps every mapped read (no cherry-picking),
and is the read set Step 3's SquiggleSeek-vs-RawHash2 head-to-head runs on.

Trim-jitter augmentation (user-required) is a Step-2 TRAINING feature (randomize
the trim start by +/- jitter so the encoder tolerates the variable adapter);
evaluation always uses the fixed trim. This builder only records `--jitter` in the
length filter + summary so Step 2 can rely on every train/val read being long
enough for the widest jitter.

DEV-ONLY: not run here. Marked-unverified spots assume the run server's pyslow5 /
pore-model behavior; confirm from the logs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for p in (str(_HERE), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dna2vec.pore_model import PoreModel  # noqa: E402
from dna2vec.signal_dataset import revcomp, preprocess_window  # noqa: E402
from upsert_signal import read_single_fasta  # noqa: E402
from real_data_eval import read_fasta_name  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="STAGE4b Step 1 — real-read dataset + split")
    p.add_argument("--truth_paf", required=True, help="minimap2-of-basecalled truth PAF (CFT073)")
    p.add_argument("--blow5", required=True, help="real reads blow5 (ecoli_R9.blow5)")
    p.add_argument("--ref", required=True, help="CFT073 reference FASTA (d2_ecoli_r94/ref.fa)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pore_model", default=None, help="ONT R9 6-mer table (else $PORE_MODEL_PATH)")
    p.add_argument("--kmer_len", type=int, default=6)
    p.add_argument("--samples_per_kmer", type=int, default=9)
    p.add_argument("--downsample_factor", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trim_fixed", type=int, default=2500)
    p.add_argument("--input_signal_len", type=int, default=2000)
    p.add_argument("--unit_length", type=int, default=300)
    p.add_argument("--max_qstart", type=int, default=50, help="max 5' soft-clip (bp), both strands")
    p.add_argument("--min_mapq", type=int, default=50)
    p.add_argument("--jitter", type=int, default=500, help="max training trim jitter (samples); sizes the length filter")
    p.add_argument("--spotcheck", type=int, default=8)
    return p.parse_args()


def parse_truth_paf(path, ref_contig):
    """read_id -> best-primary record dict {qstart,strand,tstart,tend,mapq,primary}.
    Keeps only lines whose target contig == ref_contig. 'mapped' = any kept line."""
    best = {}
    mapped = set()
    n_lines = 0
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            n_lines += 1
            qname, tname = c[0], c[5]
            if tname != ref_contig:
                continue
            try:
                qstart = int(c[2]); tstart = int(c[7]); tend = int(c[8]); mapq = int(c[11])
            except ValueError:
                continue
            strand = c[4]
            if strand not in ("+", "-"):
                continue
            primary = any(t == "tp:A:P" for t in c[12:])
            mapped.add(qname)
            rec = {"qstart": qstart, "strand": strand, "tstart": tstart,
                   "tend": tend, "mapq": mapq, "primary": primary}
            # keep the highest-mapq record per read (primary preferred on ties)
            prev = best.get(qname)
            if (prev is None or mapq > prev["mapq"]
                    or (mapq == prev["mapq"] and primary and not prev["primary"])):
                best[qname] = rec
    return best, mapped, n_lines


def scan_blow5_lengths(path, want_signals, k):
    """Single pass over the blow5. Returns ({read_id: len_raw_signal}, {id: signal}
    for the first k ids in `want_signals`). Uses only seq_reads (the API real_data_eval
    already relies on)."""
    import pyslow5
    lengths = {}
    stash = {}
    s = pyslow5.Open(path, "r")
    for rec in s.seq_reads(pA=True):
        rid = rec["read_id"]
        sig = np.asarray(rec["signal"], dtype=np.float32)
        lengths[rid] = int(sig.shape[0])
        if rid in want_signals and len(stash) < k and rid not in stash:
            stash[rid] = sig
    return lengths, stash


def _q_window(sig, trim_fixed, input_signal_len, downsample_factor):
    start = int(min(max(0, trim_fixed), max(0, len(sig) - (input_signal_len + 100))))
    seg = sig[start:]
    q, _ = preprocess_window(seg, input_signal_len, downsample_factor)
    return q


def _ref_window(reference_seq, strand, tstart, tend, unit_length, pore, args):
    L = len(reference_seq)
    if strand == "-":
        anchor = min(max(0, tend - unit_length), max(0, L - unit_length))
        bases = revcomp(reference_seq[anchor:anchor + unit_length])
    else:
        anchor = min(max(0, tstart), max(0, L - unit_length))
        bases = reference_seq[anchor:anchor + unit_length]
    ref_sig = pore.sequence_to_signal(bases)
    r, _ = preprocess_window(ref_sig, args.input_signal_len, args.downsample_factor)
    return r


def _pearson(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    min_len = args.trim_fixed + args.input_signal_len + args.jitter  # widest jittered window

    reference_seq = read_single_fasta(args.ref)
    ref_contig = read_fasta_name(args.ref)
    print(f"[stage4b] reference contig = '{ref_contig}'  ({len(reference_seq)} bp)", flush=True)

    truth, mapped, n_lines = parse_truth_paf(args.truth_paf, ref_contig)
    print(f"[stage4b] truth PAF: {n_lines} lines on '{ref_contig}'; "
          f"{len(mapped)} distinct mapped reads", flush=True)
    if not mapped:
        sys.exit("[stage4b][FATAL] no truth lines matched the reference contig — "
                 "check that truth.paf targets CFT073 (AE014075.1).")

    # filter-pass set (train/val eligibility, PAF-only checks; length added later)
    filter_pass = {rid for rid, r in truth.items()
                   if r["primary"] and r["mapq"] >= args.min_mapq and r["qstart"] <= args.max_qstart}

    lengths, stash = scan_blow5_lengths(args.blow5, filter_pass, args.spotcheck)
    universe = sorted(lengths.keys())
    print(f"[stage4b] blow5 reads = {len(universe)}", flush=True)
    if not universe:
        sys.exit("[stage4b][FATAL] no reads read from blow5.")

    # deterministic 90/5/5 split BY READ over the blow5 universe
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(universe))
    ids = [universe[i] for i in perm]
    n = len(ids); n_tr = int(n * 0.90); n_va = int(n * 0.05)
    train_split = set(ids[:n_tr]); val_split = set(ids[n_tr:n_tr + n_va]); test_split = set(ids[n_tr + n_va:])

    # ---- train/val positives: apply full filter; test: keep mapped only --------
    reasons = {"not_in_truth": 0, "not_primary": 0, "low_mapq": 0, "big_qstart": 0, "too_short": 0}

    def build_pairs(split_ids):
        pairs = []
        for rid in split_ids:
            r = truth.get(rid)
            if r is None:
                reasons["not_in_truth"] += 1; continue
            if not r["primary"]:
                reasons["not_primary"] += 1; continue
            if r["mapq"] < args.min_mapq:
                reasons["low_mapq"] += 1; continue
            if r["qstart"] > args.max_qstart:
                reasons["big_qstart"] += 1; continue
            if lengths.get(rid, 0) < min_len:
                reasons["too_short"] += 1; continue
            pairs.append((rid, r["strand"], r["tstart"], r["tend"]))
        return pairs

    train_pairs = build_pairs(train_split)
    val_pairs = build_pairs(val_split)
    test_reads = sorted(rid for rid in test_split if rid in mapped)

    def write_pairs(path, pairs):
        with open(path, "w") as f:
            f.write("read_id\tstrand\ttstart\ttend\n")
            for rid, strand, ts, te in pairs:
                f.write(f"{rid}\t{strand}\t{ts}\t{te}\n")

    write_pairs(os.path.join(args.out_dir, "train_pairs.tsv"), train_pairs)
    write_pairs(os.path.join(args.out_dir, "val_pairs.tsv"), val_pairs)
    with open(os.path.join(args.out_dir, "test_reads.txt"), "w") as f:
        f.write("\n".join(test_reads) + ("\n" if test_reads else ""))

    # ---- spot-check: aligned vs random-offset ref correlation -------------------
    pore = PoreModel(kmer_table_path=(args.pore_model or os.environ.get("PORE_MODEL_PATH")),
                     kmer_len=args.kmer_len, samples_per_kmer=args.samples_per_kmer)
    spot = []
    if pore.synthetic:
        print("[stage4b][WARN] synthetic pore model — spot-check corr is less meaningful "
              "(set --pore_model or $PORE_MODEL_PATH).", flush=True)
    train_ids_set = {rid for rid, *_ in train_pairs}
    for rid in list(stash.keys()):
        if rid not in train_ids_set:
            continue
        r = truth[rid]
        q = _q_window(stash[rid], args.trim_fixed, args.input_signal_len, args.downsample_factor)
        ra = _ref_window(reference_seq, r["strand"], r["tstart"], r["tend"], args.unit_length, pore, args)
        # random far window (>=5000 bp away, same strand) as the negative control
        far = int(rng.integers(0, max(1, len(reference_seq) - args.unit_length)))
        rr = _ref_window(reference_seq, r["strand"], far, far + args.unit_length, args.unit_length, pore, args)
        spot.append((rid, r["strand"], r["tstart"], r["tend"], r["qstart"], r["mapq"],
                     int(stash[rid].shape[0]), _pearson(q, ra), _pearson(q, rr)))
        if len(spot) >= args.spotcheck:
            break

    # ---- summary ---------------------------------------------------------------
    lines = []
    lines.append("==== STAGE4b Step 1 — dataset build summary ====")
    lines.append(f"reference contig      : {ref_contig}  ({len(reference_seq)} bp)")
    lines.append(f"blow5 reads (universe): {len(universe)}")
    lines.append(f"truth mapped reads    : {len(mapped)}")
    lines.append(f"filter-pass (PAF only): {len(filter_pass)}  "
                 f"(primary & mapq>={args.min_mapq} & qstart<={args.max_qstart})")
    lines.append(f"length filter min_len : {min_len} samples (trim {args.trim_fixed} + isl "
                 f"{args.input_signal_len} + jitter {args.jitter})")
    lines.append(f"split (90/5/5, seed {args.seed}): train={len(train_split)} val={len(val_split)} test={len(test_split)}")
    lines.append(f"POSITIVES  train={len(train_pairs)}  val={len(val_pairs)}")
    lines.append(f"TEST (mapped, unfiltered): {len(test_reads)}")
    lines.append(f"train/val drop reasons   : {reasons}")
    lines.append("")
    lines.append("spot-check (aligned vs random ref; aligned should be clearly higher):")
    lines.append("  read_id  strand  tstart  tend  qstart  mapq  siglen  corr_aligned  corr_random")
    for rid, st, ts, te, qs, mq, sl, ca, cr in spot:
        lines.append(f"  {rid}  {st}  {ts}  {te}  {qs}  {mq}  {sl}  {ca:+.3f}  {cr:+.3f}")
    if spot:
        ma = float(np.mean([s[7] for s in spot])); mr = float(np.mean([s[8] for s in spot]))
        lines.append(f"  MEAN corr_aligned={ma:+.3f}  corr_random={mr:+.3f}  (aligned >> random => alignment OK)")
    summary = "\n".join(lines)
    print(summary, flush=True)
    with open(os.path.join(args.out_dir, "split_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[stage4b] wrote train_pairs.tsv / val_pairs.tsv / test_reads.txt / split_summary.txt "
          f"-> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
