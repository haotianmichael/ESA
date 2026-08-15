"""
Sanitize an overlap PAF before feeding it to miniasm.

Two failure modes are handled:

1. Degenerate/malformed lines -- self-hits (qname==tname), zero-length or reversed
   intervals, <12 cols, non-numeric coords, or a strand that is not +/-.

2. Length inconsistency with the reads FASTA. miniasm (`-f reads.fasta`) treats the
   FASTA sequence length as authoritative and SIGABRTs when a PAF read's length /
   coordinates disagree with it. Signal-derived overlappers (Neurosamble uses
   ``signal_len // samples_per_kmer``; rawhash2 estimates bases from signal) emit
   qlen/tlen that differ from the basecalled read length -- which is exactly why
   both tools' .gfa came out empty while minimap2's (lengths straight from the
   FASTA) assembled fine. When ``--reads_fasta`` is given we therefore REWRITE each
   line's qlen/tlen to the FASTA base length and rescale the coordinates
   proportionally into that base space, so the PAF is fully consistent with ``-f``.

This only affects the assembly input (the ``*.clean.paf``); the overlap P/R/F1 is
scored on the original PAF, so those numbers are untouched.

Usage:
  python evaluate/sanitize_paf.py --in_paf in.paf --out_paf out.clean.paf [--reads_fasta reads.fa]
"""
from __future__ import annotations

import argparse


def read_fasta_lengths(path):
    """name -> sequence length in bases (first header token as the id)."""
    lengths = {}
    name = None
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = n
                name = line[1:].split()[0]
                n = 0
            else:
                n += len(line.strip())
    if name is not None:
        lengths[name] = n
    return lengths


def parse_args():
    p = argparse.ArgumentParser(description="Sanitize an overlap PAF for miniasm")
    p.add_argument("--in_paf", required=True)
    p.add_argument("--out_paf", required=True)
    p.add_argument("--reads_fasta", default=None,
                   help="rewrite qlen/tlen to FASTA base lengths + rescale coords "
                        "(and drop reads absent from the FASTA)")
    return p.parse_args()


def _rescale(x, paf_len, fasta_len):
    if paf_len <= 0:
        return 0
    v = int(round(x * fasta_len / paf_len))
    return max(0, min(v, fasta_len))


def sanitize(in_paf, out_paf, fasta_lengths=None):
    kept = dropped = 0
    with open(in_paf) as fin, open(out_paf, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                dropped += 1
                continue
            qname, tname, strand = f[0], f[5], f[4]
            if qname == tname or strand not in ("+", "-"):
                dropped += 1
                continue
            try:
                qlen, qs, qe = int(f[1]), int(f[2]), int(f[3])
                tlen, ts, te = int(f[6]), int(f[7]), int(f[8])
            except ValueError:
                dropped += 1
                continue

            if fasta_lengths is not None:
                fq = fasta_lengths.get(qname)
                ft = fasta_lengths.get(tname)
                if fq is None or ft is None or qlen <= 0 or tlen <= 0:
                    dropped += 1
                    continue
                qs, qe = _rescale(qs, qlen, fq), _rescale(qe, qlen, fq)
                ts, te = _rescale(ts, tlen, ft), _rescale(te, tlen, ft)
                qlen, tlen = fq, ft
                f[1], f[2], f[3] = str(qlen), str(qs), str(qe)
                f[6], f[7], f[8] = str(tlen), str(ts), str(te)

            if not (0 <= qs < qe <= qlen and 0 <= ts < te <= tlen):
                dropped += 1
                continue
            fout.write("\t".join(f) + "\n")
            kept += 1
    return kept, dropped


def main():
    args = parse_args()
    fasta_lengths = read_fasta_lengths(args.reads_fasta) if args.reads_fasta else None
    kept, dropped = sanitize(args.in_paf, args.out_paf, fasta_lengths)
    print(f"[sanitize] {args.in_paf}: kept={kept} dropped={dropped} -> {args.out_paf}"
          + (" (lengths/coords rescaled to FASTA)" if fasta_lengths else ""), flush=True)


if __name__ == "__main__":
    main()
