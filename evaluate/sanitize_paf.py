"""
Sanitize an overlap PAF before feeding it to miniasm.

miniasm aborts (SIGABRT) on degenerate/malformed overlap lines -- e.g. self-hits
(qname==tname), zero-length or reversed intervals, or reads not present in the
reads FASTA. rawhash2's ava PAF triggers this (its .gfa came out empty), while
minimap2/Neurosamble PAFs happen not to. This filter drops such lines uniformly so
every tool's assembly is built on the same clean footing.

Keeps a line iff: >=12 tab cols, qname!=tname, integer coords with
0<=qs<qe<=qlen and 0<=ts<te<=tlen, strand in {+,-}, and (if --reads_fasta given)
both read-ids present in the FASTA. Original line (incl. tags) is written verbatim.

Usage:
  python evaluate/sanitize_paf.py --in_paf in.paf --out_paf out.clean.paf [--reads_fasta reads.fa]
"""
from __future__ import annotations

import argparse


def read_fasta_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def parse_args():
    p = argparse.ArgumentParser(description="Sanitize an overlap PAF for miniasm")
    p.add_argument("--in_paf", required=True)
    p.add_argument("--out_paf", required=True)
    p.add_argument("--reads_fasta", default=None,
                   help="if given, drop lines whose qname/tname is not in the FASTA")
    return p.parse_args()


def sanitize(in_paf, out_paf, fasta_ids=None):
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
            if not (0 <= qs < qe <= qlen and 0 <= ts < te <= tlen):
                dropped += 1
                continue
            if fasta_ids is not None and (qname not in fasta_ids or tname not in fasta_ids):
                dropped += 1
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    return kept, dropped


def main():
    args = parse_args()
    fasta_ids = read_fasta_ids(args.reads_fasta) if args.reads_fasta else None
    kept, dropped = sanitize(args.in_paf, args.out_paf, fasta_ids)
    print(f"[sanitize] {args.in_paf}: kept={kept} dropped={dropped} -> {args.out_paf}", flush=True)


if __name__ == "__main__":
    main()
