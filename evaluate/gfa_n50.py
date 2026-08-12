"""
Tiny N50 helper for GFA unitig contiguity (Neurosamble Phase 2 head-to-head).

Reads a miniasm GFA, takes unitig lengths from each ``S`` line's ``LN:i:`` tag
(falling back to the sequence length when the tag is absent and the sequence is
present), and prints n_unitigs, total length, max, and N50. Complements the
RawHash ``compute_aun.py`` (area-under-Nx) with a single headline number.

Usage:  python evaluate/gfa_n50.py <assembly.gfa>
"""
from __future__ import annotations

import sys
from typing import List


def unitig_lengths(gfa_path: str) -> List[int]:
    lengths: List[int] = []
    with open(gfa_path) as f:
        for line in f:
            if not line.startswith("S"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            seq = parts[2]
            ln = None
            for tag in parts[3:]:
                if tag.startswith("LN:i:"):
                    try:
                        ln = int(tag[5:])
                    except ValueError:
                        ln = None
                    break
            if ln is None and seq and seq != "*":
                ln = len(seq)
            if ln is not None and ln > 0:
                lengths.append(ln)
    return lengths


def n50(lengths: List[int]) -> int:
    if not lengths:
        return 0
    s = sorted(lengths, reverse=True)
    total = sum(s)
    acc = 0
    for length in s:
        acc += length
        if acc * 2 >= total:
            return length
    return s[-1]


def main():
    if len(sys.argv) != 2:
        print("usage: python evaluate/gfa_n50.py <assembly.gfa>", file=sys.stderr)
        sys.exit(2)
    gfa = sys.argv[1]
    lengths = unitig_lengths(gfa)
    total = sum(lengths)
    mx = max(lengths) if lengths else 0
    print(f"gfa={gfa}")
    print(f"n_unitigs={len(lengths)} total_bp={total} max_bp={mx} N50={n50(lengths)}")


if __name__ == "__main__":
    main()
