"""
Reproduce the Phase-1 seed=1234 read-id subset as a plain id list.

Phase 1 (``overlap_probe.py``) chose its 4000-read subset deterministically but
did not persist the ids. STAGE A of the Phase-2 head-to-head needs that exact
fixed subset (to build a k=0 lossless subset blow5 and a subset FASTA). This
helper re-runs the identical Phase-1 selection -- same eligibility ('+'-strand
reads present in the ref PAF), same ``sorted`` + ``random.Random(seed).sample``
-- and writes one read-id per line. Pure reuse of the Phase-1 logic; nothing in
Phase 1 is modified.

Usage:
  python evaluate/emit_subset_ids.py \
      --real_reads reads.blow5 --ref_truth_paf truth.paf \
      --limit 4000 --seed 1234 --out subset_ids.txt
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from overlap_probe import parse_ref_truth_paf  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Emit the Phase-1 seed=N read-id subset.")
    p.add_argument("--real_reads", required=True)
    p.add_argument("--ref_truth_paf", required=True)
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from real_data_eval import read_blow5

    all_reads = read_blow5(args.real_reads, limit=None)
    reads_by_id = {r.id: r for r in all_reads}
    read_info = parse_ref_truth_paf(args.ref_truth_paf)

    # Identical to overlap_probe.main(): eligible = '+'-strand reads present in
    # the ref PAF, then a seeded sample over the sorted id list.
    eligible = [rid for rid in reads_by_id if read_info.get(rid, (None, None))[1] == "+"]
    eligible_sorted = sorted(eligible)
    rng = random.Random(args.seed)
    if args.limit and len(eligible_sorted) > args.limit:
        chosen = set(rng.sample(eligible_sorted, args.limit))
    else:
        chosen = set(eligible_sorted)
    ids = [rid for rid in eligible_sorted if rid in chosen]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for rid in ids:
            f.write(rid + "\n")
    print(f"[subset] wrote {len(ids)} read-ids -> {args.out} "
          f"(seed={args.seed} limit={args.limit} eligible={len(eligible_sorted)})", flush=True)


if __name__ == "__main__":
    main()
