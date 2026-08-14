"""
Deterministically sample N read-ids from a Phase-4 read_ids.txt (or any file whose
lines start with, or are, read names). Used to build a coverage-controlled subset
(e.g. ~50x from the full 451x D1 set) to validate that overlap recall scales with
``topk`` when ``topk >= coverage`` -- before committing to a full-scale rerun.

Source format: the Phase-4 ``index/read_ids.txt`` lines are ``gid<TAB>name<TAB>nsamp``;
plain one-name-per-line files also work (first whitespace/tab token is taken).

Usage:
  python evaluate/sample_read_ids.py --source OUT/index/read_ids.txt \
      --n 39000 --seed 1234 --out OUT_sub/sub50_ids.txt
"""
from __future__ import annotations

import argparse
import random


def parse_args():
    p = argparse.ArgumentParser(description="Deterministically sample N read-ids")
    p.add_argument("--source", required=True, help="read_ids.txt (gid<TAB>name<TAB>nsamp) or name-per-line")
    p.add_argument("--n", type=int, required=True, help="number of read-ids to keep")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    names = []
    seen = set()
    with open(args.source) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # gid<TAB>name<TAB>nsamp  ->  name is field 2; else first token.
            parts = line.split("\t")
            name = parts[1] if len(parts) >= 2 else parts[0].split()[0]
            if name not in seen:
                seen.add(name)
                names.append(name)

    total = len(names)
    if args.n >= total:
        chosen = names
        print(f"[sample] requested {args.n} >= available {total}; keeping all", flush=True)
    else:
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(names, args.n))
    with open(args.out, "w") as f:
        for nm in chosen:
            f.write(nm + "\n")
    print(f"[sample] wrote {len(chosen)}/{total} read-ids -> {args.out} (seed={args.seed})",
          flush=True)


if __name__ == "__main__":
    main()
