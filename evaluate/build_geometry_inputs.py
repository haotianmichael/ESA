#!/usr/bin/env python3
"""
build_geometry_inputs.py -- turn a Phase-4 encode/+index/ run into the aligned
(embeddings, metadata) pair that plot_embedding_geometry.py consumes.

Phase 4 stores, per full run:
  encode/encode_manifest.json + embeddings_shard*.f32   (float32 [n_rows, D], L2-normed)
  index/windows.npy                                     (int64 [N,2] = gid, offset_samples)
  index/read_ids.txt                                    (lines "gid<TAB>name<TAB>nsamp")
where the global row order is shard0 rows ++ shard1 rows ++ ... (== windows.npy order).

This script selects a set of window rows and writes:
  --out_emb  : float32 .npy [M, D]  (rows gathered from the shard memmaps)
  --out_meta : TSV with columns  read_id<TAB>sample_offset  (row-aligned with out_emb)

IMPORTANT — pick reads, not random windows, for a meaningful retrieval recall.
recall@k needs a read's true overlapping neighbours to be present in the index. A
uniform random window subsample destroys that density (recall -> ~0). So select a
subset of READS and keep ALL their windows: use --read_ids <file of names>, or
--sample_reads N (keeps every window of N randomly chosen reads). --limit_windows
is offered only for the figure/Spearman (not recall).

Example:
  python evaluate/build_geometry_inputs.py \
      --encode_dir $FULL/encode --index_dir $FULL/index \
      --sample_reads 40000 --seed 0 \
      --out_emb $FULL/geom_emb.npy --out_meta $FULL/geom_meta.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def die(msg, code=2):
    print(f"[error] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def parse_args():
    p = argparse.ArgumentParser(description="Build aligned embeddings+metadata for Fig.2")
    p.add_argument("--encode_dir", required=True, help="Phase-4 encode/ (manifest + shard .f32)")
    p.add_argument("--index_dir", required=True, help="Phase-4 index/ (windows.npy, read_ids.txt)")
    p.add_argument("--out_emb", required=True, help="output float32 .npy [M, D]")
    p.add_argument("--out_meta", required=True, help="output TSV: read_id<TAB>sample_offset")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--read_ids", default=None, help="file of read NAMES to keep (all their windows)")
    g.add_argument("--sample_reads", type=int, default=0, help="keep all windows of N random reads")
    g.add_argument("--limit_windows", type=int, default=0,
                   help="random window cap (figure/Spearman only -- breaks recall density)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    man_path = os.path.join(args.encode_dir, "encode_manifest.json")
    win_path = os.path.join(args.index_dir, "windows.npy")
    rid_path = os.path.join(args.index_dir, "read_ids.txt")
    for pth in (man_path, win_path, rid_path):
        if not os.path.exists(pth):
            die(f"missing required input: {pth}")

    with open(man_path) as f:
        man = json.load(f)
    D = int(man["D"])
    shards = man["shards"]

    windows = np.load(win_path)  # [N, 2] = (gid, offset)
    if windows.ndim != 2 or windows.shape[1] != 2:
        die(f"windows.npy must be [N,2]; got {windows.shape}")
    N = windows.shape[0]
    gid = windows[:, 0]
    off = windows[:, 1]

    # gid -> name
    gid2name = {}
    with open(rid_path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2:
                gid2name[int(c[0])] = c[1]
    if not gid2name:
        die(f"no gid->name entries parsed from {rid_path}")

    rng = np.random.default_rng(args.seed)

    # ---- select global window rows -------------------------------------------
    if args.read_ids:
        if not os.path.exists(args.read_ids):
            die(f"--read_ids not found: {args.read_ids}")
        keep = {ln.strip() for ln in open(args.read_ids) if ln.strip() and not ln.startswith("#")}
        names_per_row = np.array([gid2name.get(int(g), "") for g in gid], dtype=object)
        sel = np.flatnonzero(np.array([nm in keep for nm in names_per_row]))
        info_sel = f"read_ids file ({len(keep)} names)"
    elif args.sample_reads > 0:
        all_names = np.array(sorted(set(gid2name.values())))
        n = min(args.sample_reads, len(all_names))
        chosen = set(rng.choice(all_names, size=n, replace=False).tolist())
        sel = np.flatnonzero(np.array([gid2name.get(int(g), "") in chosen for g in gid]))
        info_sel = f"{n} random reads (all their windows)"
    elif args.limit_windows > 0:
        m = min(args.limit_windows, N)
        sel = np.sort(rng.choice(N, size=m, replace=False))
        info_sel = f"{m} random windows (figure/Spearman only)"
    else:
        sel = np.arange(N)
        info_sel = "ALL windows (no subset)"

    if sel.size == 0:
        die("selection is empty -- do the --read_ids names match read_ids.txt column 2?")
    sel = np.sort(sel)
    print(f"[build] selecting {sel.size} / {N} windows via {info_sel}", flush=True)

    # ---- gather embeddings from shard memmaps (global row order) --------------
    out = np.empty((sel.size, D), dtype=np.float32)
    cum = 0
    filled = 0
    for sh in shards:
        n_rows = int(sh["n_rows"])
        lo, hi = cum, cum + n_rows
        mask = (sel >= lo) & (sel < hi)
        rows_here = np.flatnonzero(mask)
        if rows_here.size:
            local = sel[rows_here] - lo  # sorted -> fast memmap reads
            mm = np.memmap(os.path.join(args.encode_dir, sh["emb_file"]),
                           dtype=np.float32, mode="r", shape=(n_rows, D))
            out[rows_here] = np.asarray(mm[local])
            filled += rows_here.size
            del mm
        cum = hi
    if filled != sel.size:
        die(f"gathered {filled} rows but selected {sel.size}; shard n_rows sum "
            f"({cum}) may not match windows.npy ({N})")

    np.save(args.out_emb, out)
    with open(args.out_meta, "w") as f:
        f.write("read_id\tsample_offset\n")
        for r in sel:
            f.write(f"{gid2name.get(int(gid[r]), '')}\t{int(off[r])}\n")

    n_reads = len({gid2name.get(int(gid[r]), "") for r in sel})
    print(f"[build] wrote {args.out_emb}  shape=({sel.size}, {D}) float32", flush=True)
    print(f"[build] wrote {args.out_meta}  ({sel.size} rows, {n_reads} reads)", flush=True)
    print("[build] next: feed these to plot_embedding_geometry.py "
          "(--embeddings out_emb.npy --metadata out_meta.tsv)", flush=True)


if __name__ == "__main__":
    main()
