"""
Neurosamble Phase 4 -- streaming IVF query + chaining -> neurosamble.paf.

Replaces the Phase-2 in-memory pair-anchor dict (O(N^2) RAM) with a per-query-read
streaming pass: memory is bounded by ONE read's anchors, not the whole N^2 pair
set. The encoder is NOT loaded here -- query vectors come straight from the encode
memmap shards.

For each query read R (windows are contiguous per read in the global row order):
  * fetch R's window vectors from its shard memmap, ``index.search(nprobe, topk)``,
  * keep neighbors whose target read T satisfies ``T != R`` AND
    ``canonical(R,T)[0] == R`` (i.e. name(R) <= name(T)) -- self-excludes and dedups
    (A,B)/(B,A) in one condition, halving work,
  * group anchors by T -> ``chain_anchors`` per (R,T) -> append surviving chains to
    the PAF (append mode; nothing accumulates across reads),
  * free R's anchor structures before the next read.

PAF format matches Phase 2: 12 std cols + ``mt:f:0.0`` tag on every line; qname/tname
are real read-ids; coords in bp = offset_samples // samples_per_kmer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from overlap_chain import chain_anchors  # noqa: E402
from overlap_probe import canonical_pair  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 streaming IVF query + chaining")
    p.add_argument("--index_dir", required=True, help="dir with ivf.index, windows.npy, read_ids.txt")
    p.add_argument("--encode_dir", required=True, help="dir with encode_manifest.json + emb shards")
    p.add_argument("--out_paf", required=True)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="faiss omp threads (0 = default)")
    p.add_argument("--samples_per_kmer", type=int, default=9)
    p.add_argument("--min_chaining_score", type=float, default=40.0)
    p.add_argument("--min_num_anchors", type=int, default=5)
    p.add_argument("--max_gap_bp", type=int, default=2500)
    p.add_argument("--bw_bp", type=int, default=5000)
    p.add_argument("--faiss_gpu", type=int, default=0,
                   help="1 = move index to GPU(s) for query (only sensible for ivfpq)")
    return p.parse_args()


def _load_read_table(path):
    """read_ids.txt lines 'gid<TAB>name<TAB>nsamp' -> (id2name, id2nsamp) dicts."""
    id2name, id2nsamp = {}, {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gid = int(parts[0])
            id2name[gid] = parts[1]
            id2nsamp[gid] = int(parts[2])
    return id2name, id2nsamp


def main():
    args = parse_args()
    spk = max(1, int(args.samples_per_kmer))

    import faiss

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    with open(os.path.join(args.encode_dir, "encode_manifest.json")) as f:
        manifest = json.load(f)
    D = int(manifest["D"])
    win_bp = max(1, int(manifest["win"]) // spk)
    shards = manifest["shards"]

    # Global-row -> shard memmap mapping (row order == shard concatenation order).
    cum = [0]
    mmaps = []
    for sh in shards:
        n = int(sh["n_rows"])
        mm = np.memmap(os.path.join(args.encode_dir, sh["emb_file"]),
                       dtype=np.float32, mode="r", shape=(n, D))
        mmaps.append(mm)
        cum.append(cum[-1] + n)
    cum = np.asarray(cum)

    windows = np.load(os.path.join(args.index_dir, "windows.npy"))   # [N,2] (gid, off)
    id2name, id2nsamp = _load_read_table(os.path.join(args.index_dir, "read_ids.txt"))
    N = windows.shape[0]

    index = faiss.read_index(os.path.join(args.index_dir, "ivf.index"))
    try:
        index.nprobe = args.nprobe
    except Exception:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
    if args.faiss_gpu:
        print("[map] moving index to GPU(s) for query", flush=True)
        index = faiss.index_cpu_to_all_gpus(index)
    print(f"[map] N_windows={N} nprobe={args.nprobe} topk={args.topk} spk={spk} "
          f"win_bp={win_bp} thresholds(mcs={args.min_chaining_score},mna={args.min_num_anchors},"
          f"gap={args.max_gap_bp},bw={args.bw_bp})", flush=True)

    def shard_rows(i, j):
        """Vectors for global rows [i, j) -- a single read is within one shard."""
        s = int(np.searchsorted(cum, i, side="right") - 1)
        return np.ascontiguousarray(mmaps[s][i - cum[s]: j - cum[s]])

    # Per-read groups = maximal runs of equal gid in row order.
    gids = windows[:, 0]
    boundaries = np.flatnonzero(np.diff(gids)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [N]])

    out = open(args.out_paf, "w")
    n_reads = 0
    n_pairs = 0
    n_chains = 0
    t0 = time.time()

    for i, j in zip(starts.tolist(), ends.tolist()):
        r_gid = int(gids[i])
        r_name = id2name.get(r_gid)
        if r_name is None:
            continue
        n_reads += 1
        qvecs = shard_rows(i, j)
        q_offs = windows[i:j, 1]
        scores, ids = index.search(qvecs, args.topk)

        anchors_by_T = {}   # t_gid -> {(q_off, t_off): score}
        for qr in range(qvecs.shape[0]):
            q_off = int(q_offs[qr])
            for score, nid in zip(scores[qr], ids[qr]):
                if nid < 0:
                    continue
                t_gid = int(windows[nid, 0])
                if t_gid == r_gid:
                    continue
                t_name = id2name.get(t_gid)
                if t_name is None or r_name > t_name:   # keep only canonical R<=T
                    continue
                t_off = int(windows[nid, 1])
                d = anchors_by_T.setdefault(t_gid, {})
                key = (q_off, t_off)
                sc = float(score)
                if key not in d or sc > d[key]:
                    d[key] = sc

        if anchors_by_T:
            qlen = max(1, id2nsamp.get(r_gid, spk) // spk)
            for t_gid, anchors in anchors_by_T.items():
                chains = chain_anchors(
                    anchors, spk, args.min_num_anchors, args.min_chaining_score,
                    max_gap_bp=args.max_gap_bp, bw_bp=args.bw_bp,
                )
                if not chains:
                    continue
                t_name = id2name[t_gid]
                tlen = max(1, id2nsamp.get(t_gid, spk) // spk)
                wrote_pair = False
                for ch in chains:
                    qs = max(0, min(ch.q_start, qlen)); qe = max(0, min(ch.q_end, qlen))
                    ts = max(0, min(ch.t_start, tlen)); te = max(0, min(ch.t_end, tlen))
                    if qe <= qs or te <= ts:
                        continue
                    span = qe - qs
                    mapq = min(60, max(1, int(round(ch.score / win_bp))))
                    cols = [r_name, qlen, qs, qe, "+", t_name, tlen, ts, te, span, span, mapq]
                    out.write("\t".join(str(c) for c in cols) + "\tmt:f:0.0\n")
                    n_chains += 1
                    wrote_pair = True
                if wrote_pair:
                    n_pairs += 1

        del anchors_by_T
        if (n_reads % 20000) == 0:
            dt = time.time() - t0
            print(f"[map] reads={n_reads} pairs={n_pairs} chains={n_chains} "
                  f"({n_reads/max(dt,1e-9):.0f} reads/s)", flush=True)

    out.close()
    query_sec = time.time() - t0

    import resource
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB on Linux

    stats = {
        "n_reads": n_reads, "n_windows": int(N),
        "n_pairs_reported": n_pairs, "n_chains": n_chains,
        "query_sec": round(query_sec, 1), "peak_rss_gb": round(peak_rss_gb, 2),
        "nprobe": args.nprobe, "topk": args.topk,
    }
    with open(os.path.join(args.index_dir, "query_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[map] DONE reads={n_reads} windows={N} pairs={n_pairs} chains={n_chains} "
          f"query={query_sec:.1f}s peak_rss={peak_rss_gb:.2f}GB -> {args.out_paf}", flush=True)


if __name__ == "__main__":
    main()
