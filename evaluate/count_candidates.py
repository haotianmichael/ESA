#!/usr/bin/env python3
"""
count_candidates.py -- candidate read-pair statistics for the Phase-4 R9.4 run.

Quantifies how many candidate read pairs each seeding proposes and their purity,
to test whether embedding seeding yields fewer / cleaner candidates than the hash
baseline. Standalone and READ-ONLY w.r.t. the experiment: it only reads
experiments/neurosamble_phase4/{index,encode,*.paf} and writes exclusively into
--out-dir (default .../neurosamble_phase4_candidates/). It never modifies pipeline
code, the running experiment, or rawhash2.

Sub-commands
------------
neurosamble-precandidates
    The precise PRE-chain candidates (Neurosamble only): reuse the exact phase-4
    retrieval/anchoring path (same IVF index + encodings, same query params read
    from query.log) up to canonical read-pair de-dup, but STOP before colinear
    chaining. Emits (read_a, read_b, n_anchors) per canonical pair + a JSON summary.

compare
    Read-only comparison at the level each tool exposes. Truth = mm2 ava-ont
    read-pair set. Reports, for {Neurosamble pre-chain candidates, Neurosamble
    final PAF, Rawsamble final PAF}: unique pairs, #true/#false, precision, recall,
    and the records-per-pair distribution. PAFs (~10GB) are streamed line-by-line.

rawsamble-loose-proxy  (optional; NOT run by the delivered commands)
    Documented convenience that re-runs rawhash2 -x ava --min-anchors 1 --min-score 0
    into OUTDIR as a LOOSE POST-CHAIN proxy (Rawsamble has no pre-chain output).

HONESTY: rawhash2 exposes NO pre-chain candidate list (no such flag; only the
post-chain PAF). So a true pre-chain candidate count exists for Neurosamble ONLY;
the cross-tool comparison is at the final-overlap level. We do not recompile or
modify rawhash2.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the EXACT phase-4 query primitives (numpy-only import; no torch/faiss here).
from overlap_map_full import load_read_table, filter_neighbors, batched_search  # noqa: E402


def die(msg, code=2):
    print(f"[error] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def info(msg):
    print(f"[cand] {msg}", flush=True)


def canonical(a, b):
    return (a, b) if a <= b else (b, a)


# --------------------------------------------------------------------------- #
# query-param discovery (do NOT guess -- read from phase-4 query.log)
# --------------------------------------------------------------------------- #
def parse_query_log(path):
    """Pull topk/nprobe/query_batch/spk/gpu_temp_mb from the phase-4 [map] line."""
    if not os.path.exists(path):
        return {}
    pat = re.compile(r"(nprobe|topk|spk|query_batch|gpu_temp_mb)=(\d+)")
    found = {}
    with open(path) as f:
        for line in f:
            if line.startswith("[map] N_windows="):
                for k, v in pat.findall(line):
                    found[k] = int(v)
                break
    return found


# --------------------------------------------------------------------------- #
# percentiles from a value->frequency histogram (memory-light for huge counts)
# --------------------------------------------------------------------------- #
def hist_stats(hist):
    total = int(sum(hist.values()))
    if total == 0:
        return {"n": 0, "min": 0, "median": 0.0, "p90": 0.0, "max": 0, "mean": 0.0}
    keys = sorted(hist.keys())
    ssum = sum(k * hist[k] for k in keys)

    def pct(q):
        # linear-interpolated percentile over the expanded (sorted) multiset
        pos = q / 100.0 * (total - 1)
        lo = int(np.floor(pos)); hi = int(np.ceil(pos)); frac = pos - lo
        # walk cumulative to find values at rank lo and hi
        want = {lo, hi}
        vals = {}
        c = 0
        for k in keys:
            c2 = c + hist[k]
            while want and min(want) < c2:
                r = min(want); vals[r] = k; want.discard(r)
            c = c2
            if not want:
                break
        return vals[lo] * (1 - frac) + vals[hi] * frac if lo != hi else float(vals[lo])

    return {"n": total, "min": keys[0], "median": pct(50), "p90": pct(90),
            "max": keys[-1], "mean": ssum / total}


# --------------------------------------------------------------------------- #
# 1) neurosamble-precandidates
# --------------------------------------------------------------------------- #
def cmd_precandidates(args):
    os.makedirs(args.out_dir, exist_ok=True)
    index_dir = os.path.join(args.phase4_dir, "index")
    encode_dir = os.path.join(args.phase4_dir, "encode")
    for pth in (os.path.join(index_dir, "ivf.index"),
                os.path.join(index_dir, "windows.npy"),
                os.path.join(index_dir, "read_ids.txt"),
                os.path.join(encode_dir, "encode_manifest.json")):
        if not os.path.exists(pth):
            die(f"missing required phase-4 input: {pth}")

    # --- query params: read from query.log unless overridden (do not guess) ---
    qlog = parse_query_log(os.path.join(args.phase4_dir, "query.log"))
    topk = args.topk if args.topk else qlog.get("topk")
    nprobe = args.nprobe if args.nprobe else qlog.get("nprobe")
    query_batch = args.query_batch if args.query_batch else qlog.get("query_batch", 65536)
    spk = args.samples_per_kmer if args.samples_per_kmer else qlog.get("spk", 9)
    if topk is None or nprobe is None:
        die("could not determine topk/nprobe from query.log; pass --topk/--nprobe explicitly")
    info(f"query params (from query.log unless overridden): topk={topk} nprobe={nprobe} "
         f"query_batch={query_batch} spk={spk}")

    import faiss
    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    with open(os.path.join(encode_dir, "encode_manifest.json")) as f:
        manifest = json.load(f)
    D = int(manifest["D"])
    shards = manifest["shards"]

    cum = [0]
    mmaps = []
    for sh in shards:
        n = int(sh["n_rows"])
        mmaps.append(np.memmap(os.path.join(encode_dir, sh["emb_file"]),
                               dtype=np.float32, mode="r", shape=(n, D)))
        cum.append(cum[-1] + n)
    cum = np.asarray(cum)

    windows = np.load(os.path.join(index_dir, "windows.npy"))
    win_gid = np.ascontiguousarray(windows[:, 0])
    win_off = np.ascontiguousarray(windows[:, 1])
    id2name, name_rank, nsamp = load_read_table(os.path.join(index_dir, "read_ids.txt"))
    N = windows.shape[0]

    index = faiss.read_index(os.path.join(index_dir, "ivf.index"))
    try:
        index.nprobe = nprobe
    except Exception:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)

    # GPU honors CUDA_VISIBLE_DEVICES (faiss only sees visible devices). CPU = slow.
    use_gpu = False
    if args.faiss_gpu != 0:
        try:
            ngpu = faiss.get_num_gpus()
        except Exception:
            ngpu = 0
        if ngpu > 0:
            try:
                co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
                res = [faiss.StandardGpuResources() for _ in range(ngpu)]
                for r in res:
                    r.setTempMemory(int(args.gpu_temp_mb) * 1024 * 1024)
                index = faiss.index_cpu_to_gpu_multiple_py(res, index, co)
                globals()["_keep_gpu_res"] = res  # keep alive
                use_gpu = True
                info(f"FAISS on GPU (CUDA_VISIBLE_DEVICES="
                     f"{os.environ.get('CUDA_VISIBLE_DEVICES','<all>')}, ngpu={ngpu})")
                try:
                    faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
                except Exception:
                    pass
            except Exception as e:  # noqa: BLE001
                info(f"[warn] GPU move failed ({e}); using CPU IVF index")
    if not use_gpu:
        info("[warn] FAISS on CPU (no GPU / --faiss-gpu 0) -- this will be SLOW on 28M vectors")

    def shard_rows(i, j):
        s = int(np.searchsorted(cum, i, side="right") - 1)
        return np.ascontiguousarray(mmaps[s][i - cum[s]: j - cum[s]])

    boundaries = np.flatnonzero(np.diff(win_gid)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [N]])

    out_tsv = os.path.join(args.out_dir, "neurosamble_precandidates.tsv.gz")
    fout = gzip.open(out_tsv, "wt")
    fout.write("read_a\tread_b\tn_anchors\n")

    anchor_hist = Counter()   # n_anchors -> #pairs
    total_pairs = 0
    n_reads = 0
    t0 = time.time()

    # Each canonical pair (R<T) is produced exactly once -- only when querying the
    # lexicographically smaller read R (filter_neighbors keeps rank[T] > rank[R]).
    # So we can stream pairs straight to disk with O(1) memory per read.
    for i, j in zip(starts.tolist(), ends.tolist()):
        r_gid = int(win_gid[i])
        r_name = id2name[r_gid]
        if r_name is None:
            continue
        n_reads += 1
        r_rank = int(name_rank[r_gid])
        qvecs = shard_rows(i, j)
        q_offs = win_off[i:j]
        scores, ids = batched_search(index, qvecs, topk, query_batch)
        nid = ids.reshape(-1); sc = scores.reshape(-1)
        qoff = np.repeat(q_offs, topk)
        t_gid, t_off, q_kept, s_kept = filter_neighbors(
            r_gid, r_rank, nid, sc, qoff, win_gid, win_off, name_rank)
        if t_gid.size:
            order = np.argsort(t_gid, kind="stable")
            tg = t_gid[order]; to = t_off[order]; qo = q_kept[order]
            seg = np.flatnonzero(np.diff(tg)) + 1
            seg_starts = np.concatenate([[0], seg])
            seg_ends = np.concatenate([seg, [tg.size]])
            for a, b in zip(seg_starts.tolist(), seg_ends.tolist()):
                t = int(tg[a])
                # n_anchors = #unique (q_off, t_off) -- exactly what chaining consumes
                uniq = {(int(qo[k]), int(to[k])) for k in range(a, b)}
                na = len(uniq)
                ra, rb = canonical(r_name, id2name[t])
                fout.write(f"{ra}\t{rb}\t{na}\n")
                anchor_hist[na] += 1
                total_pairs += 1
        if (n_reads % 20000) == 0:
            info(f"reads={n_reads} candidate_pairs={total_pairs} "
                 f"({n_reads/max(time.time()-t0,1e-9):.0f} reads/s)")
    fout.close()

    st = hist_stats(anchor_hist)
    thresholds = {str(t): int(sum(c for v, c in anchor_hist.items() if v >= t))
                  for t in (1, 2, 5, 10)}
    summary = {
        "phase4_dir": os.path.abspath(args.phase4_dir),
        "params": {"topk": topk, "nprobe": nprobe, "query_batch": query_batch, "spk": spk},
        "backend": "gpu" if use_gpu else "cpu",
        "n_reads_queried": n_reads,
        "total_candidate_pairs": total_pairs,
        "anchors_per_pair": st,
        "pairs_with_ge_min_anchors": thresholds,
        "elapsed_sec": round(time.time() - t0, 1),
        "tsv": os.path.abspath(out_tsv),
        "note": "PRE-chain Neurosamble candidates (canonical, deduped). Rawsamble has "
                "no pre-chain equivalent.",
    }
    out_json = os.path.join(args.out_dir, "neurosamble_precandidates.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    info(f"total candidate pairs = {total_pairs}")
    info(f"anchors/pair: min={st['min']} median={st['median']:.1f} p90={st['p90']:.1f} "
         f"max={st['max']} mean={st['mean']:.2f}")
    info(f"pairs >= min_anchors: {thresholds}")
    info(f"wrote {out_tsv} and {out_json}")


# --------------------------------------------------------------------------- #
# 2) compare (streaming, read-only)
# --------------------------------------------------------------------------- #
def _open(path):
    if not os.path.exists(path):
        die(f"missing input: {path}")
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_truth_pairs(path):
    """Canonical read-pair set from an ava PAF (cols 0=query, 5=target)."""
    s = set()
    n = 0
    with _open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            a, b = c[0], c[5]
            if a == b:
                continue
            s.add(canonical(a, b))
            n += 1
            if n % 20_000_000 == 0:
                info(f"  truth: read {n} lines, {len(s)} unique pairs ...")
    return s


def source_pair_counts_paf(path):
    """dict canonical_pair -> #records, streaming a (final) PAF."""
    counts = Counter()
    n = 0
    with _open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            a, b = c[0], c[5]
            if a == b:
                continue
            counts[canonical(a, b)] += 1
            n += 1
            if n % 20_000_000 == 0:
                info(f"  {os.path.basename(path)}: read {n} records, {len(counts)} pairs ...")
    return counts


def source_pair_counts_precand(path):
    """dict canonical_pair -> n_anchors, from the precandidates TSV(.gz)."""
    counts = {}
    with _open(path) as f:
        header = f.readline()  # read_a read_b n_anchors
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            counts[canonical(c[0], c[1])] = int(c[2])
    return counts


def eval_vs_truth(name, pair_counts, truth, per_pair_label):
    uniq = set(pair_counts.keys())
    n_uniq = len(uniq)
    n_true = sum(1 for p in uniq if p in truth)
    n_false = n_uniq - n_true
    precision = n_true / n_uniq if n_uniq else 0.0
    recall = n_true / len(truth) if truth else 0.0
    dist = hist_stats(Counter(pair_counts.values()))
    return {
        "source": name,
        "unique_pairs": n_uniq,
        "true_pairs": n_true,
        "false_pairs": n_false,
        "precision": precision,
        "recall": recall,
        per_pair_label: dist,
    }


def cmd_compare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    info(f"loading truth read-pair set from {args.truth} (streaming) ...")
    truth = load_truth_pairs(args.truth)
    info(f"truth unique read pairs: {len(truth)}")

    results = {"truth": os.path.abspath(args.truth),
               "n_truth_pairs": len(truth), "sources": []}

    # (i) Neurosamble PRE-chain candidates
    info("evaluating Neurosamble pre-chain candidates ...")
    pc = source_pair_counts_precand(args.neuro_precand)
    results["sources"].append(
        eval_vs_truth("neurosamble_precandidates", pc, truth, "anchors_per_pair"))
    del pc

    # (ii) Neurosamble FINAL (post-chain) PAF
    info("evaluating Neurosamble final PAF ...")
    nf = source_pair_counts_paf(args.neuro_paf)
    results["sources"].append(
        eval_vs_truth("neurosamble_final", nf, truth, "records_per_pair"))
    del nf

    # (iii) Rawsamble FINAL (post-chain) PAF
    info("evaluating Rawsamble final PAF ...")
    rf = source_pair_counts_paf(args.rawsamble_paf)
    results["sources"].append(
        eval_vs_truth("rawsamble_final", rf, truth, "records_per_pair"))
    del rf

    out_json = os.path.join(args.out_dir, "candidate_comparison.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("Candidate read-pair comparison (canonical pairs vs mm2 ava-ont truth)")
    lines.append(f"truth unique read pairs: {len(truth)}")
    lines.append("")
    lines.append("HONESTY: rawhash2 exposes NO pre-chain candidate list (no such flag; only")
    lines.append("the post-chain PAF). The pre-chain candidate count is Neurosamble-only;")
    lines.append("the cross-tool comparison is at the final-overlap level.")
    lines.append("")
    hdr = f"{'source':30s} {'unique':>12s} {'true':>12s} {'false':>12s} {'prec':>7s} {'recall':>7s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s in results["sources"]:
        lines.append(f"{s['source']:30s} {s['unique_pairs']:12d} {s['true_pairs']:12d} "
                     f"{s['false_pairs']:12d} {s['precision']:7.4f} {s['recall']:7.4f}")
    lines.append("")
    lines.append("per-pair record/anchor distribution (min / median / p90 / max / mean):")
    for s in results["sources"]:
        d = s.get("anchors_per_pair") or s.get("records_per_pair")
        lbl = "anchors" if "anchors_per_pair" in s else "records"
        lines.append(f"  {s['source']:30s} {lbl}: {d['min']} / {d['median']:.1f} / "
                     f"{d['p90']:.1f} / {d['max']} / {d['mean']:.2f}")
    txt = "\n".join(lines) + "\n"
    out_txt = os.path.join(args.out_dir, "candidate_comparison.txt")
    with open(out_txt, "w") as f:
        f.write(txt)
    print("\n" + txt, flush=True)
    info(f"wrote {out_json} and {out_txt}")


# --------------------------------------------------------------------------- #
# 3) rawsamble-loose-proxy (optional; documented; run only when invoked)
# --------------------------------------------------------------------------- #
def cmd_rawsamble_loose_proxy(args):
    os.makedirs(args.out_dir, exist_ok=True)
    import shutil
    import subprocess
    for req in (args.rawhash2_bin, args.pore, args.blow5):
        if not req or not os.path.exists(req):
            die(f"missing required input for proxy: {req}")
    idx = os.path.join(args.out_dir, "rawsamble_loose_idx")
    paf = os.path.join(args.out_dir, "rawsamble_loose.paf")
    nice = ["nice", "-19"]
    ionice = (["ionice", "-c3"] if shutil.which("ionice") else [])
    build = nice + ionice + [args.rawhash2_bin, "-x", "ava", "-t", str(args.threads),
                             "-p", args.pore, "-d", idx, args.blow5]
    mapc = nice + ionice + [args.rawhash2_bin, "-x", "ava", "--min-anchors", "1",
                            "--min-score", "0", "-t", str(args.threads), idx, args.blow5]
    info("PROXY (loose post-chain, NOT true pre-chain candidates); low priority:")
    info("  " + " ".join(build))
    info("  " + " ".join(mapc) + f"  > {paf}")
    with open(os.path.join(args.out_dir, "rawsamble_loose_index.log"), "w") as lg:
        subprocess.run(build, check=True, stdout=lg, stderr=subprocess.STDOUT)
    with open(paf, "w") as out, open(os.path.join(args.out_dir, "rawsamble_loose_map.log"), "w") as lg:
        subprocess.run(mapc, check=True, stdout=out, stderr=lg)
    info(f"wrote {paf} (LABEL: loose post-chain proxy, not pre-chain candidates)")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Candidate read-pair statistics (read-only)")
    p.add_argument("--out-dir", default="/home/nfs/mahaotian/ESA/CALL_RSA/experiments/"
                                        "neurosamble_phase4_candidates",
                   help="all outputs go here (created); never write into phase4 dir")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("neurosamble-precandidates", help="pre-chain candidates (Neurosamble)")
    a.add_argument("--phase4-dir", required=True, help="dir with index/ encode/ query.log")
    a.add_argument("--topk", type=int, default=0, help="0 = read from query.log")
    a.add_argument("--nprobe", type=int, default=0, help="0 = read from query.log")
    a.add_argument("--query-batch", type=int, default=0, help="0 = read from query.log")
    a.add_argument("--samples-per-kmer", type=int, default=0, help="0 = read from query.log (else 9)")
    a.add_argument("--faiss-gpu", type=int, default=1, help="1 = use GPU if visible; 0 = force CPU")
    a.add_argument("--gpu-temp-mb", type=int, default=8192)
    a.add_argument("--threads", type=int, default=0)
    a.set_defaults(func=cmd_precandidates)

    c = sub.add_parser("compare", help="compare pre-chain / final / rawsamble vs truth")
    c.add_argument("--neuro-precand", required=True, help="neurosamble_precandidates.tsv(.gz)")
    c.add_argument("--neuro-paf", required=True, help="phase4 neurosamble.paf (final)")
    c.add_argument("--rawsamble-paf", required=True, help="phase4 rawsamble.paf (final)")
    c.add_argument("--truth", required=True, help="phase4 mm2_overlaps.paf (ava truth)")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("rawsamble-loose-proxy",
                       help="OPTIONAL loose post-chain proxy (labels itself as such)")
    r.add_argument("--rawhash2-bin", required=True)
    r.add_argument("--pore", required=True)
    r.add_argument("--blow5", required=True)
    r.add_argument("--threads", type=int, default=8)
    r.set_defaults(func=cmd_rawsamble_loose_proxy)
    return p.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
