#!/usr/bin/env python3
"""
plot_embedding_geometry.py -- Neurosamble paper Fig. 2 ("embedding geometry").

Shows that the frozen encoder's window embeddings form an order-preserving manifold
along the genome coordinate (a color-graded loop for a circular bacterial genome),
and produces the two quantitative numbers the paper text cites:

  * Spearman rho between embedding (cosine) distance and reference-coordinate
    distance over random same-contig window pairs           -> "rho = [.]"
  * top-k retrieval recall of a true neighbouring window     -> "[.]%"

It does NOT touch encode/index/query or any pipeline logic; it only reads the
window embeddings + per-window metadata + the truth PAF you already have. No data
is fabricated: any missing/mis-shaped input is a loud error, never a fallback.

Example
-------
  python evaluate/plot_embedding_geometry.py \
      --embeddings   /path/embeddings.f32  --emb-format memmap --dim 384 --dtype float32 \
      --metadata     /path/windows.tsv     --read-id-col read_id --offset-col sample_offset \
      --truth        /path/truth.paf       --truth-format paf   --rho 9 \
      --genome-len   5231428 --circular \
      --normalize --method umap --use-faiss \
      --out-fig      geometry.pdf --out-metrics metrics.json

  # .npy embeddings + parquet metadata, t-SNE:
  python evaluate/plot_embedding_geometry.py \
      --embeddings emb.npy --metadata windows.parquet \
      --truth truth.paf --genome-len 5231428 --method tsne \
      --out-fig geometry.pdf --out-metrics metrics.json

CHECK THESE for your data:
  --read-id-col / --offset-col      column names in --metadata
  --rho                             samples-per-kmer used to build the windows (default 9)
  --emb-format / --dim / --dtype    only needed for raw memmap (not .npy)
  --truth-*-col                     only for --truth-format tsv (PAF is by column index)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def die(msg: str, code: int = 2):
    print(f"[error] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def info(msg: str):
    print(f"[geom] {msg}", flush=True)


def l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_embeddings(path, emb_format, dim, dtype):
    if not os.path.exists(path):
        die(f"--embeddings not found: {path}")
    fmt = emb_format
    if fmt == "auto":
        fmt = "npy" if path.endswith(".npy") else "memmap"

    if fmt == "npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            die(f"--embeddings .npy must be 2D (n_windows, dim); got shape {arr.shape}")
        if dim and arr.shape[1] != dim:
            die(f"--dim={dim} but .npy has {arr.shape[1]} columns")
        return arr, arr.shape[0], arr.shape[1]

    # raw memmap: need dim + dtype, infer n from file size
    if not dim:
        die("--emb-format memmap requires --dim")
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        die(f"--dtype '{dtype}' is not a valid numpy dtype")
    itemsize = np_dtype.itemsize
    nbytes = os.path.getsize(path)
    if nbytes % (dim * itemsize) != 0:
        die(f"memmap size {nbytes} not divisible by dim*itemsize={dim*itemsize}; "
            f"check --dim/--dtype")
    n = nbytes // (dim * itemsize)
    arr = np.memmap(path, dtype=np_dtype, mode="r", shape=(n, dim))
    return arr, n, dim


def load_metadata(path, read_id_col, offset_col, n_windows):
    if not os.path.exists(path):
        die(f"--metadata not found: {path}")
    ext = os.path.splitext(path)[1].lower()

    read_ids = offsets = None
    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.dtype.names is None:
            die(".npy metadata must be a structured array with named fields "
                f"(need '{read_id_col}' and '{offset_col}'); got plain array {arr.shape}")
        for c in (read_id_col, offset_col):
            if c not in arr.dtype.names:
                die(f"metadata .npy missing field '{c}'; has {arr.dtype.names}")
        read_ids = np.asarray(arr[read_id_col]).astype(str)
        offsets = np.asarray(arr[offset_col])
    elif ext in (".parquet", ".pq"):
        try:
            import pandas as pd
        except ImportError:
            die("reading .parquet needs pandas (+pyarrow); pip install pandas pyarrow")
        df = pd.read_parquet(path)
        for c in (read_id_col, offset_col):
            if c not in df.columns:
                die(f"metadata missing column '{c}'; has {list(df.columns)}")
        read_ids = df[read_id_col].to_numpy().astype(str)
        offsets = df[offset_col].to_numpy()
    elif ext in (".tsv", ".csv", ".txt"):
        sep = "," if ext == ".csv" else "\t"
        try:
            import pandas as pd
            df = pd.read_csv(path, sep=sep)
            for c in (read_id_col, offset_col):
                if c not in df.columns:
                    die(f"metadata missing column '{c}'; has {list(df.columns)}")
            read_ids = df[read_id_col].to_numpy().astype(str)
            offsets = df[offset_col].to_numpy()
        except ImportError:
            import csv
            rids, offs = [], []
            with open(path) as f:
                reader = csv.DictReader(f, delimiter=sep)
                if read_id_col not in reader.fieldnames or offset_col not in reader.fieldnames:
                    die(f"metadata missing '{read_id_col}'/'{offset_col}'; "
                        f"has {reader.fieldnames}")
                for row in reader:
                    rids.append(row[read_id_col])
                    offs.append(row[offset_col])
            read_ids = np.asarray(rids, dtype=str)
            offsets = np.asarray(offs)
    else:
        die(f"unsupported --metadata extension '{ext}' (use .tsv/.csv/.npy/.parquet)")

    try:
        offsets = offsets.astype(np.int64)
    except (ValueError, TypeError):
        die(f"could not parse '{offset_col}' as integer sample offsets")

    if len(read_ids) != n_windows:
        die(f"metadata rows ({len(read_ids)}) != n_windows embeddings ({n_windows}); "
            f"they must be row-for-row aligned")
    return read_ids, offsets


def load_truth(path, fmt, require_primary, min_mapq, cols):
    """read_id -> (tname, tstart, tend, strand, mapq). Keeps one alignment per read
    (best mapq, then longest span). With --require-primary, drops reads that are
    multi-mapped at >= min_mapq (ambiguous) and reads whose best mapq < min_mapq."""
    if not os.path.exists(path):
        die(f"--truth not found: {path}")

    per_read = {}  # rid -> list of (tname, tstart, tend, strand, mapq, span)

    def add(rid, tname, tstart, tend, strand, mapq):
        per_read.setdefault(rid, []).append(
            (tname, int(tstart), int(tend), strand, int(mapq), int(tend) - int(tstart)))

    if fmt == "paf":
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 12:
                    continue
                try:
                    add(c[0], c[5], int(c[7]), int(c[8]), c[4], int(c[11]))
                except (ValueError, IndexError):
                    continue
    elif fmt == "tsv":
        try:
            import pandas as pd
        except ImportError:
            die("reading --truth-format tsv needs pandas; pip install pandas")
        df = pd.read_csv(path, sep="\t")
        need = [cols["qname"], cols["tname"], cols["tstart"], cols["strand"]]
        for c in need:
            if c not in df.columns:
                die(f"truth tsv missing column '{c}'; has {list(df.columns)}")
        has_tend = cols["tend"] in df.columns
        has_mapq = cols["mapq"] in df.columns
        if not has_tend:
            warnings.warn(f"truth tsv has no '{cols['tend']}' column; reverse-strand "
                          f"coordinates will use tstart only (less accurate)")
        for _, r in df.iterrows():
            ts = int(r[cols["tstart"]])
            te = int(r[cols["tend"]]) if has_tend else ts
            mq = int(r[cols["mapq"]]) if has_mapq else 60
            add(str(r[cols["qname"]]), str(r[cols["tname"]]), ts, te, str(r[cols["strand"]]), mq)
    else:
        die(f"unknown --truth-format '{fmt}'")

    if not per_read:
        die(f"no alignments parsed from --truth {path}; wrong format/columns?")

    truth = {}
    n_multi = 0
    for rid, aligns in per_read.items():
        strong = [a for a in aligns if a[4] >= min_mapq]
        if require_primary:
            if len(strong) != 1:
                if len(strong) > 1:
                    n_multi += 1
                continue  # ambiguous (multi-mapped) or no confident alignment -> drop
            best = strong[0]
        else:
            best = max(aligns, key=lambda a: (a[4], a[5]))  # best mapq, then longest
        truth[rid] = best[:5]
    return truth, n_multi


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #
def compute_ref_coords(read_ids, offsets, truth, rho):
    """Absolute reference coordinate (bp) per window; np.nan where read not in truth."""
    n = len(read_ids)
    coords = np.full(n, np.nan, dtype=np.float64)
    contigs = np.empty(n, dtype=object)
    base_off = offsets.astype(np.float64) / float(rho)
    for i in range(n):
        t = truth.get(read_ids[i])
        if t is None:
            continue
        tname, tstart, tend, strand, _mapq = t
        if strand == "-":
            coords[i] = tend - base_off[i]       # reverse: mirror within aligned span
        else:
            coords[i] = tstart + base_off[i]     # forward
        contigs[i] = tname
    return coords, contigs


def circ_dist(a, b, circular, genome_len):
    d = np.abs(a - b)
    if circular and genome_len:
        d = np.minimum(d, genome_len - d)
    return d


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def spearman_metric(emb, kept_idx, coords, contigs, args, rng):
    from scipy.stats import spearmanr

    # group kept windows by contig
    by_contig = {}
    for local, gi in enumerate(kept_idx):
        by_contig.setdefault(contigs[gi], []).append(gi)
    by_contig = {k: np.asarray(v, dtype=np.int64) for k, v in by_contig.items()
                 if len(v) >= 2}
    if not by_contig:
        die("no contig has >=2 kept windows; cannot sample pairs")

    weights = np.array([len(v) * (len(v) - 1) for v in by_contig.values()], dtype=np.float64)
    weights /= weights.sum()
    names = list(by_contig.keys())
    alloc = np.floor(weights * args.n_pairs).astype(np.int64)

    idx_a, idx_b = [], []
    for name, npc in zip(names, alloc):
        if npc <= 0:
            continue
        pool = by_contig[name]
        m = len(pool)
        ia = rng.integers(0, m, size=npc)
        ib = rng.integers(0, m, size=npc)
        same = ia == ib
        while same.any():
            ib[same] = rng.integers(0, m, size=int(same.sum()))
            same = ia == ib
        idx_a.append(pool[ia])
        idx_b.append(pool[ib])
    idx_a = np.concatenate(idx_a)
    idx_b = np.concatenate(idx_b)
    info(f"spearman: sampled {len(idx_a)} same-contig pairs across {len(names)} contig(s)")

    # gather only the rows we need (fancy-index the memmap -> RAM)
    va = l2norm_rows(np.asarray(emb[idx_a], dtype=np.float32))
    vb = l2norm_rows(np.asarray(emb[idx_b], dtype=np.float32))
    cos_sim = np.sum(va * vb, axis=1)
    cos_dist = 1.0 - cos_sim
    dcoord = circ_dist(coords[idx_a], coords[idx_b], args.circular, args.genome_len)

    rho, pval = spearmanr(cos_dist, dcoord)
    return {"spearman_rho": float(rho), "spearman_p": float(pval),
            "n_pairs": int(len(idx_a)), "n_contigs": len(names)}


def retrieval_metric(emb, kept_idx, coords, read_ids, args, rng):
    ks = sorted(int(x) for x in args.k.split(","))
    kmax = max(ks)

    # index set (cap for memory), queries drawn from it
    idx_all = np.asarray(kept_idx, dtype=np.int64)
    if len(idx_all) > args.max_index:
        idx_index = rng.choice(idx_all, size=args.max_index, replace=False)
    else:
        idx_index = idx_all
    idx_index = np.sort(idx_index)
    info(f"retrieval: index set = {len(idx_index)} windows (cap --max-index={args.max_index})")

    X = l2norm_rows(np.asarray(emb[idx_index], dtype=np.float32))
    coords_idx = coords[idx_index]
    reads_idx = read_ids[idx_index]

    nq = min(args.n_queries, len(idx_index))
    q_local = rng.choice(len(idx_index), size=nq, replace=False)
    Q = X[q_local]

    # retrieve extra to survive same-read exclusion
    search_k = min(len(idx_index), kmax + args.exclude_buffer)

    if args.use_faiss:
        try:
            import faiss
            index = faiss.IndexFlatIP(X.shape[1])
            index.add(X)
            _, I = index.search(Q, search_k)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"FAISS unavailable/failed ({e}); using sklearn NearestNeighbors")
            args.use_faiss = False
    if not args.use_faiss:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=search_k, metric="cosine", algorithm="brute")
        nn.fit(X)
        _, I = nn.kneighbors(Q)

    hits = {k: 0 for k in ks}
    for qi in range(nq):
        q_row = q_local[qi]
        q_read = reads_idx[q_row]
        q_coord = coords_idx[q_row]
        # exclude self + same-read neighbours, preserve rank order
        neigh = [j for j in I[qi] if j != q_row and reads_idx[j] != q_read]
        if not neigh:
            continue
        for k in ks:
            topk = neigh[:k]
            if not topk:
                continue
            d = circ_dist(q_coord, coords_idx[np.asarray(topk)], args.circular, args.genome_len)
            if np.any(d <= args.tol_bp):
                hits[k] += 1
    recall = {f"recall@{k}": hits[k] / nq for k in ks}
    recall["n_queries"] = int(nq)
    recall["search_k"] = int(search_k)
    recall["backend"] = "faiss" if args.use_faiss else "sklearn"
    return recall


# --------------------------------------------------------------------------- #
# projection + figure
# --------------------------------------------------------------------------- #
def project_2d(V, method, seed):
    if method == "umap":
        try:
            import umap  # noqa: F401
            reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed)
            return reducer.fit_transform(V), "umap"
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"umap-learn unavailable ({e}); falling back to t-SNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(30, max(5, V.shape[0] // 100))
        ts = TSNE(n_components=2, metric="cosine", init="random",
                  perplexity=perp, random_state=seed)
        return ts.fit_transform(V), "tsne"
    die(f"unknown --method '{method}'")


def make_figure(emb, kept_idx, coords, args, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.asarray(kept_idx, dtype=np.int64)
    if len(idx) > args.subsample:
        idx = rng.choice(idx, size=args.subsample, replace=False)
    V = np.asarray(emb[idx], dtype=np.float32)
    if args.normalize:
        V = l2norm_rows(V)
    c_mb = coords[idx] / 1e6

    info(f"projecting {len(idx)} windows via {args.method} ...")
    xy, used = project_2d(V, args.method, args.seed)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))  # single-column width
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c_mb, s=2, linewidths=0,
                    cmap="viridis", alpha=0.8, rasterized=True)
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("reference coordinate (Mb)")
    fig.tight_layout()

    fig.savefig(args.out_fig, bbox_inches="tight")
    png = os.path.splitext(args.out_fig)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    info(f"wrote {args.out_fig} and {png} (projection: {used})")
    return used


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Neurosamble Fig.2 embedding-geometry figure + metrics")
    # inputs
    p.add_argument("--embeddings", required=True)
    p.add_argument("--emb-format", choices=["auto", "npy", "memmap"], default="auto")
    p.add_argument("--dim", type=int, default=384, help="embedding dim (required for memmap)")
    p.add_argument("--dtype", default="float32", help="memmap dtype (default float32)")
    p.add_argument("--normalize", action="store_true", help="L2-normalize rows before use")
    p.add_argument("--metadata", required=True)
    p.add_argument("--read-id-col", default="read_id")
    p.add_argument("--offset-col", default="sample_offset")
    p.add_argument("--truth", required=True)
    p.add_argument("--truth-format", choices=["paf", "tsv"], default="paf")
    p.add_argument("--require-primary", action="store_true",
                   help="keep only reads with a single alignment at >= --min-mapq")
    p.add_argument("--min-mapq", type=int, default=60)
    p.add_argument("--rho", type=int, default=9, help="samples-per-kmer (offset->bp)")
    # truth tsv column names (only for --truth-format tsv)
    p.add_argument("--truth-qname-col", default="read_id")
    p.add_argument("--truth-tname-col", default="ref_name")
    p.add_argument("--truth-tstart-col", default="ref_start")
    p.add_argument("--truth-tend-col", default="ref_end")
    p.add_argument("--truth-strand-col", default="strand")
    p.add_argument("--truth-mapq-col", default="mapq")
    # geometry / circularity
    p.add_argument("--genome-len", type=float, default=0.0,
                   help="genome length (bp) for circular distance; required if --circular")
    p.add_argument("--circular", dest="circular", action="store_true", default=True)
    p.add_argument("--no-circular", dest="circular", action="store_false")
    # figure
    p.add_argument("--method", choices=["umap", "tsne"], default="umap")
    p.add_argument("--subsample", type=int, default=30000)
    p.add_argument("--out-fig", default="geometry.pdf")
    # metrics
    p.add_argument("--n-pairs", type=int, default=200000)
    p.add_argument("--k", default="1,5,10")
    p.add_argument("--tol-bp", type=float, default=50.0)
    p.add_argument("--n-queries", type=int, default=5000)
    p.add_argument("--max-index", type=int, default=500000)
    p.add_argument("--exclude-buffer", type=int, default=256,
                   help="extra neighbours retrieved so same-read exclusion still leaves k")
    p.add_argument("--use-faiss", action="store_true", help="use FAISS for retrieval (else sklearn)")
    p.add_argument("--out-metrics", default="metrics.json")
    p.add_argument("--seed", type=int, default=0)
    # what to run
    p.add_argument("--skip-figure", action="store_true")
    p.add_argument("--skip-metrics", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.circular and not args.genome_len:
        warnings.warn("--circular set but --genome-len is 0; treating distances as linear")
        args.circular = False

    emb, n_windows, dim = load_embeddings(args.embeddings, args.emb_format, args.dim, args.dtype)
    info(f"embeddings: n_windows={n_windows} dim={dim} dtype={emb.dtype} format-ok")

    read_ids, offsets = load_metadata(args.metadata, args.read_id_col, args.offset_col, n_windows)
    info(f"metadata: {len(read_ids)} rows; offset range "
         f"[{int(offsets.min())}, {int(offsets.max())}]")

    tcols = {"qname": args.truth_qname_col, "tname": args.truth_tname_col,
             "tstart": args.truth_tstart_col, "tend": args.truth_tend_col,
             "strand": args.truth_strand_col, "mapq": args.truth_mapq_col}
    truth, n_multi = load_truth(args.truth, args.truth_format, args.require_primary,
                                args.min_mapq, tcols)
    info(f"truth: {len(truth)} reads kept" +
         (f" ({n_multi} multi-mapped dropped)" if args.require_primary else ""))

    coords, contigs = compute_ref_coords(read_ids, offsets, truth, args.rho)
    kept_idx = np.flatnonzero(~np.isnan(coords))
    if kept_idx.size == 0:
        die("no windows survived truth filtering (read_ids in metadata do not match truth "
            "qnames?). Check --read-id-col and the truth qname column.")

    kept_contigs = {contigs[i] for i in kept_idx}
    cmin, cmax = np.nanmin(coords[kept_idx]), np.nanmax(coords[kept_idx])
    n_reads_kept = len({read_ids[i] for i in kept_idx})
    info("---- sanity ----")
    info(f"windows kept: {kept_idx.size} / {n_windows}")
    info(f"reads kept:   {n_reads_kept}")
    info(f"contigs:      {len(kept_contigs)} -> {sorted(kept_contigs)[:5]}"
         f"{' ...' if len(kept_contigs) > 5 else ''}")
    info(f"coord range:  {cmin/1e6:.3f} .. {cmax/1e6:.3f} Mb")
    if args.genome_len:
        info(f"genome-len:   {args.genome_len/1e6:.3f} Mb  circular={args.circular}")
        if cmax > args.genome_len * 1.05 or cmin < -0.05 * args.genome_len:
            warnings.warn("computed coordinates fall outside [0, genome-len]; check --rho / "
                          "strand handling / that --genome-len matches this reference")

    results = {"n_windows": int(n_windows), "n_windows_kept": int(kept_idx.size),
               "n_reads_kept": int(n_reads_kept), "n_contigs": int(len(kept_contigs)),
               "coord_min_bp": float(cmin), "coord_max_bp": float(cmax),
               "rho": args.rho, "circular": bool(args.circular),
               "genome_len": float(args.genome_len), "seed": args.seed}

    if not args.skip_metrics:
        info("computing Spearman(embedding cos-dist, |delta ref-coord|) ...")
        sp = spearman_metric(emb, kept_idx, coords, contigs, args, rng)
        results.update(sp)
        info(f"  Spearman rho = {sp['spearman_rho']:.4f}  (p={sp['spearman_p']:.2e}, "
             f"n_pairs={sp['n_pairs']})")

        info("computing top-k retrieval recall (same-read excluded) ...")
        rc = retrieval_metric(emb, kept_idx, coords, read_ids, args, rng)
        results.update(rc)
        for key in sorted(k for k in rc if k.startswith("recall@")):
            info(f"  {key} = {100*rc[key]:.2f}%  (tol={args.tol_bp:g}bp, "
                 f"backend={rc['backend']})")

        with open(args.out_metrics, "w") as f:
            json.dump(results, f, indent=2)
        info(f"wrote {args.out_metrics}")

    if not args.skip_figure:
        used = make_figure(emb, kept_idx, coords, args, rng)
        results["projection_method"] = used
        if not args.skip_metrics:
            with open(args.out_metrics, "w") as f:
                json.dump(results, f, indent=2)

    info("done.")


if __name__ == "__main__":
    main()
