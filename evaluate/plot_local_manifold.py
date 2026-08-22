#!/usr/bin/env python3
"""
plot_local_manifold.py -- ESA-style "fragment vs read" LOCAL embedding manifold
(mirror of ESA Fig. 4), a companion to plot_embedding_geometry.py (which it does
NOT modify). Read-only except for --out-fig / --out-metrics.

Why local, not global: over the whole genome, far-apart windows saturate in
cosine distance, so the global UMAP collapses to a blob and Spearman flattens.
Restricting to ONE short region (~30 kb: big enough to show an ordered trajectory,
small enough to avoid cosine saturation) recovers the order-preserving structure.

The figure overlays, in ONE shared 2D UMAP space, two marker types coloured by
reference coordinate (shared viridis colorbar):
  '|'  reference windows  = clean pore-model expected-signal windows tiled in order
                            along the region (the ESA "fragments").
  '_'  read windows       = real read windows whose true reference coordinate lies
                            in the region (reuse of build_geometry_inputs.py output).
Both go through the SAME frozen encoder used by the Phase-4 encode step.

Sources
-------
Read windows ('_'): reuse the already-built aligned inputs
  --embeddings geom_emb.npy  --metadata geom_meta.tsv  --truth truth.paf
  coordinate computed EXACTLY as plot_embedding_geometry.py (same --rho + strand:
  forward tstart+off/rho, reverse tend-off/rho); keep only in-region windows.
  Reads are NOT re-encoded.
Reference windows ('|'): expected signal for ref[region] via the pipeline's
  PoreModel.sequence_to_signal, tiled with the same --w/--stride, encoded by the
  same frozen encoder (load_encoder). Coord = region_start + offset_samples/rho.

Example
-------
  python evaluate/plot_local_manifold.py \
      --embeddings   $FULL/geom_emb.npy --metadata $FULL/geom_meta.tsv \
      --truth        $DATA/truth.paf    --rho 9 --kmer-len 6 \
      --ref-fasta    $DATA/ref_CFT073_full.fasta \
      --region-contig AE014075.1 --region-start 1000000 --region-end 1030000 \
      --pore-model   /path/r9.4_6mer.model \
      --encoder-ckpt experiments/stage4b_finetune/real_encoder_v1.pt \
      --w 2000 --stride 1000 --method umap --seed 0 \
      --out-fig local_manifold.pdf --out-metrics local_metrics.json

CONFIRM for your repo/data (commented inline below):
  * encoder import + checkpoint         -> load_encoder / --encoder-ckpt
  * pore-model class + table path/kmer  -> PoreModel / --pore-model / --kmer-len
  * window length + stride              -> --w / --stride (Phase-2/4 used 2000/1000)
  * metadata column names               -> --read-id-col / --offset-col
  * truth qname must match metadata read_id
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reuse the exact loaders / coordinate logic from the global figure script
# (numpy-only at import; torch/faiss/matplotlib are lazy there too).
from plot_embedding_geometry import (  # noqa: E402
    die,
    info,
    l2norm_rows,
    load_embeddings,
    load_metadata,
    load_truth,
    compute_ref_coords,
)


# --------------------------------------------------------------------------- #
def read_fasta_contig(path, contig):
    """Return the sequence (uppercase str) of one contig from a (multi-)FASTA.

    Contig-aware (read_single_fasta in the repo concatenates ALL records, which is
    wrong for a multi-contig reference), so we select by the header's first token.
    """
    if not os.path.exists(path):
        die(f"--ref-fasta not found: {path}")
    seqs = {}
    cur = None
    parts = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(parts).upper()
                cur = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
    if cur is not None:
        seqs[cur] = "".join(parts).upper()
    if contig not in seqs:
        die(f"--region-contig '{contig}' not in {path}; available: {list(seqs)[:10]}")
    return seqs[contig]


def build_reference_windows(ref_seq, region_start, region_end, pore_model, model,
                            rho, w, stride):
    """Clean pore-model expected-signal windows tiled along ref[start:end].

    Returns (emb [R,D] float32, coords_bp [R]). Each window's coordinate is
    region_start + offset_samples/rho (base offset from the region start).
    """
    from overlap_index import tile_read  # numpy-only

    region_seq = ref_seq[region_start:region_end]
    if len(region_seq) < pore_model.kmer_len + 1:
        die(f"region [{region_start},{region_end}) too short for kmer_len="
            f"{pore_model.kmer_len}")
    expected = pore_model.sequence_to_signal(region_seq)  # 1-D expected signal
    if expected.size == 0:
        die("pore model produced empty expected signal for the region")

    tiles = tile_read(expected, win=w, stride=stride)
    if not tiles:
        die("no reference windows tiled from the region (region shorter than window?)")
    offs = np.array([off for off, _ in tiles], dtype=np.float64)
    wins = [win for _, win in tiles]
    coords = region_start + offs / float(rho)          # absolute bp on the contig
    emb = model.encode(wins)                            # SAME frozen encoder, normalized
    return np.ascontiguousarray(emb, dtype=np.float32), coords


def local_spearman(emb_read, coords_read, n_pairs, seed):
    """Spearman(cosine dist, |delta ref-coord|) over random in-region READ pairs.

    Same pairing/Spearman logic as plot_embedding_geometry.py, but LINEAR (no
    circular wrap -- the region is short) and only over in-region read windows.
    """
    from scipy.stats import spearmanr

    m = emb_read.shape[0]
    if m < 2:
        return {"local_spearman_rho": float("nan"), "local_spearman_p": float("nan"),
                "n_pairs": 0}
    rng = np.random.default_rng(seed)
    npairs = min(n_pairs, m * (m - 1) // 2)
    ia = rng.integers(0, m, size=npairs)
    ib = rng.integers(0, m, size=npairs)
    same = ia == ib
    while same.any():
        ib[same] = rng.integers(0, m, size=int(same.sum()))
        same = ia == ib
    va = l2norm_rows(emb_read[ia])
    vb = l2norm_rows(emb_read[ib])
    cos_dist = 1.0 - np.sum(va * vb, axis=1)
    dcoord = np.abs(coords_read[ia] - coords_read[ib])   # LINEAR (no circular)
    rho, p = spearmanr(cos_dist, dcoord)
    return {"local_spearman_rho": float(rho), "local_spearman_p": float(p),
            "n_pairs": int(npairs)}


def make_figure(emb_ref, coords_ref, emb_read, coords_read, method, seed, out_fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    V = l2norm_rows(np.vstack([emb_ref, emb_read]).astype(np.float32))
    n_ref = emb_ref.shape[0]

    used = method
    xy = None
    if method == "umap":
        try:
            import umap
            xy = umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(V)
        except Exception as e:  # noqa: BLE001
            import warnings
            warnings.warn(f"umap-learn unavailable ({e}); falling back to t-SNE")
            method = "tsne"
    if xy is None:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, V.shape[0] // 100))
        xy = TSNE(n_components=2, metric="cosine", init="random",
                  perplexity=perp, random_state=seed).fit_transform(V)
        used = "tsne"

    xy_ref, xy_read = xy[:n_ref], xy[n_ref:]
    coords_all_kb = np.concatenate([coords_ref, coords_read]) / 1e3
    vmin, vmax = float(coords_all_kb.min()), float(coords_all_kb.max())

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(xy_ref[:, 0], xy_ref[:, 1], c=coords_ref / 1e3, cmap=cmap, norm=norm,
               marker="|", s=44, linewidths=0.9, alpha=0.95, rasterized=True)
    ax.scatter(xy_read[:, 0], xy_read[:, 1], c=coords_read / 1e3, cmap=cmap, norm=norm,
               marker="_", s=44, linewidths=0.9, alpha=0.7, rasterized=True)
    ax.axis("off")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("reference coordinate (kb)")

    handles = [
        Line2D([0], [0], marker="|", color="0.25", linestyle="none",
               markersize=8, markeredgewidth=1.2, label="reference (pore-model)"),
        Line2D([0], [0], marker="_", color="0.25", linestyle="none",
               markersize=8, markeredgewidth=1.2, label="read"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=7, frameon=False)
    fig.tight_layout()

    fig.savefig(out_fig, bbox_inches="tight")
    png = os.path.splitext(out_fig)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    info(f"wrote {out_fig} and {png} (projection: {used})")
    return used


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="ESA-style local fragment-vs-read manifold (Fig.4)")
    # read-window inputs (reuse build_geometry_inputs.py output)
    p.add_argument("--embeddings", required=True, help="geom_emb.npy (or memmap) [n,dim]")
    p.add_argument("--emb-format", choices=["auto", "npy", "memmap"], default="auto")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--metadata", required=True, help="geom_meta.tsv (read_id, sample_offset)")
    p.add_argument("--read-id-col", default="read_id")
    p.add_argument("--offset-col", default="sample_offset")
    p.add_argument("--truth", required=True, help="read->reference PAF")
    p.add_argument("--truth-format", choices=["paf", "tsv"], default="paf")
    p.add_argument("--require-primary", action="store_true")
    p.add_argument("--min-mapq", type=int, default=60)
    p.add_argument("--rho", type=int, default=9, help="samples-per-kmer (offset->bp) AND pore dwell")
    # region
    p.add_argument("--region-contig", required=True, help="target/contig name (truth tname)")
    p.add_argument("--region-start", type=int, required=True, help="bp, inclusive")
    p.add_argument("--region-end", type=int, required=True, help="bp, exclusive (~30kb span)")
    # reference-window (pore-model) inputs
    p.add_argument("--ref-fasta", required=True)
    p.add_argument("--pore-model", default=None,
                   help="ONT k-mer table; falls back to $PORE_MODEL_PATH (else synthetic -- "
                        "synthetic is a plumbing aid, NOT valid for a real figure)")
    p.add_argument("--kmer-len", type=int, default=6, help="6 for R9, 9 for R10")
    p.add_argument("--w", type=int, default=2000, help="window length in samples (encoder input)")
    p.add_argument("--stride", type=int, default=1000, help="tiling stride in samples")
    p.add_argument("--encoder-ckpt", default="experiments/stage4b_finetune/real_encoder_v1.pt",
                   help="frozen encoder checkpoint (same one the pipeline uses)")
    p.add_argument("--device", default="cuda:0")
    # figure / metrics
    p.add_argument("--method", choices=["umap", "tsne"], default="umap")
    p.add_argument("--max-read-windows", type=int, default=4000,
                   help="subsample read windows in-region for legibility")
    p.add_argument("--n-pairs", type=int, default=100000, help="local Spearman pair sample")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-fig", default="local_manifold.pdf")
    p.add_argument("--out-metrics", default="local_metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.region_end <= args.region_start:
        die("--region-end must be > --region-start")
    rng = np.random.default_rng(args.seed)
    region_len = args.region_end - args.region_start

    # ---- read windows: reuse aligned inputs; compute coords; keep in-region ----
    emb_all, n_windows, dim = load_embeddings(args.embeddings, args.emb_format,
                                              args.dim, args.dtype)
    info(f"read embeddings: n_windows={n_windows} dim={dim}")
    read_ids, offsets = load_metadata(args.metadata, args.read_id_col, args.offset_col,
                                      n_windows)
    tcols = {"qname": "read_id", "tname": "ref_name", "tstart": "ref_start",
             "tend": "ref_end", "strand": "strand", "mapq": "mapq"}
    truth, _ = load_truth(args.truth, args.truth_format, args.require_primary,
                          args.min_mapq, tcols)
    info(f"truth: {len(truth)} reads")
    info("computing read-window reference coordinates (this scans all rows)...")
    coords, contigs = compute_ref_coords(read_ids, offsets, truth, args.rho)

    in_region = np.flatnonzero(
        (contigs == args.region_contig)
        & (coords >= args.region_start) & (coords < args.region_end))
    if in_region.size == 0:
        die(f"no read windows in {args.region_contig}:{args.region_start}-{args.region_end}. "
            f"Check --region-contig matches truth tname and --region-start/-end units (bp).")
    if in_region.size > args.max_read_windows:
        in_region = np.sort(rng.choice(in_region, size=args.max_read_windows, replace=False))
    emb_read = l2norm_rows(np.asarray(emb_all[in_region], dtype=np.float32)) \
        if args.normalize else np.asarray(emb_all[in_region], dtype=np.float32)
    coords_read = coords[in_region]

    # ---- reference windows: pore-model expected signal -> frozen encoder --------
    import torch  # noqa: F401  (encoder side)
    from pilot_recall import load_encoder            # CONFIRM: same encoder as Phase-4 encode
    from dna2vec.pore_model import PoreModel          # CONFIRM: pipeline's pore model

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"
        info("CUDA not available; encoder on CPU (few hundred ref windows, fine)")
    if not os.path.exists(args.encoder_ckpt):
        die(f"--encoder-ckpt not found: {args.encoder_ckpt}")
    model, cfg = load_encoder(args.encoder_ckpt, device, args)

    pore_model = PoreModel(kmer_table_path=args.pore_model, kmer_len=args.kmer_len,
                           samples_per_kmer=args.rho)
    if pore_model.synthetic:
        import warnings
        warnings.warn("PoreModel is SYNTHETIC (no --pore-model / $PORE_MODEL_PATH); "
                      "reference windows are NOT valid for a real figure")

    ref_seq = read_fasta_contig(args.ref_fasta, args.region_contig)
    if args.region_end > len(ref_seq):
        die(f"--region-end {args.region_end} exceeds contig length {len(ref_seq)}")
    emb_ref, coords_ref = build_reference_windows(
        ref_seq, args.region_start, args.region_end, pore_model, model,
        args.rho, args.w, args.stride)
    if args.normalize:
        emb_ref = l2norm_rows(emb_ref)

    # ---- sanity ---------------------------------------------------------------
    info("---- sanity ----")
    info(f"region: {args.region_contig}:{args.region_start}-{args.region_end} "
         f"({region_len} bp = {region_len/1e3:.1f} kb)")
    info(f"reference windows (|): {emb_ref.shape[0]}")
    info(f"read windows (_) kept: {emb_read.shape[0]}"
         + (f" (subsampled to {args.max_read_windows})"
            if in_region.size == args.max_read_windows else ""))
    info(f"coord range read : {coords_read.min()/1e3:.2f}..{coords_read.max()/1e3:.2f} kb")
    info(f"coord range ref  : {coords_ref.min()/1e3:.2f}..{coords_ref.max()/1e3:.2f} kb")
    if emb_read.shape[0] < 50:
        import warnings
        warnings.warn(f"only {emb_read.shape[0]} read windows in region (<50) -- region too "
                      f"small, wrong contig, or low coverage; figure may be uninformative")

    # ---- local Spearman (linear, in-region read windows) ----------------------
    sp = local_spearman(emb_read, coords_read, args.n_pairs, args.seed)
    info(f"local Spearman rho = {sp['local_spearman_rho']:.4f} "
         f"(p={sp['local_spearman_p']:.2e}, n_pairs={sp['n_pairs']})")

    # ---- figure ---------------------------------------------------------------
    used = make_figure(emb_ref, coords_ref, emb_read, coords_read,
                       args.method, args.seed, args.out_fig)

    results = {
        "region_contig": args.region_contig, "region_start": args.region_start,
        "region_end": args.region_end, "region_len_bp": int(region_len),
        "n_reference_windows": int(emb_ref.shape[0]),
        "n_read_windows": int(emb_read.shape[0]),
        "read_coord_min_kb": float(coords_read.min() / 1e3),
        "read_coord_max_kb": float(coords_read.max() / 1e3),
        "rho": args.rho, "kmer_len": args.kmer_len, "w": args.w, "stride": args.stride,
        "pore_model_synthetic": bool(pore_model.synthetic),
        "projection_method": used, "seed": args.seed, **sp,
    }
    with open(args.out_metrics, "w") as f:
        json.dump(results, f, indent=2)
    info(f"wrote {args.out_metrics}")
    info("done.")


if __name__ == "__main__":
    main()
