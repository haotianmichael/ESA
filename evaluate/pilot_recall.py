"""
M0 go/no-go pilot for signal-domain (learned RawHash) seeding.

One command answers the single life-or-death question: can a learned encoder
close the domain gap between a real noisy query signal and the reference's clean
*expected* signal, well enough to retrieve the right coordinate by ANN search?

It prints recall@{1,5,10,50} for three settings, which MUST order as
``trained > untrained > random`` to justify continuing to M1:

  * random    - retrieve random reference coordinates (chance floor)
  * untrained - freshly-initialized SignalEncoder (architecture only)
  * trained   - the same encoder after InfoNCE contrastive training on
                (query signal, reference expected signal) pairs

Everything heavy is reused: ``FaissStore`` (via ``SignalFaissStore``, unedited),
``ContrastiveTrainer`` and ``AveragePooler`` (unedited). No absolute paths.

Example::

    python evaluate/pilot_recall.py --reference_fasta /path/ecoli.fasta \\
        --pore_model /path/r9.4_6mer.model --device cuda:0 --train_steps 2000

    # smoke-test the plumbing with no external binary / pore-model table:
    python evaluate/pilot_recall.py --reference_fasta /path/ecoli.fasta --use_synthetic
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make `dna2vec` importable without installation / absolute paths.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from dna2vec.pore_model import PoreModel  # noqa: E402
from dna2vec.config_schema import SignalModelConfigSchema  # noqa: E402
from dna2vec.signal_encoder import signal_encoder_from_config  # noqa: E402
from dna2vec.signal_dataset import SignalPairDataset, signal_collate  # noqa: E402
from dna2vec.simulate_signal import (  # noqa: E402
    simulate_mapped_signals,
    simulate_synthetic_signals,
)

from inference_signal import SignalEvalModel  # noqa: E402
from upsert_signal import read_single_fasta, build_signal_reference_index  # noqa: E402
from signal_faiss_store import SignalFaissStore  # noqa: E402
from refine import dtw_refine_one, dtw_rerank_one  # noqa: E402
from baselines_cascade import cascade_oracle_positions, cascade_real_positions  # noqa: E402
from paf_io import write_ground_truth_paf, write_mapping_paf, mapq_from_scores  # noqa: E402


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Signal-seeding M0 recall pilot")
    p.add_argument("--reference_fasta", type=str, required=True,
                   help="Single-record FASTA to build the reference index from.")
    p.add_argument("--pore_model", type=str, default=None,
                   help="ONT k-mer pore-model table (falls back to $PORE_MODEL_PATH, then synthetic).")
    p.add_argument("--kmer_len", type=int, default=6)
    p.add_argument("--samples_per_kmer", type=int, default=9)
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--ref_bp", type=int, default=0,
                   help="Truncate the reference to this many bp for the pilot (0 = full genome). "
                        "The SAME truncated sequence is fed to squigulator, so coordinates stay aligned.")
    p.add_argument("--unit_length", type=int, default=300)
    p.add_argument("--overlap", type=int, default=285,
                   help="Index tiling overlap; default 285 (stride 15) removes the "
                        "'best window not aligned to read start' ceiling artifact.")
    p.add_argument("--index_stride", type=int, default=0,
                   help="Index tiling step; 0 = unit_length - overlap. Set to decouple "
                        "index density from --overlap if VRAM is tight.")
    p.add_argument("--forward_only", type=int, default=0,
                   help="1 = keep only '+'-strand reads. Default 0: both strands (the index "
                        "carries revcomp windows tagged strand='-').")
    p.add_argument("--both_strands", type=int, default=1,
                   help="1 = index reverse-complement windows too (needed for '-' reads).")

    # --- hard-negative mining (Stage-1 precision lever) ---
    p.add_argument("--hard_negatives", type=int, default=8,
                   help="H near-coordinate hard negatives per anchor. 0 = fall back to "
                        "the unmodified ContrastiveTrainer path (ablation baseline).")
    p.add_argument("--hard_neg_min_bp", type=int, default=30)
    p.add_argument("--hard_neg_max_bp", type=int, default=300)

    # --- checkpoint (train/eval separation) ---
    p.add_argument("--save_encoder", type=str, default=None)
    p.add_argument("--load_encoder", type=str, default=None,
                   help="If given and exists, skip training and evaluate this checkpoint.")

    # --- noise sweep (fixed model, vary squigulator noise) ---
    p.add_argument("--amp_noise", type=float, default=None)
    p.add_argument("--dwell_std", type=float, default=None)

    p.add_argument("--results_csv", type=str, default=None,
                   help="CSV to append recall@k results to (default: evaluate/signal_pilot_results.csv).")

    # --- Step 2a: DTW refinement -> single mapping (P/R/F1) ---
    p.add_argument("--refine", type=str, default="none", choices=["none", "dtw"],
                   help="'dtw' = DTW-refine top-k candidates into one mapping; "
                        "'none' = single mapping is the top-1 retrieval (behavior unchanged).")
    p.add_argument("--refine_topk", type=int, default=20,
                   help="Number of retrieval candidates whose span the subsequence DTW searches.")
    p.add_argument("--refine_dtw_ds", type=int, default=0,
                   help="Mean-downsample factor before DTW; 0 = samples_per_kmer (k-mer aligned, "
                        "fast + exact).")
    p.add_argument("--refine_ctx_margin", type=int, default=90,
                   help="bp margin added around the candidate span for subsequence DTW.")
    p.add_argument("--method_name", type=str, default="SquiggleSeek",
                   help="Method label used in metrics CSV / logs.")
    p.add_argument("--metrics_csv", type=str, default=None,
                   help="CSV for per-arm P/R/F1 (default: evaluate/squiggleseek_metrics.csv).")
    p.add_argument("--map_threshold", type=float, default=None,
                   help="Confidence threshold: reads scoring below it are 'unmapped' "
                        "(excluded from precision, still in recall denominator). None = map all.")
    p.add_argument("--score_source", type=str, default="dtw", choices=["cosine", "dtw"],
                   help="Confidence score for --map_threshold / the PR curve (evaluated on "
                        "the dtw-rerank arm; dtw = -DTW cost).")

    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_query", type=int, default=5000)
    p.add_argument("--read_length_bp", type=int, default=300)

    p.add_argument("--encoder_type", type=str, default="mamba",
                   choices=["mamba", "transformer", "cnn_rnn"])
    p.add_argument("--input_signal_len", type=int, default=2000)
    p.add_argument("--downsample_factor", type=int, default=5)
    p.add_argument("--n_blocks", type=int, default=6)

    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.05)

    p.add_argument("--tol_bp", type=int, default=15)
    p.add_argument("--probe", type=int, default=1,
                   help="1 = print a one-batch collapse probe (input/embedding across-batch std) "
                        "before training, to localize a stuck ln(N) loss.")
    p.add_argument("--use_synthetic", action="store_true",
                   help="Use the synthetic pore-model simulator instead of squigulator.")
    p.add_argument("--squigulator_profile", type=str, default="dna-r9-min")
    p.add_argument("--seed", type=int, default=42)

    # --- Step 2b: cascade baseline (basecall -> minimap2) ---
    p.add_argument("--baseline", type=str, default="none", choices=["none", "cascade"],
                   help="'cascade' = also run basecall->minimap2 baselines head-to-head.")
    p.add_argument("--cascade_oracle", type=int, default=1,
                   help="Run the oracle line (true reference substring -> minimap2). "
                        "No basecaller needed; upper bound.")
    p.add_argument("--cascade_real", type=int, default=0,
                   help="Run the real line (basecall the query BLOW5 -> minimap2). "
                        "Requires --basecaller_cmd.")
    p.add_argument("--minimap2_bin", type=str, default="minimap2")
    p.add_argument("--basecaller_cmd", type=str, default=None,
                   help="Shell template with {blow5} and {fastq}, e.g. "
                        "'buttery-eel -i {blow5} -o {fastq} -g <server> --config <cfg> --port 5000'.")
    p.add_argument("--head2head_csv", type=str, default=None,
                   help="CSV for the SquiggleSeek/cascade head-to-head (default: "
                        "evaluate/squiggleseek_head2head.csv).")

    # --- Step 2c: emit PAFs for the RawHash head-to-head (uncalled pafstats) ---
    p.add_argument("--paf_out_dir", type=str, default=None,
                   help="If set, write ground_truth.paf + squiggleseek.paf (retrieval-top1) here, "
                        "plus copies of ref.fasta and reads.blow5 (feed both to rawhash2).")
    return p.parse_args()


def make_signal_model(args, device, trained_encoder=None):
    cfg = SignalModelConfigSchema(
        encoder_type=args.encoder_type,
        downsample_factor=args.downsample_factor,
        n_mamba_blocks=args.n_blocks,
        input_signal_len=args.input_signal_len,
    )
    if trained_encoder is None:
        encoder, pooling = signal_encoder_from_config(cfg)
    else:
        encoder, pooling = trained_encoder, cfg.pooling
    model = SignalEvalModel(
        encoder=encoder, pooling=pooling, device=device,
        input_signal_len=args.input_signal_len,
        downsample_factor=args.downsample_factor,
        embedding_dim=cfg.embedding_dim,
    )
    return model, cfg


def resolve_index_stride(args):
    return args.index_stride if args.index_stride else (args.unit_length - args.overlap)


def build_store(signal_model, name, device, reference_seq, pore_model, args):
    stride = resolve_index_stride(args)
    store = SignalFaissStore(signal_model=signal_model, index_name=name, device=device)
    store.drop_table()  # start clean for a reproducible pilot
    store = SignalFaissStore(signal_model=signal_model, index_name=name, device=device)
    build_signal_reference_index(
        reference_seq=reference_seq, pore_model=pore_model, signal_model=signal_model,
        store=store, unit_length=args.unit_length, stride=stride,
        both_strands=bool(args.both_strands),
    )
    return store


def _covers(coord, true_coord, unit_length, tol_bp):
    """A retrieved window is correct if its interval covers the read's start.

    Interval-coverage (with small tol slack) is the semantically right hit
    criterion: under overlapping tiling every genomic position is covered by at
    least one window, so a perfect encoder can reach ~100% recall — unlike a
    point-distance criterion, whose ceiling is tol/stride.
    """
    return (coord - tol_bp) <= true_coord < (coord + unit_length + tol_bp)


def _within(sel, true_coord, tol_bp):
    """Symmetric criterion for a bp-level POINT estimate (DTW refine output)."""
    return abs(sel - true_coord) <= tol_bp


def _first_hit_rank(cands, true_coord, unit_length, tol_bp):
    """1-based rank of the first covering candidate, or 0 if none."""
    for j, c in enumerate(cands):
        if _covers(int(c), true_coord, unit_length, tol_bp):
            return j + 1
    return 0


def evaluate_recall(store, eval_reads, topk_list, tol_bp, unit_length):
    signals = [r.signal for r in eval_reads]
    coords = [r.reference_start for r in eval_reads]
    max_k = max(topk_list)
    results = store.query_batch(signals, coords, top_k=max_k)

    hits = {k: 0 for k in topk_list}
    mrr = 0.0
    for res in results:
        true_coord = res["index"]
        cand = [m["metadata"]["coord"] for m in res["matches"]]
        rank = _first_hit_rank(cand, true_coord, unit_length, tol_bp)
        if rank:
            mrr += 1.0 / rank
        for k in topk_list:
            if rank and rank <= k:
                hits[k] += 1
    n = len(eval_reads)
    return {k: hits[k] / n for k in topk_list}, mrr / n


def evaluate_random(reference_len, eval_reads, unit_length, stride, topk_list, tol_bp, seed):
    rng = np.random.default_rng(seed)
    step = max(1, stride)
    starts = np.arange(0, reference_len - unit_length + 1, step)
    max_k = max(topk_list)
    hits = {k: 0 for k in topk_list}
    mrr = 0.0
    for r in eval_reads:
        picks = rng.choice(starts, size=min(max_k, len(starts)), replace=False)
        rank = _first_hit_rank([int(c) for c in picks], r.reference_start, unit_length, tol_bp)
        if rank:
            mrr += 1.0 / rank
        for k in topk_list:
            if rank and rank <= k:
                hits[k] += 1
    n = len(eval_reads)
    return {k: hits[k] / n for k in topk_list}, mrr / n


def run_collapse_probe(encoder, pooling, dataloader, device, args):
    """One-batch collapse probe (localizes a stuck ln(N) loss). Handles both the
    2-tuple (no hard-neg) and 3-tuple (hard-neg) collate outputs."""
    if not getattr(args, "probe", 1):
        return
    batch = next(iter(dataloader))
    xb1, xb2 = batch[0], batch[1]

    def _s(t):
        t = t.float()
        return (f"shape={tuple(t.shape)}  global_std={t.std().item():.4f}  "
                f"per-sample-mean_std={t.mean(1).std().item():.4f}")

    print("[probe] x_1 query :", _s(xb1["signal"]), flush=True)
    print("[probe] x_2 ref   :", _s(xb2["signal"]), flush=True)
    print("[probe] mask sums x1:", xb1["attention_mask"].sum(1)[:5].tolist(),
          " x2:", xb2["attention_mask"].sum(1)[:5].tolist(), flush=True)
    with torch.no_grad():
        e = encoder.to(device).eval()
        h1 = e(**{k: v.to(device) for k, v in xb1.items()})
        h2 = e(**{k: v.to(device) for k, v in xb2.items()})
        y1 = pooling(h1, attention_mask=xb1["attention_mask"].to(device))
        y2 = pooling(h2, attention_mask=xb2["attention_mask"].to(device))
    print("[probe] y_1 across-batch std:", y1.std(0).mean().item(), flush=True)
    print("[probe] y_2 across-batch std:", y2.std(0).mean().item(),
          "  <- ~0 means the reference side collapsed", flush=True)
    encoder.train()


def _encode_pool_norm(encoder, pooling, x, device):
    h = encoder(**{k: v.to(device) for k, v in x.items()})
    y = pooling(h, attention_mask=x["attention_mask"].to(device))
    return torch.nn.functional.normalize(y, dim=-1)


def train_encoder_hardneg(encoder, pooling, dataset, device, args):
    """InfoNCE with near-coordinate hard negatives, implemented HERE (not in
    trainer.py). Per anchor the logits are [in-batch positives | H hard negs]:
    logits[i] = [cos(y1_i, y2_j) for all j] ++ [cos(y1_i, yneg_i,h) for all h],
    label = i (positive on the diagonal). Used when --hard_negatives > 0."""
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import OneCycleLR

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=signal_collate)
    run_collapse_probe(encoder, pooling, dataloader, device, args)

    encoder.to(device).train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.train_steps + 1)
    ce = torch.nn.CrossEntropyLoss()
    temp = args.temperature

    data_iter = iter(dataloader)
    for step in range(args.train_steps):
        x1, x2, xneg = next(data_iter)
        y1 = _encode_pool_norm(encoder, pooling, x1, device)   # [B, D]
        y2 = _encode_pool_norm(encoder, pooling, x2, device)   # [B, D]

        B, H, L = xneg["signal"].shape
        T = xneg["attention_mask"].shape[2]
        flat = {
            "signal": xneg["signal"].reshape(B * H, L),
            "attention_mask": xneg["attention_mask"].reshape(B * H, T),
        }
        yneg = _encode_pool_norm(encoder, pooling, flat, device).reshape(B, H, -1)  # [B,H,D]

        logits_pos = (y1 @ y2.t()) / temp                              # [B, B]
        logits_neg = torch.einsum("bd,bhd->bh", y1, yneg) / temp       # [B, H]
        logits = torch.cat([logits_pos, logits_neg], dim=1)           # [B, B+H]
        labels = torch.arange(B, device=device)
        loss = ce(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"[train] step {step} loss {loss.item():.4f} lr {lr:.2e} "
                  f"(hardneg H={H})", flush=True)

    encoder.eval()
    return encoder


_ENCODER_CFG_FIELDS = [
    "encoder_type", "conv_channels_1", "conv_channels_2", "conv_kernel_1",
    "downsample_factor", "n_mamba_blocks", "d_state", "d_conv", "expand",
    "num_heads", "dropout", "input_signal_len", "embedding_dim",
]


def save_encoder(path, encoder, cfg):
    payload = {f: getattr(cfg, f) for f in _ENCODER_CFG_FIELDS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": encoder.state_dict(), "signal_config": payload}, path)
    print(f"[info] saved encoder -> {path}", flush=True)


def load_encoder(path, device, args):
    ckpt = torch.load(path, map_location="cpu")
    cfg = SignalModelConfigSchema(**ckpt["signal_config"])
    encoder, pooling = signal_encoder_from_config(cfg)
    encoder.load_state_dict(ckpt["model"])
    model = SignalEvalModel(
        encoder=encoder, pooling=pooling, device=device,
        input_signal_len=cfg.input_signal_len, downsample_factor=cfg.downsample_factor,
        embedding_dim=cfg.embedding_dim,
    )
    print(f"[info] loaded encoder <- {path} (skipping training)", flush=True)
    return model, cfg


def _csv_needs_header(csv_path, header):
    """True if we should write ``header``. If the file exists with a DIFFERENT
    header (schema changed across versions), archive it to .bak and start fresh
    so one file never mixes two schemas (which misaligns the columns)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return True
    with open(csv_path) as f:
        first = f.readline().rstrip("\r\n")
    if first == ",".join(str(h) for h in header):
        return False
    bak = csv_path.with_suffix(csv_path.suffix + ".bak")
    i = 1
    while bak.exists():
        bak = csv_path.with_suffix(csv_path.suffix + f".bak{i}")
        i += 1
    csv_path.rename(bak)
    print(f"[info] {csv_path.name} schema changed; archived old file -> {bak.name}", flush=True)
    return True


def _prf(correct, mapped, n):
    precision = correct / mapped if mapped else 0.0
    recall = correct / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def emit_pafs(store, eval_reads, reference_seq, args, out_dir, squig_fasta, eval_blow5):
    """Write ground_truth.paf + squiggleseek.paf (retrieval-top1) and copy the
    ref.fasta / reads.blow5 so RawHash2 runs on the identical inputs (Step 2c)."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    spk = args.samples_per_kmer

    ref_out = os.path.join(out_dir, "ref.fasta")
    shutil.copy(squig_fasta, ref_out)
    blow5_out = os.path.join(out_dir, "reads.blow5")
    if os.path.exists(eval_blow5):
        shutil.copy(eval_blow5, blow5_out)

    write_ground_truth_paf(eval_reads, reference_seq, spk,
                           os.path.join(out_dir, "ground_truth.paf"))

    # SquiggleSeek main arm = retrieval-top1 (pure seeding, vs RawHash pure seeding).
    # ALL reads are written with their cosine score as a tag; the confidence
    # threshold sweep (work-point matching) happens in rawhash_compare.
    signals = [r.signal for r in eval_reads]
    coords = [r.reference_start for r in eval_reads]
    results = store.query_batch(signals, coords, top_k=1)
    quads = []
    for r, res in zip(eval_reads, results):
        if not res["matches"]:
            continue
        m0 = res["matches"][0]
        strand = m0["metadata"].get("strand", "+")
        quads.append((r, m0["metadata"]["coord"], strand, float(m0["score"])))
    mapqs = mapq_from_scores([s for *_, s in quads])
    entries = [(r, pos, mq, strand, score)
               for (r, pos, strand, score), mq in zip(quads, mapqs)]
    write_mapping_paf(entries, reference_seq, spk, os.path.join(out_dir, "squiggleseek.paf"))

    print(f"[paf] out_dir = {out_dir}", flush=True)
    print(f"[paf]   ref.fasta   = {ref_out}", flush=True)
    print(f"[paf]   reads.blow5 = {blow5_out}  (feed this SAME blow5 to rawhash2)", flush=True)
    print(f"[paf]   ground_truth.paf, squiggleseek.paf ({len(entries)}/{len(eval_reads)} mapped)",
          flush=True)


def score_positions(positions, eval_reads, unit_length, tol_bp):
    """Score a list of reported start positions (None = unmapped) under _covers."""
    n = len(eval_reads)
    mapped = correct = 0
    for pos, r in zip(positions, eval_reads):
        if pos is None:
            continue
        mapped += 1
        if _covers(pos, r.reference_start, unit_length, tol_bp):
            correct += 1
    p, r_, f = _prf(correct, mapped, n)
    return {"precision": p, "recall": r_, "f1": f, "n_mapped": mapped, "n": n}


def append_head2head_csv(csv_path, args, rows):
    """rows: list of (method, metrics dict). One line per method."""
    import csv
    from datetime import datetime

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "method", "amp_noise", "dwell_std", "precision", "recall",
              "f1", "n_mapped", "n", "ref_bp", "tol_bp", "n_query"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = _csv_needs_header(csv_path, header)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for method, m in rows:
            w.writerow([ts, method, args.amp_noise, args.dwell_std,
                        f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                        m["n_mapped"], m["n"], args.ref_bp, args.tol_bp, args.n_query])
    print(f"[info] appended {len(rows)} head-to-head rows -> {csv_path}", flush=True)


def evaluate_arms(store, eval_reads, reference_seq, pore_model, args):
    """Score three arms separately (never merged) under the RIGHT criterion:

      retrieval-top1 : top-1 window coord          (_covers, window)
      dtw-rerank     : DTW-chosen candidate coord   (_covers, window)
      dtw-refine     : DTW bp point estimate        (_within, symmetric)

    Also returns diagnostics: signed (refine - true) errors on the
    rerank-correct subset, a 5-bin histogram, fallback rate, and per-read
    (score, refine_correct) for the PR curve.
    """
    signals = [r.signal for r in eval_reads]
    coords = [r.reference_start for r in eval_reads]
    topk = max(args.refine_topk, 1)
    results = store.query_batch(signals, coords, top_k=topk)
    n = len(eval_reads)
    do_dtw = args.refine == "dtw"

    top1_c = rerank_c = refine_c = refine_mapped = fallback = 0
    signed, per_read, samples = [], [], []
    for res in results:
        true = res["index"]
        cand = [m["metadata"]["coord"] for m in res["matches"]]
        cos1 = float(res["matches"][0]["score"]) if res["matches"] else 0.0
        if not cand:
            continue
        top1_ok = _covers(cand[0], true, args.unit_length, args.tol_bp)
        top1_c += top1_ok
        if not do_dtw:
            per_read.append((cos1, top1_ok))
            continue
        # rerank first, then anchor the bp-refine on the reranked window (A1)
        rr, rr_cost = dtw_rerank_one(res["query"], cand[:topk], reference_seq, pore_model,
                                     ds=args.refine_dtw_ds)
        rr_ok = _covers(rr, true, args.unit_length, args.tol_bp)
        rerank_c += rr_ok
        rf, _, fuse = dtw_refine_one(res["query"], rr, reference_seq, pore_model,
                                     args.unit_length, ds=args.refine_dtw_ds,
                                     ctx_margin=args.refine_ctx_margin)
        fallback += int(fuse)
        refine_mapped += 1
        rf_ok = _within(rf, true, args.tol_bp)
        refine_c += rf_ok
        if rr_ok:
            signed.append(rf - true)
        # PR curve is evaluated on the dtw-rerank arm (main metric) with the
        # DTW cost as confidence (A2); cosine kept for comparison.
        score = (-rr_cost) if args.score_source == "dtw" else cos1
        per_read.append((score, rr_ok))
        if len(samples) < 6:
            samples.append((true, rr, rf, rr_ok, rf_ok))

    arms = {"retrieval-top1": (_prf(top1_c, n, n), "_covers")}
    if do_dtw:
        arms["dtw-rerank"] = (_prf(rerank_c, n, n), "_covers")
        arms["dtw-refine"] = (_prf(refine_c, refine_mapped, n), "_within")

    signed = np.array(signed) if signed else np.array([0])
    hist = [int(np.sum(signed < -15)), int(np.sum((signed >= -15) & (signed < -5))),
            int(np.sum((signed >= -5) & (signed <= 5))), int(np.sum((signed > 5) & (signed <= 15))),
            int(np.sum(signed > 15))]
    return {
        "arms": arms, "n": n, "samples": samples, "per_read": per_read,
        "fallback": fallback, "refine_mapped": refine_mapped,
        "signed": {"median": float(np.median(signed)), "mean": float(signed.mean()),
                   "std": float(signed.std()), "p10": float(np.percentile(signed, 10)),
                   "p90": float(np.percentile(signed, 90)), "hist": hist},
    }


def pr_curve(per_read, n, n_points=10):
    """Sweep the confidence score -> list of (threshold, n_mapped, precision, recall, f1)."""
    if not per_read:
        return []
    scores = sorted(s for s, _ in per_read)
    qs = np.linspace(0, 1, n_points)
    thresholds = sorted({float(np.quantile(scores, q)) for q in qs})
    rows = []
    for t in thresholds:
        mapped = [(s, ok) for s, ok in per_read if s >= t]
        correct = sum(ok for _, ok in mapped)
        p, r, f = _prf(correct, len(mapped), n)
        rows.append((t, len(mapped), p, r, f))
    return rows


def append_metrics_csv(csv_path, args, method, result):
    """Append one row PER ARM (arm + criterion columns; never merged)."""
    import csv
    from datetime import datetime

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "method", "arm", "criterion", "refine", "refine_topk",
              "precision", "recall", "f1", "n_mapped", "n", "fallback",
              "err_median", "err_mean", "amp_noise", "dwell_std", "hard_negatives",
              "overlap", "index_stride", "ref_bp", "unit_length", "tol_bp"]
    stride = resolve_index_stride(args)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sg = result["signed"]
    write_header = _csv_needs_header(csv_path, header)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for arm, ((p, r, f1), crit) in result["arms"].items():
            mapped = result["refine_mapped"] if arm == "dtw-refine" else result["n"]
            w.writerow([ts, method, arm, crit, args.refine, args.refine_topk,
                        f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", mapped, result["n"],
                        result["fallback"], f"{sg['median']:.1f}", f"{sg['mean']:.1f}",
                        args.amp_noise, args.dwell_std, args.hard_negatives, args.overlap,
                        stride, args.ref_bp, args.unit_length, args.tol_bp])
    print(f"[info] appended {len(result['arms'])} arm rows -> {csv_path}", flush=True)


def train_encoder(encoder, pooling, dataset, device, args):
    """Contrastive training via the reused ContrastiveTrainer (InfoNCE)."""
    os.environ.setdefault("WANDB_MODE", "disabled")
    import wandb
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import OneCycleLR

    from dna2vec.trainer import ContrastiveTrainer
    from dna2vec.similarity import SimilarityWithTemperature
    from dna2vec.config_schema import ConfigSchema

    wandb.init(mode="disabled")

    # Make training loss visible WITHOUT forking trainer.py: ContrastiveTrainer
    # only reports via wandb.log(...), and wandb is disabled here, so route that
    # call to stdout. This preserves the "trainer.py unchanged" reuse guarantee
    # while letting us tell "not learning" apart from "learned but misaligned".
    def _stdout_log(d, *a, **k):
        if isinstance(d, dict) and "loss" in d:
            print(f"[train] step {d.get('step', '?')} loss {d['loss']:.4f} "
                  f"lr {d.get('lr', float('nan')):.2e}", flush=True)
    wandb.log = _stdout_log

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=signal_collate)
    run_collapse_probe(encoder, pooling, dataloader, device, args)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.train_steps + 1)
    similarity = SimilarityWithTemperature(temperature=args.temperature)
    loss = torch.nn.CrossEntropyLoss()

    config = ConfigSchema()
    config.training_config.device = torch.device(device)
    config.training_config.save_path = Path(__file__).resolve().parent / "signal_checkpoints"

    trainer = ContrastiveTrainer(
        encoder=encoder, pooling=pooling, similarity=similarity, loss=loss,
        optimizer=optimizer, train_dataloader=dataloader, scheduler=scheduler,
        device=torch.device(device), config=config, tokenizer=None,
    )
    trainer.train(max_steps=args.train_steps, log_interval=100)
    encoder.eval()
    return encoder


def write_single_record_fasta(seq, path):
    """Write ``seq`` as a single-record FASTA so squigulator's coordinates map
    directly onto our in-memory ``reference_seq`` (fixes the truncation /
    multi-record coordinate mismatch)."""
    with open(path, "w") as f:
        f.write(">ref\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")
    return path


def simulate_reads(args, reference_seq, pore_model, n_reads, seed, tag, squig_fasta,
                   work_dir=None):
    if args.use_synthetic:
        reads = simulate_synthetic_signals(
            reference_seq=reference_seq, pore_model=pore_model,
            n_reads=n_reads, read_length_bp=args.read_length_bp, seed=seed,
        )
    else:
        reads = simulate_mapped_signals(
            reference_genome=squig_fasta, n_reads=n_reads,
            read_length_bp=args.read_length_bp, profile=args.squigulator_profile,
            seed=seed, amp_noise=args.amp_noise, dwell_std=args.dwell_std,
            work_dir=work_dir,
        )
    if args.forward_only:
        reads = [r for r in reads if r.strand == "+"]
    return reads


def print_table(name, recall, mrr, topk_list):
    cells = "  ".join(f"@{k}={recall[k]*100:5.1f}%" for k in topk_list)
    print(f"  {name:<10s}  {cells}   MRR={mrr:.3f}")


def append_results_csv(csv_path, args, topk_list, rows):
    """Append one row per arm with recall@k, MRR and key hyperparameters."""
    import csv
    from datetime import datetime

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    hparams = ["hard_negatives", "overlap", "index_stride", "amp_noise", "dwell_std",
               "ref_bp", "n_train", "train_steps", "unit_length", "tol_bp"]
    header = (["timestamp", "arm"] + [f"recall@{k}" for k in topk_list] + ["mrr"] + hparams)
    stride = resolve_index_stride(args)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write_header = _csv_needs_header(csv_path, header)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for arm, recall, mrr in rows:
            row = [ts, arm] + [f"{recall[k]:.4f}" for k in topk_list] + [f"{mrr:.4f}"]
            row += [args.hard_negatives, args.overlap, stride, args.amp_noise,
                    args.dwell_std, args.ref_bp, args.n_train, args.train_steps,
                    args.unit_length, args.tol_bp]
            w.writerow(row)
    print(f"[info] appended {len(rows)} rows -> {csv_path}", flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    topk_list = [1, 5, 10, 20, 50, 75, 100]

    pore_model = PoreModel(
        kmer_table_path=args.pore_model, kmer_len=args.kmer_len,
        samples_per_kmer=args.samples_per_kmer,
    )
    if pore_model.synthetic:
        print("[warn] No pore-model table -> SYNTHETIC pore model. "
              "Plumbing smoke-test only; not a valid measurement.")

    reference_seq = read_single_fasta(args.reference_fasta)
    if args.ref_bp and len(reference_seq) > args.ref_bp:
        reference_seq = reference_seq[: args.ref_bp]
    stride = resolve_index_stride(args)
    print(f"[info] reference length = {len(reference_seq)} bp; device = {device}; "
          f"index stride = {stride}; hard_negatives = {args.hard_negatives}")

    # Feed squigulator the EXACT sequence we index, as a single record, so read
    # coordinates align with our reference_seq (coord-frame guarantee).
    import tempfile
    squig_fasta = write_single_record_fasta(
        reference_seq, os.path.join(tempfile.mkdtemp(prefix="pilot_ref_"), "ref.fasta")
    )

    # Query reads: disjoint train / eval sets. The eval reads use a persistent
    # work dir so their BLOW5 survives for the cascade-real basecaller.
    eval_work_dir = tempfile.mkdtemp(prefix="pilot_eval_")
    eval_blow5 = os.path.join(eval_work_dir, "reads.blow5")
    train_reads = simulate_reads(args, reference_seq, pore_model, args.n_train, args.seed, "train", squig_fasta)
    eval_reads = simulate_reads(args, reference_seq, pore_model, args.n_query, args.seed + 1, "eval", squig_fasta,
                                work_dir=eval_work_dir)
    print(f"[info] simulated {len(train_reads)} train / {len(eval_reads)} eval reads "
          f"(forward_only={bool(args.forward_only)})")
    n_distinct = len(set(r.reference_start for r in eval_reads))
    print(f"[info] distinct eval coords = {n_distinct} / {len(eval_reads)} "
          f"(should be ~= n_query; ==1 means the coordinate bug is back)")

    # --- random baseline ---
    rec_random, mrr_random = evaluate_random(
        len(reference_seq), eval_reads, args.unit_length, stride,
        topk_list, args.tol_bp, args.seed,
    )

    # --- untrained encoder ---
    untrained_model, _ = make_signal_model(args, device)
    store_u = build_store(untrained_model, "signal-pilot-untrained", device,
                          reference_seq, pore_model, args)
    rec_untrained, mrr_untrained = evaluate_recall(
        store_u, eval_reads, topk_list, args.tol_bp, args.unit_length)

    # --- trained (or loaded) encoder ---
    if args.load_encoder and os.path.exists(args.load_encoder):
        trained_model, cfg = load_encoder(args.load_encoder, device, args)
    else:
        trained_model, cfg = make_signal_model(args, device)
        dataset = SignalPairDataset(
            query_signals=[r.signal for r in train_reads],
            query_coords=[r.reference_start for r in train_reads],
            query_strands=[r.strand for r in train_reads],
            query_ends=[r.reference_end for r in train_reads],
            reference_seq=reference_seq, pore_model=pore_model,
            unit_length=args.unit_length,
            input_signal_len=args.input_signal_len, downsample_factor=args.downsample_factor,
            samples_per_kmer=args.samples_per_kmer,
            hard_negatives=args.hard_negatives,
            hard_neg_min_bp=args.hard_neg_min_bp, hard_neg_max_bp=args.hard_neg_max_bp,
        )
        if args.hard_negatives > 0:
            trained_encoder = train_encoder_hardneg(
                trained_model.encoder, trained_model.pooling, dataset, device, args)
        else:
            trained_encoder = train_encoder(
                trained_model.encoder, trained_model.pooling, dataset, device, args)
        trained_model.encoder = trained_encoder
        if args.save_encoder:
            save_encoder(args.save_encoder, trained_model.encoder, cfg)

    store_t = build_store(trained_model, "signal-pilot-trained", device,
                          reference_seq, pore_model, args)
    rec_trained, mrr_trained = evaluate_recall(
        store_t, eval_reads, topk_list, args.tol_bp, args.unit_length)

    # --- report ---
    print("\n================ recall (tol +/-%dbp, coverage) ================" % args.tol_bp)
    print_table("random", rec_random, mrr_random, topk_list)
    print_table("untrained", rec_untrained, mrr_untrained, topk_list)
    print_table("trained", rec_trained, mrr_trained, topk_list)
    print("================================================================")
    verdict = (rec_trained[10] > rec_untrained[10] > rec_random[10])
    print(f"[go/no-go] trained > untrained > random @10: "
          f"{'GO' if verdict else 'NO-GO (investigate)'}")

    csv_path = args.results_csv or (Path(__file__).resolve().parent / "signal_pilot_results.csv")
    append_results_csv(csv_path, args, topk_list, [
        ("random", rec_random, mrr_random),
        ("untrained", rec_untrained, mrr_untrained),
        ("trained", rec_trained, mrr_trained),
    ])

    # --- Step 2a: three arms, each under its own criterion ---
    res = evaluate_arms(store_t, eval_reads, reference_seq, pore_model, args)
    print(f"\n======== {args.method_name} mapping arms (refine={args.refine}, "
          f"topk={args.refine_topk}) ========")
    for arm, ((p, r, f1), crit) in res["arms"].items():
        mapped = res["refine_mapped"] if arm == "dtw-refine" else res["n"]
        print(f"  {arm:<14s} [{crit:<8s}]  P={p*100:5.1f}%  R={r*100:5.1f}%  "
              f"F1={f1*100:5.1f}%   (mapped {mapped}/{res['n']})")
    print(f"  sanity: retrieval-top1 (_covers) should match recall@1 = {rec_trained[1]*100:.1f}%")
    if args.refine == "dtw":
        sg = res["signed"]
        print(f"  [diag] refine signed err (rerank-correct subset): median={sg['median']:+.0f} "
              f"mean={sg['mean']:+.1f} std={sg['std']:.0f} p10={sg['p10']:+.0f} p90={sg['p90']:+.0f}")
        print(f"  [diag] err hist [<-15, -15..-5, -5..5, 5..15, >15]: {sg['hist']}")
        print(f"  [diag] fuse triggered (|refine-rerank|>unit): {res['fallback']}/{res['refine_mapped']}")
        print("  (true, rerank_coord, refine_bp, rerank_ok, refine_ok) samples:")
        for tc, rr, rf, rok, fok in res["samples"]:
            print(f"    true={tc:>9d}  rerank={rr:>9d}({'H' if rok else '.'})  "
                  f"refine={rf:>9d}({'H' if fok else '.'})")
        rows = pr_curve(res["per_read"], res["n"])
        print(f"  [PR curve on dtw-rerank, score={args.score_source}] threshold  n_mapped  P      R      F1")
        for t, nm, p, r, f1 in rows:
            print(f"    {t:>10.4f}  {nm:>7d}  {p*100:5.1f}  {r*100:5.1f}  {f1*100:5.1f}")
    print("================================================================")

    metrics_csv = args.metrics_csv or (Path(__file__).resolve().parent / "squiggleseek_metrics.csv")
    append_metrics_csv(metrics_csv, args, args.method_name, res)

    # --- Step 2c: emit PAFs for the RawHash head-to-head ---
    if args.paf_out_dir:
        emit_pafs(store_t, eval_reads, reference_seq, args, args.paf_out_dir,
                  squig_fasta, eval_blow5)

    # --- Step 2b: cascade baselines, head-to-head under the same criterion ---
    if args.baseline == "cascade":
        if "dtw-rerank" in res["arms"]:
            (p, r, f1), _ = res["arms"]["dtw-rerank"]
            ss_method = "SquiggleSeek(dtw-rerank)"
        else:
            (p, r, f1), _ = res["arms"]["retrieval-top1"]
            ss_method = "SquiggleSeek(top1)"
        ss_metrics = {"precision": p, "recall": r, "f1": f1,
                      "n_mapped": res["n"], "n": res["n"]}

        import tempfile
        bwork = tempfile.mkdtemp(prefix="cascade_")
        h2h = [(ss_method, ss_metrics)]

        if args.cascade_oracle:
            oracle_pos = cascade_oracle_positions(
                eval_reads, reference_seq, squig_fasta, bwork, args.minimap2_bin)
            h2h.append(("cascade-oracle",
                        score_positions(oracle_pos, eval_reads, args.unit_length, args.tol_bp)))

        if args.cascade_real:
            if args.use_synthetic:
                print("[warn] --cascade_real needs a real BLOW5; skipped under --use_synthetic.")
            elif not args.basecaller_cmd:
                print("[warn] --cascade_real needs --basecaller_cmd; skipped.")
            else:
                real_pos = cascade_real_positions(
                    eval_reads, eval_blow5, squig_fasta, bwork,
                    args.basecaller_cmd, args.minimap2_bin)
                h2h.append(("cascade-real",
                            score_positions(real_pos, eval_reads, args.unit_length, args.tol_bp)))

        print("\n======== Step 2b head-to-head (_covers, tol +/-%dbp, amp_noise=%s dwell_std=%s) ========"
              % (args.tol_bp, args.amp_noise, args.dwell_std))
        for method, m in h2h:
            print(f"  {method:<24s}  P={m['precision']*100:5.1f}%  R={m['recall']*100:5.1f}%  "
                  f"F1={m['f1']*100:5.1f}%   (mapped {m['n_mapped']}/{m['n']})")
        print("=================================================================")
        h2h_csv = args.head2head_csv or (Path(__file__).resolve().parent / "squiggleseek_head2head.csv")
        append_head2head_csv(h2h_csv, args, h2h)


if __name__ == "__main__":
    main()
