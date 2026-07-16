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
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--forward_only", type=int, default=1,
                   help="1 = keep only '+'-strand reads (forward-only index). M0 default; "
                        "strand-aware dual-index is M1.")

    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_query", type=int, default=2000)
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
    p.add_argument("--use_synthetic", action="store_true",
                   help="Use the synthetic pore-model simulator instead of squigulator.")
    p.add_argument("--squigulator_profile", type=str, default="dna-r9-min")
    p.add_argument("--seed", type=int, default=42)
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


def build_store(signal_model, name, device, reference_seq, pore_model, args):
    store = SignalFaissStore(signal_model=signal_model, index_name=name, device=device)
    store.drop_table()  # start clean for a reproducible pilot
    store = SignalFaissStore(signal_model=signal_model, index_name=name, device=device)
    build_signal_reference_index(
        reference_seq=reference_seq, pore_model=pore_model, signal_model=signal_model,
        store=store, unit_length=args.unit_length, overlap=args.overlap,
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


def evaluate_recall(store, eval_reads, topk_list, tol_bp, unit_length):
    signals = [r.signal for r in eval_reads]
    coords = [r.reference_start for r in eval_reads]
    max_k = max(topk_list)
    results = store.query_batch(signals, coords, top_k=max_k)

    hits = {k: 0 for k in topk_list}
    for res in results:
        true_coord = res["index"]
        cand = [m["metadata"]["coord"] for m in res["matches"]]
        for k in topk_list:
            if any(_covers(c, true_coord, unit_length, tol_bp) for c in cand[:k]):
                hits[k] += 1
    n = len(eval_reads)
    return {k: hits[k] / n for k in topk_list}


def evaluate_random(reference_len, eval_reads, unit_length, overlap, topk_list, tol_bp, seed):
    rng = np.random.default_rng(seed)
    step = max(1, unit_length - overlap)
    starts = np.arange(0, reference_len - unit_length + 1, step)
    max_k = max(topk_list)
    hits = {k: 0 for k in topk_list}
    for r in eval_reads:
        picks = rng.choice(starts, size=min(max_k, len(starts)), replace=False)
        for k in topk_list:
            if any(_covers(int(c), r.reference_start, unit_length, tol_bp) for c in picks[:k]):
                hits[k] += 1
    n = len(eval_reads)
    return {k: hits[k] / n for k in topk_list}


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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=signal_collate)
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


def simulate_reads(args, reference_seq, pore_model, n_reads, seed, tag, squig_fasta):
    if args.use_synthetic:
        reads = simulate_synthetic_signals(
            reference_seq=reference_seq, pore_model=pore_model,
            n_reads=n_reads, read_length_bp=args.read_length_bp, seed=seed,
        )
    else:
        reads = simulate_mapped_signals(
            reference_genome=squig_fasta, n_reads=n_reads,
            read_length_bp=args.read_length_bp, profile=args.squigulator_profile,
            seed=seed,
        )
    if args.forward_only:
        reads = [r for r in reads if r.strand == "+"]
    return reads


def print_table(name, recall, topk_list):
    cells = "  ".join(f"@{k}={recall[k]*100:5.1f}%" for k in topk_list)
    print(f"  {name:<10s}  {cells}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    topk_list = [1, 5, 10, 50]

    pore_model = PoreModel(
        kmer_table_path=args.pore_model, kmer_len=args.kmer_len,
        samples_per_kmer=args.samples_per_kmer,
    )
    if pore_model.synthetic:
        print("[warn] No pore-model table -> SYNTHETIC pore model. "
              "Plumbing smoke-test only; not a valid go/no-go measurement.")

    reference_seq = read_single_fasta(args.reference_fasta)
    if args.ref_bp and len(reference_seq) > args.ref_bp:
        reference_seq = reference_seq[: args.ref_bp]
    print(f"[info] reference length = {len(reference_seq)} bp; device = {device}")

    # Feed squigulator the EXACT sequence we index, as a single record, so its
    # PAF coordinates align with our reference_seq (fixes the coord-frame bug).
    import tempfile
    squig_fasta = write_single_record_fasta(
        reference_seq, os.path.join(tempfile.mkdtemp(prefix="pilot_ref_"), "ref.fasta")
    )

    # Query reads: disjoint train / eval sets.
    train_reads = simulate_reads(args, reference_seq, pore_model, args.n_train, args.seed, "train", squig_fasta)
    eval_reads = simulate_reads(args, reference_seq, pore_model, args.n_query, args.seed + 1, "eval", squig_fasta)
    print(f"[info] simulated {len(train_reads)} train / {len(eval_reads)} eval reads "
          f"(forward_only={bool(args.forward_only)})")

    # --- random baseline ---
    rec_random = evaluate_random(
        len(reference_seq), eval_reads, args.unit_length, args.overlap,
        topk_list, args.tol_bp, args.seed,
    )

    # --- untrained encoder ---
    untrained_model, _ = make_signal_model(args, device)
    store_u = build_store(untrained_model, "signal-pilot-untrained", device,
                          reference_seq, pore_model, args)
    rec_untrained = evaluate_recall(store_u, eval_reads, topk_list, args.tol_bp, args.unit_length)

    # --- trained encoder ---
    trained_model, cfg = make_signal_model(args, device)
    dataset = SignalPairDataset(
        query_signals=[r.signal for r in train_reads],
        query_coords=[r.reference_start for r in train_reads],
        reference_seq=reference_seq, pore_model=pore_model,
        input_signal_len=args.input_signal_len, downsample_factor=args.downsample_factor,
        samples_per_kmer=args.samples_per_kmer,
    )
    trained_encoder = train_encoder(
        trained_model.encoder, trained_model.pooling, dataset, device, args
    )
    trained_model.encoder = trained_encoder
    store_t = build_store(trained_model, "signal-pilot-trained", device,
                          reference_seq, pore_model, args)
    rec_trained = evaluate_recall(store_t, eval_reads, topk_list, args.tol_bp, args.unit_length)

    # --- report ---
    print("\n================ M0 recall (tol +/-%dbp) ================" % args.tol_bp)
    print_table("random", rec_random, topk_list)
    print_table("untrained", rec_untrained, topk_list)
    print_table("trained", rec_trained, topk_list)
    print("=========================================================")
    verdict = (rec_trained[10] > rec_untrained[10] > rec_random[10])
    print(f"[go/no-go] trained > untrained > random @10: "
          f"{'GO' if verdict else 'NO-GO (investigate before M1)'}")


if __name__ == "__main__":
    main()
