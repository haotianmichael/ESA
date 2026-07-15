# Signal-domain seeding (learned RawHash) — M0 pilot

This adds a **signal-domain** seeding path on top of the existing FAISS-ESA
framework: instead of aligning *bases*, it encodes raw nanopore **current
signal** with a Mamba2 encoder and retrieves reference coordinates by ANN
search — **no basecalling**. Reference bases are turned into their *expected*
signal with an ONT pore model (RawHash-style), and a contrastively-trained
encoder is asked to close the gap between noisy query signal and clean expected
signal.

**Scope = M0 only:** a single go/no-go recall pilot. It is not basecalling, not
chaining, not a full mapper. If M0 fails, stop.

## What was added (all new; three reuse files are untouched)

| File | Role |
|---|---|
| `src/dna2vec/signal_encoder.py` | `SignalEncoder`: Conv front-end + N× Mamba2 (M2Caller-style). `encoder_type` ∈ {mamba, transformer, cnn_rnn}. |
| `src/dna2vec/pore_model.py` | `PoreModel.sequence_to_signal`: bases → expected signal via ONT k-mer table. |
| `src/dna2vec/simulate_signal.py` | `simulate_mapped_signals` (squigulator) + `simulate_synthetic_signals` (fallback). |
| `src/dna2vec/signal_dataset.py` | `SignalPairDataset` (Siamese positive pairs) + `signal_collate` + preprocessing. |
| `src/dna2vec/config_schema.py` | `SignalModelConfigSchema` (additive). |
| `evaluate/inference_signal.py` | `SignalEvalModel.encode(signals) -> [N, D]`. |
| `evaluate/signal_faiss_store.py` | `SignalFaissStore(FaissStore)`: injects the signal model + `add_embeddings`. |
| `evaluate/upsert_signal.py` | Build the reference expected-signal index. |
| `evaluate/pilot_recall.py` | **The M0 deliverable.** |

**Reuse guarantees (verified `git diff` empty):** `evaluate/faiss_store.py`,
`src/dna2vec/trainer.py`, `src/dna2vec/model.py` (`AveragePooler`) are not
modified. `SignalEncoder` honors the trainer contract
(`encoder(**x) -> [B,T,D]`, `pooling(last_hidden, attention_mask) -> [B,D]`), so
`ContrastiveTrainer` runs unchanged; the mask is provided at the **down-sampled
T** resolution. Mamba2 comes from the official `mamba-ssm` (no hand-written SSM).

## Install

```bash
pip install -e ".[signal]"     # mamba-ssm, causal-conv1d, pyslow5, fastdtw
```

Then obtain two external artifacts (neither is bundled):

1. **ONT R9.4 6-mer pore model** — e.g. from `CMU-SAFARI/RawHash` (or
   remora/bonito). A whitespace table of `<kmer> <level_pA> [<stdv>]` per line.
   Pass via `--pore_model` or `$PORE_MODEL_PATH`.
2. **squigulator** binary (`hasindu2008/squigulator`) on `$PATH`.

## Run

Real go/no-go (recommended):
```bash
python evaluate/pilot_recall.py \
    --reference_fasta /path/to/single_record.fasta \
    --pore_model /path/to/r9.4_6mer.model \
    --device cuda:0 --train_steps 2000
```

Plumbing smoke-test (no squigulator / no pore-model table — synthetic, **not a
valid measurement**):
```bash
python evaluate/pilot_recall.py --reference_fasta /path/to/ref.fasta --use_synthetic
```

Output is a recall@{1,5,10,50} table for **random / untrained / trained**. The
go/no-go criterion is `trained > untrained > random` (checked at @10).

## Two integration points to confirm on your machine

The sandbox has no torch/mamba/GPU/squigulator, so these two external glue
points are written to the documented interface but must be matched to your
installed versions:

- **squigulator flags** (`simulate_signal.py`): `--profile` and `--paf` (the
  ground-truth PAF option) are parameterized — check `squigulator --help` and
  adjust `profile` / `paf_option` if your build differs.
- **pore-model table format** (`pore_model.py`): parser expects `<kmer> <level>`
  columns and skips headers; verify against your table's layout.

## If M0 is NO-GO

Per the plan: try query-side augmentation, or inject simulated noise on the
reference side (squigulator noise presets) to shrink the domain gap, before
abandoning. Do **not** build M1 until M0 passes.
