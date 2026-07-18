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
    --reference_fasta /path/to/genome.fasta \
    --pore_model /path/to/r9.4_6mer.model \
    --device cuda:0 --train_steps 3000 \
    --input_signal_len 3000 --tol_bp 15
    # note: do NOT pass --ref_bp (defaults to full genome; see coordinate alignment below)
```

Plumbing smoke-test (no squigulator / no pore-model table — synthetic, **not a
valid measurement**):
```bash
python evaluate/pilot_recall.py --reference_fasta /path/to/ref.fasta --use_synthetic
```

Output is a recall@{1,5,10,50} table for **random / untrained / trained**. The
go/no-go criterion is `trained > untrained > random` (checked at @10).

### Real-path correctness (three fixes baked in)

The real (squigulator) path has three ways to silently read 0% that the
synthetic path bypasses; all are handled:

1. **Coordinate alignment.** squigulator reads the whole FASTA, so truncating
   the in-memory reference would desync coordinates. The pilot writes the exact
   sequence it indexes (optionally truncated via `--ref_bp`) as a single-record
   FASTA and feeds *that* to squigulator, so PAF coordinates map onto the index.
   `--ref_bp` defaults to `0` (full genome).
2. **Strand.** squigulator emits ~50% reverse-strand reads whose signal is the
   reverse-complement waveform; against a forward-only index they cannot match.
   Strand is parsed from the PAF and, by default (`--forward_only 1`), only `+`
   reads are kept. (Strand-aware dual-strand indexing is an M1 item.)
3. **Hit criterion.** With `unit_length=300, overlap=150` (stride 150) a
   point-distance `±tol_bp` criterion caps recall at ~`tol/stride` (~20%). The
   pilot instead uses **interval coverage** — a window is correct if it covers
   the read start — so overlapping tiling admits a ~100% ceiling.

## Two integration points to confirm on your machine

The sandbox has no torch/mamba/GPU/squigulator, so these two external glue
points are written to the documented interface but must be matched to your
installed versions:

- **squigulator flags** (`simulate_signal.py`): `--profile` and `--paf` (the
  ground-truth PAF option) are parameterized — check `squigulator --help` and
  adjust `profile` / `paf_option` if your build differs.
- **pore-model table format** (`pore_model.py`): parser expects `<kmer> <level>`
  columns and skips headers; verify against your table's layout.

### Span invariant + loss visibility

Two things that otherwise silently kill recall:

- **Span invariant:** the training reference span equals the index window span
  (`SignalPairDataset(unit_length=...)` uses `win_bp = unit_length`). If they
  differ, the encoder is trained to match a reference vector length that does
  not exist in the index and recall can fall *below* random.
- **Loss visibility:** `ContrastiveTrainer` only logs via `wandb.log` (disabled
  here). The pilot routes that to stdout (`[train] step .. loss .. lr ..`)
  without editing `trainer.py`, so you can tell "not learning" (loss flat) from
  "learned but misaligned" (loss drops, recall still low).

## If M0 is NO-GO

First read the loss:

- **loss drops + recall rises** → GO, proceed to M1.
- **loss drops, recall ≈ 0** → real domain gap (clean vs noisy + fixed dwell vs
  variable dwell time-warp). Only now tune training: more `--train_steps`,
  higher `--lr`, near-position hard negatives, denser tiling
  (`--overlap 285`), or a small squigulator `--dwell-std` probe to test whether
  time-warp is the wall.
- **loss flat** → training-side problem (pairs/lr/optimizer); fix that first.

Do **not** tune training hyperparameters before the loss is visible, and do
**not** build M1 until M0 passes.

## M1 Stage 1 — precision + noise sweep + reusable scaffold

M0 passed (full E. coli: trained recall@10 ≈ 60% ≫ untrained ≫ random). Stage 1
pushes precision with **hard negatives + denser tiling** and turns the pilot
into a checkpointable, noise-sweepable harness. All features are flags:

| Flag | Default | Purpose |
|---|---|---|
| `--hard_negatives H` | 8 | H near-coordinate hard negatives per anchor. **0 = fall back to the unmodified `trainer.py` path** (ablation baseline). |
| `--hard_neg_min_bp` / `--hard_neg_max_bp` | 30 / 300 | offset range for hard negatives. |
| `--overlap` | 285 | index tiling overlap (stride 15 at unit_length 300). |
| `--index_stride` | 0 (=unit_length−overlap) | decouple index density from `--overlap` if VRAM is tight. |
| `--save_encoder PATH` / `--load_encoder PATH` | – | save encoder+config / skip training and evaluate a checkpoint. |
| `--amp_noise` / `--dwell_std` | profile default | appended to squigulator (`--amp-noise` / `--dwell-std`) for the noise sweep. |
| `--results_csv PATH` | evaluate/signal_pilot_results.csv | append 3-arm recall@{1,5,10,20,50,75,100} + MRR + hyperparams. |

Hard-negative InfoNCE is implemented **in the pilot**, not in `trainer.py`: per
anchor the logits are `[in-batch positives | H hard negatives]`, label on the
diagonal. With `H=0` the dataset yields 2-tuples and training goes through the
untouched `ContrastiveTrainer`.

Suggested runs:
```bash
# precision: hard negatives + dense tiling, save the encoder
python evaluate/pilot_recall.py --reference_fasta <genome.fasta> \
    --pore_model $PORE_MODEL_PATH --device cuda:0 --train_steps 3000 \
    --input_signal_len 3000 --hard_negatives 8 --overlap 285 \
    --save_encoder evaluate/signal_checkpoints/hn8.pt

# ablation: no hard negatives (unmodified trainer path)
python evaluate/pilot_recall.py ... --hard_negatives 0

# noise sweep on a fixed model (no retrain)
python evaluate/pilot_recall.py ... --load_encoder evaluate/signal_checkpoints/hn8.pt \
    --amp_noise 1.5 --dwell_std 4.0
```

**Validation gates (self-check each as you go):**
1. With hard negatives: probe `y_2 across-batch std > 0`, `distinct eval coords ≈
   n_query`, loss falls from ~ln(batch). Hard negatives make loss converge
   slower to a **higher** floor (not 1e-4) — that is correct; the earlier
   instant-0 loss was the task being too easy.
2. `--load_encoder` reproducibility: same checkpoint evaluated twice gives
   identical recall.
3. `git diff` empty for `faiss_store.py` / `trainer.py` / `AveragePooler`.
4. No hardcoded absolute paths.

Out of scope for Stage 1 (a later round): SW/DTW refinement, cascade baseline
(basecall→minimap2), RawHash comparison, encoder ablation.

## SquiggleSeek Stage 2 — single mapping + baselines

The method is named **SquiggleSeek**. Stage 2 makes SquiggleSeek and the
baselines each emit ONE best mapping under the same reads / true coords /
`_covers` criterion, for a head-to-head precision/recall/F1 noise sweep. Built
strictly in sub-steps; **only 2a is implemented so far.**

### Step 2a — DTW refinement (done)

Collapse top-k retrieval to a single **bp-level** mapping via **subsequence
DTW**: render the expected signal of the reference region spanned by the top-k
candidates, align the query as a subsequence (`dtaidistance`), and read off where
it best matches → `region_start + offset`. Picking a window *coord* by global
DTW failed (68% vs 97% top-1) because near-duplicate stride-15 windows blur ~±30
bp > tol; subsequence DTW instead reports the aligned position to ~0 bp.
Downsampling by `samples_per_kmer` (one point per k-mer) is both fast and exact.
Then score RawHash-style: correct if the reported position covers the true coord;
`precision = correct/mapped`, `recall = correct/total`, `F1` harmonic mean.

Flags: `--refine dtw|none` (default none), `--refine_topk` (20),
`--refine_dtw_ds` (0 = samples_per_kmer), `--refine_ctx_margin` (90),
`--method_name` (SquiggleSeek), `--metrics_csv`.

```bash
python evaluate/pilot_recall.py --reference_fasta <genome.fasta> \
    --pore_model $PORE_MODEL_PATH --device cuda:0 \
    --load_encoder evaluate/signal_checkpoints/hn8.pt --batch_size 32 \
    --refine dtw --refine_topk 20
```
Prints SquiggleSeek P/R/F1, `(true_coord, selected_coord, hit?)` samples, and the
single-mapping-recall-vs-recall@1 sanity line; appends a row to
`evaluate/squiggleseek_metrics.csv` (schema: method, refine, precision, recall,
f1, n_mapped, amp_noise, dwell_std, … — the unified head-to-head schema that 2b
/2c and the noise-sweep driver will reuse). `--refine none` reports the top-1
retrieval unchanged.

Validation gates: single-mapping recall ≈ (≥) retrieval recall@1; printed
(true, selected) pairs are close; `--refine none` matches prior behavior.

Steps 2b (cascade basecall→minimap2 + oracle) and 2c (RawHash) are **not yet
implemented** — pending confirmation of 2a's P/R/F1.
