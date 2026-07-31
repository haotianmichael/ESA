#!/usr/bin/env bash
# =============================================================================
# SquiggleSeek vs RawHash2 head-to-head — from scratch, one shot (Step 2c fix).
#
# Does EVERYTHING: generates a reference, simulates reads (squigulator), builds
# the both-strand FAISS index + trains the encoder, exports PAFs, runs RawHash2
# on the identical blow5, then scores with the builtin locus scorer AND the real
# UNCALLED pafstats (pure-python, no HDF5 build). Work-point matching (PR sweep +
# recall at RawHash's precision) is printed by rawhash_compare.
#
# Usage:
#     export PATH=$PATH:/path/to/squigulator
#     export PORE_MODEL_PATH=/path/to/r9_6mer_pore_model.txt
#     nohup bash evaluate/run_head2head.sh > head2head_nohup.out 2>&1 &
#
# Everything below the config block is automatic. All logs -> $OUT/run.log; the
# final numbers are also collected into $OUT/SUMMARY.txt.
# =============================================================================
set -u
set -o pipefail

# ---- config: paths are wired for this server; override any via env ----------
# Layout on this box:  /home/nfs/mahaotian/ESA/{squigulator,Rawhash2,UNCALLED,...}
BASE="${BASE:-/home/nfs/mahaotian/ESA}"
export PATH="$PATH:$BASE/squigulator"                       # squigulator on PATH
export PORE_MODEL_PATH="${PORE_MODEL_PATH:-$BASE/squigulator/r9_6mer_pore_model.txt}"
# rawhash2: use $RAWHASH2 if set, else PATH, else auto-find under Rawhash2/
if [ -z "${RAWHASH2:-}" ]; then
  RAWHASH2="$(command -v rawhash2 2>/dev/null || \
              find "$BASE/Rawhash2" -maxdepth 4 -type f -name rawhash2 2>/dev/null | head -1)"
  RAWHASH2="${RAWHASH2:-rawhash2}"
fi
# RawHash2 needs a pore model to build its index from a FASTA. Prefer its own
# official R9.4 450bps 6-mer model (matches squigulator dna-r9-min); avoid the
# r9.2 / RNA variants that also ship as template_median68pA.model. Don't tune the
# opponent's config; fall back to ours only if nothing is found.
if [ -z "${RAWHASH_PORE:-}" ]; then
  _cands="$(find "$BASE/Rawhash2" -maxdepth 6 -type f -name 'template_median68pA.model' 2>/dev/null)"
  RAWHASH_PORE="$(echo "$_cands" | grep -i 'r9.4_180mv_450bps_6mer' | head -1)"
  [ -z "$RAWHASH_PORE" ] && RAWHASH_PORE="$(echo "$_cands" | grep -iv 'rna' | grep -i 'r9.4' | head -1)"
  [ -z "$RAWHASH_PORE" ] && RAWHASH_PORE="$(echo "$_cands" | head -1)"
  RAWHASH_PORE="${RAWHASH_PORE:-$PORE_MODEL_PATH}"
fi

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"           # repo root (auto)
OUT="${OUT:-$REPO/head2head_out}"
REF_FASTA="${REF_FASTA:-}"           # real genome FASTA (single record). If set, used
                                     # verbatim (skip random gen) — e.g. full 4.64Mb E. coli.
REF_BP="${REF_BP:-1000000}"          # random reference length (bp) when REF_FASTA unset
FAST_SWEEP="${FAST_SWEEP:-0}"        # 1 = skip random/untrained baselines (noise-sweep speedup)
FAISS_CPU="${FAISS_CPU:-0}"          # 1 = FAISS index on CPU (avoids GPU-OOM on full-genome index)
N_TRAIN="${N_TRAIN:-20000}"          # training reads
TRAIN_STEPS="${TRAIN_STEPS:-2000}"   # optimizer steps (2000 = original behavior)
N_QUERY="${N_QUERY:-5000}"           # eval/query reads (the head-to-head set)
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
# GPU memory: the hard-neg encode pushes B*H length-2000 signals through Mamba2
# in one backward. B=64,H=8 -> 512 seqs OOMs a 32GB V100 (Triton autotuner spike).
# B=16,H=8 -> 128 seqs fits. Drop BATCH_SIZE further (8) if it still OOMs.
BATCH_SIZE="${BATCH_SIZE:-16}"
HARD_NEG="${HARD_NEG:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FORWARD_ONLY="${FORWARD_ONLY:-0}"    # 1 = keep only '+' reads (truth + eval); Gate-2 A
BOTH_STRANDS="${BOTH_STRANDS:-1}"    # 1 = index revcomp windows too (needed for '-' reads)
ENCODER_TYPE="${ENCODER_TYPE:-mamba}"  # mamba | transformer | cnn_rnn (encoder ablation)
OVERLAP="${OVERLAP:-285}"            # index tiling overlap; stride = unit_length - overlap (tiling ablation)
SKIP_RAWHASH="${SKIP_RAWHASH:-0}"    # 1 = SquiggleSeek PR sweep only (fast A/B/C sweeps)
AMP_NOISE="${AMP_NOISE:-}"           # squigulator --amp-noise (blank = its default); noise sweep
DWELL_STD="${DWELL_STD:-}"           # squigulator --dwell-std  (blank = its default, ~4); noise sweep
RAWHASH_PRESET="${RAWHASH_PRESET:-sensitive}"  # check Rawhash2/test/ for the R9 preset
THREADS="${THREADS:-32}"
LOAD_ENCODER="${LOAD_ENCODER:-}"     # path to a saved encoder to skip training (optional)
SAVE_ENCODER="${SAVE_ENCODER:-$OUT/encoder.pt}"
# -----------------------------------------------------------------------------

mkdir -p "$OUT"
LOG="$OUT/run.log"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"
# tee all stdout/stderr to the run log
exec > >(tee -a "$LOG") 2>&1

say()  { echo -e "\n========== $* =========="; }
fail() { echo "FATAL: $*"; exit 1; }

say "0. environment check  ($(date))"
echo "REPO=$REPO  OUT=$OUT  REF_BP=$REF_BP  N_QUERY=$N_QUERY  SEED=$SEED"
command -v "$PYTHON" >/dev/null || fail "python not found ($PYTHON)"
command -v squigulator >/dev/null || fail "squigulator not on PATH (export PATH=\$PATH:/path/to/squigulator)"
[ -n "${PORE_MODEL_PATH:-}" ] || fail "PORE_MODEL_PATH not set (export PORE_MODEL_PATH=/path/to/r9_6mer_pore_model.txt)"
[ -f "$PORE_MODEL_PATH" ]     || fail "pore model file missing: $PORE_MODEL_PATH"
"$PYTHON" -c "import pyslow5" 2>/dev/null || fail "pyslow5 missing (pip install pyslow5)"
HAVE_RAWHASH=1; command -v "$RAWHASH2" >/dev/null || { echo "WARN: rawhash2 not found ($RAWHASH2) — will still produce SquiggleSeek PAF; RawHash steps skipped."; HAVE_RAWHASH=0; }
echo "squigulator: $(command -v squigulator)"
echo "pore model : $PORE_MODEL_PATH"
echo "rawhash2   : $(command -v "$RAWHASH2" 2>/dev/null || echo MISSING)"

# -----------------------------------------------------------------------------
say "1. reference + pafstats.py"
if [ -n "$REF_FASTA" ]; then
  [ -s "$REF_FASTA" ] || fail "REF_FASTA not found: $REF_FASTA"
  REF_FA="$REF_FASTA"                        # real genome, used verbatim
  echo "using real reference: $REF_FA ($(grep -vc '^>' "$REF_FA" | tr -d ' ') seq lines)"
else
  REF_FA="$OUT/ref.fa"
  if [ ! -s "$REF_FA" ]; then
    "$PYTHON" - "$REF_BP" "$SEED" "$REF_FA" <<'PY'
import sys, random
n, seed, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
random.seed(seed)
seq = ''.join(random.choice('ACGT') for _ in range(n))
with open(path, 'w') as f:
    f.write('>ref\n')
    for i in range(0, len(seq), 80):
        f.write(seq[i:i+80] + '\n')
print(f"wrote {path}: {len(seq)} bp")
PY
  else
    echo "reuse existing $REF_FA"
  fi
fi

PAFSTATS_PY="$OUT/pafstats.py"
if [ ! -s "$PAFSTATS_PY" ]; then
  URL="https://raw.githubusercontent.com/skovaka/UNCALLED/master/uncalled/pafstats.py"
  ( command -v wget >/dev/null && wget -q -O "$PAFSTATS_PY" "$URL" ) \
    || ( command -v curl >/dev/null && curl -sS -o "$PAFSTATS_PY" "$URL" ) \
    || echo "WARN: could not download pafstats.py — the pafstats cross-check will be skipped."
fi
[ -s "$PAFSTATS_PY" ] && echo "pafstats.py ready: $PAFSTATS_PY" || { echo "no pafstats.py"; PAFSTATS_PY=""; }

# -----------------------------------------------------------------------------
say "2. pilot — index (both strands) + train + export PAFs"
if [ -s "$OUT/ground_truth.paf" ] && [ -s "$OUT/squiggleseek.paf" ] && [ -s "$OUT/reads.blow5" ] \
   && [ "${FORCE_PILOT:-0}" != "1" ]; then
  echo "reuse existing pilot outputs (ground_truth.paf / squiggleseek.paf / reads.blow5)."
  echo "  -> skipping train+index+simulate. Set FORCE_PILOT=1 to redo the pilot."
else
  PILOT_ARGS=(
    --reference_fasta "$REF_FA"
    --pore_model "$PORE_MODEL_PATH"
    --ref_bp 0
    --refine none
    --n_train "$N_TRAIN" --n_query "$N_QUERY" --train_steps "$TRAIN_STEPS"
    --batch_size "$BATCH_SIZE" --hard_negatives "$HARD_NEG"
    --forward_only "$FORWARD_ONLY" --both_strands "$BOTH_STRANDS"
    --encoder_type "$ENCODER_TYPE" --overlap "$OVERLAP"
    --fast_sweep "$FAST_SWEEP" --faiss_cpu "$FAISS_CPU"
    --seed "$SEED"
    --paf_out_dir "$OUT"
  )
  [ -n "$AMP_NOISE" ] && PILOT_ARGS+=( --amp_noise "$AMP_NOISE" )
  [ -n "$DWELL_STD" ] && PILOT_ARGS+=( --dwell_std "$DWELL_STD" )
  echo "batch_size=$BATCH_SIZE  hard_negatives=$HARD_NEG  forward_only=$FORWARD_ONLY  both_strands=$BOTH_STRANDS  encoder_type=$ENCODER_TYPE  overlap=$OVERLAP  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  if [ -n "$LOAD_ENCODER" ] && [ -f "$LOAD_ENCODER" ]; then
    PILOT_ARGS+=( --load_encoder "$LOAD_ENCODER" )
    echo "loading encoder: $LOAD_ENCODER (training skipped)"
  else
    PILOT_ARGS+=( --save_encoder "$SAVE_ENCODER" )
    echo "training a fresh encoder -> $SAVE_ENCODER"
  fi
  "$PYTHON" "$REPO/evaluate/pilot_recall.py" "${PILOT_ARGS[@]}" || fail "pilot_recall.py failed"
fi

[ -s "$OUT/ground_truth.paf" ] || fail "ground_truth.paf not produced"
[ -s "$OUT/squiggleseek.paf" ] || fail "squiggleseek.paf not produced"
[ -s "$OUT/reads.blow5" ]      || fail "reads.blow5 not produced"
echo "truth reads   : $(wc -l < "$OUT/ground_truth.paf")"
echo "squiggleseek  : $(wc -l < "$OUT/squiggleseek.paf") mapped"

# -----------------------------------------------------------------------------
MATCH_ARGS=( --sweep SquiggleSeek --match_to RawHash2 )
if [ "$HAVE_RAWHASH" -eq 1 ] && [ "$SKIP_RAWHASH" != "1" ]; then
  say "3. RawHash2 on the identical reads.blow5 (preset=$RAWHASH_PRESET)"
  echo "rawhash2 pore model: $RAWHASH_PORE"
  "$RAWHASH2" -x "$RAWHASH_PRESET" -p "$RAWHASH_PORE" -t "$THREADS" -d "$OUT/ref.idx" "$OUT/ref.fasta" \
    || fail "rawhash2 index build failed"
  "$RAWHASH2" -x "$RAWHASH_PRESET" -t "$THREADS" "$OUT/ref.idx" "$OUT/reads.blow5" \
    > "$OUT/rawhash2.paf" || fail "rawhash2 mapping failed"
  echo "rawhash2 lines: $(wc -l < "$OUT/rawhash2.paf")"
  PAF_ARGS=( --paf "SquiggleSeek=$OUT/squiggleseek.paf" --paf "RawHash2=$OUT/rawhash2.paf" )
else
  if [ "$SKIP_RAWHASH" = "1" ]; then
    say "3. RawHash2 SKIPPED (SKIP_RAWHASH=1) — SquiggleSeek PR sweep only (Gate-2 A/B/C)"
  else
    say "3. RawHash2 SKIPPED (binary missing) — scoring SquiggleSeek only"
  fi
  PAF_ARGS=( --paf "SquiggleSeek=$OUT/squiggleseek.paf" )
  MATCH_ARGS=( --sweep SquiggleSeek )   # no RawHash target; read recall@P from the sweep table
fi

# -----------------------------------------------------------------------------
run_scorer() {  # $1 = tag, rest = extra args
  local tag="$1"; shift
  say "score: $tag"
  {
    echo "########## SCORER: $tag ##########"
    "$PYTHON" "$REPO/evaluate/rawhash_compare.py" \
      --truth "$OUT/ground_truth.paf" \
      "${PAF_ARGS[@]}" \
      "${MATCH_ARGS[@]}" \
      "$@"
  } 2>&1 | tee -a "$SUMMARY"
}

say "4. scoring — builtin + real pafstats"
run_scorer "builtin" --scorer builtin
if [ -n "$PAFSTATS_PY" ]; then
  run_scorer "pafstats (real UNCALLED, no HDF5)" --scorer pafstats --uncalled "$PAFSTATS_PY"
else
  echo "pafstats cross-check skipped (no pafstats.py)" | tee -a "$SUMMARY"
fi

# -----------------------------------------------------------------------------
say "5. per-strand recall (forward vs reverse, builtin locus criterion)"
"$PYTHON" - "$REPO/evaluate" "$OUT/ground_truth.paf" "$OUT/squiggleseek.paf" <<'PY' 2>&1 | tee -a "$SUMMARY"
import sys, os
sys.path.insert(0, sys.argv[1])
from rawhash_compare import _read_paf, _score_truth_tool
truth = _read_paf(sys.argv[2]); tool = _read_paf(sys.argv[3])
for strand, label in (("+", "forward"), ("-", "reverse")):
    sub = {q: v for q, v in truth.items() if v[3] == strand}
    if not sub:
        print(f"{label:8s}: no truth reads"); continue
    m = _score_truth_tool(sub, tool, require_strand=True)
    print(f"{label:8s}: n={len(sub):6d}  TP={m['tp']:6d}  FP={m['fp']:5d}  FN={m['fn']:5d}  "
          f"P={m['precision']*100:5.1f}  R={m['recall']*100:5.1f}  F1={m['f1']*100:5.1f}")
PY

say "DONE  ($(date))"
echo "full log : $LOG"
echo "summary  : $SUMMARY"
echo
echo "==== collect these 3 things from $SUMMARY ===="
echo "  A. work-point matched table + PR sweep points (per scorer)"
echo "  B. builtin vs pafstats head-to-head tables side by side (should agree <1pp)"
echo "  C. full both-strand head-to-head + 'extra' column, and per-strand recall (step 5)"
