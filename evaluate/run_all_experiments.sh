#!/usr/bin/env bash
# =============================================================================
# SquiggleSeek — retrain EVERYTHING and run all paper experiments (orchestrator).
#
# One resumable driver for the four stages, all on the LOCKED canonical config
# (2-GPU DDP training, effective contrastive batch 48, input_signal_len 3000,
# 20000 steps, full 4.64 Mb E. coli genome, hard negatives, seed 42):
#
#   STAGE 1  canonical DDP training  -> $CKPT (+ default-noise head-to-head)
#   STAGE 2  noise sweep (loads $CKPT, single-GPU) vs RawHash2  -> noise_sweep.csv
#   STAGE 3  five ablations, each DDP-training its own encoder at batch 48
#   STAGE 4  real-data head-to-head (loads $CKPT, single-GPU) vs RawHash2
#
# The canonical checkpoint is trained ONCE in STAGE 1 and reused by 2 & 4;
# input_signal_len is restored from the checkpoint by load_encoder, so those
# stages inherit isl=3000. The canonical model is ALSO the mamba/H8/stride15 point
# of the STAGE 3 ablation grid (not retrained).
#
# Resumable: a stage whose output already exists is skipped, so re-running after a
# crash continues where it stopped. Select stages with STAGES="1 3".
#
# Usage (run server, conda env py310, 2x V100 32GB):
#     nohup bash evaluate/run_all_experiments.sh > run_all_nohup.out 2>&1 &
#
# NOTHING here is a "cheaper" fallback: every value below is the locked config.
# =============================================================================
set -u
set -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# ---- locked config (env-overridable, but overriding breaks comparability) ----
BASE="${BASE:-/home/nfs/mahaotian/ESA}"
ENCODER_TYPE_MAIN="${ENCODER_TYPE_MAIN:-mamba}"
INPUT_SIGNAL_LEN="${INPUT_SIGNAL_LEN:-2000}"   # samples/window (LOCKED 2000 = verified A1_v3; 3000 dropped recall@1 to 66%)
N_MAMBA_BLOCKS="${N_MAMBA_BLOCKS:-6}"    # encoder depth (LOCKED 6 = verified A1_v3 default; history shows 6 throughout)
HARD_NEG_MAIN="${HARD_NEG_MAIN:-8}"
OVERLAP_MAIN="${OVERLAP_MAIN:-285}"
N_TRAIN="${N_TRAIN:-100000}"
N_QUERY="${N_QUERY:-5000}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
SEED="${SEED:-42}"
FAISS_CPU="${FAISS_CPU:-1}"
NPROC="${NPROC:-2}"

# ---- paths (already provisioned on the run server; referenced, never made) ----
REF_KECOLI="${REF_KECOLI:-$BASE/CALL_ESA/ecoli.fna}"                       # full 4.64 Mb E. coli K-12
REF_REAL="${REF_REAL:-$BASE/CALL_ESA/data/d2_ecoli_r94/ref.fa}"           # real-data reference
REAL_BLOW5="${REAL_BLOW5:-$BASE/CALL_ESA/data/d2_ecoli_r94/ecoli_R9.blow5}"
REAL_TRUTH="${REAL_TRUTH:-$BASE/CALL_ESA/data/d2_ecoli_r94/truth.paf}"
CKPT="${CKPT:-$BASE/CALL_ESA/canonical_encoder.pt}"
OUT_ROOT="${OUT_ROOT:-$BASE/CALL_ESA/experiments}"

# real-data leader/adapter trim (validated: 2500 samples for R9 E. coli)
TRIM_MODE="${TRIM_MODE:-fixed}"
TRIM_FIXED="${TRIM_FIXED:-2500}"
REAL_LIMIT="${REAL_LIMIT:-0}"

# noise grid (headline figure): dwell fixed at squigulator default (4), amp swept
AMP_LIST="${AMP_LIST:-default 1.5 2.0 3.0 5.0}"
DWELL_LIST="${DWELL_LIST:-4}"

# STAGE-1 reproducibility gate: the pilot's own trained>untrained>random go/no-go
# is the pass/fail signal. GATE_R1_MIN is only a low collapse-floor on the strict
# +/-15bp trained recall@1 (a much smaller number than head-to-head @all recall),
# used to catch a totally-collapsed run — NOT to enforce ~99%.
GATE_R1_MIN="${GATE_R1_MIN:-30}"

STAGES="${STAGES:-1 2 3 4}"
PYTHON="${PYTHON:-python}"

# squigulator on PATH (training/noise sweep simulate reads)
export PATH="$PATH:$BASE/squigulator"

mkdir -p "$OUT_ROOT"
MASTER_LOG="$OUT_ROOT/run_all.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

say()  { echo -e "\n########################################  $*  ########################################"; }
fail() { echo "FATAL: $*"; exit 1; }
have_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

say "run_all_experiments  ($(date))"
echo "BASE=$BASE  OUT_ROOT=$OUT_ROOT  CKPT=$CKPT"
echo "locked: encoder=$ENCODER_TYPE_MAIN isl=$INPUT_SIGNAL_LEN n_mamba_blocks=$N_MAMBA_BLOCKS H=$HARD_NEG_MAIN overlap=$OVERLAP_MAIN"
echo "        n_train=$N_TRAIN train_steps=$TRAIN_STEPS batch=$BATCH_SIZE seed=$SEED nproc=$NPROC"
echo "STAGES=$STAGES  amp={$AMP_LIST} dwell={$DWELL_LIST}"

# ---- fail-fast: the four data files must exist (never regenerate them) --------
command -v squigulator >/dev/null || fail "squigulator not on PATH ($BASE/squigulator)"
command -v torchrun    >/dev/null || fail "torchrun not found (conda env py310 not active?)"
[ -s "$REF_KECOLI" ] || fail "REF_KECOLI missing: $REF_KECOLI"
[ -s "$REF_REAL" ]   || fail "REF_REAL missing: $REF_REAL"
[ -s "$REAL_BLOW5" ] || fail "REAL_BLOW5 missing: $REAL_BLOW5"
[ -s "$REAL_TRUTH" ] || fail "REAL_TRUTH missing: $REAL_TRUTH"
echo "data OK: $REF_KECOLI ; $REF_REAL ; $REAL_BLOW5 ; $REAL_TRUTH"

# =============================================================================
# STAGE 1 — canonical DDP training + default-noise head-to-head
# =============================================================================
STAGE1_OUT="$OUT_ROOT/stage1_main"
if have_stage 1; then
  if [ -s "$CKPT" ] && [ -s "$STAGE1_OUT/SUMMARY.txt" ]; then
    say "STAGE 1 SKIP — $CKPT and $STAGE1_OUT/SUMMARY.txt already exist"
  else
    say "STAGE 1 — canonical DDP training (torchrun --nproc_per_node=$NPROC) -> $CKPT"
    REF_FASTA="$REF_KECOLI" REF_BP=0 \
    OUT="$STAGE1_OUT" SAVE_ENCODER="$CKPT" \
    ENCODER_TYPE="$ENCODER_TYPE_MAIN" HARD_NEG="$HARD_NEG_MAIN" OVERLAP="$OVERLAP_MAIN" \
    INPUT_SIGNAL_LEN="$INPUT_SIGNAL_LEN" N_MAMBA_BLOCKS="$N_MAMBA_BLOCKS" BATCH_SIZE="$BATCH_SIZE" \
    N_TRAIN="$N_TRAIN" N_QUERY="$N_QUERY" TRAIN_STEPS="$TRAIN_STEPS" \
    FORWARD_ONLY=0 BOTH_STRANDS=1 FAST_SWEEP=0 FAISS_CPU="$FAISS_CPU" \
    NPROC="$NPROC" SEED="$SEED" SKIP_RAWHASH=0 \
      bash "$HERE/run_head2head.sh" || fail "STAGE 1 head-to-head failed"

    [ -s "$CKPT" ] || fail "STAGE 1 did not produce the checkpoint: $CKPT"

    # reproducibility gate. PRIMARY signal = the pilot's own '[go/no-go] ... GO'
    # line (trained > untrained > random @10). SECONDARY = trained recall@1 from
    # the 'recall (tol +/-15bp, coverage)' table — parsed with sed so the '1' in
    # '@1=' is NOT mistaken for the value (the old bug that read 66.0% as 1%). Read
    # NEITHER the PR-sweep rows nor any '@P>=...' work-point row.
    RUNLOG="$STAGE1_OUT/run.log"
    GO_LINE="$(grep '\[go/no-go\]' "$RUNLOG" 2>/dev/null | tail -1)"
    R1="$(grep -E '^[[:space:]]+trained[[:space:]]' "$RUNLOG" 2>/dev/null \
          | tail -1 | sed -nE 's/.*@1=[[:space:]]*([0-9]+(\.[0-9]+)?).*/\1/p')"
    echo "[gate] STAGE 1 go/no-go='${GO_LINE:-<none>}'  trained recall@1=${R1:-<unparsed>}% (collapse floor ${GATE_R1_MIN}%)"
    if echo "$GO_LINE" | grep -q 'NO-GO'; then
      fail "STAGE 1 GATE: pilot reported NO-GO (trained NOT > untrained > random @10). STOPPING — investigate before running stages 2-4."
    elif echo "$GO_LINE" | grep -q ': GO'; then
      echo "[gate] PASS — pilot go/no-go = GO; proceeding to stages 2-4."
    elif [ -n "$R1" ]; then
      # no go/no-go line (e.g. a --fast_sweep run): fall back to the collapse floor.
      awk -v r="$R1" -v m="$GATE_R1_MIN" 'BEGIN{exit !(r+0 >= m+0)}' \
        && echo "[gate] PASS — recall@1 ${R1}% >= collapse floor ${GATE_R1_MIN}%." \
        || fail "STAGE 1 GATE: recall@1 ${R1}% below collapse floor ${GATE_R1_MIN}% (run looks collapsed). STOPPING."
    else
      echo "[gate][WARN] could not read go/no-go or recall@1 from $RUNLOG — inspect it manually before trusting the checkpoint."
    fi
  fi
fi

# =============================================================================
# STAGE 2 — noise sweep (loads $CKPT, single-GPU) vs RawHash2
# =============================================================================
STAGE2_OUT="$OUT_ROOT/stage2_noise"
NOISE_CSV="$OUT_ROOT/noise_sweep.csv"
if have_stage 2; then
  if [ -s "$NOISE_CSV" ]; then
    say "STAGE 2 SKIP — $NOISE_CSV already exists"
  else
    [ -s "$CKPT" ] || fail "STAGE 2 needs the canonical checkpoint: $CKPT (run STAGE 1 first)"
    say "STAGE 2 — noise sweep loading $CKPT -> $NOISE_CSV"
    CHECKPOINT="$CKPT" SWEEP_OUT="$STAGE2_OUT" CSV="$NOISE_CSV" \
    REF_FASTA="$REF_KECOLI" REF_BP=0 \
    AMP_LIST="$AMP_LIST" DWELL_LIST="$DWELL_LIST" \
    N_QUERY="$N_QUERY" SEED="$SEED" FAISS_CPU="$FAISS_CPU" \
      bash "$HERE/run_noise_sweep.sh" || fail "STAGE 2 noise sweep failed"
  fi
fi

# =============================================================================
# STAGE 3 — five ablations, each DDP-training its OWN encoder (batch 48, isl 3000,
# 20000 steps, full genome, seed 42). Only the single ablated knob changes.
# mamba/H8/stride15 is STAGE 1 (not retrained). SKIP_RAWHASH=1 (SquiggleSeek only).
# =============================================================================
run_ablation() {  # $1=tag  rest=EXTRA env assignments handled by caller
  local tag="$1"; shift
  local out="$OUT_ROOT/stage3_$tag"
  local ckpt="$out/encoder.pt"
  if [ -s "$out/SUMMARY.txt" ]; then
    say "STAGE 3 [$tag] SKIP — $out/SUMMARY.txt exists"
    return 0
  fi
  say "STAGE 3 [$tag] — DDP training its own encoder -> $ckpt"
  # shared locked config; caller has exported the single ablated override
  REF_FASTA="$REF_KECOLI" REF_BP=0 \
  OUT="$out" SAVE_ENCODER="$ckpt" \
  INPUT_SIGNAL_LEN="$INPUT_SIGNAL_LEN" N_MAMBA_BLOCKS="$N_MAMBA_BLOCKS" BATCH_SIZE="$BATCH_SIZE" \
  N_TRAIN="$N_TRAIN" N_QUERY="$N_QUERY" TRAIN_STEPS="$TRAIN_STEPS" \
  FORWARD_ONLY=0 BOTH_STRANDS=1 FAST_SWEEP=0 FAISS_CPU="$FAISS_CPU" \
  NPROC="$NPROC" SEED="$SEED" SKIP_RAWHASH=1 \
  ENCODER_TYPE="${AB_ENCODER_TYPE:-$ENCODER_TYPE_MAIN}" \
  HARD_NEG="${AB_HARD_NEG:-$HARD_NEG_MAIN}" \
  OVERLAP="${AB_OVERLAP:-$OVERLAP_MAIN}" \
    bash "$HERE/run_head2head.sh" || fail "STAGE 3 [$tag] failed"
}

if have_stage 3; then
  say "STAGE 3 — ablations (canonical mamba/H8/stride15 = STAGE 1)"
  ( AB_ENCODER_TYPE=transformer; export AB_ENCODER_TYPE; run_ablation enc_transformer )
  ( AB_ENCODER_TYPE=cnn_rnn;     export AB_ENCODER_TYPE; run_ablation enc_cnnrnn )
  ( AB_HARD_NEG=0;               export AB_HARD_NEG;     run_ablation hn0 )
  ( AB_HARD_NEG=16;              export AB_HARD_NEG;     run_ablation hn16 )
  ( AB_OVERLAP=150;              export AB_OVERLAP;      run_ablation stride150 )
fi

# =============================================================================
# STAGE 4 — real-data head-to-head (loads $CKPT, single-GPU) vs RawHash2
# =============================================================================
STAGE4_OUT="$OUT_ROOT/stage4_real"
if have_stage 4; then
  if [ -s "$STAGE4_OUT/SUMMARY_real.txt" ]; then
    say "STAGE 4 SKIP — $STAGE4_OUT/SUMMARY_real.txt exists"
  else
    [ -s "$CKPT" ] || fail "STAGE 4 needs the canonical checkpoint: $CKPT (run STAGE 1 first)"
    say "STAGE 4 — real data loading $CKPT -> $STAGE4_OUT"
    REAL_READS="$REAL_BLOW5" REAL_REF="$REF_REAL" TRUTH_PAF="$REAL_TRUTH" \
    CHECKPOINT="$CKPT" OUT="$STAGE4_OUT" \
    TRIM_MODE="$TRIM_MODE" TRIM_FIXED="$TRIM_FIXED" LIMIT="$REAL_LIMIT" \
    FAISS_CPU="$FAISS_CPU" \
      bash "$HERE/run_real_data.sh" || fail "STAGE 4 real-data failed"
  fi
fi

# =============================================================================
say "DONE  ($(date))  — collect these results"
echo "  STAGE 1 : $STAGE1_OUT/SUMMARY.txt   (+ run.log for trained recall@1)"
echo "  STAGE 2 : $NOISE_CSV                 (per-point dirs under $STAGE2_OUT/)"
echo "  STAGE 3 : $OUT_ROOT/stage3_{enc_transformer,enc_cnnrnn,hn0,hn16,stride150}/SUMMARY.txt"
echo "  STAGE 4 : $STAGE4_OUT/SUMMARY_real.txt"
echo "  checkpoint: $CKPT"
