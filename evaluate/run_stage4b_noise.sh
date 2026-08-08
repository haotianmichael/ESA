#!/usr/bin/env bash
# =============================================================================
# STAGE4b Step 4 — real-read noise-robustness head-to-head (SquiggleSeek v2 vs
# RawHash2). One driver: build a test-only blow5 once, then for each noise level k
# add additive Gaussian noise (sigma = k*MAD on the RAW signal, nested draws), run
# the full head-to-head on the SAME noised file for both tools, record P/R/F1, and
# plot F1-vs-k. Everything isolated under experiments/stage4b_noise/.
#
#   x-axis = additive Gaussian noise on REAL reads (x read MAD) — NOT squigulator
#   amp_noise. This is a separate panel from the STAGE2 (simulated) sweep.
#
# k=0 is the SELF-CHECK: it must reproduce Step 3 (SS~91.6 / RawHash2~97.2). If it
# does not, the subset/writer is buggy and the run aborts before the sweep.
#
# Usage:
#   nohup bash evaluate/run_stage4b_noise.sh > stage4b_noise.out 2>&1 &
# =============================================================================
set -u
set -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BASE="${BASE:-/home/nfs/mahaotian/ESA}"
export PORE_MODEL_PATH="${PORE_MODEL_PATH:-$BASE/squigulator/r9_6mer_pore_model.txt}"
PYTHON="${PYTHON:-python}"

D="$BASE/CALL_ESA/data/d2_ecoli_r94"
FULL_B5="$D/ecoli_R9.blow5"
REF="$D/ref.fa"
DATA="$BASE/CALL_ESA/experiments/stage4b_data"
TEST_IDS="$DATA/test_reads.txt"
V2="$BASE/CALL_ESA/experiments/stage4b_finetune/real_encoder_v1.pt"   # F1=91.6 checkpoint

OUT_ROOT="${OUT_ROOT:-$BASE/CALL_ESA/experiments/stage4b_noise}"
NOISE_CSV="${NOISE_CSV:-$REPO/evaluate/stage4b_noise.csv}"
K_LIST="${K_LIST:-0.0 0.5 1.0 1.5 2.0}"
BASE_SEED="${BASE_SEED:-42}"
EXPECT_N="${EXPECT_N:-17570}"          # Step 3 test read count
GATE_RH_MIN="${GATE_RH_MIN:-95}"       # k=0 RawHash2 F1 must be ~97.2
GATE_SS_MIN="${GATE_SS_MIN:-88}"       # k=0 SquiggleSeek F1 must be ~91.6

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1
say()  { echo -e "\n########## $* ##########"; }
fail() { echo "FATAL: $*"; exit 1; }

say "STAGE4b noise sweep  ($(date))"
echo "v2 checkpoint = $V2"
echo "k list = {$K_LIST}  base_seed=$BASE_SEED  csv=$NOISE_CSV"
[ -s "$FULL_B5" ] || fail "full blow5 missing: $FULL_B5"
[ -s "$REF" ]     || fail "reference missing: $REF"
[ -s "$TEST_IDS" ]|| fail "test read-id list missing: $TEST_IDS (run Step 1)"
[ -s "$V2" ]      || fail "v2 checkpoint missing: $V2 (run Step 2)"

# reuse Step 3's test truth if present, else regenerate from the full truth PAF
TEST_TRUTH="$BASE/CALL_ESA/experiments/stage4b_realtest/test_truth.paf"
if [ ! -s "$TEST_TRUTH" ]; then
  TEST_TRUTH="$OUT_ROOT/test_truth.paf"
  echo "[stage4b-noise] regenerating test truth -> $TEST_TRUTH"
  grep -Ff "$TEST_IDS" "$D/truth.paf" > "$TEST_TRUTH" || fail "could not build test_truth.paf"
fi
echo "test truth = $TEST_TRUTH ($(wc -l < "$TEST_TRUTH") lines)"

# ---- build the test-only blow5 ONCE (slow5tools if available, else pyslow5) ----
TESTB5="$OUT_ROOT/test.blow5"
say "0. build test-only blow5"
if [ ! -s "$TESTB5" ]; then
  if command -v slow5tools >/dev/null 2>&1; then
    echo "slow5tools found -> trying 'slow5tools get' for the subset"
    slow5tools index "$FULL_B5" >/dev/null 2>&1 || true
    if ! ( slow5tools get "$FULL_B5" --list "$TEST_IDS" -o "$TESTB5" >/dev/null 2>&1 && [ -s "$TESTB5" ] ); then
      echo "slow5tools get failed -> falling back to pyslow5 writer"; rm -f "$TESTB5"
      "$PYTHON" "$HERE/stage4b_noise_blow5.py" --in_blow5 "$FULL_B5" --out_blow5 "$TESTB5" \
        --k 0 --read_ids "$TEST_IDS" || fail "pyslow5 subset failed"
    fi
  else
    echo "slow5tools not found -> pyslow5 subset"
    "$PYTHON" "$HERE/stage4b_noise_blow5.py" --in_blow5 "$FULL_B5" --out_blow5 "$TESTB5" \
      --k 0 --read_ids "$TEST_IDS" || fail "pyslow5 subset failed"
  fi
else
  echo "reuse existing $TESTB5"
fi
N_TEST="$("$PYTHON" -c "import pyslow5,sys; s=pyslow5.Open(sys.argv[1],'r'); print(sum(1 for _ in s.seq_reads(pA=False)))" "$TESTB5")"
echo "test.blow5 reads = $N_TEST  (expect $EXPECT_N)"
[ "$N_TEST" = "$EXPECT_N" ] || fail "test.blow5 read count $N_TEST != $EXPECT_N — subset bug, stop."

# ---- sweep ------------------------------------------------------------------
for k in $K_LIST; do
  KD="$OUT_ROOT/k${k}"
  say "noise level k=$k -> $KD"
  mkdir -p "$KD"
  NB5="$KD/noised.blow5"
  # k=0 goes through the pyslow5 writer too (lossless) so the self-check validates it
  "$PYTHON" "$HERE/stage4b_noise_blow5.py" --in_blow5 "$TESTB5" --out_blow5 "$NB5" \
    --k "$k" --base_seed "$BASE_SEED" || fail "noise write failed at k=$k"

  REAL_READS="$NB5" REAL_REF="$REF" TRUTH_PAF="$TEST_TRUTH" READ_IDS="$TEST_IDS" \
  CHECKPOINT="$V2" OUT="$KD" TRIM_MODE=fixed TRIM_FIXED=2500 LIMIT=0 FAISS_CPU=1 \
  HEAD2HEAD_CSV="$KD/head2head.csv" \
    bash "$HERE/run_real_data.sh" || fail "head-to-head failed at k=$k"

  "$PYTHON" "$HERE/stage4b_noise_point.py" --summary "$KD/SUMMARY_real.txt" --k "$k" --csv "$NOISE_CSV" \
    || fail "scoring/append failed at k=$k"

  # ---- k=0 GATE: must reproduce Step 3 ----
  if [ "$k" = "0.0" ] || [ "$k" = "0" ]; then
    read SSF1 RHF1 < <(awk '$1=="SquiggleSeek" && NF==8 {ss=$8} $1=="RawHash2" && NF==8 {rh=$8} END{print ss, rh}' "$KD/SUMMARY_real.txt")
    echo "[gate] k=0 SquiggleSeek F1=$SSF1  RawHash2 F1=$RHF1  (expect ~91.6 / ~97.2)"
    awk -v v="$RHF1" -v m="$GATE_RH_MIN" 'BEGIN{exit !(v+0>=m+0)}' \
      || fail "k=0 RawHash2 F1=$RHF1 < $GATE_RH_MIN — subset/writer bug (does not reproduce Step 3). STOP."
    awk -v v="$SSF1" -v m="$GATE_SS_MIN" 'BEGIN{exit !(v+0>=m+0)}' \
      || fail "k=0 SquiggleSeek F1=$SSF1 < $GATE_SS_MIN — subset/writer bug (does not reproduce Step 3). STOP."
    echo "[gate] k=0 reproduces Step 3 — noising pipeline OK, continuing sweep."
  fi
done

# ---- plot -------------------------------------------------------------------
say "plot F1-vs-k"
PNG="$OUT_ROOT/noise_curve_real.png"
"$PYTHON" "$HERE/plot_stage4b_noise.py" --csv "$NOISE_CSV" --out "$PNG" \
  || echo "[warn] plotting failed (matplotlib?); CSV is still at $NOISE_CSV"

say "DONE  ($(date))"
echo "CSV   -> $NOISE_CSV"
echo "curve -> $PNG"
echo "per-k dirs -> $OUT_ROOT/k*/  (SUMMARY_real.txt each)"
echo "== read: does RawHash2 F1 fall FASTER than SquiggleSeek as k rises? is there a crossover? =="
