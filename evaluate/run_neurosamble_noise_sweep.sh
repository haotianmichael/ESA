#!/usr/bin/env bash
# =============================================================================
# Neurosamble Phase 3 -- event-noise robustness sweep (headline figure).
#
# Adds event-mode block-constant noise (sigma = k * MAD(raw)) on top of REAL
# R9.4 reads and, at each k, reruns the read<->read head-to-head against a FIXED
# overlap truth built ONCE from the CLEAN basecalled reads (we never re-basecall).
# Primary metric: overlap recall / F1 vs k -- Neurosamble (continuous embedding
# retrieval) should degrade slower than Rawsamble (discrete hashing).
#
# The reads are real, so k=0 already contains full physical device noise: this is
# "real device noise -> real + extra event noise", a controlled stress test. k=0
# is a LOSSLESS copy and MUST reproduce the Phase 2 numbers (75.1 / 54.5).
#
# Positional args:
#   1 OUTDIR              output root
#   2 CLEAN_SUBSET_BLOW5  k=0 lossless subset blow5 (noise source)
#   3 SUBSET_FASTA        clean basecalled subset reads (FIXED truth + miniasm seq)
#   4 REF                 full CFT073 (AE014075.1) reference (chained %)
#   5 READ_IDS            fixed subset read-id list
#   6 PORE                ONT pore model (rawhash2 -p)
#   7 RAWHASH2_BIN        rawhash2 binary
#   8 THREADS             thread count
#   9 SCRIPTS_DIR         RawHash test/scripts dir
#  10 NOISE_BLOCK         (optional, default 9) event-mode block length
#  11 K_LIST              (optional, default "0 0.05 0.1 0.15 0.2 0.25 0.4")
#
# Required env: LOAD_ENCODER=<encoder .pt>
# Optional env: MINIASM, MINIMAP2, PYTHON, DEVICE, SEED (calib, default 1234),
#               NOISE_BASE_SEED (default 42)
#
# Isolation: one sub-dir per k (OUTDIR/k{k}/); the fixed truth and calibration
# are computed once and never overwritten.
# =============================================================================
set -euo pipefail

if [[ $# -lt 9 ]]; then
  echo "usage: $0 OUTDIR CLEAN_SUBSET_BLOW5 SUBSET_FASTA REF READ_IDS PORE RAWHASH2_BIN THREADS SCRIPTS_DIR [NOISE_BLOCK] [K_LIST]" >&2
  echo "       (env: LOAD_ENCODER=<encoder.pt> required)" >&2
  exit 2
fi

OUTDIR="$1"; CLEAN_SUBSET_BLOW5="$2"; SUBSET_FASTA="$3"; REF="$4"; READ_IDS="$5"
PORE="$6"; RAWHASH2_BIN="$7"; THREADS="$8"; SCRIPTS_DIR="$9"
NOISE_BLOCK="${10:-9}"
K_LIST="${11:-0 0.05 0.1 0.15 0.2 0.25 0.4}"

: "${LOAD_ENCODER:?set LOAD_ENCODER=<path to encoder .pt>}"
MINIASM="${MINIASM:-miniasm}"
MINIMAP2="${MINIMAP2:-minimap2}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-1234}"
NOISE_BASE_SEED="${NOISE_BASE_SEED:-42}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # evaluate/
mkdir -p "$OUTDIR"
echo "[sweep] OUTDIR=$OUTDIR"
echo "[sweep] NOISE_BLOCK=$NOISE_BLOCK  K_LIST=[$K_LIST]"
echo "[sweep] SEED(calib)=$SEED  NOISE_BASE_SEED=$NOISE_BASE_SEED  block/mode=event/$NOISE_BLOCK"

# --------------------------------------------------------------------------- #
# Step 0 -- intrinsic event-noise calibration (measure only)
# --------------------------------------------------------------------------- #
echo "[sweep] === Step 0: calibrate intrinsic event noise (r_real) ==="
R_REAL_FILE="$OUTDIR/r_real_L${NOISE_BLOCK}.txt"
"$PYTHON" "$HERE/neurosamble_noise_calib.py" \
  --blow5 "$CLEAN_SUBSET_BLOW5" --block "$NOISE_BLOCK" --n 200 \
  --seed "$SEED" --out "$R_REAL_FILE"
R_REAL="$(tail -n 1 "$R_REAL_FILE")"
echo "[sweep] r_real=$R_REAL (physically-plausible band: k <= r_real)"

# --------------------------------------------------------------------------- #
# Step 1 -- build the overlap truth ONCE from the CLEAN reads (fixed across k)
# --------------------------------------------------------------------------- #
FIXED_TRUTH="$OUTDIR/mm2_overlaps.paf"
if [[ -s "$FIXED_TRUTH" ]]; then
  echo "[sweep] === Step 1: reusing existing fixed truth $FIXED_TRUTH ==="
else
  echo "[sweep] === Step 1: building fixed clean overlap truth ==="
  "$MINIMAP2" -x ava-ont --for-only -t "$THREADS" "$SUBSET_FASTA" "$SUBSET_FASTA" \
    > "$FIXED_TRUTH" 2> "$OUTDIR/mm2_overlaps.log"
fi
echo "[sweep] fixed truth lines: $(wc -l < "$FIXED_TRUTH")"

# --------------------------------------------------------------------------- #
# Step 2 -- sweep k
# --------------------------------------------------------------------------- #
for k in $K_LIST; do
  echo ""
  echo "[sweep] ===================== k=$k ====================="
  KD="$OUTDIR/k${k}"
  mkdir -p "$KD"
  NOISED="$KD/subset_L${NOISE_BLOCK}_k${k}.blow5"

  echo "[sweep] noise blow5 (event, block=$NOISE_BLOCK, k=$k) -> $NOISED"
  "$PYTHON" "$HERE/stage4b_noise_blow5.py" \
    --in_blow5 "$CLEAN_SUBSET_BLOW5" --out_blow5 "$NOISED" \
    --k "$k" --noise_mode event --block "$NOISE_BLOCK" \
    --read_ids "$READ_IDS" --base_seed "$NOISE_BASE_SEED" \
    2>&1 | tee "$KD/noise_write_k${k}.log"

  echo "[sweep] head-to-head at k=$k (fixed clean truth)"
  LOAD_ENCODER="$LOAD_ENCODER" DO_ASSEMBLY=1 MIN_CHAINING_SCORE=40 \
  MINIASM="$MINIASM" MINIMAP2="$MINIMAP2" PYTHON="$PYTHON" DEVICE="$DEVICE" \
  FIXED_TRUTH_PAF="$FIXED_TRUTH" \
  bash "$HERE/run_neurosamble_head2head.sh" \
    "$KD" "$NOISED" "$SUBSET_FASTA" "$REF" "$READ_IDS" \
    "$PORE" "$RAWHASH2_BIN" "$THREADS" "$SCRIPTS_DIR" \
    2>&1 | tee "$KD/head2head_k${k}.log"
done

# --------------------------------------------------------------------------- #
# Step 3-4 -- aggregate CSV + curves
# --------------------------------------------------------------------------- #
echo ""
echo "[sweep] === Step 3-4: aggregate + plot ==="
"$PYTHON" "$HERE/neurosamble_noise_aggregate.py" \
  --outdir "$OUTDIR" --block "$NOISE_BLOCK" --seed "$SEED" \
  --r_real "$R_REAL" --k_list "$K_LIST"

echo ""
echo "[sweep] DONE. Key outputs:"
echo "  $OUTDIR/noise_sweep_L${NOISE_BLOCK}.csv   (metadata line has r_real)"
echo "  $OUTDIR/noise_curve_recall_L${NOISE_BLOCK}.png"
echo "  $OUTDIR/noise_curve_f1_L${NOISE_BLOCK}.png"
echo "  r_real=$R_REAL   (k<=r_real = within intrinsic device noise)"
