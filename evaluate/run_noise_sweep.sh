#!/usr/bin/env bash
# =============================================================================
# SquiggleSeek vs RawHash2 — noise sweep (Stage 3, paper figure E1).
#
# Fixes ONE clean checkpoint (Gate-2 C: both-strand, batch 48; NOT retrained per
# noise point — the honest setting, RawHash isn't noise-tuned either) and sweeps
# squigulator noise. At each point the SAME seed generates the reads, so the true
# coordinate set is identical across noise levels (paired comparison); only the
# signal changes. Each point runs the full head-to-head (SquiggleSeek + RawHash2
# on the same blow5, same ground truth) via run_head2head.sh, then scores it.
#
# Usage (after Gate-2 C has produced h2h_C/encoder.pt):
#     nohup bash evaluate/run_noise_sweep.sh > noise_sweep_nohup.out 2>&1 &
#
# Grid defaults to the dwell_std=4 row (5 amp points) as the prompt asks; widen
# with DWELL_LIST="4 8 12" once the trend is clear.
# =============================================================================
set -u
set -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BASE="${BASE:-/home/nfs/mahaotian/ESA}"
CHECKPOINT="${CHECKPOINT:-$BASE/CALL_ESA/h2h_C/encoder.pt}"   # Gate-2 C checkpoint
SWEEP_OUT="${SWEEP_OUT:-$BASE/CALL_ESA/noise_sweep}"
CSV="${CSV:-$REPO/evaluate/noise_sweep.csv}"
PYTHON="${PYTHON:-python}"
AMP_LIST="${AMP_LIST:-default 1.5 2.0 3.0 5.0}"   # squigulator --amp-noise
DWELL_LIST="${DWELL_LIST:-4}"                     # 4 = squigulator default; "4 8 12" = full 2D grid
REF_FASTA="${REF_FASTA:-}"          # real genome (e.g. full E. coli); empty = random ref of REF_BP
REF_BP="${REF_BP:-1000000}"
N_QUERY="${N_QUERY:-5000}"
N_TRAIN="${N_TRAIN:-200}"          # tiny: we load a checkpoint, no training happens
SEED="${SEED:-42}"

mkdir -p "$SWEEP_OUT"
LOG="$SWEEP_OUT/sweep.log"
exec > >(tee -a "$LOG") 2>&1

echo "========== noise sweep ($(date)) =========="
[ -f "$CHECKPOINT" ] || { echo "FATAL: checkpoint not found: $CHECKPOINT (run Gate-2 C first)"; exit 1; }
echo "checkpoint = $CHECKPOINT"
echo "grid: amp = {$AMP_LIST}  x  dwell = {$DWELL_LIST}"
echo "ref_bp=$REF_BP  n_query=$N_QUERY  seed=$SEED  csv=$CSV"

for dwell in $DWELL_LIST; do
  for amp in $AMP_LIST; do
    tag="amp_${amp}_dwell_${dwell}"
    pdir="$SWEEP_OUT/$tag"
    echo -e "\n############################ noise point: $tag ############################"
    env_amp="";   [ "$amp"   != "default" ] && env_amp="$amp"
    env_dwell=""; [ "$dwell" != "4" ]       && env_dwell="$dwell"   # 4 = squigulator default

    # full head-to-head at this noise point: C checkpoint (no retrain), RawHash on,
    # SAME seed/ref. RawHash's command line + preset + pore model are unchanged.
    LOAD_ENCODER="$CHECKPOINT" \
    FORWARD_ONLY=0 BOTH_STRANDS=1 SKIP_RAWHASH=0 FAST_SWEEP=1 FAISS_CPU="${FAISS_CPU:-1}" \
    REF_FASTA="$REF_FASTA" REF_BP="$REF_BP" N_QUERY="$N_QUERY" N_TRAIN="$N_TRAIN" SEED="$SEED" \
    AMP_NOISE="$env_amp" DWELL_STD="$env_dwell" \
    OUT="$pdir" \
      bash "$HERE/run_head2head.sh" || { echo "!! point $tag failed, skipping"; continue; }

    "$PYTHON" "$HERE/noise_point_eval.py" \
      --truth "$pdir/ground_truth.paf" --ss "$pdir/squiggleseek.paf" \
      --rh "$pdir/rawhash2.paf" --amp "$amp" --dwell "$dwell" \
      --csv "$CSV" --ref_bp "$REF_BP" --checkpoint "$CHECKPOINT"
  done
done

echo -e "\n================ TRUTH-INVARIANCE (gate 1) — coords must match across amp ================"
grep "\[gate1\]" "$LOG" || true
echo -e "\n================ NOISE SWEEP TREND ================"
printf "  %-9s %-6s | %-8s %-12s | %-8s %-8s | %s\n" amp dwell SS_F1 SS_R@P99.9 RH_F1 RH_R dF1
grep "\[trend\]" "$LOG" | sed 's/\[trend\] //' || true
echo -e "\nCSV -> $CSV"
echo "per-point dirs -> $SWEEP_OUT/amp_*_dwell_*/  (SUMMARY.txt each)"
echo "== how to read: RawHash F1 should fall faster than SquiggleSeek as amp rises =="
