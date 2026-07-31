#!/usr/bin/env bash
# =============================================================================
# Real-data head-to-head: external blow5 + external reference (no squigulator).
#
# SquiggleSeek (real_data_eval.py, --load_encoder the trained encoder) and
# RawHash2 map the SAME real blow5; ground truth = minimap2-of-basecalled reads
# (RawHash's own convention) or a user PAF; rawhash_compare scores both.
#
# Truth is basecall-defined here — UNFAVOURABLE to SquiggleSeek but fair; as-is.
#
# Usage (set the paths, then):
#   REAL_READS=/path/ecoli_R9.blow5 REAL_REF=/path/ecoli.fna \
#   CHECKPOINT=/path/encoder_final_mamba_v3.pt \
#   TRUTH_PAF=/path/truth.paf \                # OR: BASECALL_CMD='buttery-eel -i {blow5} -o {fastq} ...'
#   LIMIT=100 \                                # smoke test on 100 reads; unset for full
#   nohup bash evaluate/run_real_data.sh > real.out 2>&1 &
# =============================================================================
set -u
set -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BASE="${BASE:-/home/nfs/mahaotian/ESA}"

REAL_READS="${REAL_READS:?set REAL_READS=/path/to/real.blow5}"
REAL_REF="${REAL_REF:?set REAL_REF=/path/to/reference.fasta}"
CHECKPOINT="${CHECKPOINT:-$BASE/CALL_ESA/evaluate/signal_checkpoints/encoder_final_mamba_v3.pt}"
OUT="${OUT:-$BASE/CALL_ESA/real_data_out}"
PYTHON="${PYTHON:-python}"
LIMIT="${LIMIT:-0}"                 # 0 = all reads; set 100 for a smoke test
FAISS_CPU="${FAISS_CPU:-1}"
PORE_MODEL="${PORE_MODEL_PATH:-$BASE/squigulator/r9_6mer_pore_model.txt}"
export PORE_MODEL_PATH="$PORE_MODEL"

TRUTH_PAF="${TRUTH_PAF:-}"          # ready-made truth, OR use BASECALL_CMD
BASECALL_CMD="${BASECALL_CMD:-}"
MINIMAP2="${MINIMAP2:-minimap2}"

RAWHASH2="${RAWHASH2:-}"
if [ -z "$RAWHASH2" ]; then
  RAWHASH2="$(command -v rawhash2 2>/dev/null || \
              find "$BASE/Rawhash2" -maxdepth 4 -type f -name rawhash2 2>/dev/null | head -1)"
  RAWHASH2="${RAWHASH2:-rawhash2}"
fi
if [ -z "${RAWHASH_PORE:-}" ]; then
  _cands="$(find "$BASE/Rawhash2" -maxdepth 6 -type f -name 'template_median68pA.model' 2>/dev/null)"
  RAWHASH_PORE="$(echo "$_cands" | grep -i 'r9.4_180mv_450bps_6mer' | head -1)"
  [ -z "$RAWHASH_PORE" ] && RAWHASH_PORE="$(echo "$_cands" | grep -iv 'rna' | grep -i 'r9.4' | head -1)"
  [ -z "$RAWHASH_PORE" ] && RAWHASH_PORE="$(echo "$_cands" | head -1)"
  RAWHASH_PORE="${RAWHASH_PORE:-$PORE_MODEL}"
fi
RAWHASH_PRESET="${RAWHASH_PRESET:-sensitive}"
THREADS="${THREADS:-32}"

mkdir -p "$OUT"
LOG="$OUT/run.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo -e "\n========== $* =========="; }
fail(){ echo "FATAL: $*"; exit 1; }

say "0. config ($(date))"
echo "reads=$REAL_READS  ref=$REAL_REF  checkpoint=$CHECKPOINT  limit=$LIMIT"
echo "rawhash2=$RAWHASH2  pore=$RAWHASH_PORE  preset=$RAWHASH_PRESET"
[ -s "$REAL_READS" ] || fail "REAL_READS not found: $REAL_READS"
[ -s "$REAL_REF" ]   || fail "REAL_REF not found: $REAL_REF"
[ -s "$CHECKPOINT" ] || fail "CHECKPOINT not found: $CHECKPOINT"
"$PYTHON" -c "import pyslow5" 2>/dev/null || fail "pyslow5 missing (pip install pyslow5)"

say "1. SquiggleSeek on real reads + ground truth"
SS_ARGS=( --real_reads "$REAL_READS" --real_reference "$REAL_REF"
          --load_encoder "$CHECKPOINT" --out_dir "$OUT"
          --faiss_cpu "$FAISS_CPU" --limit "$LIMIT" --minimap2_bin "$MINIMAP2" )
if [ -n "$TRUTH_PAF" ]; then
  SS_ARGS+=( --truth_paf "$TRUTH_PAF" )
elif [ -n "$BASECALL_CMD" ]; then
  SS_ARGS+=( --basecall_cmd "$BASECALL_CMD" )
else
  fail "provide TRUTH_PAF=... or BASECALL_CMD='...{blow5}...{fastq}...'"
fi
"$PYTHON" "$HERE/real_data_eval.py" "${SS_ARGS[@]}" || fail "real_data_eval.py failed"

[ -s "$OUT/squiggleseek_real.paf" ]  || fail "squiggleseek_real.paf not produced"
[ -s "$OUT/ground_truth_real.paf" ]  || fail "ground_truth_real.paf empty (basecall/minimap2 issue?)"

say "2. RawHash2 on the identical reads.blow5 (preset=$RAWHASH_PRESET)"
command -v "$RAWHASH2" >/dev/null 2>&1 || [ -x "$RAWHASH2" ] || fail "rawhash2 not found: $RAWHASH2"
"$RAWHASH2" -x "$RAWHASH_PRESET" -p "$RAWHASH_PORE" -t "$THREADS" -d "$OUT/ref.idx" "$OUT/ref.fasta" \
  || fail "rawhash2 index build failed"
"$RAWHASH2" -x "$RAWHASH_PRESET" -t "$THREADS" "$OUT/ref.idx" "$OUT/reads.blow5" \
  > "$OUT/rawhash2_real.paf" || fail "rawhash2 mapping failed"
echo "rawhash2 lines: $(wc -l < "$OUT/rawhash2_real.paf")"

say "3. score both vs the real ground truth (builtin locus criterion)"
"$PYTHON" "$HERE/rawhash_compare.py" \
  --truth "$OUT/ground_truth_real.paf" \
  --paf "SquiggleSeek=$OUT/squiggleseek_real.paf" \
  --paf "RawHash2=$OUT/rawhash2_real.paf" \
  --sweep SquiggleSeek --match_to RawHash2 --scorer builtin | tee "$OUT/SUMMARY_real.txt"

say "DONE  ($(date))"
echo "summary -> $OUT/SUMMARY_real.txt ; full log -> $LOG"
echo "NOTE: truth = basecall-defined (RawHash's home turf); report honestly."
