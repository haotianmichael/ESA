#!/usr/bin/env bash
# =============================================================================
# Neurosamble Phase 2 -- clean-data read<->read overlap head-to-head.
#
# Compares Neurosamble (frozen encoder -> read-window FAISS -> colinear chaining)
# against Rawsamble (rawhash2 -x ava) on the IDENTICAL subset signals, both
# scored against a minimap2 ava-ont overlap truth. Then (Stage B) assembles both
# with miniasm and reports contiguity + chained-read %.
#
# Positional args (as specified):
#   1 OUTDIR        output root (a timestamped run-dir is created under it)
#   2 SUBSET_BLOW5  k=0 lossless subset blow5 (built in STAGE A from the id list)
#   3 SUBSET_FASTA  basecalled subset reads FASTA (qnames MUST match blow5 ids)
#   4 REF           reference FASTA (for the minimap2 true-mappings)
#   5 READ_IDS      the fixed subset read-id list (Phase-1 seed=1234)
#   6 PORE          ONT pore model (rawhash2 -p)
#   7 RAWHASH2_BIN  rawhash2 binary
#   8 THREADS       thread count
#   9 SCRIPTS_DIR   RawHash repo test/scripts dir (pafstats.py, analyze_gfa.sh,
#                   compute_aun.py, evaluate_gfa.py, run_minimap2_multimap.sh)
#
# Required env:
#   LOAD_ENCODER    path to the fine-tuned encoder .pt (Neurosamble needs it)
# Optional env:
#   DO_ASSEMBLY=1        STAGE B (assembly + chained %); set 0 for STAGE A only
#   MIN_CHAINING_SCORE=40  chaining-score threshold for overlap_map.py (sweepable)
#   MINIMAP2=minimap2  MINIASM=miniasm  PYTHON=python  DEVICE=cuda:0
#
# Isolation: every output lands in OUTDIR/run_<timestamp>/ so prior results are
# never overwritten.
# =============================================================================
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 OUTDIR SUBSET_BLOW5 SUBSET_FASTA REF READ_IDS PORE RAWHASH2_BIN THREADS SCRIPTS_DIR" >&2
  echo "       (env: LOAD_ENCODER=<encoder.pt> required; DO_ASSEMBLY=0 for Stage A only)" >&2
  exit 2
fi

OUTDIR="$1"; SUBSET_BLOW5="$2"; SUBSET_FASTA="$3"; REF="$4"; READ_IDS="$5"
PORE="$6"; RAWHASH2_BIN="$7"; THREADS="$8"; SCRIPTS_DIR="$9"

: "${LOAD_ENCODER:?set LOAD_ENCODER=<path to encoder .pt>}"
DO_ASSEMBLY="${DO_ASSEMBLY:-1}"
MIN_CHAINING_SCORE="${MIN_CHAINING_SCORE:-40}"
MINIMAP2="${MINIMAP2:-minimap2}"
MINIASM="${MINIASM:-miniasm}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
# Phase 3: when set, reuse this pre-built overlap truth instead of rebuilding it
# from SUBSET_FASTA. The noise sweep builds the mm2 truth ONCE from the CLEAN
# basecalled reads and must keep it FIXED across all k (we never re-basecall).
FIXED_TRUTH_PAF="${FIXED_TRUTH_PAF:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # evaluate/
RUN="${OUTDIR}/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
echo "[h2h] run dir: $RUN"
echo "[h2h] DO_ASSEMBLY=$DO_ASSEMBLY  MIN_CHAINING_SCORE=$MIN_CHAINING_SCORE"

# --------------------------------------------------------------------------- #
# 1) Neurosamble: overlap_map.py -> neurosamble.paf
# --------------------------------------------------------------------------- #
echo "[h2h] === Neurosamble overlap_map ==="
"$PYTHON" "$HERE/overlap_map.py" \
  --real_reads   "$SUBSET_BLOW5" \
  --load_encoder "$LOAD_ENCODER" \
  --reads_fasta  "$SUBSET_FASTA" \
  --read_ids     "$READ_IDS" \
  --out_dir      "$RUN" \
  --min_chaining_score "$MIN_CHAINING_SCORE" \
  --device "$DEVICE" --faiss_cpu 1 \
  2>&1 | tee "$RUN/neurosamble_map.log"
NEURO_PAF="$RUN/neurosamble.paf"

# --------------------------------------------------------------------------- #
# 2) Rawsamble: rawhash2 -x ava on the SAME subset signals
# --------------------------------------------------------------------------- #
echo "[h2h] === Rawsamble (rawhash2 -x ava) ==="
"$RAWHASH2_BIN" -x ava -t "$THREADS" -p "$PORE" -d "$RUN/rawsamble_idx" "$SUBSET_BLOW5" \
  2>&1 | tee "$RUN/rawsamble_index.log"
"$RAWHASH2_BIN" -x ava -t "$THREADS" "$RUN/rawsamble_idx" "$SUBSET_BLOW5" \
  > "$RUN/rawsamble.paf" 2> "$RUN/rawsamble_map.log"
RAW_PAF="$RUN/rawsamble.paf"

# --------------------------------------------------------------------------- #
# 3) Overlap TRUTH: minimap2 ava-ont (forward-only) on the subset FASTA
# --------------------------------------------------------------------------- #
if [[ -n "$FIXED_TRUTH_PAF" && -s "$FIXED_TRUTH_PAF" ]]; then
  echo "[h2h] === using FIXED overlap truth: $FIXED_TRUTH_PAF ==="
  cp "$FIXED_TRUTH_PAF" "$RUN/mm2_overlaps.paf"   # snapshot into the run dir for provenance
else
  echo "[h2h] === minimap2 ava-ont overlap truth ==="
  "$MINIMAP2" -x ava-ont --for-only -t "$THREADS" "$SUBSET_FASTA" "$SUBSET_FASTA" \
    > "$RUN/mm2_overlaps.paf" 2> "$RUN/mm2_overlaps.log"
fi
TRUTH_PAF="$RUN/mm2_overlaps.paf"

# --------------------------------------------------------------------------- #
# 4) Overlap scoring: both tools vs the mm2 truth (P/R/F1 + shared/unique)
#    pafstats.py requires an mt:f: tag on the test PAF (overlap_map adds it;
#    rawhash2 emits it natively).
# --------------------------------------------------------------------------- #
echo "[h2h] === pafstats: Neurosamble vs truth ==="
"$PYTHON" "$SCRIPTS_DIR/pafstats.py" "$NEURO_PAF" "$TRUTH_PAF" \
  > "$RUN/pafstats_neurosamble.out" 2> "$RUN/pafstats_neurosamble.err" || true
echo "[h2h] === pafstats: Rawsamble vs truth ==="
"$PYTHON" "$SCRIPTS_DIR/pafstats.py" "$RAW_PAF" "$TRUTH_PAF" \
  > "$RUN/pafstats_rawsamble.out" 2> "$RUN/pafstats_rawsamble.err" || true

echo "[h2h] ---- pafstats (Neurosamble) ----"; cat "$RUN/pafstats_neurosamble.err" || true
echo "[h2h] ---- pafstats (Rawsamble)  ----"; cat "$RUN/pafstats_rawsamble.err"  || true

if [[ "$DO_ASSEMBLY" == "0" ]]; then
  echo "[h2h] STAGE A complete (overlap scoring only). Set DO_ASSEMBLY=1 for STAGE B."
  echo "[h2h] outputs under: $RUN"
  exit 0
fi

# =========================================================================== #
# STAGE B: assembly (miniasm) + contiguity + chained-read %
# =========================================================================== #
echo "[h2h] === STAGE B: miniasm assembly ==="
for tag in neurosamble rawsamble mm2; do
  case "$tag" in
    neurosamble) PAF="$NEURO_PAF" ;;
    rawsamble)   PAF="$RAW_PAF" ;;
    mm2)         PAF="$TRUTH_PAF" ;;
  esac
  "$MINIASM" -f "$SUBSET_FASTA" "$PAF" > "$RUN/${tag}.gfa" 2> "$RUN/${tag}_miniasm.log" || true
done

echo "[h2h] === contiguity (analyze_gfa.sh + compute_aun.py + N50) ==="
for tag in neurosamble rawsamble mm2; do
  GFA="$RUN/${tag}.gfa"
  [[ -s "$GFA" ]] || { echo "[h2h] $GFA empty; skipping"; continue; }
  echo "---- $tag ----"                                              | tee -a "$RUN/contiguity.out"
  bash "$SCRIPTS_DIR/analyze_gfa.sh" "$GFA"                           2>&1 | tee -a "$RUN/contiguity.out" || true
  "$PYTHON" "$SCRIPTS_DIR/compute_aun.py" "$GFA"                      2>&1 | tee -a "$RUN/contiguity.out" || true
  "$PYTHON" "$HERE/gfa_n50.py" "$GFA"                                 2>&1 | tee -a "$RUN/contiguity.out"
done

echo "[h2h] === chained read % (run_minimap2_multimap.sh + evaluate_gfa.py) ==="
# RawHash signature: run_minimap2_multimap.sh OUTDIR READS REF THREAD
# (it writes ${OUTDIR}/true_mappings.paf itself -- do NOT redirect stdout).
bash "$SCRIPTS_DIR/run_minimap2_multimap.sh" "$RUN" "$SUBSET_FASTA" "$REF" "$THREADS" \
  2> "$RUN/true_mappings.log" || true
for tag in neurosamble rawsamble; do
  GFA="$RUN/${tag}.gfa"
  [[ -s "$GFA" ]] || { echo "[h2h] $GFA empty; skip chained% for $tag"; continue; }
  echo "---- $tag ----"                                                    | tee -a "$RUN/chained_reads.out"
  "$PYTHON" "$SCRIPTS_DIR/evaluate_gfa.py" "$GFA" "$RUN/true_mappings.paf"  2>&1 | tee -a "$RUN/chained_reads.out" || true
done

# --------------------------------------------------------------------------- #
# Final summary: cat every headline result so the whole run is visible at once
# (the pafstats throughput ZeroDivisionError -- from our mt:f:0.0 placeholder --
# lands AFTER the metrics, so grepping the metric lines keeps the summary clean).
# --------------------------------------------------------------------------- #
echo ""
echo "############################ SUMMARY ############################"
echo "run dir: $RUN"
echo "----- overlap P/R/F1 : Neurosamble -----"
grep -E 'TP:|Precision|Recall|F1 Score' "$RUN/pafstats_neurosamble.err" 2>/dev/null || true
echo "----- overlap P/R/F1 : Rawsamble  -----"
grep -E 'TP:|Precision|Recall|F1 Score' "$RUN/pafstats_rawsamble.err" 2>/dev/null || true
echo "----- assembly GFAs -----"
ls -l "$RUN"/*.gfa 2>/dev/null || true
echo "----- contiguity (analyze_gfa / AUN / N50) -----"
cat "$RUN/contiguity.out" 2>/dev/null || true
echo "----- chained read % -----"
cat "$RUN/chained_reads.out" 2>/dev/null || true
echo "################################################################"
echo "[h2h] STAGE B complete. All outputs under: $RUN"
