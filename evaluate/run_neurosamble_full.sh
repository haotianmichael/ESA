#!/usr/bin/env bash
# =============================================================================
# Neurosamble Phase 4 -- FULL-SCALE all-vs-all head-to-head (no subsampling).
#
# Full D1 read set (~353k reads). Neurosamble map step is the scale path:
#   2-GPU sharded encode -> CPU IVF index (checkpointed) -> streaming query.
# Rawsamble / minimap2-truth / miniasm / pafstats / evaluate_gfa wiring mirrors
# run_neurosamble_head2head.sh. All outputs under OUTDIR; encode+index are
# checkpointed so a re-run RESUMES rather than recomputes. The encoder is NOT
# loaded during the query phase.
#
# Positional args:
#   1 OUTDIR         output root (checkpoints live here; not timestamped)
#   2 REAL_BLOW5     full blow5 (ALL reads)
#   3 READS_FASTA    full basecalled reads (truth + miniasm sequences)
#   4 REF            full CFT073 (AE014075.1) reference (chained %)
#   5 PORE           ONT pore model (rawhash2 -p)
#   6 RAWHASH2_BIN   rawhash2 binary
#   7 THREADS        thread count (faiss + tools)
#   8 SCRIPTS_DIR    RawHash test/scripts dir
#   9 NUM_GPUS       (optional, default 2)
#  10 NPROBE         (optional, default 64)
#  11 INDEX_TYPE     (optional, default ivfflat; or ivfpq)
#  12 DO_ASSEMBLY    (optional, default 1)
#
# Required env: LOAD_ENCODER=<encoder .pt>
# Optional env: MINIASM, MINIMAP2, PYTHON, TORCHRUN, SAMPLES_PER_KMER (9)
# =============================================================================
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "usage: $0 OUTDIR REAL_BLOW5 READS_FASTA REF PORE RAWHASH2_BIN THREADS SCRIPTS_DIR [NUM_GPUS] [NPROBE] [INDEX_TYPE] [DO_ASSEMBLY]" >&2
  echo "       (env: LOAD_ENCODER=<encoder.pt> required)" >&2
  exit 2
fi

OUTDIR="$1"; REAL_BLOW5="$2"; READS_FASTA="$3"; REF="$4"
# Derive phase tag from output dir name (e.g. neurosamble_phase5 -> phase5)
PHASE_TAG="$(basename "$OUTDIR")"
PORE="$5"; RAWHASH2_BIN="$6"; THREADS="$7"; SCRIPTS_DIR="$8"
NUM_GPUS="${9:-2}"; NPROBE="${10:-64}"; INDEX_TYPE="${11:-ivfflat}"; DO_ASSEMBLY="${12:-1}"

: "${LOAD_ENCODER:?set LOAD_ENCODER=<path to encoder .pt>}"
MINIASM="${MINIASM:-miniasm}"
MINIMAP2="${MINIMAP2:-minimap2}"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
SPK="${SAMPLES_PER_KMER:-9}"
RAWHASH_PRESET="${RAWHASH_PRESET:-}"   # set to "--r10" for R10.4.1 data (else R9 defaults)
# topk MUST scale with coverage for all-vs-all: each window has ~coverage true
# neighbors, so a fixed small topk caps recall at ~topk/coverage. Configurable.
TOPK="${TOPK:-10}"
FAISS_GPU="${FAISS_GPU:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # evaluate/
mkdir -p "$OUTDIR" "$OUTDIR/encode" "$OUTDIR/index"
NEURO_PAF="$OUTDIR/neurosamble.paf"
RAW_PAF="$OUTDIR/rawsamble.paf"
TRUTH_PAF="$OUTDIR/mm2_overlaps.paf"
echo "[full] OUTDIR=$OUTDIR NUM_GPUS=$NUM_GPUS NPROBE=$NPROBE TOPK=$TOPK INDEX_TYPE=$INDEX_TYPE FAISS_GPU=$FAISS_GPU DO_ASSEMBLY=$DO_ASSEMBLY THREADS=$THREADS SPK=$SPK"

# --------------------------------------------------------------------------- #
# 1) Neurosamble scale path: encode (2 GPU) -> IVF (CPU) -> streaming query
# --------------------------------------------------------------------------- #
ENCODE_SEC=0; INDEX_SEC=0
if [[ "${REUSE_NEURO_PAF:-0}" == "1" && -s "$NEURO_PAF" ]]; then
  echo "[full] REUSE_NEURO_PAF=1 and $NEURO_PAF present -> skip encode/index/query"
else
  echo "[full] === encode (2-GPU sharded) ==="
  T_ENC0=$SECONDS
  "$TORCHRUN" --nproc_per_node="$NUM_GPUS" "$HERE/overlap_encode_mp.py" \
    --real_reads "$REAL_BLOW5" --load_encoder "$LOAD_ENCODER" \
    --out_dir "$OUTDIR/encode" --win 2000 --stride 1000 \
    2>&1 | tee "$OUTDIR/encode.log"
  ENCODE_SEC=$((SECONDS - T_ENC0))

  echo "[full] === IVF index build (CPU, checkpointed) ==="
  T_IDX0=$SECONDS
  "$PYTHON" "$HERE/overlap_index_ivf.py" \
    --encode_dir "$OUTDIR/encode" --out_dir "$OUTDIR/index" \
    --index_type "$INDEX_TYPE" --threads "$THREADS" \
    2>&1 | tee "$OUTDIR/index.log"
  INDEX_SEC=$((SECONDS - T_IDX0))

  echo "[full] === streaming query + chaining ==="
  "$PYTHON" "$HERE/overlap_map_full.py" \
    --index_dir "$OUTDIR/index" --encode_dir "$OUTDIR/encode" \
    --out_paf "$NEURO_PAF" --nprobe "$NPROBE" --topk "$TOPK" \
    --threads "$THREADS" --samples_per_kmer "$SPK" --faiss_gpu "$FAISS_GPU" \
    --query_batch "${QUERY_BATCH:-65536}" --gpu_temp_mb "${GPU_TEMP_MB:-8192}" \
    2>&1 | tee "$OUTDIR/query.log"
fi

# --------------------------------------------------------------------------- #
# 2) Rawsamble on the SAME full blow5
# --------------------------------------------------------------------------- #
echo "[full] === Rawsamble (rawhash2 -x ava) ==="
if [[ -s "$RAW_PAF" ]]; then
  echo "[full] reuse existing rawsamble.paf (topk-independent): $RAW_PAF"
else
  "$RAWHASH2_BIN" -x ava $RAWHASH_PRESET -t "$THREADS" -p "$PORE" -d "$OUTDIR/rawsamble_idx" "$REAL_BLOW5" \
    2>&1 | tee "$OUTDIR/rawsamble_index.log"
  "$RAWHASH2_BIN" -x ava $RAWHASH_PRESET -t "$THREADS" "$OUTDIR/rawsamble_idx" "$REAL_BLOW5" \
    > "$RAW_PAF" 2> "$OUTDIR/rawsamble_map.log"
fi

# --------------------------------------------------------------------------- #
# 3) Overlap truth (minimap2 ava-ont, forward-only) on the full FASTA
# --------------------------------------------------------------------------- #
echo "[full] === minimap2 ava-ont overlap truth ==="
if [[ -s "$TRUTH_PAF" ]]; then
  echo "[full] reuse existing mm2_overlaps.paf (topk-independent truth): $TRUTH_PAF"
else
  "$MINIMAP2" -x ava-ont --for-only -t "$THREADS" "$READS_FASTA" "$READS_FASTA" \
    > "$TRUTH_PAF" 2> "$OUTDIR/mm2_overlaps.log"
fi

# --------------------------------------------------------------------------- #
# 4) Overlap scoring
# --------------------------------------------------------------------------- #
echo "[full] === pafstats: Neurosamble vs truth ==="
"$PYTHON" "$SCRIPTS_DIR/pafstats.py" "$NEURO_PAF" "$TRUTH_PAF" \
  > "$OUTDIR/pafstats_neurosamble.out" 2> "$OUTDIR/pafstats_neurosamble.err" || true
echo "[full] === pafstats: Rawsamble vs truth ==="
"$PYTHON" "$SCRIPTS_DIR/pafstats.py" "$RAW_PAF" "$TRUTH_PAF" \
  > "$OUTDIR/pafstats_rawsamble.out" 2> "$OUTDIR/pafstats_rawsamble.err" || true
echo "[full] ---- pafstats (Neurosamble) ----"; cat "$OUTDIR/pafstats_neurosamble.err" || true
echo "[full] ---- pafstats (Rawsamble)  ----"; cat "$OUTDIR/pafstats_rawsamble.err"  || true

if [[ "$DO_ASSEMBLY" != "0" ]]; then
  # ------------------------------------------------------------------------- #
  # 5) Assembly + contiguity
  # ------------------------------------------------------------------------- #
  echo "[full] === miniasm assembly ==="
  # FAIRNESS: every tool sanitizes with --reads_fasta, which now applies ONE global
  # scale c=median(fasta_len/native_len) -- a similarity transform that preserves
  # each tool's overlap geometry (topology unchanged) and only shifts the overall
  # scale into base space. Per-tool c differs (mm2 ~1.0 identity; Neurosamble ~0.78;
  # rawsamble its own), so none is stretched relative to its own reads. mm2 also
  # gets miniasm -f reads.fasta (real contig sequences); the signal-domain tools
  # assemble without -f (lengths from the rescaled PAF).
  for tag in neurosamble rawsamble mm2; do
    case "$tag" in
      neurosamble) PAF="$NEURO_PAF" ;;
      rawsamble)   PAF="$RAW_PAF" ;;
      mm2)         PAF="$TRUTH_PAF" ;;
    esac
    CLEAN="$OUTDIR/${tag}.clean.paf"
    GFA="$OUTDIR/${tag}.gfa"
    if [[ "$tag" == "mm2" ]]; then
      "$PYTHON" "$HERE/sanitize_paf.py" --in_paf "$PAF" \
        --out_paf "$CLEAN" --reads_fasta "$READS_FASTA" \
        2>&1 | tee -a "$OUTDIR/sanitize.log"
      "$MINIASM" -f "$READS_FASTA" "$CLEAN" \
        > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
    else
      # Global-scale sanitize (single median constant c; preserves overlap geometry,
      # only shifts scale into base space). Assemble on the rescaled native coords
      # without -f (lengths come from the rescaled PAF).
      "$PYTHON" "$HERE/sanitize_paf.py" --in_paf "$PAF" \
        --out_paf "$CLEAN" --reads_fasta "$READS_FASTA" \
        2>&1 | tee -a "$OUTDIR/sanitize.log"
      "$MINIASM" "$CLEAN" > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
      if [[ ! -s "$GFA" ]]; then
        # Fallback: some miniasm builds need -f. Build a placeholder FASTA from the
        # NATIVE clean.paf read lengths (N x length) -- never the basecalled
        # reads.fasta, so no rescaling of the signal-domain coordinates.
        echo "[full] $tag: miniasm w/o -f gave empty gfa; retrying with placeholder FASTA" \
          | tee -a "$OUTDIR/${tag}_miniasm.log"
        PLACE="$OUTDIR/${tag}.placeholder.fasta"
        "$PYTHON" -c '
import sys
clean, out = sys.argv[1], sys.argv[2]
L = {}
with open(clean) as f:
    for line in f:
        c = line.rstrip("\n").split("\t")
        if len(c) < 9:
            continue
        try:
            ql, tl = int(c[1]), int(c[6])
        except ValueError:
            continue
        if ql > L.get(c[0], 0):
            L[c[0]] = ql
        if tl > L.get(c[5], 0):
            L[c[5]] = tl
with open(out, "w") as w:
    for name, n in L.items():
        w.write(">" + name + "\n" + "N" * n + "\n")
' "$CLEAN" "$PLACE" 2>&1 | tee -a "$OUTDIR/${tag}_miniasm.log" || true
        "$MINIASM" -f "$PLACE" "$CLEAN" \
          > "$GFA" 2> "$OUTDIR/${tag}_miniasm.log" || true
      fi
    fi
  done

  echo "[full] === contiguity (analyze_gfa.sh + compute_aun.py + N50) ==="
  for tag in neurosamble rawsamble mm2; do
    GFA="$OUTDIR/${tag}.gfa"
    [[ -s "$GFA" ]] || { echo "[full] $GFA empty; skipping"; continue; }
    echo "---- $tag ----"                                          | tee -a "$OUTDIR/contiguity.out"
    bash "$SCRIPTS_DIR/analyze_gfa.sh" "$GFA"                       2>&1 | tee -a "$OUTDIR/contiguity.out" || true
    "$PYTHON" "$SCRIPTS_DIR/compute_aun.py" "$GFA"                  2>&1 | tee -a "$OUTDIR/contiguity.out" || true
    "$PYTHON" "$HERE/gfa_n50.py" "$GFA"                            2>&1 | tee -a "$OUTDIR/contiguity.out"
  done

  # ------------------------------------------------------------------------- #
  # 6) Chained read %
  # ------------------------------------------------------------------------- #
  echo "[full] === chained read % ==="
  bash "$SCRIPTS_DIR/run_minimap2_multimap.sh" "$OUTDIR" "$READS_FASTA" "$REF" "$THREADS" \
    2> "$OUTDIR/true_mappings.log" || true
  for tag in neurosamble rawsamble; do
    GFA="$OUTDIR/${tag}.gfa"
    [[ -s "$GFA" ]] || { echo "[full] $GFA empty; skip chained% for $tag"; continue; }
    echo "---- $tag ----"                                                        | tee -a "$OUTDIR/chained_reads.out"
    "$PYTHON" "$SCRIPTS_DIR/evaluate_gfa.py" "$GFA" "$OUTDIR/true_mappings.paf"   2>&1 | tee -a "$OUTDIR/chained_reads.out" || true
  done
fi

# --------------------------------------------------------------------------- #
# 7) SUMMARY + ${PHASE_TAG}_summary.csv
# --------------------------------------------------------------------------- #
SUMMARY_CSV="${OUTDIR}/${PHASE_TAG}_summary.csv"
echo "[full] === summary ==="
"$PYTHON" "$HERE/overlap_full_summary.py" \
  --run_dir "$OUTDIR" --encode_sec "$ENCODE_SEC" --index_sec "$INDEX_SEC" \
  --out_csv "$SUMMARY_CSV" \
  2>&1 | tee "$OUTDIR/summary.log"

echo ""
echo "############################ ${PHASE_TAG^^} SUMMARY ############################"
echo "OUTDIR=$OUTDIR  encode_sec=$ENCODE_SEC  index_sec=$INDEX_SEC"
cat "$SUMMARY_CSV" 2>/dev/null || true
echo "########################################################################"
echo "[full] DONE. Outputs under: $OUTDIR"
