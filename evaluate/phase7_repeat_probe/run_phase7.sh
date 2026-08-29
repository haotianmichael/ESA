#!/usr/bin/env bash
# =============================================================================
# Phase 7 -- repeat / multi-locus probe orchestrator (reuses Phase-4 outputs).
#
# Offline feasibility gate: does the Neurosamble all-vs-all overlap graph already
# encode genomic multiplicity? Produces a fresh reads->REF alignment (WITH
# secondaries) as ground truth, then calls repeat_probe.py. Does NOT rebuild the
# FAISS index, re-encode, or touch the Phase-4 experiment; writes only into OUT_DIR.
#
# Args:  PHASE4_DIR  REF  READS_FASTA  MINIMAP2_BIN  OUT_DIR  THREADS
#   PHASE4_DIR   : dir containing neurosamble.paf (the canonical ava overlap graph)
#   REF          : CFT073 reference (AE014075.1) fasta
#   READS_FASTA  : the same reads that fed Phase-4 encode (ids must match the PAF)
#   MINIMAP2_BIN : minimap2 binary
#   OUT_DIR      : e.g. CALL_RSA/experiments/phase7_repeat_probe  (created here)
#   THREADS      : threads for minimap2
# =============================================================================
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 PHASE4_DIR REF READS_FASTA MINIMAP2_BIN OUT_DIR THREADS" >&2
  exit 2
fi

PHASE4_DIR="$1"; REF="$2"; READS_FASTA="$3"; MINIMAP2_BIN="$4"; OUT_DIR="$5"; THREADS="$6"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEURO_PAF="$PHASE4_DIR/neurosamble.paf"
for req in "$NEURO_PAF" "$REF" "$READS_FASTA"; do
  [[ -e "$req" ]] || { echo "[phase7][error] missing input: $req" >&2; exit 2; }
done
# minimap2 may be a PATH command (conda env) OR an explicit file path -- accept both.
if ! command -v "$MINIMAP2_BIN" >/dev/null 2>&1 && [[ ! -x "$MINIMAP2_BIN" ]]; then
  echo "[phase7][error] minimap2 not found on PATH or as an executable: $MINIMAP2_BIN" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_phase7.log"
exec > >(tee -a "$LOG") 2>&1
echo "[phase7] start $(date -u +%FT%TZ)"
echo "[phase7] PHASE4_DIR=$PHASE4_DIR"
echo "[phase7] OUT_DIR=$OUT_DIR THREADS=$THREADS"

# ---- 1) reads -> REF alignment WITH secondaries (repeat copies surface) ------
REF_PAF="$OUT_DIR/reads_to_ref.paf"
if [[ -s "$REF_PAF" ]]; then
  echo "[phase7] reusing existing $REF_PAF"
else
  echo "[phase7] === minimap2 reads->REF (map-ont, secondaries) ==="
  "$MINIMAP2_BIN" -x map-ont -N 10 -p 0.5 --secondary=yes -t "$THREADS" \
    "$REF" "$READS_FASTA" > "$REF_PAF" 2> "$OUT_DIR/minimap2.log"
fi
echo "[phase7] reads_to_ref.paf lines: $(wc -l < "$REF_PAF")"

# ---- 2) repeat probe ---------------------------------------------------------
echo "[phase7] === repeat_probe.py ==="
python "$HERE/repeat_probe.py" \
  --neuro_paf "$NEURO_PAF" \
  --ref_paf "$REF_PAF" \
  --out_dir "$OUT_DIR" \
  --locus_gap 20000 \
  --mapq_thr 5

echo "[phase7] done $(date -u +%FT%TZ). Outputs in $OUT_DIR"
