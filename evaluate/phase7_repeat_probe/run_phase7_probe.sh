#!/usr/bin/env bash
# =============================================================================
# Phase 7 (generalized) -- run the repeat/multi-locus probe on ANY overlap graph
# (Neurosamble or Rawsamble) and ANY dataset (E. coli / yeast). Offline, read-only
# w.r.t. the source experiment; writes only into OUT_SUBDIR. No re-encode / no FAISS.
#
# Args:  GRAPH_PAF  READS_FASTA  REF  MM2_BIN  OUT_SUBDIR  THREADS  GRAPH_LABEL  DATASET  [REF_PAF]
#   GRAPH_PAF    : overlap-graph PAF (neurosamble.paf OR rawsamble.paf)
#   READS_FASTA  : the reads that produced GRAPH_PAF (ids must match)
#   REF          : reference fasta for this dataset
#   MM2_BIN      : minimap2 (PATH command or file path)
#   OUT_SUBDIR   : e.g. .../phase7_repeat_probe/rawsamble_ecoli
#   THREADS      : minimap2 threads
#   GRAPH_LABEL  : neurosamble | rawsamble
#   DATASET      : ecoli | yeast
#   REF_PAF      : (optional) existing reads_to_ref.paf to REUSE (skip minimap2)
# =============================================================================
set -euo pipefail

if [[ $# -lt 8 || $# -gt 9 ]]; then
  echo "usage: $0 GRAPH_PAF READS_FASTA REF MM2_BIN OUT_SUBDIR THREADS GRAPH_LABEL DATASET [REF_PAF]" >&2
  exit 2
fi

GRAPH_PAF="$1"; READS_FASTA="$2"; REF="$3"; MM2_BIN="$4"; OUT_SUBDIR="$5"
THREADS="$6"; GRAPH_LABEL="$7"; DATASET="$8"; REF_PAF_IN="${9:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for req in "$GRAPH_PAF" "$READS_FASTA" "$REF"; do
  [[ -e "$req" ]] || { echo "[phase7][error] missing input: $req" >&2; exit 2; }
done
if ! command -v "$MM2_BIN" >/dev/null 2>&1 && [[ ! -x "$MM2_BIN" ]]; then
  echo "[phase7][error] minimap2 not found on PATH or as an executable: $MM2_BIN" >&2; exit 2
fi

mkdir -p "$OUT_SUBDIR"
LOG="$OUT_SUBDIR/run_phase7_probe.log"
exec > >(tee -a "$LOG") 2>&1
echo "[phase7] start $(date -u +%FT%TZ)  graph=$GRAPH_LABEL dataset=$DATASET"
echo "[phase7] OUT_SUBDIR=$OUT_SUBDIR THREADS=$THREADS"

# ---- reads -> REF alignment (reuse if REF_PAF passed & non-empty) -------------
if [[ -n "$REF_PAF_IN" && -s "$REF_PAF_IN" ]]; then
  REF_PAF="$REF_PAF_IN"
  echo "[phase7] REUSING existing reads_to_ref.paf (not re-running minimap2): $REF_PAF"
else
  REF_PAF="$OUT_SUBDIR/reads_to_ref.paf"
  if [[ -s "$REF_PAF" ]]; then
    echo "[phase7] reusing existing $REF_PAF"
  else
    echo "[phase7] === minimap2 reads->REF (map-ont, secondaries) ==="
    "$MM2_BIN" -x map-ont -N 10 -p 0.5 --secondary=yes -t "$THREADS" \
      "$REF" "$READS_FASTA" > "$REF_PAF" 2> "$OUT_SUBDIR/minimap2.log"
  fi
fi
echo "[phase7] reads_to_ref.paf lines: $(wc -l < "$REF_PAF")"

# ---- probe (repeat_probe.py performs the LOUD read-id overlap assert: it prints
#      the overlap % and exits non-zero if <30%, which aborts this script via -e) -
echo "[phase7] === repeat_probe.py ($GRAPH_LABEL / $DATASET) ==="
python "$HERE/repeat_probe.py" \
  --graph_paf "$GRAPH_PAF" \
  --ref_paf "$REF_PAF" \
  --out_dir "$OUT_SUBDIR" \
  --graph_label "$GRAPH_LABEL" \
  --dataset "$DATASET" \
  --locus_gap 20000 \
  --mapq_thr 5

echo "[phase7] done $(date -u +%FT%TZ). Outputs in $OUT_SUBDIR"
