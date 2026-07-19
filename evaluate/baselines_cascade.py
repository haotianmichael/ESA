"""
SquiggleSeek Step 2b — cascade baseline (basecall -> minimap2).

Two lines, both scored later by the SAME read-ID true coords + ``_covers`` the
pilot uses (this module only returns reported positions):

* cascade-oracle : "perfect basecall" = the read's true reference substring
  (from the read ID) fed to minimap2. No basecaller needed; an upper bound that
  should NOT degrade with noise (isolates basecalling error from mapping error).
* cascade-real   : basecall the query BLOW5 (buttery-eel, via a user-supplied
  command template so we don't hard-code version-specific flags) -> FASTQ ->
  minimap2.

Reference for minimap2 = the same single-record FASTA fed to squigulator, so its
``reference_start`` is in the same coordinate frame as the read-ID truth.
"""
from __future__ import annotations

import os
import subprocess
from typing import List, Optional

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def run_minimap2(ref_fasta, query_path, out_sam, minimap2_bin="minimap2", preset="map-ont"):
    cmd = [minimap2_bin, "-a", "-x", preset, str(ref_fasta), str(query_path)]
    with open(out_sam, "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.DEVNULL, check=True)
    return out_sam


def parse_primary_starts(sam_path) -> dict:
    """query_name -> reference_start for primary, mapped alignments only."""
    import pysam

    starts = {}
    with pysam.AlignmentFile(str(sam_path), "r") as sam:
        for a in sam:
            if a.is_unmapped or a.is_secondary or a.is_supplementary:
                continue
            starts[a.query_name] = int(a.reference_start)
    return starts


def _write_oracle_fasta(eval_reads, reference_seq, path):
    with open(path, "w") as f:
        for i, r in enumerate(eval_reads):
            end = r.reference_end if r.reference_end is not None else (r.reference_start + 300)
            seq = reference_seq[r.reference_start:end]
            if r.strand == "-":
                seq = revcomp(seq)
            if seq:
                f.write(f">read_{i}\n{seq}\n")
    return path


def cascade_oracle_positions(eval_reads, reference_seq, ref_fasta, work_dir,
                             minimap2_bin="minimap2") -> List[Optional[int]]:
    """Map the true reference substrings with minimap2. Returns per-read start
    (None = unmapped), aligned to eval_reads order."""
    os.makedirs(work_dir, exist_ok=True)
    reads_fa = os.path.join(work_dir, "oracle_reads.fasta")
    out_sam = os.path.join(work_dir, "oracle.sam")
    _write_oracle_fasta(eval_reads, reference_seq, reads_fa)
    run_minimap2(ref_fasta, reads_fa, out_sam, minimap2_bin)
    starts = parse_primary_starts(out_sam)
    return [starts.get(f"read_{i}") for i in range(len(eval_reads))]


def cascade_real_positions(eval_reads, blow5_path, ref_fasta, work_dir,
                           basecaller_cmd, minimap2_bin="minimap2") -> List[Optional[int]]:
    """Basecall the query BLOW5 then map with minimap2.

    ``basecaller_cmd`` is a shell template with ``{blow5}`` and ``{fastq}``
    placeholders, e.g.::

        "buttery-eel -i {blow5} -o {fastq} -g /path/dorado_server \
             --config dna_r9.4.1_450bps_fast.cfg --port 5000 --use_tcp"

    Basecalled read ids are the squigulator read ids, so positions are matched
    back to eval_reads by ``r.id``.
    """
    os.makedirs(work_dir, exist_ok=True)
    fastq = os.path.join(work_dir, "reads.fastq")
    out_sam = os.path.join(work_dir, "cascade_real.sam")
    cmd = basecaller_cmd.format(blow5=blow5_path, fastq=fastq)
    subprocess.run(cmd, shell=True, check=True)
    run_minimap2(ref_fasta, fastq, out_sam, minimap2_bin)
    starts = parse_primary_starts(out_sam)
    return [starts.get(r.id) for r in eval_reads]
