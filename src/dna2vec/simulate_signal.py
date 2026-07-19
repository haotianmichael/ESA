"""
Signal-domain read simulation, mirroring ``simulate.py`` (which wraps ART).

``simulate_mapped_signals`` shells out to **squigulator**
(``hasindu2008/squigulator``) to produce raw nanopore signal for reads sampled
from a reference, together with ground-truth reference coordinates (parsed from
squigulator's PAF output). Signal is read back from the SLOW5/BLOW5 file via
``pyslow5``.

Because squigulator + pyslow5 are external/optional, a deterministic synthetic
fallback (``simulate_synthetic_signals``) is also provided: it renders reads
from the pore model and adds Gaussian noise. That fallback exercises the full
encode -> FAISS -> recall plumbing but is NOT a substitute for real signal when
measuring the go/no-go domain gap.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SignalReadAndRef:
    signal: np.ndarray          # 1-D raw/pA current samples
    reference_start: int        # ground-truth genomic start (bp, forward-strand coord)
    id: str
    reference_name: Optional[str] = None
    strand: str = "+"           # '+' or '-' (PAF column 5)
    reference_end: Optional[int] = None  # ground-truth end (bp); for the oracle sequence


# --------------------------------------------------------------------------- #
# Real simulation via squigulator
# --------------------------------------------------------------------------- #
def simulate_mapped_signals(
    reference_genome: str,
    n_reads: int,
    read_length_bp: int,
    profile: str = "dna-r9-min",
    work_dir: Optional[str] = None,
    seed: Optional[int] = None,
    squigulator_bin: str = "squigulator",
    paf_option: str = "--paf",
    amp_noise: Optional[float] = None,
    dwell_std: Optional[float] = None,
    extra_args: Optional[List[str]] = None,
) -> List[SignalReadAndRef]:
    """
    Simulate raw signal reads with ground-truth coordinates.

    Note: squigulator's exact flags vary by version; ``profile`` and
    ``paf_option`` are exposed so they can be matched to the installed binary
    without code changes (see ``squigulator --help``).
    """
    try:
        import pyslow5
    except ImportError as e:  # pragma: no cover - env dependent
        raise ImportError(
            "simulate_mapped_signals needs pyslow5 to read squigulator output: "
            "pip install pyslow5"
        ) from e

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="squig_")
    os.makedirs(work_dir, exist_ok=True)
    blow5 = os.path.join(work_dir, "reads.blow5")
    paf = os.path.join(work_dir, "reads.paf")

    cmd = [
        squigulator_bin, str(reference_genome),
        "-x", profile,
        "-o", blow5,
        "-n", str(n_reads),
        "-r", str(read_length_bp),
        paf_option, paf,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if amp_noise is not None:
        cmd += ["--amp-noise", str(amp_noise)]
    if dwell_std is not None:
        cmd += ["--dwell-std", str(dwell_std)]
    if extra_args:
        cmd += list(extra_args)

    subprocess.run(cmd, check=True)

    # IMPORTANT: squigulator's --paf reports the signal-vs-read alignment, whose
    # target start is always 0 (the read starts at sample 0 of its own signal) —
    # it is NOT a genomic coordinate. The true genomic start/strand are encoded
    # in the read id, e.g. "S1_1!NC_000913.3!1956875!1958160!-":
    #   parts[-4]=contig, parts[-3]=start, parts[-2]=end, parts[-1]=strand.
    # The PAF is still written above and can be parsed via _parse_paf_coords for
    # debugging, but coordinates here come from the read id.
    reads: List[SignalReadAndRef] = []
    s = pyslow5.Open(blow5, "r")
    for read in s.seq_reads(pA=True):
        rid = read["read_id"]
        parts = str(rid).split("!")
        if len(parts) < 4:  # not squigulator's !-encoded id -> skip
            continue
        tname = parts[-4]
        tstart = int(parts[-3])
        tend = int(parts[-2])
        strand = parts[-1]
        reads.append(
            SignalReadAndRef(
                signal=np.asarray(read["signal"], dtype=np.float32),
                reference_start=tstart,
                id=str(rid),
                reference_name=tname,
                strand=strand,
                reference_end=tend,
            )
        )
    return reads


def _parse_paf_coords(paf_path: str) -> dict:
    """read_id -> (target_start, target_name, strand) from a PAF file.

    Kept for debugging only. NOTE: squigulator's PAF target start is the
    signal-vs-read offset (always 0), not a genomic coordinate — do not use it
    for ground truth. Genomic coordinates come from the read id instead (see
    ``simulate_mapped_signals``).
    """
    coords = {}
    with open(paf_path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            qname, strand, tname, tstart = parts[0], parts[4], parts[5], parts[7]
            coords[qname] = (int(tstart), tname, strand)
    return coords


# --------------------------------------------------------------------------- #
# Synthetic fallback (no external binary required)
# --------------------------------------------------------------------------- #
def simulate_synthetic_signals(
    reference_seq: str,
    pore_model,
    n_reads: int,
    read_length_bp: int,
    noise_std: float = 0.3,
    seed: Optional[int] = None,
) -> List[SignalReadAndRef]:
    """Render reads from the pore model with additive Gaussian noise (smoke test only)."""
    rng = np.random.default_rng(seed)
    reads: List[SignalReadAndRef] = []
    max_start = len(reference_seq) - read_length_bp - 1
    if max_start <= 0:
        raise ValueError("Reference shorter than read_length_bp.")

    for i in range(n_reads):
        start = int(rng.integers(0, max_start))
        bases = reference_seq[start : start + read_length_bp]
        clean = pore_model.sequence_to_signal(bases)
        # Noise is applied in the pore-model's pA scale.
        noise = rng.normal(0.0, noise_std * pore_model._level_std, size=clean.shape)
        signal = (clean + noise).astype(np.float32)
        reads.append(
            SignalReadAndRef(
                signal=signal, reference_start=start, id=f"synth_{i}", reference_name="ref",
                reference_end=start + read_length_bp,
            )
        )
    return reads
