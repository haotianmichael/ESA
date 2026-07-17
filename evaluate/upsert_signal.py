"""
Build the reference *expected-signal* FAISS index for signal-domain seeding.

The reference genome is windowed by coordinate (``unit_length`` bp, ``overlap``
bp, same tiling idea as ``stage_upstream.py``); each window's bases are turned
into an expected signal by the pore model, embedded by the (possibly untrained)
signal encoder, and stored in a ``SignalFaissStore`` with ``{'coord': start}``
metadata. The reference is static, so this is built once and reused.
"""
from __future__ import annotations

from typing import List

import numpy as np
from tqdm import tqdm


def read_single_fasta(path: str) -> str:
    """Concatenate all sequence lines of a FASTA into one uppercase string."""
    seq_parts: List[str] = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip())
    return "".join(seq_parts).upper()


def build_signal_reference_index(
    reference_seq: str,
    pore_model,
    signal_model,
    store,
    unit_length: int = 300,
    stride: int = 150,
    namespace: str = "ref",
    encode_batch: int = 2048,
):
    """Window the reference, embed expected signals, and populate ``store``.

    ``stride`` is the index tiling step (decoupled from training). A small
    stride (e.g. 15) removes the "best window not aligned to the read start"
    ceiling artifact at the cost of more windows.
    """
    step = max(1, stride)
    starts = list(range(0, len(reference_seq) - unit_length + 1, step))

    buf_signals: List[np.ndarray] = []
    buf_meta: List[dict] = []

    def flush():
        if not buf_signals:
            return
        vecs = signal_model.encode(buf_signals)
        store.add_embeddings(vecs, list(buf_meta), namespace=namespace, save=False)
        buf_signals.clear()
        buf_meta.clear()

    for start in tqdm(starts, desc="reference windows"):
        bases = reference_seq[start : start + unit_length]
        buf_signals.append(pore_model.sequence_to_signal(bases))
        buf_meta.append({"coord": start, "metadata": namespace, "text": ""})
        if len(buf_signals) >= encode_batch:
            flush()
    flush()

    store._save()
    print(store.describe_index_stats())
    return store
