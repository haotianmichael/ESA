"""
Read-side window tiling + all-vs-all read-window FAISS index (Neurosamble Phase 1).

This is the *build-only* half of the read<->read overlap probe, à la Rawsamble's
all-vs-all (ava) seeding. It exists to answer ONE gating question upstream of any
overlapper: does the frozen encoder produce dense, accurate anchors when BOTH
sides of a comparison are noisy real reads? It deliberately does NOTHING else --
no chaining, no miniasm, no assembly PAF.

Reuse (frozen, unmodified):
  - ``SignalEvalModel.encode`` (inference_signal.py): batched z-norm + fixed
    window + T-mask; returns L2-normalized [W, D] embeddings.
  - ``SignalFaissStore.add_embeddings`` / ``query_batch`` (signal_faiss_store.py
    -> faiss_store.py): upsert precomputed vectors + coordinate metadata, then
    flat cosine search. ``git diff`` of both stores stays empty.
  - ``preprocess_window`` (dna2vec.signal_dataset) is applied *inside* encode();
    ``tile_read`` therefore only slices the raw recorded signal.

Same-strand only, exactly like Rawsamble's ava: we tile the recorded signal and
never add reverse-complement windows (a read and its RC are not both recorded, so
mixing them would only manufacture false, unchainable anchors).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def tile_read(signal, win: int = 2000, stride: int = 1000) -> List[Tuple[int, np.ndarray]]:
    """Slide a fixed-length window over the FULL read.

    Unlike the read->reference mapping path (which truncates each read to the
    first ``input_signal_len`` samples), here we tile the ENTIRE read so an
    overlap anywhere along two reads can surface anchors.

    Returns a list of ``(offset_samples, window_signal)``. The last partial
    window is kept (zero-padded downstream by ``preprocess_window``) iff its
    length ``>= win // 2``, otherwise it is dropped.
    """
    signal = np.asarray(signal, dtype=np.float32)
    n = int(signal.shape[0])
    win = int(win)
    stride = int(stride)
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive")

    windows: List[Tuple[int, np.ndarray]] = []
    if n <= 0:
        return windows

    off = 0
    while off < n:
        w = signal[off : off + win]
        if w.shape[0] == win:
            windows.append((off, w))
            off += stride
        else:
            # Trailing partial window: keep only if long enough to be meaningful.
            if w.shape[0] >= win // 2:
                windows.append((off, w))
            break
    return windows


def build_readwindow_index(
    reads,
    model,
    store,
    win: int = 2000,
    stride: int = 1000,
    namespace: str = "reads",
) -> Tuple[int, int]:
    """Tile every read, encode its windows, and upsert them into ``store``.

    For each read we tile -> ``model.encode(windows)`` (batched inside encode) ->
    ``store.add_embeddings(vecs, metas)`` where each meta is::

        {"read_id": rid, "win_idx": j, "offset": off_samples,
         "strand": "+", "metadata": namespace, "text": ""}

    Same-strand only (no reverse-complement windows). Persistence is skipped
    (``save=False``): the probe builds and queries the index in one process, so
    keeping it purely in memory avoids stale on-disk reloads and disk churn.

    Returns ``(n_windows, n_reads)`` and prints the index size.
    """
    n_windows = 0
    n_reads = 0
    for r in reads:
        rid = r.id
        tiles = tile_read(r.signal, win=win, stride=stride)
        if not tiles:
            continue
        offs = [int(off) for off, _ in tiles]
        wins = [w for _, w in tiles]
        vecs = model.encode(wins)  # [W, D], already L2-normalized
        metas = [
            {
                "read_id": rid,
                "win_idx": j,
                "offset": offs[j],
                "strand": "+",
                "metadata": namespace,
                "text": "",
            }
            for j in range(len(offs))
        ]
        store.add_embeddings(vecs, metas, namespace=namespace, save=False)
        n_windows += len(wins)
        n_reads += 1

    print(
        f"[index] read-window index built (same-strand): "
        f"n_windows={n_windows} n_reads={n_reads}",
        flush=True,
    )
    return n_windows, n_reads
