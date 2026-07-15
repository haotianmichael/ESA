"""
Signal-domain vector store: a thin subclass of ``FaissStore``.

The whole point is maximal reuse — index creation, GPU placement, cosine /
IndexFlatIP, namespaces, persistence, batched search and top-k merging all come
from ``FaissStore`` unchanged (``git diff evaluate/faiss_store.py`` stays empty).
This subclass only:

  1. injects a ``SignalEvalModel`` as ``self.model`` (so the inherited
     ``_embed`` / ``query_batch`` work on raw signals), and
  2. adds ``add_embeddings`` to upsert *precomputed* reference vectors (reference
     embeddings are computed once by the signal model, not re-derived from text).
"""
from __future__ import annotations

import os
from typing import List, Optional

import faiss
import numpy as np

from faiss_store import FaissStore, _default_index_dir


class SignalFaissStore(FaissStore):
    def __init__(
        self,
        signal_model,
        index_name: str,
        device="cpu",
        metric: str = "cosine",
        persist_dir: Optional[str] = None,
    ):
        # Replicate FaissStore.__init__ but inject a signal model instead of
        # constructing a text EvalModel/Baseline.
        self.model = signal_model

        self.metric = metric.lower()
        self._normalize = self.metric == "cosine"
        self._higher_is_better = self.metric in ("cosine", "dotproduct")

        index_name = self._sanitize_index_name(index_name)
        self.index_name = index_name

        try:
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception:
            self.dimension = 384

        self.persist_dir = persist_dir or _default_index_dir()
        self.index_path = os.path.join(self.persist_dir, index_name)

        self._init_gpu(device)

        self.namespaces = {}
        self._load()

    def add_embeddings(
        self,
        vectors: np.ndarray,
        metadatas: List[dict],
        namespace: str = "",
        save: bool = True,
    ):
        """Upsert precomputed reference vectors with their coordinate metadata."""
        entry = self._get_or_create_namespace(namespace)
        xc = np.ascontiguousarray(vectors, dtype=np.float32)
        if xc.ndim == 1:
            xc = xc.reshape(1, -1)
        if self._normalize:
            faiss.normalize_L2(xc)
        entry["index"].add(xc)
        entry["metadata"].extend(metadatas)
        if save:
            self._save()
