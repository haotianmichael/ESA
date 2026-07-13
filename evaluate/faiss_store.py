"""
Local FAISS store for DNA2Vec embeddings.

Drop-in replacement for the previous cloud-hosted ``PineconeStore``. Instead of
upserting vectors into a remote Pinecone index, this keeps everything on the
local filesystem: each index is persisted under ``FAISS_INDEX_DIR`` (defaults to
``<this_dir>/faiss_indexes``) so that the two-step "upsert then evaluate"
workflow still works across separate processes.

The public surface (``trigger_pinecone_upsertion``, ``query_batch``,
``drop_table`` and the Pinecone-shaped query results) mirrors the old store so
none of the downstream evaluation scripts need to change.
"""
import os
import re
import string
import random
import pickle
from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:  # torch is only needed to detect tensor-typed embeddings
    torch = None

from inference_models import EvalModel, Baseline


def _default_index_dir() -> str:
    return os.environ.get(
        "FAISS_INDEX_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_indexes"),
    )


class FaissStore:
    # Default namespace used when vectors are upserted without partitioning,
    # mirroring Pinecone's empty-string default namespace.
    DEFAULT_NAMESPACE = ""

    def __init__(
        self,
        device,
        index_name: str,
        metric: str = "cosine",
        model_params=None,
        baseline: bool = False,
        baseline_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        if model_params is None and not baseline:
            raise ValueError("Model params are empty.")
        if baseline:
            self.model = Baseline(
                option=baseline_name,
                device=device,
            )
        else:
            self.model = EvalModel(
                model_params["tokenizer"],
                model_params["model"],
                model_params["pooling"],
                device=device,
            )

        self.metric = metric.lower()
        # Cosine similarity is implemented as inner product over L2-normalized
        # vectors. Both cosine and dotproduct rank higher scores first; L2 ranks
        # smaller distances first.
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

        # namespace -> {"index": faiss.Index, "metadata": list[dict]}
        self.namespaces = {}
        self._load()

    # ------------------------------------------------------------------ #
    # GPU helpers
    # ------------------------------------------------------------------ #
    def _init_gpu(self, device):
        """Enable GPU-backed search when a CUDA device and faiss-gpu are present.

        Falls back to CPU transparently so the store still works with faiss-cpu
        or when no GPU is available.
        """
        self._gpu_id = 0
        self._gpu_resources = None
        self.use_gpu = False

        device_str = str(device)
        if "cuda" not in device_str:
            return
        if ":" in device_str:
            try:
                self._gpu_id = int(device_str.split(":")[-1])
            except ValueError:
                self._gpu_id = 0

        gpu_available = (
            hasattr(faiss, "StandardGpuResources")
            and getattr(faiss, "get_num_gpus", lambda: 0)() > 0
        )
        if not gpu_available:
            return
        if self._gpu_id >= faiss.get_num_gpus():
            self._gpu_id = 0
        self._gpu_resources = faiss.StandardGpuResources()
        self.use_gpu = True

    def _to_gpu(self, index):
        if not self.use_gpu:
            return index
        return faiss.index_cpu_to_gpu(self._gpu_resources, self._gpu_id, index)

    @staticmethod
    def _to_cpu(index):
        """Return a CPU copy of a (possibly GPU-resident) index for saving."""
        if hasattr(faiss, "index_gpu_to_cpu") and hasattr(index, "getDevice"):
            try:
                return faiss.index_gpu_to_cpu(index)
            except Exception:
                return index
        return index

    # ------------------------------------------------------------------ #
    # Naming / persistence helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sanitize_index_name(index_name: str) -> str:
        """Sanitize index name to a filesystem-safe, lowercase-hyphen form."""
        sanitized = index_name.lower()
        sanitized = re.sub(r"[^a-z0-9-]", "-", sanitized)
        sanitized = re.sub(r"-+", "-", sanitized).strip("-")
        if not re.match(r"^[a-z0-9-]+$", sanitized):
            raise ValueError(
                f"Invalid index name: {sanitized}. Must contain only lowercase "
                "alphanumeric characters or hyphens."
            )
        return sanitized

    def _new_index(self) -> faiss.Index:
        if self.metric == "l2" or self.metric == "euclidean":
            index = faiss.IndexFlatL2(self.dimension)
        else:
            # cosine (normalized) and dotproduct both use inner product.
            index = faiss.IndexFlatIP(self.dimension)
        # Exact flat search on GPU: fast and lossless for this data scale.
        return self._to_gpu(index)

    def _get_or_create_namespace(self, namespace: str):
        entry = self.namespaces.get(namespace)
        if entry is None:
            entry = {"index": self._new_index(), "metadata": []}
            self.namespaces[namespace] = entry
        return entry

    def _load(self):
        meta_file = os.path.join(self.index_path, "store.pkl")
        if not os.path.exists(meta_file):
            return
        with open(meta_file, "rb") as f:
            saved = pickle.load(f)
        self.metric = saved.get("metric", self.metric)
        self.dimension = saved.get("dimension", self.dimension)
        self._normalize = self.metric == "cosine"
        self._higher_is_better = self.metric in ("cosine", "dotproduct")
        for ns, info in saved["namespaces"].items():
            index = faiss.read_index(os.path.join(self.index_path, info["file"]))
            self.namespaces[ns] = {
                "index": self._to_gpu(index),
                "metadata": info["metadata"],
            }

    def _save(self):
        os.makedirs(self.index_path, exist_ok=True)
        namespaces_meta = {}
        for i, (ns, entry) in enumerate(self.namespaces.items()):
            file_name = f"ns_{i}.faiss"
            faiss.write_index(
                self._to_cpu(entry["index"]),
                os.path.join(self.index_path, file_name),
            )
            namespaces_meta[ns] = {"file": file_name, "metadata": entry["metadata"]}
        with open(os.path.join(self.index_path, "store.pkl"), "wb") as f:
            pickle.dump(
                {
                    "metric": self.metric,
                    "dimension": self.dimension,
                    "namespaces": namespaces_meta,
                },
                f,
            )

    # ------------------------------------------------------------------ #
    # Embedding helper
    # ------------------------------------------------------------------ #
    def _embed(self, texts) -> np.ndarray:
        xc = self.model.encode(texts)
        if torch is not None and isinstance(xc, torch.Tensor):
            xc = xc.detach().cpu().numpy()
        elif not isinstance(xc, np.ndarray):
            xc = np.asarray(xc)
        xc = np.ascontiguousarray(xc, dtype=np.float32)
        if xc.ndim == 1:
            xc = xc.reshape(1, -1)
        if self._normalize:
            faiss.normalize_L2(xc)
        return xc

    # ------------------------------------------------------------------ #
    # Data streaming (unchanged from the Pinecone version)
    # ------------------------------------------------------------------ #
    @staticmethod
    def batched_data_generator(file_path, batch_size):
        """Generator to stream many snippets from a pickled text dump.

        Args:
            file_path (str): Location of file to stream individual snippets.
            batch_size (int): Stream load.

        Yields:
            tuple(list(dict), str): batch of records and its namespace.
        """
        namespace = ""
        with open(file_path, "rb") as f:
            list_of_objects = pickle.load(f)
            batch = []

            for unit in list_of_objects:
                if namespace == "":
                    namespace = unit["metadata"]

                if namespace != unit["metadata"] or len(batch) >= batch_size:
                    yield batch, namespace
                    batch = [unit]
                    namespace = unit["metadata"]
                else:
                    batch.append(unit)

            if batch:
                yield batch, namespace

    @staticmethod
    def generate_random_string(length: int = 20):
        letters = string.ascii_lowercase
        return "".join(random.choice(letters) for _ in range(length))

    # ------------------------------------------------------------------ #
    # Upsert / index building
    # ------------------------------------------------------------------ #
    def trigger_pinecone_upsertion(
        self, file_paths: list, batch_size: int = 100, add_namespace=False
    ):
        """Build the local FAISS index from one or more pickled data dumps."""
        for file_path in file_paths:
            batches = FaissStore.batched_data_generator(file_path, batch_size)

            for _, (batch, namespace) in tqdm(enumerate(batches)):
                target_namespace = namespace if add_namespace else self.DEFAULT_NAMESPACE
                entry = self._get_or_create_namespace(target_namespace)

                texts = [record["text"] for record in batch]
                xc = self._embed(texts)

                entry["index"].add(xc)
                # Store the full record dict as metadata; downstream code reads
                # metadata["position"], metadata["metadata"], metadata["text"].
                entry["metadata"].extend(batch)

        self._save()
        print(self.describe_index_stats())

    def describe_index_stats(self):
        namespaces = {
            ns: {"vector_count": entry["index"].ntotal}
            for ns, entry in self.namespaces.items()
        }
        total = sum(entry["index"].ntotal for entry in self.namespaces.values())
        return {
            "dimension": self.dimension,
            "total_vector_count": total,
            "namespaces": namespaces,
        }

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def _search_namespace(self, namespace: str, xqs: np.ndarray, top_k: int):
        """Search a single namespace, returning (scores, ids) arrays."""
        entry = self.namespaces.get(namespace)
        if entry is None or entry["index"].ntotal == 0:
            n = xqs.shape[0]
            empty_scores = np.full((n, top_k), -np.inf, dtype=np.float32)
            empty_ids = np.full((n, top_k), -1, dtype=np.int64)
            return empty_scores, empty_ids
        k = min(top_k, entry["index"].ntotal)
        distances, ids = entry["index"].search(xqs, k)
        if not self._higher_is_better:
            # Convert L2 distances into a descending-sortable score.
            distances = -distances
        return distances, ids

    def _collect_matches(self, namespace_hits, row: int, top_k: int):
        """Merge per-namespace hits for a single query row into ranked matches."""
        candidates = []
        for namespace, (scores, ids) in namespace_hits:
            metadata = self.namespaces[namespace]["metadata"]
            for score, idx in zip(scores[row], ids[row]):
                if idx < 0:
                    continue
                candidates.append((float(score), namespace, int(idx), metadata[idx]))
        candidates.sort(key=lambda c: c[0], reverse=True)
        matches = []
        for score, namespace, idx, meta in candidates[:top_k]:
            matches.append(
                {
                    "id": f"{namespace}:{idx}" if namespace else str(idx),
                    "score": score,
                    "metadata": meta,
                }
            )
        return matches

    def query_batch(
        self,
        queries,
        indices,
        top_k=5,
        hotstart_list=None,
        meta_dict=None,
        prioritize=False,
    ):
        xqs = self._embed(queries)
        n = len(queries)
        results = [None] * n

        if prioritize and hotstart_list is not None:
            # Restrict each query to its own hotstart namespace. Group rows by
            # namespace so each namespace is searched in a single batched call.
            rows_by_namespace = {}
            for i, namespace in enumerate(hotstart_list):
                target = namespace if namespace in self.namespaces else None
                rows_by_namespace.setdefault(target, []).append(i)

            for namespace, rows in rows_by_namespace.items():
                sub = xqs[rows]
                if namespace is None:
                    # Unknown namespace: fall back to searching every namespace.
                    hits = [
                        (ns, self._search_namespace(ns, sub, top_k))
                        for ns in self.namespaces
                    ]
                else:
                    hits = [(namespace, self._search_namespace(namespace, sub, top_k))]

                for local_row, global_row in enumerate(rows):
                    matches = self._collect_matches(hits, local_row, top_k)
                    results[global_row] = {
                        "matches": matches,
                        "query": queries[global_row],
                        "index": indices[global_row],
                    }
        else:
            # No namespace preference: search across all namespaces and merge.
            hits = [
                (ns, self._search_namespace(ns, xqs, top_k))
                for ns in self.namespaces
            ]
            for row in range(n):
                matches = self._collect_matches(hits, row, top_k)
                results[row] = {
                    "matches": matches,
                    "query": queries[row],
                    "index": indices[row],
                }

        return results

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #
    def drop_table(self):
        """Delete the persisted index from disk and clear it in memory."""
        import shutil

        self.namespaces = {}
        if os.path.isdir(self.index_path):
            shutil.rmtree(self.index_path)


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(description="Local FAISS parse and management.")
    parser.add_argument('--reupload', help="Should we reupload the data?", type=str, choices=['y', 'n'])
    parser.add_argument('--drop', help="Should we drop the data?", type=str, choices=['y', 'n'])
    parser.add_argument('--inputpath', help="Splice input file", type=str)
    parser.add_argument('--indexname', help="Index name", type=str)
    args = parser.parse_args()
    # fmt: on

    random.seed(42)

    faiss_obj = FaissStore(
        device="cuda:3",
        index_name=args.indexname,
    )

    if args.drop == "y":
        faiss_obj.drop_table()
