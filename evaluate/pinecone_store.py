"""
Backwards-compatible shim.

The vector store now runs locally on FAISS instead of the cloud-hosted
Pinecone service (see ``faiss_store.py``). This module is kept so that existing
imports such as ``from pinecone_store import PineconeStore`` continue to work;
``PineconeStore`` is simply an alias for the local ``FaissStore``.
"""
from faiss_store import FaissStore
from inference_models import EvalModel, Baseline

# Alias for backwards compatibility with the previous Pinecone-based API.
PineconeStore = FaissStore

__all__ = ["FaissStore", "PineconeStore", "EvalModel", "Baseline"]


if __name__ == "__main__":
    import argparse
    import random

    # fmt: off
    parser = argparse.ArgumentParser(description="Local FAISS parse and management.")
    parser.add_argument('--reupload', help="Should we reupload the data?", type=str, choices=['y', 'n'])
    parser.add_argument('--drop', help="Should we drop the data?", type=str, choices=['y', 'n'])
    parser.add_argument('--inputpath', help="Splice input file", type=str)
    parser.add_argument('--indexname', help="Index name", type=str)
    args = parser.parse_args()
    # fmt: on

    random.seed(42)

    store = PineconeStore(
        device="cuda:3",
        index_name=args.indexname,
    )

    if args.drop == "y":
        store.drop_table()
