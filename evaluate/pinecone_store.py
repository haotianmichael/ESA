"""
Pinecone store for DNA2Vec embeddings

Updated for new Pinecone client (v3.0+)
"""
import string
import re
from pinecone import Pinecone, ServerlessSpec
import torch
import numpy as np
from tqdm import tqdm
import random
from inference_models import EvalModel, Baseline
from typing import Optional
import concurrent.futures


class PineconeStore:
    def __init__(
        self,
        device,
        index_name: str,
        metric: str = "cosine",
        model_params = None,
        baseline: bool = False,
        baseline_name: Optional[str] = None
    ):
        if model_params is None and not baseline:
            raise ValueError("Model params are empty.")
        if baseline:
            self.model = Baseline(
                option = baseline_name,
                device = device
            )
        else:
            self.model = EvalModel(
                model_params["tokenizer"],
                model_params["model"],
                model_params["pooling"],
                device = device
            )

        # Sanitize index name to ensure it meets Pinecone requirements
        index_name = self._sanitize_index_name(index_name)
        
        if "config-" in index_name: # premium account
            self.api_key = "pcsk_6YUeeT_Lba6F6sD3Vwo6RDqrgLj8bUqWem4vYbLSFvq73Pts9qyUM3m9boP6eAVMbFdjko"
            self.environment = "us-east-1"
        else:
            raise NotImplementedError("Name not identified.")
        
        self.initialize_pinecone_upsertion(metric, index_name)
        self.index_name = index_name

    @staticmethod
    def _sanitize_index_name(index_name: str) -> str:
        """Sanitize index name to meet Pinecone requirements (lowercase alphanumeric or hyphen)."""
        # Convert to lowercase and replace invalid characters with hyphens
        sanitized = index_name.lower()
        sanitized = re.sub(r'[^a-z0-9-]', '-', sanitized)
        # Remove consecutive hyphens and leading/trailing hyphens
        sanitized = re.sub(r'-+', '-', sanitized).strip('-')
        if not re.match(r'^[a-z0-9-]+$', sanitized):
            raise ValueError(f"Invalid index name: {sanitized}. Must contain only lowercase alphanumeric characters or hyphens.")
        return sanitized

    def initialize_pinecone_upsertion(
        self, 
        metric: str, 
        index_name: str
    ):
        # Initialize Pinecone client with new syntax
        self.pc = Pinecone(api_key=self.api_key)

        # Check if index exists
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
        except Exception:
            # Fallback for different API versions
            existing_indexes = self.pc.list_indexes()
        
        if index_name not in existing_indexes:
            print(f"Creating new index, {index_name}")
            
            try:
                dimension = self.model.get_sentence_embedding_dimension()
            except:
                dimension = 384
            
            # Create index with new syntax
            try:
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.environment
                    )
                )
            except Exception as e:
                print(f"Failed to create index with ServerlessSpec, trying alternative method: {e}")
                # Fallback with spec included
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.environment
                    )
                )

        # Connect to the index
        self.index = self.pc.Index(index_name)

    @staticmethod
    def batched_data_generator(file_path, batch_size):
        """Generator to stream many snippets from text file

        Args:
            file_path (str): Location of file to stream individual snippets
            batch_size (int): stream load

        Yields:
            list(str): list of strings
        """
        import pickle
        
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

    def trigger_pinecone_upsertion(self, file_paths: list, 
                                   batch_size: int = 100, add_namespace=False):
        from tqdm import tqdm
        
        for file_path in file_paths:
            batches = PineconeStore.batched_data_generator(file_path, batch_size)

            for _, (batch, namespace) in tqdm(enumerate(batches)):
                ids = [PineconeStore.generate_random_string() for _ in range(len(batch))]

                # create metadata batch - we can add context here
                metadatas = batch
                texts = [text["text"] for text in batch]
                # create embeddings
                xc = self.model.encode(texts)

                # create records list for upsert
                if isinstance(xc, torch.Tensor):
                    xc = xc.tolist()
                elif isinstance(xc, np.ndarray):
                    xc = xc.tolist()
                records = list(zip(ids, xc, metadatas))
                
                # upsert to Pinecone
                if add_namespace:
                    self.index.upsert(vectors=records, namespace=namespace)
                else:
                    self.index.upsert(vectors=records)

        # check number of records in the index
        print(self.index.describe_index_stats())

    def query_batch(self, queries, indices, top_k=5, hotstart_list=None, meta_dict=None, prioritize=False):
        
        # create the query vector
        xqs = self.model.encode(queries)
        if isinstance(xqs, torch.Tensor):
            xqs = xqs.tolist()
        elif isinstance(xqs, np.ndarray):
            xqs = xqs.tolist()
        
        all_results = []
        
        def query_single(xq, query, index, single_hotstart):
            if not prioritize:
                xc = self.index.query(vector=xq, top_k=top_k, include_metadata=True)
            else:
                xc = self.index.query(vector=xq, top_k=top_k, include_metadata=True,
                                      namespace=single_hotstart)
            
            xc["query"] = query
            xc["index"] = index
            return xc

        if hotstart_list is None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(query_single, xq, query, index, None) \
                    for xq, query, index in zip(xqs, queries, indices)]
            
            for future in concurrent.futures.as_completed(futures):
                all_results.append(future.result())
                
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(query_single, xq, query, index, single_hotstart) \
                    for xq, query, index, single_hotstart in zip(xqs, queries, indices, hotstart_list)]

            for future in concurrent.futures.as_completed(futures):
                all_results.append(future.result())

        return all_results
    
    def drop_table(self):
        """Delete the index"""
        self.pc.delete_index(self.index_name)


if __name__ == "__main__":
    import argparse
    import random

    # fmt: off
    parser = argparse.ArgumentParser(description="Pinecone parse and upload.")
    parser.add_argument('--reupload', help="Should we reupload the data?", type=str, choices=['y','n'])
    parser.add_argument('--drop', help="Should we drop the data?", type=str, choices=['y','n'])
    parser.add_argument('--inputpath', help="Splice input file", type=str)
    parser.add_argument('--indexname', help="Index name", type=str)
    
    args = parser.parse_args()
    # fmt: on

    random.seed(42)

    pinecone_obj = PineconeStore(
        device="cuda:3", 
        index_name=args.indexname
    )

    if args.drop == "y":
        pinecone_obj.drop_table()