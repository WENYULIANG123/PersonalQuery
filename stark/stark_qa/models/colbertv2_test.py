"""
TEST VERSION: ColBERTv2 that processes only FIRST 100 DOCUMENTS

This is a modified version of ColBERTv2 for testing purposes.
It only processes the first 100 documents to allow quick testing.
"""

import os
import os.path as osp
import subprocess
from typing import Any, Union, List, Dict, Optional
from collections import defaultdict

import torch
from tqdm import tqdm

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

from stark_qa.models.base import ModelForSTaRKQA
from stark_qa import load_qa


class Colbertv2Test(ModelForSTaRKQA):
    """
    TEST VERSION: ColBERTv2 Model for STaRK QA - ONLY FIRST 100 DOCUMENTS

    This model integrates the ColBERTv2 dense retrieval model to rank candidates based on their relevance
    to a query from a question-answering dataset.

    *** TEST MODE: Only processes first 100 documents for quick testing ***
    """

    url = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz"

    def __init__(self,
                 skb: Any,
                 dataset_name: str,
                 human_generated_eval: bool,
                 add_rel: bool = False,
                 download_dir: str = 'output',
                 save_dir: str = 'output/colbertv2_test_100',  # Test directory
                 nbits: int = 2,
                 k: int = 100,
                 max_docs: int = 100):  # NEW: Limit to 100 documents

        super(Colbertv2Test, self).__init__(skb)

        self.max_docs = max_docs  # Store the limit
        self.k = k
        self.nbits = nbits

        print(f"🧪 TEST MODE: ColBERTv2 limited to first {max_docs} documents only!")

        query_tsv_name = 'query_hg.tsv' if human_generated_eval else 'query.tsv'
        self.exp_name = dataset_name + '_hg' if human_generated_eval else dataset_name

        # Create experiment directory
        self.experiments_dir = download_dir
        self.query_tsv_path = osp.join(self.experiments_dir, self.exp_name, query_tsv_name)
        self.doc_tsv_path = osp.join(self.experiments_dir, self.exp_name, 'doc.tsv')

        # Load the question-answer dataset and check for required files
        qa_dataset = load_qa(dataset_name, human_generated_eval=human_generated_eval)
        self._check_query_csv(qa_dataset, self.query_tsv_path)
        self._check_doc_csv(skb, self.doc_tsv_path, add_rel)

        # Download and set up the ColBERTv2 model
        self._download()

        # Load the queries and documents into ColBERTv2 format
        self.queries = Queries(self.query_tsv_path)
        self.collection = Collection(self.doc_tsv_path)

        # Prepare the indexer and build the index
        self._prepare_indexer()

        # Run the model and store the results
        self.score_dict = self.run_all()

    def _check_doc_csv(self, skb: Any, doc_tsv_path: str, add_rel: bool) -> None:
        """
        Check if the document TSV file exists; if not, create it from the knowledge base.
        *** TEST MODE: Only first 100 documents ***
        """
        indices = skb.candidate_ids[:self.max_docs]  # LIMIT to first max_docs documents
        print(f"🧪 TEST MODE: Processing only first {len(indices)} documents (out of {len(skb.candidate_ids)} total)")

        self.docid2pid = {idx: i for i, idx in enumerate(indices)}
        self.pid2docid = {i: idx for i, idx in enumerate(indices)}

        if not osp.exists(doc_tsv_path):
            print(f"📝 TEST MODE: Creating document TSV with {len(indices)} documents...")
            lines = []
            for idx in tqdm(indices, desc="Creating test doc TSV"):
                try:
                    doc_text = skb.get_doc_info(idx, add_rel=add_rel, compact=True)
                    lines.append(f"{idx}\t{doc_text.replace(chr(10), ' ').replace(chr(9), ' ')}")
                except Exception as e:
                    print(f"Warning: Failed to get document {idx}: {e}")

            with open(doc_tsv_path, 'w') as file:
                file.write('\n'.join(lines))
            print(f"✅ TEST MODE: Created {len(lines)} document entries in {doc_tsv_path}")
        else:
            print(f'Loaded existing documents from {doc_tsv_path} (limited to {len(indices)} docs)')

    # ... 其余方法与原始ColBERTv2相同，但会自动使用限制的文档集合 ...
