import os
import pandas as pd
import ast
from typing import Union
from stark_qa.retrieval import *


def load_qa(name: str,
            root: Union[str, None] = None,
            human_generated_eval: bool = False) -> STaRKDataset:
    """
    Load the QA dataset.

    Args:
        name (str): Name of the dataset. One of 'amazon', 'prime', or 'mag'.
        root (Union[str, None]): Root directory to store the dataset. If not provided, the default Hugging Face cache path is used.
        human_generated_eval (bool): Whether to use human-generated evaluation data. Default is False.

    Returns:
        STaRKDataset: The loaded STaRK dataset.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    assert name in REGISTERED_DATASETS, f"Unknown dataset {name}"
    
    if root is not None:
        if not os.path.isabs(root):
            root = os.path.abspath(root)

    return STaRKDataset(name, root,
                        human_generated_eval=human_generated_eval)


def load_custom_qa_dataset(csv_path, dataframe=None):
    """Load QA dataset directly from CSV file or DataFrame."""
    if dataframe is not None:
        df = dataframe
    else:
        df = pd.read_csv(csv_path)

    class CustomQADataset:
        def __init__(self, dataframe):
            self.data = []
            for idx, row in dataframe.iterrows():
                answers = ast.literal_eval(row['answer_ids']) if isinstance(row['answer_ids'], str) else row['answer_ids']
                self.data.append((row['query'], int(row['id']), answers, None))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def get_idx_split(self, test_ratio=1.0):
            import torch
            total = len(self.data)
            indices = torch.tensor(list(range(total)))
            return {'variants': indices, 'test': indices, 'val': indices, 'train': indices}

        def get_query_by_qid(self, qid):
            """Get query text by query id."""
            for query, query_id, answers, meta in self.data:
                if query_id == qid:
                    return query
            raise ValueError(f"Query ID {qid} not found in dataset")

    return CustomQADataset(df)

