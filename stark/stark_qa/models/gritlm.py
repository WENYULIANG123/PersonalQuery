import os.path as osp
import torch
from typing import Any, Union, List, Dict
from tqdm import tqdm
from stark_qa.models.base import ModelForSTaRKQA
from stark_qa.evaluator import Evaluator
from transformers import AutoTokenizer, AutoModel


class GritLM(ModelForSTaRKQA):
    """
    GritLM-7B model for knowledge base retrieval.

    This model uses GritLM-7B as an encoder for both queries and documents,
    performing similarity search in the embedding space.
    """

    def __init__(self,
                 skb: Any,
                 query_emb_dir: str,
                 candidates_emb_dir: str,
                 emb_model: str = 'Muennighoff/SGPT-7B-weightedmean-nli-bitfit',
                 device: str = 'cuda') -> None:
        """
        Initialize the GritLM model.

        Args:
            skb (Any): Knowledge base containing semi-structured data.
            query_emb_dir (str): Directory where query embeddings are stored.
            candidates_emb_dir (str): Directory where candidate embeddings are stored.
            emb_model (str, optional): GritLM model name. Defaults to 'Muennighoff/SGPT-7B-weightedmean-nli-bitfit'.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        super(GritLM, self).__init__(skb, query_emb_dir=query_emb_dir)
        self.emb_model = emb_model
        self.candidates_emb_dir = candidates_emb_dir
        self.device = device
        self.evaluator = Evaluator(self.candidate_ids, device)

        # Load GritLM model and tokenizer
        print(f"Loading GritLM model: {emb_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
        self.model = AutoModel.from_pretrained(emb_model)
        self.model.eval()
        self.model.to(device)

        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load candidate embeddings
        candidate_emb_path = osp.join(candidates_emb_dir, 'candidate_emb_dict.pt')
        if osp.exists(candidate_emb_path):
            candidate_emb_dict = torch.load(candidate_emb_path, map_location='cpu')
            print(f'Loaded candidate_emb_dict from {candidate_emb_path}!')

            assert len(candidate_emb_dict) == len(self.candidate_ids), "Mismatch in candidate embedding count."

            # Stack candidate embeddings into a tensor
            candidate_embs = [candidate_emb_dict[idx].view(1, -1) for idx in self.candidate_ids]
            self.candidate_embs = torch.cat(candidate_embs, dim=0).to(device)
        else:
            print(f"Warning: Candidate embeddings not found at {candidate_emb_path}")
            print("Will encode documents on-the-fly...")
            self.candidate_embs = None

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using GritLM with mean pooling."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                              padding=True, max_length=512)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            # Mean pooling
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

            # Normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            return pooled.cpu()

    def forward(self,
                query: Union[str, List[str]],
                query_id: Union[int, List[int]],
                **kwargs) -> Dict:
        """
        Perform forward pass for GritLM retrieval.

        Args:
            query (Union[str, List[str]]): Query text(s).
            query_id (Union[int, List[int]]): Query ID(s).

        Returns:
            Dict: Dictionary containing retrieval scores for each candidate.
        """
        # Handle batch queries
        if isinstance(query, list):
            # For batch processing, encode each query separately
            all_scores = {}
            for q, qid in zip(query, query_id):
                query_emb = self.encode_text(q)

                if self.candidate_embs is not None:
                    # Use pre-computed embeddings
                    scores = torch.matmul(self.candidate_embs, query_emb.T).squeeze(-1)
                    scores = scores.cpu().numpy()
                else:
                    # Encode candidates on-the-fly (fallback)
                    scores = {}
                    for idx, candidate_id in enumerate(self.candidate_ids):
                        doc_text = self.skb.get_doc_info(candidate_id, add_rel=False, compact=True)
                        doc_emb = self.encode_text(doc_text)
                        score = torch.dot(query_emb.squeeze(), doc_emb.squeeze()).item()
                        scores[candidate_id] = score

                # Convert to dict format expected by evaluator
                pred_dict = {cid: float(score) for cid, score in zip(self.candidate_ids, scores)}
                all_scores[qid] = pred_dict

            return all_scores

        else:
            # Single query
            query_emb = self.encode_text(query)

            if self.candidate_embs is not None:
                # Use pre-computed embeddings
                scores = torch.matmul(self.candidate_embs, query_emb.T).squeeze(-1)
                scores = scores.cpu().numpy()
            else:
                # Encode candidates on-the-fly (fallback)
                scores = {}
                for idx, candidate_id in enumerate(self.candidate_ids):
                    doc_text = self.skb.get_doc_info(candidate_id, add_rel=False, compact=True)
                    doc_emb = self.encode_text(doc_text)
                    score = torch.dot(query_emb.squeeze(), doc_emb.squeeze()).item()
                    scores[candidate_id] = score

            # Convert to dict format expected by evaluator
            pred_dict = {cid: float(score) for cid, score in zip(self.candidate_ids, scores)}
            return pred_dict








