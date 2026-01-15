import os
import os.path as osp
import torch
import numpy as np
from typing import Any, Union, List, Dict
from tqdm import tqdm
from stark_qa.models.base import ModelForSTaRKQA
from stark_qa.evaluator import Evaluator
from transformers import AutoTokenizer, AutoModel


class ColBERT(ModelForSTaRKQA):
    """
    ColBERT model for knowledge base retrieval.

    This model uses ColBERT approach with document chunking and MaxSim aggregation
    for fine-grained retrieval using BERT-base as approximation.
    """

    def __init__(self,
                 skb: Any,
                 query_emb_dir: str,
                 candidates_emb_dir: str,
                 emb_model: str = 'bert-base-uncased',
                 device: str = 'cuda') -> None:
        """
        Initialize the ColBERT model.

        Args:
            skb (Any): Knowledge base containing semi-structured data.
            query_emb_dir (str): Directory where query embeddings are stored.
            candidates_emb_dir (str): Directory where candidate embeddings are stored.
            emb_model (str, optional): ColBERT model name. Defaults to 'bert-base-uncased'.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        super(ColBERT, self).__init__(skb, query_emb_dir=query_emb_dir)
        self.emb_model = emb_model
        self.candidates_emb_dir = candidates_emb_dir
        self.device = device
        self.evaluator = Evaluator(self.candidate_ids, device)

        # Load ColBERT model and tokenizer (BERT-base approximation)
        print(f"Loading ColBERT model: {emb_model}")
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
        print(f"🔍 [ColBERT Init] Checking embeddings at: {candidate_emb_path}")
        print(f"🔍 [ColBERT Init] Embeddings file exists: {osp.exists(candidate_emb_path)}")
        print(f"🔍 [ColBERT Init] Candidate IDs count: {len(self.candidate_ids) if hasattr(self, 'candidate_ids') else 'Not set'}")

        candidate_emb_dict = None  # Initialize to None
        if osp.exists(candidate_emb_path):
            try:
                loaded_data = torch.load(candidate_emb_path, map_location='cpu', weights_only=False)
                print(f'🔍 Loaded data keys: {list(loaded_data.keys()) if isinstance(loaded_data, dict) else "not dict"}')

                # Check if this is our saved format with 'candidate_emb_dict' key
                if isinstance(loaded_data, dict) and 'candidate_emb_dict' in loaded_data:
                    candidate_emb_dict = loaded_data['candidate_emb_dict']
                    print(f'✅ Extracted candidate_emb_dict with {len(candidate_emb_dict)} entries!')
                else:
                    # Assume the loaded data is directly the candidate_emb_dict
                    candidate_emb_dict = loaded_data
                print(f'✅ Loaded candidate_emb_dict with {len(candidate_emb_dict)} entries!')

                # Validate embeddings - check if it's not empty and has reasonable content
                if len(candidate_emb_dict) == 0:
                    print("⚠️  Embeddings file exists but is empty! Will regenerate.")
                    candidate_emb_dict = None
                elif len(candidate_emb_dict) < 100 and len(self.candidate_ids) >= 100:
                    print(f"⚠️  Embeddings file has only {len(candidate_emb_dict)} entries, expected more! Will regenerate.")
                    candidate_emb_dict = None
                elif len(candidate_emb_dict) < 10 and len(self.candidate_ids) < 100:  # Very suspicious if less than 10 documents
                    print(f"⚠️  Embeddings file has very few entries ({len(candidate_emb_dict)})! Will regenerate.")
                    candidate_emb_dict = None
                else:
                    # Quick validation - check if embeddings look reasonable
                    sample_key = list(candidate_emb_dict.keys())[0]
                    sample_value = candidate_emb_dict[sample_key]
                    if isinstance(sample_value, (list, tuple)) and len(sample_value) == 0:
                        print("⚠️  Embeddings appear to be empty lists! Will regenerate.")
                        candidate_emb_dict = None
                    elif hasattr(sample_value, 'shape') and sample_value.shape[0] == 0:
                        print("⚠️  Embeddings appear to be empty arrays! Will regenerate.")
                        candidate_emb_dict = None
            except Exception as e:
                print(f"⚠️  Error loading embeddings: {e}. Will regenerate.")
                candidate_emb_dict = None

        if candidate_emb_dict is None:
            print("🔄 Embeddings validation failed, will regenerate...")
            # Fall through to generation code below
        else:
            # Process chunk embeddings for ColBERT
            self.doc_chunks_embeddings = []
            self.doc_ids = []
            self.chunk_info = []

            # Get the actual document IDs that have embeddings
            available_doc_ids = set(candidate_emb_dict.keys())

            print(f"🔄 Processing embeddings for {len(available_doc_ids)} documents with available embeddings...")


            processed_docs = 0
            for doc_id in available_doc_ids:  # Only process docs that actually have embeddings
                    chunks_data = candidate_emb_dict[doc_id]
                    if isinstance(chunks_data, list):  # Multiple chunks
                        for chunk_idx, chunk_emb in enumerate(chunks_data):
                            self.doc_chunks_embeddings.append(chunk_emb)
                            self.doc_ids.append(doc_id)
                            self.chunk_info.append({
                                'doc_id': doc_id,
                                'chunk_idx': chunk_idx,
                                'chunk_info': {}
                            })
                    else:  # Single embedding, create one chunk
                        self.doc_chunks_embeddings.append(chunks_data)
                        self.doc_ids.append(doc_id)
                        self.chunk_info.append({
                            'doc_id': doc_id,
                            'chunk_idx': 0,
                            'chunk_info': {}
                        })
                    processed_docs += 1

            print(f"✅ Processed {len(self.doc_chunks_embeddings)} chunks for {len(set(self.doc_ids))} documents")
            print(f"📊 Successfully loaded embeddings for {processed_docs} documents")

            # Check if we successfully loaded embeddings
            if len(self.doc_chunks_embeddings) > 0:
                return  # Successfully loaded, exit

        # If we reach here, embeddings need to be generated
        print(f"Warning: Candidate embeddings not found or invalid at {candidate_emb_path}")
        print("Generating ColBERT chunk embeddings...")

        # Generate embeddings for all candidates with batch processing
        self.doc_chunks_embeddings = []
        self.doc_ids = []
        self.chunk_info = []

        # Process in batches to avoid memory issues
        batch_size = 500  # Process 500 documents at a time (reduced for speed)
        save_interval = 2000  # Save every 2000 documents (more frequent)

        for batch_start in range(0, len(self.candidate_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(self.candidate_ids))
            batch_candidates = self.candidate_ids[batch_start:batch_end]

            print(f"🔄 Processing batch {batch_start//batch_size + 1}: documents {batch_start}-{batch_end-1}")

            batch_chunks_embeddings = []
            batch_doc_ids = []
            batch_chunk_info = []

            for idx_in_batch, candidate_id in enumerate(batch_candidates):
                global_idx = batch_start + idx_in_batch
            doc_text = self.skb.get_doc_info(candidate_id, add_rel=False, compact=True)

            # Debug: Check document text
            if global_idx < 10:  # Only print first 10 documents
                print(f"🔍 Doc {candidate_id}: text length = {len(doc_text) if doc_text else 0}, first 100 chars: '{doc_text[:100] if doc_text else 'None'}'")

            # Split document into chunks
            chunks = [doc_text[i:i+256] for i in range(0, len(doc_text), 200)] if doc_text else []
            if not chunks:
                chunks = [doc_text] if doc_text else []

            doc_chunks = []
            doc_chunk_info = []

            for chunk_idx, chunk in enumerate(chunks):
                if chunk and chunk.strip():  # Skip empty chunks
                    try:
                        chunk_emb = self.encode_text_colbert(chunk)
                        doc_chunks.append(chunk_emb)
                        doc_chunk_info.append({
                            'chunk_idx': chunk_idx,
                            'chunk_text': chunk,
                            'num_tokens': len(chunk_emb)
                        })
                    except Exception as e:
                        print(f"Warning: Failed to encode chunk {chunk_idx} for doc {candidate_id}: {e}")
                        continue
                else:
                    if global_idx < 10:  # Debug empty chunks
                        print(f"⚠️  Doc {candidate_id} chunk {chunk_idx} is empty")

            # Always add document to the batch, even if it has no chunks
            # This ensures ALL candidate documents have embeddings (even if empty)
            if doc_chunks:
                # Document has valid chunks
                batch_chunks_embeddings.extend(doc_chunks)
                batch_doc_ids.extend([candidate_id] * len(doc_chunks))
                batch_chunk_info.extend(doc_chunk_info)
            else:
                # Document has no valid chunks - create a minimal representation
                # This ensures the document still exists in our embedding collection
                empty_emb = torch.zeros((1, 768), dtype=torch.float32)  # Minimal BERT-sized embedding
                batch_chunks_embeddings.append(empty_emb)
                batch_doc_ids.append(candidate_id)
                batch_chunk_info.append({
                    'chunk_idx': 0,
                    'chunk_text': doc_text or "",  # Empty string if no text
                    'num_tokens': 1  # Minimal token count
                })

            if (global_idx + 1) % 100 == 0:
                print(f"Processed {global_idx + 1}/{len(self.candidate_ids)} documents, {len(self.doc_chunks_embeddings) + len(batch_chunks_embeddings)} total chunks")

            # Add batch results to main lists
            self.doc_chunks_embeddings.extend(batch_chunks_embeddings)
            self.doc_ids.extend(batch_doc_ids)
            self.chunk_info.extend(batch_chunk_info)

            # Clear GPU cache periodically to prevent memory accumulation
            if (batch_end) % save_interval == 0 or batch_end == len(self.candidate_ids):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"Generated {len(self.doc_chunks_embeddings)} chunks for {len(set(self.doc_ids))} documents")

        # Final save of all embeddings
        self._save_embeddings_final(candidates_emb_dir)

    def _save_embeddings_final(self, candidates_emb_dir):
        """Save the final embeddings directly."""
        candidate_emb_path = osp.join(candidates_emb_dir, 'candidate_emb_dict.pt')
        os.makedirs(candidates_emb_dir, exist_ok=True)

        # Create the candidate embeddings dictionary
        candidate_emb_dict = {}
        for doc_id, emb, info in zip(self.doc_ids, self.doc_chunks_embeddings, self.chunk_info):
            if doc_id not in candidate_emb_dict:
                candidate_emb_dict[doc_id] = []
            candidate_emb_dict[doc_id].append({'embedding': emb, 'info': info})

        # Save the embeddings
        save_data = {
            'candidate_emb_dict': candidate_emb_dict,
            'processed_count': len(candidate_emb_dict),
            'total_candidates': len(self.candidate_ids),
            'model': self.emb_model,
            'device': str(self.device)
        }

        torch.save(save_data, candidate_emb_path)
        print(f"✅ Saved ColBERT embeddings to {candidate_emb_path}")
        print(f"   📊 Total documents: {len(candidate_emb_dict)}")
        print(f"   📊 Total chunks: {sum(len(chunks) for chunks in candidate_emb_dict.values())}")

    def encode_text_colbert(self, text: str) -> torch.Tensor:
        """Encode text using ColBERT approach (one vector per token)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                              padding=True, max_length=512)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

            # Remove batch dimension
            embeddings = embeddings[0]

            # Get attention mask to exclude padding tokens
            attention_mask = inputs['attention_mask'][0]

            # Only keep non-padding tokens
            embeddings = embeddings[attention_mask == 1]

            # Normalize embeddings (ColBERT style)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy()

    def calculate_maxsim(self, query_embedding, doc_embedding):
        """Calculate MaxSim score between query and document embeddings using GPU acceleration."""
        # query_embedding: [query_len, dim] - numpy array
        # doc_embedding: [doc_len, dim] - numpy array

        # Convert to torch tensors and move to GPU
        query_tensor = torch.from_numpy(query_embedding).to(self.device)  # [query_len, dim]
        doc_tensor = torch.from_numpy(doc_embedding).to(self.device)      # [doc_len, dim]

        # Compute cosine similarity matrix using torch (much faster on GPU)
        # query_tensor @ doc_tensor.T gives [query_len, doc_len] similarity matrix
        sim_matrix = torch.matmul(query_tensor, doc_tensor.T)  # [query_len, doc_len]

        # MaxSim: max over query tokens (dim=0), then max over doc tokens (dim=0)
        max_sim = torch.max(sim_matrix).item()  # Get scalar value

        return max_sim

    def forward(self,
                query: Union[str, List[str]],
                query_id: Union[int, List[int]],
                **kwargs) -> Dict:
        """
        Perform forward pass for ColBERT retrieval.

        Args:
            query (Union[str, List[str]]): Query text(s).
            query_id (Union[int, List[int]]): Query ID(s).

        Returns:
            Dict: Dictionary containing retrieval scores for each candidate.
        """
        # Handle batch queries
        if isinstance(query, list):
            all_scores = {}
            for q, qid in zip(query, query_id):
                # Encode query
                query_embedding = self.encode_text_colbert(q)

                # Debug: Check query embedding shape
                if qid == 0:  # Only print for first query
                    print(f"🔍 Query {qid} embedding shape: {query_embedding.shape}")
                    print(f"🔍 Query {qid} sample values: {query_embedding[0, :3] if len(query_embedding) > 0 else 'empty'}")

                if self.doc_chunks_embeddings is not None:
                    # Use pre-computed embeddings with GPU acceleration
                    doc_scores = {}
                    doc_chunk_counts = {}

                    for doc_id in self.candidate_ids:
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = []
                            doc_chunk_counts[doc_id] = 0

                    # Convert query embedding to tensor once for GPU acceleration
                    query_tensor = torch.from_numpy(query_embedding).to(self.device)

                    # Process each chunk with GPU acceleration
                    for chunk_idx, chunk_doc_id in enumerate(self.doc_ids):
                        if chunk_doc_id in self.candidate_ids:
                            chunk_embedding = self.doc_chunks_embeddings[chunk_idx]
                            maxsim_score = self.calculate_maxsim(query_embedding, chunk_embedding)
                            doc_scores[chunk_doc_id].append(maxsim_score)
                            doc_chunk_counts[chunk_doc_id] += 1

                    # Aggregate scores per document (max pooling across chunks)
                    final_doc_scores = {}
                    for doc_id in doc_scores:
                        if doc_scores[doc_id]:
                            final_doc_scores[doc_id] = max(doc_scores[doc_id])
                        else:
                            final_doc_scores[doc_id] = 0.0

                    pred_dict = {cid: float(final_doc_scores.get(cid, 0.0)) for cid in self.candidate_ids}

                    # Debug: Check if all scores are the same
                    scores = list(pred_dict.values())
                    unique_scores = set(scores)
                    if len(unique_scores) <= 5:  # If very few unique scores
                        print(f"⚠️  Warning: Only {len(unique_scores)} unique scores found: {sorted(unique_scores)[:10]}")
                        # Check if top scores are all the same
                        sorted_scores = sorted(scores, reverse=True)
                        if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
                            print("⚠️  Warning: Top scores are identical!")
                else:
                    # Encode candidates on-the-fly (fallback)
                    pred_dict = {}
                    for idx, candidate_id in enumerate(self.candidate_ids):
                        doc_text = self.skb.get_doc_info(candidate_id, add_rel=False, compact=True)
                        # Simple chunking for fallback
                        chunks = [doc_text[i:i+256] for i in range(0, len(doc_text), 200)]
                        max_score = 0.0
                        for chunk in chunks:
                            try:
                                doc_embedding = self.encode_text_colbert(chunk)
                                score = self.calculate_maxsim(query_embedding, doc_embedding)
                                max_score = max(max_score, score)
                            except:
                                continue
                        pred_dict[candidate_id] = float(max_score)

                all_scores[qid] = pred_dict

            return all_scores

        else:
            # Single query
            query_embedding = self.encode_text_colbert(query)

            if self.doc_chunks_embeddings is not None:
                # Use pre-computed embeddings
                doc_scores = {}
                doc_chunk_counts = {}

                for doc_id in self.candidate_ids:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = []
                        doc_chunk_counts[doc_id] = 0

                # Process each chunk
                for chunk_idx, (chunk_doc_id, chunk_start, chunk_end) in enumerate(zip(self.doc_ids, [0]*len(self.doc_ids), [len(chunk) for chunk in self.doc_chunks_embeddings])):
                    if chunk_doc_id in self.candidate_ids:
                        chunk_embedding = self.doc_chunks_embeddings[chunk_idx]
                        maxsim_score = self.calculate_maxsim(query_embedding, chunk_embedding)
                        doc_scores[chunk_doc_id].append(maxsim_score)
                        doc_chunk_counts[chunk_doc_id] += 1

                # Aggregate scores per document (max pooling across chunks)
                final_doc_scores = {}
                for doc_id in doc_scores:
                    if doc_scores[doc_id]:
                        final_doc_scores[doc_id] = max(doc_scores[doc_id])
                    else:
                        final_doc_scores[doc_id] = 0.0

                pred_dict = {cid: float(final_doc_scores.get(cid, 0.0)) for cid in self.candidate_ids}
            else:
                # Encode candidates on-the-fly (fallback)
                pred_dict = {}
                for idx, candidate_id in enumerate(self.candidate_ids):
                    doc_text = self.skb.get_doc_info(candidate_id, add_rel=False, compact=True)
                    # Simple chunking for fallback
                    chunks = [doc_text[i:i+256] for i in range(0, len(doc_text), 200)]
                    max_score = 0.0
                    for chunk in chunks:
                        try:
                            doc_embedding = self.encode_text_colbert(chunk)
                            score = self.calculate_maxsim(query_embedding, doc_embedding)
                            max_score = max(max_score, score)
                        except:
                            continue
                    pred_dict[candidate_id] = float(max_score)

            return pred_dict
