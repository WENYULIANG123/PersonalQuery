import os
import os.path as osp
import random
import sys
import argparse
import pandas as pd

import torch
from tqdm import tqdm

sys.path.append('.')
from stark_qa import load_skb, load_qa
from stark_qa.tools.llm_lib.get_llm_embeddings import get_llm_embeddings

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset and embedding model selection
    parser.add_argument('--dataset', default='amazon', choices=['amazon', 'prime', 'mag'])
    parser.add_argument('--emb_model', default='text-embedding-ada-002',
                        choices=[
                            'text-embedding-ada-002',
                            'text-embedding-3-small',
                            'text-embedding-3-large',
                            'voyage-large-2-instruct',
                            'GritLM/GritLM-7B',
                            'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
                            'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
                            'alibaba-nlp/gte-base-en-v1.5',  # Faster alternative
                            'Qwen/Qwen2.5-0.5B-Instruct',  # Added Qwen2.5-0.5B
                            'Qwen/Qwen2.5-1.5B-Instruct',  # Added Qwen2.5-1.5B
                            'Qwen/Qwen2.5-3B-Instruct',     # Added Qwen2.5-3B
                            'bert-base-uncased',  # Added for ColBERT compatibility
                            'facebook/dpr-ctx_encoder-single-nq-base',  # Added for DPR support (RoBERTa)
                            'sentence-transformers/facebook-dpr-ctx_encoder-multiset-base',  # Sentence-transformers DPR context encoder
                            'sentence-transformers/facebook-dpr-question_encoder-multiset-base',  # Sentence-transformers DPR question encoder
                            'castorini/ance-msmarco-passage'  # Added for ANCE support (RoBERTa-base)
                            ]
                        )

    # Mode settings
    parser.add_argument('--mode', default='doc', choices=['doc', 'query'])

    # Path settings
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--emb_dir", default="emb/", type=str)
    parser.add_argument("--dataset_root", default=None, type=str, help="Custom dataset root directory for variants")
    parser.add_argument("--csv_file", default=None, type=str, help="Direct CSV file path for query generation (bypasses dataset_root)")

    # Text settings
    parser.add_argument('--add_rel', action='store_true', default=False, help='add relation to the text')
    parser.add_argument('--compact', action='store_true', default=False, help='make the text compact when input to the model')

    # Evaluation settings
    parser.add_argument("--human_generated_eval", action="store_true", help="if mode is `query`, then generating query embeddings on human generated evaluation split")

    # Batch and node settings
    parser.add_argument("--batch_size", default=100, type=int)

    # encode kwargs
    parser.add_argument("--n_max_nodes", default=None, type=int, metavar="ENCODE")
    parser.add_argument("--device", default=None, type=str, metavar="ENCODE")
    parser.add_argument("--peft_model_name", default=None, type=str, help="llm2vec pdft model", metavar="ENCODE")
    parser.add_argument("--instruction", type=str, help="gritl/llm2vec instruction", metavar="ENCODE")

    args = parser.parse_args()

    # Create encode_kwargs based on the custom metavar "ENCODE"
    encode_kwargs = {k: v for k, v in vars(args).items() if v is not None and parser._option_string_actions[f'--{k}'].metavar == "ENCODE"}

    return args, encode_kwargs
    

if __name__ == '__main__':
    args, encode_kwargs = parse_args()

    # Smart embedding directory selection
    if args.mode == 'query':
        if args.dataset_root is None:
            # Standard query embeddings - use the canonical path
            emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, 'query_human_generated_eval')
            csv_cache = osp.join(args.data_dir, args.dataset, 'query_human_generated_eval.csv')
        else:
            # Custom dataset (variants) - check for existing embeddings
            dataset_name = osp.basename(args.dataset_root)

            # First, try to reuse the full variants embeddings (most efficient)
            full_variants_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, 'query_no_rel_no_compact_stark_variants_dataset')
            if osp.exists(osp.join(full_variants_dir, 'query_emb_dict.pt')):
                print(f"✅ Found existing full variants embeddings, reusing: {full_variants_dir}")
                emb_dir = full_variants_dir
            else:
                # Fallback to strategy-specific directory
                emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f'query_no_rel_no_compact_{dataset_name}')

            csv_cache = osp.join(args.data_dir, args.dataset, f'query_{dataset_name}.csv')
    else:
        # Document embeddings - use standard logic
        mode_surfix = '_no_rel' if not args.add_rel else ''
        mode_surfix += '_no_compact' if not args.compact else ''
        emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f'{args.mode}{mode_surfix}')
        csv_cache = osp.join(args.data_dir, args.dataset, f'{args.mode}{mode_surfix}.csv')

    print(f'Embedding directory: {emb_dir}')
    print(f'Cache file: {csv_cache}')
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_cache), exist_ok=True)

    if args.mode == 'doc':
        skb = load_skb(args.dataset)
        lst = skb.candidate_ids
        emb_path = osp.join(emb_dir, 'candidate_emb_dict.pt')
    elif args.mode == 'query':
        if args.csv_file is not None:
            # Load directly from CSV file (no dataset creation needed)
            from stark_qa.load_qa import load_custom_qa_dataset
            qa_dataset = load_custom_qa_dataset(args.csv_file)
            lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
            emb_path = osp.join(emb_dir, f'query_emb_dict_{osp.basename(args.csv_file).replace(".csv", "")}.pt')
        elif args.dataset_root is not None:
            # Load custom dataset for variants
            from stark_qa.load_qa import load_custom_qa_dataset
            csv_path = osp.join(args.dataset_root, "qa", "amazon", "stark_qa", "stark_qa.csv")
            qa_dataset = load_custom_qa_dataset(csv_path)
            lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
            emb_path = osp.join(emb_dir, 'query_emb_dict.pt')
        else:
            # Load standard dataset
            qa_dataset = load_qa(args.dataset, human_generated_eval=args.human_generated_eval)
            lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
            emb_path = osp.join(emb_dir, 'query_emb_dict.pt')
    random.shuffle(lst)
    
    # Load existing embeddings if they exist
    if osp.exists(emb_path):
        emb_dict = torch.load(emb_path)
        exist_emb_indices = list(emb_dict.keys())
        print(f'Loaded existing embeddings from {emb_path}. Size: {len(emb_dict)}')
    else:
        emb_dict = {}
        exist_emb_indices = []

    # Load existing document cache if it exists (only for doc mode)
    if args.mode == 'doc' and osp.exists(csv_cache):
        df = pd.read_csv(csv_cache)
        cache_dict = dict(zip(df['index'], df['text']))

        # Ensure that the indices in the cache match the expected indices
        assert set(cache_dict.keys()) == set(lst), 'Indices in cache do not match the candidate indices.'

        indices = list(set(lst) - set(exist_emb_indices))
        texts = [cache_dict[idx] for idx in tqdm(indices, desc="Filtering docs for new embeddings")]
    else:
        indices = lst
        texts = [qa_dataset.get_query_by_qid(idx) if args.mode == 'query'
                 else skb.get_doc_info(idx, add_rel=args.add_rel, compact=args.compact) for idx in tqdm(indices, desc="Gathering docs")]
        if args.mode == 'doc':
            df = pd.DataFrame({'index': indices, 'text': texts})
            df.to_csv(csv_cache, index=False)

    print(f'Generating embeddings for {len(texts)} texts...')
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+args.batch_size]
        batch_embs = get_llm_embeddings(batch_texts, args.emb_model, **encode_kwargs)
        batch_embs = batch_embs.view(len(batch_texts), -1).cpu()
            
        batch_indices = indices[i:i+args.batch_size]
        for idx, emb in zip(batch_indices, batch_embs):
            emb_dict[idx] = emb.view(1, -1)
        
    torch.save(emb_dict, emb_path)
    print(f'Saved {len(emb_dict)} embeddings to {emb_path}!')
