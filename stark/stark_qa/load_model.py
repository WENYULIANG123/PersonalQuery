from stark_qa.models import *


def load_model(args, skb, **kwargs):
    model_name = args.model
    assert model_name in REGISTERED_MODELS, f'{model_name} not implemented'

    if model_name == 'BM25':
        return BM25(skb)
    if model_name == 'Colbertv2':
        try:
            # Try absolute import first, then relative import
            try:
                from stark_qa.models.colbertv2 import Colbertv2
            except ImportError:
                from .models.colbertv2 import Colbertv2
            return Colbertv2(skb,
                             dataset_name=args.dataset,
                             save_dir=args.output_dir,
                             download_dir=args.download_dir,
                             experiments_dir=args.output_dir,  # Use same directory for experiments
                             human_generated_eval=args.split=='human_generated_eval',
                             strategy=getattr(args, 'strategy', 'original'),
                             dataset_root=getattr(args, 'dataset_root', None),
                             **kwargs
                             )
        except ImportError:
            raise ImportError("Please install the colbert package using `pip install colbert-ai`.")
    elif model_name == 'ColBERT':
        return ColBERT(
            skb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir,
            candidates_emb_dir=args.node_emb_dir,
            device=args.device
        )
    elif model_name == 'GritLM':
        return GritLM(
            skb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir,
            candidates_emb_dir=args.node_emb_dir,
            device=args.device
        )
    elif model_name == 'VSS':
        return VSS(
            skb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            device=args.device
        )
    if model_name == 'MultiVSS':
        return MultiVSS(
            skb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            chunk_emb_dir=args.chunk_emb_dir,
            aggregate=args.aggregate,
            chunk_size=args.chunk_size,
            max_k=args.multi_vss_topk,
            device=args.device
        )
    if model_name == 'LLMReranker':
        return LLMReranker(skb, 
                           emb_model=args.emb_model,
                           llm_model=args.llm_model,
                           query_emb_dir=args.query_emb_dir, 
                           candidates_emb_dir=args.node_emb_dir,
                           max_cnt = args.max_retry,
                           max_k=args.llm_topk,
                           device=args.device
                           )