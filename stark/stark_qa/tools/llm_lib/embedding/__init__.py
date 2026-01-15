from .gritlm import get_gritlm_embeddings
from .llm2vec import get_llm2vec_embeddings
from .openai import get_openai_embeddings
from .voyage import get_voyage_embeddings
from .gte_qwen import get_gte_qwen_embeddings
from .dpr import get_dpr_embeddings
from .qwen import get_qwen_embeddings

REGISTERED_EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "voyage-large-2-instruct",
    "GritLM/GritLM-*",
    "McGill-NLP/LLM2Vec-*",
    "Alibaba-NLP/gte-Qwen2-*",
    "Qwen/Qwen2.5-*",
    "facebook/dpr-ctx_encoder-single-nq-base",
    "sentence-transformers/facebook-dpr-ctx_encoder-multiset-base",
    "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
]