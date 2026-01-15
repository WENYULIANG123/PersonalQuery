import torch
from typing import List, Union
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

loaded_dpr_models = {}


def get_dpr_embeddings(text: Union[str, List[str]],
                      model_name: str = 'facebook/dpr-ctx_encoder-single-nq-base',
                      device: str = 'cuda',
                      max_length: int = 512,
                      **kwargs) -> torch.Tensor:
    """
    Get DPR context encoder embeddings for the given text.

    Args:
        text (Union[str, List[str]]): The input text to be embedded.
        model_name (str): The DPR model to use for embedding.
        device (str): Device to run the model on.
        max_length (int): Maximum sequence length.

    Returns:
        torch.Tensor: The embedding(s) of the input text(s).
    """

    if model_name in loaded_dpr_models:
        tokenizer, model = loaded_dpr_models[model_name]
    else:
        # Handle ANCE models (castorini/ance-msmarco-passage)
        if 'castorini/ance' in model_name:
            from transformers import AutoTokenizer, AutoModel
            print(f"🔧 Loading ANCE model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        # Handle sentence-transformers DPR models
        elif 'sentence-transformers' in model_name:
            # For sentence-transformers models, try using Auto classes
            from transformers import AutoTokenizer, AutoModel
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                # Use mean pooling for sentence-transformers models
                model.pooling_mode = 'mean'
            except Exception as e:
                # Fallback to DPR classes if Auto classes fail
                print(f"Warning: Failed to load {model_name} with Auto classes, trying DPR classes: {e}")
                tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
                model = DPRContextEncoder.from_pretrained(model_name)
        else:
            # Standard DPR models
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
            model = DPRContextEncoder.from_pretrained(model_name)

        model.eval()
        model.to(device)

        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        loaded_dpr_models[model_name] = (tokenizer, model)

    if isinstance(text, str):
        text = [text]

    embeddings = []
    for t in text:
        inputs = tokenizer(t, return_tensors="pt", truncation=True,
                          padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

            # Handle different pooling methods for different model types
            if 'castorini/ance' in model_name:
                # ANCE models use pooler_output (trained for retrieval)
                emb = outputs.pooler_output
                print(f"🔧 Using pooler_output for ANCE model")
            elif 'sentence-transformers' in model_name:
                # For sentence-transformers models, use pooler_output (better for retrieval)
                # Previously used mean pooling, but pooler_output is trained for similarity
                emb = outputs.pooler_output
                print(f"🔧 Using pooler_output for sentence-transformers DPR model")
            else:
                # Standard DPR models use pooler_output
                emb = outputs.pooler_output

            embeddings.append(emb.cpu())

    if embeddings:
        result = torch.cat(embeddings, dim=0)  # [num_texts, hidden_size]
    else:
        # Fallback for empty input (DPR has 768 hidden size)
        result = torch.empty(0, 768)

    return result



