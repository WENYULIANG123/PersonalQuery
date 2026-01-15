import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoModel

loaded_bert_models = {}


def get_bert_base_uncased_embeddings(text: Union[str, List[str]],
                                    model_name: str = 'bert-base-uncased',
                                    device: str = 'cuda',
                                    max_length: int = 512,
                                    **kwargs) -> torch.Tensor:
    """
    Get BERT-base-uncased embeddings for the given text.

    This is a simplified implementation that returns mean-pooled embeddings.
    For ColBERT-style embeddings, the model handles chunking separately.

    Args:
        text (Union[str, List[str]]): The input text to be embedded.
        model_name (str): The model to use for embedding (default: bert-base-uncased).
        device (str): Device to run the model on.
        max_length (int): Maximum sequence length.

    Returns:
        torch.Tensor: The embedding(s) of the input text(s).
    """

    if model_name in loaded_bert_models:
        tokenizer, model = loaded_bert_models[model_name]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)

        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        loaded_bert_models[model_name] = (tokenizer, model)

    if isinstance(text, str):
        text = [text]

    embeddings = []
    for t in text:
        inputs = tokenizer(t, return_tensors="pt", truncation=True,
                          padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of the last hidden state
            emb = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_size]
            embeddings.append(emb.cpu())

    if embeddings:
        result = torch.cat(embeddings, dim=0)  # [num_texts, hidden_size]
    else:
        # Fallback for empty input
        result = torch.empty(0, 768)  # BERT-base has 768 hidden size

    return result
