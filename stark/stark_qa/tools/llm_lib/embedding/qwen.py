import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

def get_qwen_embeddings(texts: List[str], model_name: str, **kwargs) -> torch.Tensor:
    """
    Get embeddings from Qwen models using mean pooling of last hidden states.

    Args:
        texts: List of input texts
        model_name: Qwen model name (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')
        **kwargs: Additional arguments

    Returns:
        torch.Tensor: Embeddings with shape (len(texts), hidden_size)
    """
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    # Set model to evaluation mode
    model.eval()

    embeddings = []

    for text in texts:
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        # Use last hidden states and perform mean pooling
        last_hidden_states = outputs.last_hidden_state

        # Create attention mask for mean pooling
        attention_mask = inputs['attention_mask']
        attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

        # Mean pooling
        masked_embeddings = last_hidden_states * attention_mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        embedding = summed / summed_mask

        # Normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=1)

        embeddings.append(embedding.cpu())

    # Concatenate all embeddings
    if embeddings:
        return torch.cat(embeddings, dim=0)
    else:
        return torch.empty(0, model.config.hidden_size)



















