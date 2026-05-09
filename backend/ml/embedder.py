"""
Article embedder using sentence-transformers.

Produces 768-dimensional dense embeddings suitable for FAISS indexing.
Uses mean-pooling over the last hidden state (proven more stable than [CLS]).

The embedder is used for:
  1. Building the FAISS index at ingestion time
  2. Encoding query articles at inference time for evidence retrieval
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool token embeddings, respecting padding.
    token_embeddings: (batch, seq, hidden)
    attention_mask:   (batch, seq)
    Returns: (batch, hidden)
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class ArticleEmbedder:
    """
    Wraps a sentence-transformer model for batch-efficient embedding.

    Normalises embeddings to unit norm so FAISS inner-product search
    is equivalent to cosine similarity.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        model_name = model_name or settings.EMBEDDER_MODEL_NAME
        self.device = torch.device(settings.DEVICE)

        logger.info("Loading embedder", model=model_name, device=settings.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        logger.info("Embedder ready")

    @torch.no_grad()
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed one or more texts.

        Args:
            texts: Single string or list of strings.
            batch_size: Number of texts per forward pass.
            normalize: L2-normalize embeddings (required for cosine FAISS).

        Returns:
            np.ndarray of shape (n_texts, embedding_dim) as float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            encoding = self.tokenizer(
                batch,
                max_length=settings.MAX_SEQ_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            embeddings = _mean_pool(outputs.last_hidden_state, attention_mask)

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience method for single-text embedding. Returns shape (dim,)."""
        return self.embed([text])[0]
