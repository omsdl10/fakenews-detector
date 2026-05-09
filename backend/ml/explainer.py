"""
Explainability module combining two complementary techniques:

1. Attention-based importance  — fast, approximate
   Extract the CLS token's attention to every input token in the last
   transformer layer. Average over heads. Gives intuitive word salience.

2. SHAP (KernelSHAP via transformers-interpret)  — slower, more principled
   Shapley values for each token's contribution to the predicted class.
   Run on demand when deep_explain=True.

Output schema:
    [{"token": "misleading", "score": 0.87, "type": "attention"}, ...]
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple
import numpy as np
import torch

from backend.core.logging import get_logger
from backend.ml.dataset import LABEL_NAMES

logger = get_logger(__name__)


def _clean_token(token: str) -> str:
    """Strip RoBERTa's Ġ prefix (subword marker) for readability."""
    return token.replace("Ġ", " ").replace("Ċ", "\n").strip()


def explain_with_attention(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
    top_k: int = 20,
) -> List[Dict]:
    """
    Fast attention-based token importance.

    Strategy:
      - Run forward pass with output_attentions=True
      - Take the last layer's CLS-to-token attention (averaged over heads)
      - Normalise to [0, 1] range
      - Return top_k tokens with highest scores (excluding special tokens)

    Returns list of {"token": str, "score": float, "start": int, "end": int}
    sorted by score descending.
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # Last layer attention: (1, heads, seq, seq)
    # Take CLS (row 0) attention to all tokens, mean over heads
    last_attn = outputs.attentions[-1][0]  # (heads, seq, seq)
    cls_attn = last_attn[:, 0, :].mean(dim=0)  # (seq,)
    scores = cls_attn.cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Normalise to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # Build results, skip special tokens
    SPECIAL = {"<s>", "</s>", "<pad>", "<mask>", "<unk>"}
    results = []
    for idx, (token, score, offsets) in enumerate(
        zip(tokens, scores, offset_mapping)
    ):
        if token in SPECIAL:
            continue
        results.append({
            "token": _clean_token(token),
            "score": round(float(score), 4),
            "char_start": offsets[0],
            "char_end": offsets[1],
        })

    # Sort by score descending, return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def explain_with_shap(
    model,
    tokenizer,
    text: str,
    target_label: int,
    device: torch.device,
    max_length: int = 256,  # SHAP is slow — use shorter sequences
) -> List[Dict]:
    """
    SHAP-based token attribution using the transformers-interpret library.

    Provides more principled attributions than attention but is ~10–50× slower.
    Intended for deep-dive analysis, not real-time inference.

    Returns list of {"token": str, "score": float} sorted by |score| descending.
    """
    try:
        from transformers_interpret import SequenceClassificationExplainer
    except ImportError:
        logger.warning(
            "transformers-interpret not installed — falling back to attention",
            install="pip install transformers-interpret",
        )
        return []

    # transformers-interpret expects a pipeline-style model; wrap it
    class _WrappedModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.config = inner.config
            self.num_labels = inner.NUM_LABELS

        def forward(self, input_ids, attention_mask=None, labels=None, **_):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    try:
        wrapped = _WrappedModel(model).to(device)
        explainer = SequenceClassificationExplainer(wrapped, tokenizer)
        word_attrs = explainer(text, index=target_label)
    except Exception as exc:
        logger.warning(
            "SHAP explanation failed — falling back to attention",
            error=str(exc),
        )
        return []

    SPECIAL = {"<s>", "</s>", "<pad>"}
    results = [
        {
            "token": _clean_token(word),
            "score": round(float(score), 4),
        }
        for word, score in word_attrs
        if word not in SPECIAL
    ]

    # Sort by absolute importance
    results.sort(key=lambda x: abs(x["score"]), reverse=True)
    return results[:30]


def build_explanation(
    model,
    tokenizer,
    text: str,
    predicted_label: str,
    probabilities: Dict[str, float],
    device: torch.device,
    deep_explain: bool = False,
) -> Dict:
    """
    Assemble the full explanation object returned to the client.

    Returns:
    {
        "method": "attention" | "shap",
        "token_importance": [...],
        "reasoning_summary": str,
    }
    """
    label_idx = LABEL_NAMES.index(predicted_label) if predicted_label in LABEL_NAMES else 0

    if deep_explain:
        token_importance = explain_with_shap(
            model, tokenizer, text, label_idx, device
        )
        method = "shap"
        if not token_importance:
            # Fall back to attention if SHAP unavailable
            token_importance = explain_with_attention(model, tokenizer, text, device)
            method = "attention"
    else:
        token_importance = explain_with_attention(model, tokenizer, text, device)
        method = "attention"

    # Generate a natural-language reasoning summary
    top_tokens = [t["token"] for t in token_importance[:5] if t["token"].strip()]
    confidence = max(probabilities.values())

    reasoning_parts = []
    if predicted_label == "fake":
        reasoning_parts.append(
            f"The article shows characteristics commonly associated with fake news "
            f"(confidence: {confidence:.0%})."
        )
    elif predicted_label == "real":
        reasoning_parts.append(
            f"The article shows characteristics of credible reporting "
            f"(confidence: {confidence:.0%})."
        )
    else:
        reasoning_parts.append(
            f"The article contains mixed signals and cannot be confidently classified "
            f"(top confidence: {confidence:.0%})."
        )

    if top_tokens:
        reasoning_parts.append(
            f"Key influencing terms: {', '.join(repr(t) for t in top_tokens)}."
        )

    return {
        "method": method,
        "token_importance": token_importance,
        "reasoning_summary": " ".join(reasoning_parts),
    }
