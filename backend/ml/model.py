"""
RoBERTa-based fake news classifier.

Architecture:
  - RoBERTa base encoder (12 layers, 768 hidden dim)
  - Dropout for regularisation
  - Linear classification head (768 → 3 classes)
  - Softmax for calibrated probabilities

The model is trained with label smoothing and class-weighted cross-entropy
to handle the real-world imbalance between fake/real/uncertain examples.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.ml.dataset import LABEL_NAMES

logger = get_logger(__name__)
settings = get_settings()


class FakeNewsClassifier(nn.Module):
    """
    Custom RoBERTa sequence classifier.

    We extend nn.Module rather than RobertaForSequenceClassification so we can:
      - attach custom dropout rates
      - output raw attention weights for explainability
      - easily swap the encoder backbone later
    """

    NUM_LABELS = 3  # fake=0, real=1, uncertain=2

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
        class_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=self.NUM_LABELS,
            output_attentions=True,  # expose attention for explainability
        )
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.NUM_LABELS)

        # Weighted cross-entropy handles class imbalance
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float)
            if class_weights
            else None
        )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        # Use [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)             # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,  # (layers, batch, heads, seq, seq)
        )

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return averaged last-layer attention weights.
        Shape: (batch, seq_len, seq_len) — averaged over all heads.
        """
        with torch.no_grad():
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        # Last layer, average over heads: (batch, seq, seq)
        last_layer_attn = outputs.attentions[-1]
        return last_layer_attn.mean(dim=1)


# ── Inference wrapper ─────────────────────────────────────────────────────────

class FakeNewsInference:
    """
    Stateless inference wrapper. Handles:
      - tokenization
      - batched inference
      - probability calibration
      - label thresholding (uncertain band)
    """

    UNCERTAIN_THRESHOLD = 0.55  # below this confidence → "uncertain"

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.device = torch.device(settings.DEVICE)
        model_path = model_path or settings.CLASSIFIER_MODEL_PATH
        weights_path = Path(model_path) / "pytorch_model.bin"
        self.is_finetuned = weights_path.exists()

        logger.info("Loading classifier", path=model_path, device=settings.DEVICE)

        if self.is_finetuned:
            self.model = self._load_finetuned(model_path)
        else:
            logger.warning(
                "Fine-tuned model not found, loading base weights",
                path=model_path,
            )
            self.model = FakeNewsClassifier(settings.CLASSIFIER_MODEL_NAME)

        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            settings.CLASSIFIER_MODEL_NAME
        )
        logger.info("Classifier ready")

    def _load_finetuned(self, model_path: str) -> FakeNewsClassifier:
        model = FakeNewsClassifier(settings.CLASSIFIER_MODEL_NAME)
        state = torch.load(
            Path(model_path) / "pytorch_model.bin",
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state)
        return model

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(path) / "pytorch_model.bin")
        self.tokenizer.save_pretrained(path)
        logger.info("Model saved", path=path)

    @torch.no_grad()
    def predict(self, text: str, source_domain: Optional[str] = None) -> Dict:
        """
        Classify a single article text.

        Returns:
            {
              "label": "fake" | "real" | "uncertain",
              "confidence": float,
              "credibility_score": float (0-100),
              "probabilities": {"fake": float, "real": float, "uncertain": float},
              "latency_ms": float,
            }
        """
        start = time.perf_counter()

        if not self.is_finetuned:
            trusted_domains = {
                "apnews.com",
                "bbc.com",
                "bbc.co.uk",
                "bloomberg.com",
                "economist.com",
                "ft.com",
                "indianexpress.com",
                "npr.org",
                "nytimes.com",
                "reuters.com",
                "telegraphindia.com",
                "theguardian.com",
                "thehindu.com",
                "washingtonpost.com",
            }
            normalised_domain = (source_domain or "").lower().removeprefix("www.")
            is_trusted_source = any(
                normalised_domain == domain
                or normalised_domain.endswith(f".{domain}")
                for domain in trusted_domains
            )
            latency_ms = (time.perf_counter() - start) * 1000
            if is_trusted_source:
                return {
                    "label": "real",
                    "confidence": 0.75,
                    "credibility_score": 82.0,
                    "probabilities": {
                        "fake": 0.05,
                        "real": 0.75,
                        "uncertain": 0.20,
                    },
                    "latency_ms": round(latency_ms, 2),
                    "fallback_reason": (
                        f"No fine-tuned classifier is installed; using trusted "
                        f"publisher fallback for {normalised_domain}."
                    ),
                }
            return {
                "label": "uncertain",
                "confidence": 0.0,
                "credibility_score": 50.0,
                "probabilities": {
                    "fake": 0.0,
                    "real": 0.0,
                    "uncertain": 1.0,
                },
                "latency_ms": round(latency_ms, 2),
                "fallback_reason": (
                    "No fine-tuned classifier is installed, and this source is "
                    "not in the trusted publisher fallback list."
                ),
            }

        encoding = self.tokenizer(
            text,
            max_length=settings.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(output.logits, dim=-1)[0].cpu().tolist()

        fake_p, real_p, uncertain_p = probs
        top_label_idx = int(torch.argmax(output.logits[0]).item())
        confidence = probs[top_label_idx]

        # If confidence is below threshold, force "uncertain"
        label = LABEL_NAMES[top_label_idx]
        if confidence < self.UNCERTAIN_THRESHOLD:
            label = "uncertain"

        # Credibility score: 0 (definitely fake) → 100 (definitely real)
        # Formula: weighted blend of real and uncertain probs
        credibility_score = round((real_p * 100) + (uncertain_p * 30), 1)
        credibility_score = min(max(credibility_score, 0.0), 100.0)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "credibility_score": credibility_score,
            "probabilities": {
                "fake": round(fake_p, 4),
                "real": round(real_p, 4),
                "uncertain": round(uncertain_p, 4),
            },
            "latency_ms": round(latency_ms, 2),
        }

    def get_token_ids_and_tokens(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Return input_ids tensor and human-readable tokens for explainability."""
        encoding = self.tokenizer(
            text,
            max_length=settings.MAX_SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0].tolist()
        )
        return encoding["input_ids"].to(self.device), tokens
