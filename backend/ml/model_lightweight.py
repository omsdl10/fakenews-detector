"""
Lightweight fake-news inference fallback for constrained deployments.

This module intentionally avoids importing PyTorch or transformers so small
hosts can run the API without loading the full ML stack.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


TRUSTED_DOMAINS = {
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


class FakeNewsInference:
    """Source-reputation fallback with the same interface as the full model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        model_path = model_path or settings.CLASSIFIER_MODEL_PATH
        weights_path = Path(model_path) / "pytorch_model.bin"
        self.is_finetuned = False
        self.is_lightweight = True
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        if weights_path.exists():
            logger.warning(
                "Fine-tuned weights exist but lightweight mode is enabled; "
                "skipping heavyweight model load",
                path=model_path,
            )
        else:
            logger.warning(
                "Lightweight mode enabled; using source-reputation fallback",
                path=model_path,
            )

    def predict(self, text: str, source_domain: Optional[str] = None) -> Dict:
        start = time.perf_counter()
        normalised_domain = (source_domain or "").lower().removeprefix("www.")
        is_trusted_source = any(
            normalised_domain == domain
            or normalised_domain.endswith(f".{domain}")
            for domain in TRUSTED_DOMAINS
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
                    f"Lightweight deployment mode is using trusted publisher "
                    f"fallback for {normalised_domain}."
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
                "Lightweight deployment mode is enabled, and this source is "
                "not in the trusted publisher fallback list."
            ),
        }
