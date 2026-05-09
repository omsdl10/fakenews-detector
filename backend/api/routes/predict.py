"""
POST /api/v1/predict
Classifies an article as fake/real/uncertain with explanations and evidence.
"""

import hashlib
import json
import re
import time
from typing import Optional
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import (
    get_cache,
    get_current_user_optional,
    get_inference_model,
    get_retriever,
)
from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.db.crud import create_article, create_prediction
from backend.db.session import get_db
from backend.ml.explainer import build_explanation
from backend.ml.model import FakeNewsInference
from backend.retrieval.search import EvidenceRetriever
from backend.scraper.article_scraper import ArticleScraper
from backend.schemas.request_response import (
    EvidenceItem,
    ExplanationResponse,
    PredictRequest,
    PredictResponse,
    TokenImportance,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = get_logger(__name__)
settings = get_settings()


def _cache_key(text: str) -> str:
    """Deterministic cache key from article text hash."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"predict:{digest}"


async def _scrape_url(url: str) -> str:
    """Extract article text from a URL using the configured scraper."""
    try:
        result = ArticleScraper().scrape(str(url))
        return result.text
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract article from URL: {exc}",
        )


def _text_from_url_fallback(url: str) -> str:
    """Create minimal readable context when a publisher blocks scraping."""
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-3:]
    readable = " ".join(slug)
    readable = re.sub(r"[-_]+", " ", readable)
    readable = re.sub(r"\bcid\b|\b\d+\b", " ", readable)
    readable = re.sub(r"\s+", " ", readable).strip()
    domain = parsed.netloc.replace("www.", "")
    return (
        f"Article URL from {domain}. "
        f"Readable URL context: {readable or parsed.path}."
    )


@router.post(
    "/",
    response_model=PredictResponse,
    summary="Classify a news article",
    responses={
        200: {"description": "Classification result with explanation and evidence"},
        401: {"description": "Invalid or missing auth token"},
        422: {"description": "Validation error"},
    },
)
async def predict(
    request: PredictRequest,
    db: AsyncSession = Depends(get_db),
    cache=Depends(get_cache),
    model: FakeNewsInference = Depends(get_inference_model),
    retriever: EvidenceRetriever = Depends(get_retriever),
    current_user: Optional[dict] = Depends(get_current_user_optional),
) -> PredictResponse:
    """
    Classify a news article as **fake**, **real**, or **uncertain**.

    - Accepts raw text or a URL (scraped automatically)
    - Returns prediction + confidence + credibility score
    - Highlights key tokens driving the prediction
    - Retrieves semantically similar articles as supporting/contradicting evidence
    """
    t_start = time.perf_counter()

    # ── 1. Resolve article text ────────────────────────────────────────────
    source_domain = None
    if request.url:
        source_domain = urlparse(str(request.url)).netloc.replace("www.", "").lower()

    if request.url and not request.text:
        logger.info("Scraping URL", url=str(request.url))
        try:
            article_text = await _scrape_url(str(request.url))
        except HTTPException:
            if model.is_finetuned:
                raise
            logger.warning(
                "URL scrape failed — using source fallback",
                url=str(request.url),
                source_domain=source_domain,
            )
            article_text = _text_from_url_fallback(str(request.url))
    else:
        article_text = request.text

    # ── 2. Check cache ─────────────────────────────────────────────────────
    cache_key = _cache_key(article_text)
    if cache:
        cached = await cache.get(cache_key)
        if cached:
            logger.info("Cache hit", key=cache_key)
            # Still need to return a proper Pydantic model
            return PredictResponse(**json.loads(cached))

    # ── 3. ML inference ────────────────────────────────────────────────────
    prediction = model.predict(article_text, source_domain=source_domain)

    # ── 4. Explainability ──────────────────────────────────────────────────
    explanation_data = build_explanation(
        model=model.model,
        tokenizer=model.tokenizer,
        text=article_text,
        predicted_label=prediction["label"],
        probabilities=prediction["probabilities"],
        device=model.device,
        deep_explain=request.deep_explain,
    )
    if not model.is_finetuned:
        fallback_reason = prediction.get(
            "fallback_reason",
            "A fine-tuned fake-news classifier is not installed.",
        )
        if prediction["label"] == "real":
            explanation_data["reasoning_summary"] = (
                f"{fallback_reason} This is a source-reputation signal, not a "
                "full content-level fact check. Add trained weights at "
                "models/fakenews_roberta/pytorch_model.bin for model-based "
                "fake/real predictions."
            )
        else:
            explanation_data["reasoning_summary"] = (
                f"{fallback_reason} The app is avoiding a fake/real guess "
                "until trained weights are available at "
                "models/fakenews_roberta/pytorch_model.bin."
            )

    # ── 5. Evidence retrieval ──────────────────────────────────────────────
    evidence_items: list[EvidenceItem] = []
    if request.retrieve_evidence:
        raw_evidence = retriever.retrieve(
            query_text=article_text,
            top_k=settings.TOP_K_EVIDENCE,
            query_label=prediction["label"],
        )
        evidence_items = [EvidenceItem(**e) for e in raw_evidence]

    # ── 6. Persist to database ─────────────────────────────────────────────
    source_url_str = str(request.url) if request.url else None
    db_article = await create_article(
        db,
        text=article_text,
        source_url=source_url_str,
    )

    user_id = UUID(current_user["sub"]) if current_user else None
    total_latency = (time.perf_counter() - t_start) * 1000

    db_prediction = await create_prediction(
        db,
        article_id=db_article.id,
        label=prediction["label"],
        confidence=prediction["confidence"],
        credibility_score=prediction["credibility_score"],
        probabilities=prediction["probabilities"],
        token_importance=explanation_data["token_importance"],
        evidence_ids=[e.article_id for e in evidence_items],
        user_id=user_id,
        latency_ms=round(total_latency, 2),
    )

    # ── 7. Build response ──────────────────────────────────────────────────
    response = PredictResponse(
        article_id=db_article.id,
        prediction_id=db_prediction.id,
        label=prediction["label"],
        confidence=prediction["confidence"],
        credibility_score=prediction["credibility_score"],
        probabilities=prediction["probabilities"],
        explanation=ExplanationResponse(
            method=explanation_data["method"],
            token_importance=[
                TokenImportance(**t) for t in explanation_data["token_importance"]
            ],
            reasoning_summary=explanation_data["reasoning_summary"],
        ),
        evidence=evidence_items,
        latency_ms=round(total_latency, 2),
    )

    # ── 8. Cache the result ────────────────────────────────────────────────
    if cache:
        await cache.setex(
            cache_key,
            settings.CACHE_TTL,
            response.model_dump_json(),
        )

    logger.info(
        "Prediction complete",
        label=prediction["label"],
        confidence=prediction["confidence"],
        latency_ms=round(total_latency, 2),
    )
    return response
