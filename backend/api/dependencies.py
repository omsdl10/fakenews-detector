"""
FastAPI dependency providers.
Singletons are created once at startup and shared across requests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.security import decode_access_token
from backend.db.session import get_db
from backend.ml.model import FakeNewsInference
from backend.ml.embedder import ArticleEmbedder
from backend.retrieval.faiss_store import FAISSStore
from backend.retrieval.search import EvidenceRetriever

logger = get_logger(__name__)
settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)


# ── ML model (loaded once, shared) ───────────────────────────────────────────

_inference_model: Optional[FakeNewsInference] = None
_embedder: Optional[ArticleEmbedder] = None
_faiss_store: Optional[FAISSStore] = None
_retriever: Optional[EvidenceRetriever] = None


def get_inference_model() -> FakeNewsInference:
    global _inference_model
    if _inference_model is None:
        _inference_model = FakeNewsInference()
    return _inference_model


def get_embedder() -> ArticleEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ArticleEmbedder()
    return _embedder


def get_faiss_store() -> FAISSStore:
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSStore()
    return _faiss_store


def get_retriever(
    faiss_store: FAISSStore = Depends(get_faiss_store),
    embedder: ArticleEmbedder = Depends(get_embedder),
) -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever(faiss_store=faiss_store, embedder=embedder)
    return _retriever


# ── Redis cache ───────────────────────────────────────────────────────────────

_redis_client: Optional[aioredis.Redis] = None


async def get_cache() -> Optional[aioredis.Redis]:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await _redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning("Redis unavailable — running without cache", error=str(e))
            _redis_client = None
    return _redis_client


# ── Auth ──────────────────────────────────────────────────────────────────────

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[dict]:
    """Return user payload if valid JWT is provided, else None (unauthenticated allowed)."""
    if not credentials:
        return None
    payload = decode_access_token(credentials.credentials)
    return payload


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """Require a valid JWT. Raise 401 if missing or invalid."""
    payload = await get_current_user_optional(credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload
