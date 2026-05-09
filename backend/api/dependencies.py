"""
FastAPI dependency providers.
Singletons are created once at startup and shared across requests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.core.security import decode_access_token
from backend.db.session import get_db

if get_settings().LIGHTWEIGHT_MODE:
    from backend.ml.model_lightweight import FakeNewsInference
else:
    from backend.ml.model import FakeNewsInference

logger = get_logger(__name__)
settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)


# ── ML model (loaded once, shared) ───────────────────────────────────────────

_inference_model: Optional[FakeNewsInference] = None
_embedder: Optional[Any] = None
_faiss_store: Optional[Any] = None
_retriever: Optional[Any] = None


def get_inference_model() -> FakeNewsInference:
    global _inference_model
    if _inference_model is None:
        _inference_model = FakeNewsInference()
    return _inference_model


def get_embedder() -> Any:
    global _embedder
    if settings.LIGHTWEIGHT_MODE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence embeddings are disabled in lightweight deployment mode.",
        )
    if _embedder is None:
        from backend.ml.embedder import ArticleEmbedder

        _embedder = ArticleEmbedder()
    return _embedder


def get_faiss_store() -> Any:
    global _faiss_store
    if settings.LIGHTWEIGHT_MODE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence search is disabled in lightweight deployment mode.",
        )
    if _faiss_store is None:
        from backend.retrieval.faiss_store import FAISSStore

        _faiss_store = FAISSStore()
    return _faiss_store


def get_retriever(
    faiss_store: Any = Depends(get_faiss_store),
    embedder: Any = Depends(get_embedder),
) -> Any:
    global _retriever
    if settings.LIGHTWEIGHT_MODE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence retrieval is disabled in lightweight deployment mode.",
        )
    if _retriever is None:
        from backend.retrieval.search import EvidenceRetriever

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
