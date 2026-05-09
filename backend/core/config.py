"""
Application configuration loaded from environment variables.
All secrets/paths are injected via .env — never hardcoded.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "FakeNews Detector"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 h

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/fakenews"

    # ── Redis ────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # seconds

    # ── Model ────────────────────────────────────────────────────────────────
    CLASSIFIER_MODEL_NAME: str = "roberta-base"
    CLASSIFIER_MODEL_PATH: str = "models/fakenews_roberta"
    EMBEDDER_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    MAX_SEQ_LENGTH: int = 512
    INFERENCE_BATCH_SIZE: int = 8
    DEVICE: str = "cpu"  # "cuda" when GPU is available
    LIGHTWEIGHT_MODE: bool = False  # skip heavyweight ML loads for small hosts

    # ── FAISS ────────────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "models/faiss_index/articles.index"
    FAISS_METADATA_PATH: str = "models/faiss_index/metadata.json"
    EMBEDDING_DIM: int = 768
    TOP_K_EVIDENCE: int = 5

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True  # structured JSON logs in production

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — only one Settings object for the process lifetime."""
    return Settings()
