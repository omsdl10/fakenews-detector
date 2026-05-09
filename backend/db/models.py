"""
ORM models for PostgreSQL (async SQLAlchemy 2.x).
Tables: articles, predictions, users.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    predictions = relationship("Prediction", back_populates="user")


class Article(Base):
    """
    Stores full article text and metadata.
    Each article can have one or more predictions and is indexed in FAISS.
    """

    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(512), nullable=True)
    text = Column(Text, nullable=False)
    source_url = Column(String(2048), nullable=True, index=True)
    source_domain = Column(String(255), nullable=True, index=True)
    faiss_id = Column(Integer, nullable=True, index=True)  # FAISS row id
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    predictions = relationship("Prediction", back_populates="article")

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", " ")
        return f"<Article id={self.id} snippet='{snippet}...'>"


class Prediction(Base):
    """
    Stores classifier output for an article, including explanation data.
    label: 0=fake, 1=real, 2=uncertain
    """

    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    label = Column(String(16), nullable=False)       # "fake" | "real" | "uncertain"
    confidence = Column(Float, nullable=False)         # 0.0 – 1.0
    credibility_score = Column(Float, nullable=False)  # 0–100 composite score
    probabilities = Column(JSON, nullable=False)       # {"fake":..., "real":..., "uncertain":...}
    token_importance = Column(JSON, nullable=True)     # [{token, score}, ...]
    evidence_ids = Column(JSON, nullable=True)         # [article_id, ...]

    model_version = Column(String(64), nullable=False, default="roberta-base-v1")
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    article = relationship("Article", back_populates="predictions")
    user = relationship("User", back_populates="predictions")
