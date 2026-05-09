"""
CRUD helpers for Articles and Predictions.
All operations are async and receive an injected session.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Article, Prediction, User
from backend.core.logging import get_logger

logger = get_logger(__name__)


# ── Article ───────────────────────────────────────────────────────────────────

async def create_article(
    db: AsyncSession,
    text: str,
    title: Optional[str] = None,
    source_url: Optional[str] = None,
    source_domain: Optional[str] = None,
    faiss_id: Optional[int] = None,
) -> Article:
    article = Article(
        text=text,
        title=title,
        source_url=source_url,
        source_domain=source_domain,
        faiss_id=faiss_id,
    )
    db.add(article)
    await db.flush()  # get the generated id without committing
    logger.info("Article created", article_id=str(article.id))
    return article


async def get_article(db: AsyncSession, article_id: UUID) -> Optional[Article]:
    result = await db.execute(select(Article).where(Article.id == article_id))
    return result.scalar_one_or_none()


async def get_articles_by_faiss_ids(
    db: AsyncSession, faiss_ids: List[int]
) -> List[Article]:
    result = await db.execute(
        select(Article).where(Article.faiss_id.in_(faiss_ids))
    )
    return list(result.scalars().all())


async def update_article_faiss_id(
    db: AsyncSession, article_id: UUID, faiss_id: int
) -> None:
    article = await get_article(db, article_id)
    if article:
        article.faiss_id = faiss_id
        await db.flush()


# ── Prediction ────────────────────────────────────────────────────────────────

async def create_prediction(
    db: AsyncSession,
    article_id: UUID,
    label: str,
    confidence: float,
    credibility_score: float,
    probabilities: dict,
    token_importance: Optional[list] = None,
    evidence_ids: Optional[list] = None,
    user_id: Optional[UUID] = None,
    model_version: str = "roberta-base-v1",
    latency_ms: Optional[float] = None,
) -> Prediction:
    prediction = Prediction(
        article_id=article_id,
        user_id=user_id,
        label=label,
        confidence=confidence,
        credibility_score=credibility_score,
        probabilities=probabilities,
        token_importance=token_importance,
        evidence_ids=evidence_ids,
        model_version=model_version,
        latency_ms=latency_ms,
    )
    db.add(prediction)
    await db.flush()
    logger.info("Prediction saved", prediction_id=str(prediction.id), label=label)
    return prediction


async def get_predictions_for_article(
    db: AsyncSession, article_id: UUID
) -> List[Prediction]:
    result = await db.execute(
        select(Prediction)
        .where(Prediction.article_id == article_id)
        .order_by(Prediction.created_at.desc())
    )
    return list(result.scalars().all())


# ── User ──────────────────────────────────────────────────────────────────────

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, email: str, hashed_password: str) -> User:
    user = User(email=email, hashed_password=hashed_password)
    db.add(user)
    await db.flush()
    return user
