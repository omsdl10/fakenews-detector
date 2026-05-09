"""
Build (or rebuild) the FAISS index from articles stored in PostgreSQL.

Usage:
    python scripts/build_faiss_index.py --batch_size 64

This script is idempotent: re-running it recreates the index from scratch.
Run this after bulk-importing articles or when the embedding model changes.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sqlalchemy import select

from backend.core.logging import get_logger, setup_logging
from backend.db.models import Article
from backend.db.session import AsyncSessionFactory, init_db
from backend.ml.embedder import ArticleEmbedder
from backend.retrieval.faiss_store import FAISSStore

setup_logging()
logger = get_logger(__name__)


async def build_index(batch_size: int = 64) -> None:
    await init_db()

    embedder = ArticleEmbedder()
    store = FAISSStore()  # starts fresh (empty)

    async with AsyncSessionFactory() as db:
        result = await db.execute(select(Article))
        articles = list(result.scalars().all())

    if not articles:
        logger.warning("No articles found in database. Upload some articles first.")
        return

    logger.info("Building FAISS index", total_articles=len(articles))

    for batch_start in range(0, len(articles), batch_size):
        batch = articles[batch_start : batch_start + batch_size]
        texts = [a.text for a in batch]

        embeddings = embedder.embed(texts, batch_size=batch_size)  # (n, 768)

        metas = [
            {
                "article_id": str(a.id),
                "title": a.title or "",
                "text": a.text,
                "source_url": a.source_url or "",
                "source_domain": a.source_domain or "",
            }
            for a in batch
        ]

        store.add(embeddings, metas)

        pct = min((batch_start + batch_size) / len(articles) * 100, 100)
        logger.info(
            "Indexing progress",
            processed=min(batch_start + batch_size, len(articles)),
            total=len(articles),
            pct=round(pct, 1),
        )

    store.persist()
    logger.info(
        "FAISS index built and saved",
        n_vectors=store.size,
        index_path=str(store.index_path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from DB articles")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    asyncio.run(build_index(args.batch_size))
