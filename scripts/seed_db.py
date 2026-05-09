"""
Seed the database and FAISS index with a small set of real/fake articles
for local development and testing.

Usage:
    python scripts/seed_db.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logging import get_logger, setup_logging
from backend.db.crud import create_article, update_article_faiss_id
from backend.db.session import AsyncSessionFactory, init_db
from backend.ml.embedder import ArticleEmbedder
from backend.retrieval.faiss_store import FAISSStore

setup_logging()
logger = get_logger(__name__)

SEED_ARTICLES = [
    {
        "title": "FDA approves new mRNA vaccine after Phase 3 trials",
        "text": (
            "The Food and Drug Administration announced today the approval of a new "
            "mRNA-based vaccine following successful Phase 3 clinical trials involving "
            "over 40,000 participants. The trials demonstrated 94.5% efficacy against "
            "severe disease, with the most common side effects being mild arm soreness "
            "and temporary fatigue. The independent safety monitoring board reviewed all "
            "data and found no serious adverse events attributable to the vaccine. "
            "The decision was based on peer-reviewed evidence published in the New England "
            "Journal of Medicine and endorsed by a panel of 20 independent experts."
        ),
        "label": "real",
        "source_url": "https://fda.gov/example/vaccine-approval",
    },
    {
        "title": "BREAKING: Government admits 5G towers causing mass illness",
        "text": (
            "EXCLUSIVE: Whistleblowers from inside the CDC have leaked documents proving "
            "that 5G towers are deliberately designed to weaken the human immune system. "
            "The leaked files, obtained by our investigative team, show that Big Pharma "
            "is working with telecommunications giants to create a permanent market for "
            "medications. The mainstream media refuses to cover this story because they "
            "are all owned by the same shadowy globalist elite. Share this before it gets "
            "deleted! Your life depends on it. The government has been suppressing this "
            "truth for decades while millions suffer."
        ),
        "label": "fake",
        "source_url": "https://conspiracysite.example/5g-truth",
    },
    {
        "title": "Climate study shows accelerating Arctic ice loss",
        "text": (
            "A comprehensive 30-year study published in Nature Climate Change documents "
            "a 13% per decade decline in Arctic sea ice extent. Researchers from NASA "
            "and the National Snow and Ice Data Center analysed satellite data from 1993 "
            "to 2023, finding that the rate of ice loss has accelerated significantly "
            "since 2007. The findings are consistent with climate models predicting "
            "ice-free Arctic summers by mid-century if current emission trajectories "
            "continue. The study's methodology was independently verified by three "
            "separate research groups."
        ),
        "label": "real",
        "source_url": "https://nasa.gov/example/arctic-study",
    },
    {
        "title": "Scientists CONFIRM: Drinking bleach cures cancer — they don't want you to know",
        "text": (
            "A team of rogue scientists have confirmed what alternative health practitioners "
            "have known for decades: diluted bleach, when consumed in small amounts, "
            "destroys cancer cells. The pharmaceutical industry has aggressively suppressed "
            "this research because it would cost them trillions of dollars in lost revenue. "
            "Hundreds of miraculous recoveries have been reported by individuals who tried "
            "this protocol. The mainstream medical establishment refuses to acknowledge "
            "the evidence because they are funded by the same corporations that profit "
            "from keeping people sick."
        ),
        "label": "fake",
        "source_url": "https://altmedsite.example/bleach-cure",
    },
    {
        "title": "Federal Reserve raises interest rates by 25 basis points",
        "text": (
            "The Federal Open Market Committee voted Wednesday to raise the federal funds "
            "rate target by 25 basis points to a range of 5.25% to 5.50%, the highest "
            "level in 22 years. Fed Chair Jerome Powell said the decision reflects the "
            "committee's continued commitment to returning inflation to its 2% target. "
            "The vote was 11-1, with Governor Michelle Bowman dissenting in favour of a "
            "larger 50 basis point increase. Financial markets had priced in the move "
            "with near certainty ahead of the announcement."
        ),
        "label": "real",
        "source_url": "https://reuters.example/fed-rates",
    },
]


async def seed() -> None:
    await init_db()
    embedder = ArticleEmbedder()
    store = FAISSStore()

    async with AsyncSessionFactory() as db:
        for item in SEED_ARTICLES:
            article = await create_article(
                db,
                text=item["text"],
                title=item["title"],
                source_url=item.get("source_url"),
            )
            faiss_id = store.add(
                embeddings=embedder.embed_single(item["text"]).reshape(1, -1),
                article_metas=[{
                    "article_id": str(article.id),
                    "title": item["title"],
                    "text": item["text"],
                    "source_url": item.get("source_url", ""),
                    "label": item["label"],
                }],
            )[0]
            await update_article_faiss_id(db, article.id, faiss_id)
            logger.info(
                "Seeded article",
                title=item["title"][:50],
                label=item["label"],
                faiss_id=faiss_id,
            )

    store.persist()
    logger.info("Seeding complete", total=len(SEED_ARTICLES))


if __name__ == "__main__":
    asyncio.run(seed())
