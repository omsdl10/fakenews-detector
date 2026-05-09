"""
POST /api/v1/upload-article   — ingest an article into the corpus
GET  /api/v1/search-evidence  — semantic evidence search
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_user, get_retriever
from backend.core.logging import get_logger
from backend.db.crud import create_article, update_article_faiss_id
from backend.db.session import get_db
from backend.retrieval.search import EvidenceRetriever
from backend.schemas.request_response import (
    SearchEvidenceRequest,
    SearchEvidenceResponse,
    EvidenceItem,
    UploadArticleRequest,
    UploadArticleResponse,
)

upload_router = APIRouter(prefix="/upload-article", tags=["Upload"])
evidence_router = APIRouter(prefix="/search-evidence", tags=["Evidence"])
logger = get_logger(__name__)


@upload_router.post(
    "/",
    response_model=UploadArticleResponse,
    summary="Upload an article into the searchable corpus",
    dependencies=[Depends(get_current_user)],  # auth required
)
async def upload_article(
    request: UploadArticleRequest,
    db: AsyncSession = Depends(get_db),
    retriever: EvidenceRetriever = Depends(get_retriever),
) -> UploadArticleResponse:
    """
    Index an article for evidence retrieval.

    - Stores article text in PostgreSQL
    - Embeds article and adds to FAISS index
    - Persists FAISS index to disk

    Requires authentication (Bearer token).
    """
    source_url_str = str(request.source_url) if request.source_url else None

    # 1. Save to DB
    db_article = await create_article(
        db,
        text=request.text,
        title=request.title,
        source_url=source_url_str,
    )

    # 2. Embed + add to FAISS
    faiss_id = retriever.index_article(
        article_id=str(db_article.id),
        text=request.text,
        title=request.title,
        source_url=source_url_str,
        label=request.label_ground_truth,
    )

    # 3. Save FAISS id back to DB row
    await update_article_faiss_id(db, db_article.id, faiss_id)

    logger.info(
        "Article uploaded",
        article_id=str(db_article.id),
        faiss_id=faiss_id,
    )

    return UploadArticleResponse(
        article_id=db_article.id,
        faiss_id=faiss_id,
    )


@evidence_router.post(
    "/",
    response_model=SearchEvidenceResponse,
    summary="Search for evidence articles semantically similar to a query",
)
async def search_evidence(
    request: SearchEvidenceRequest,
    retriever: EvidenceRetriever = Depends(get_retriever),
) -> SearchEvidenceResponse:
    """
    Semantic search over the indexed article corpus.

    Returns the top-k articles most semantically similar to the input query text.
    No authentication required — this endpoint is public.
    """
    raw_results = retriever.retrieve(
        query_text=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    results = [EvidenceItem(**r) for r in raw_results]

    logger.info(
        "Evidence search complete",
        query_length=len(request.query),
        n_results=len(results),
    )

    return SearchEvidenceResponse(
        query_length=len(request.query),
        results=results,
        total_found=len(results),
    )
