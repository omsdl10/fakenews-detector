"""
Evidence retrieval: given a query article, find semantically similar
articles that either support or contradict its claims.

The contradiction vs. support signal is estimated by comparing the
classifier labels of the query and retrieved articles.
"""

from __future__ import annotations

from typing import List, Optional

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.ml.embedder import ArticleEmbedder
from backend.retrieval.faiss_store import FAISSStore

logger = get_logger(__name__)
settings = get_settings()


RELATION_LABELS = {
    ("fake", "real"): "contradicts",
    ("real", "fake"): "contradicts",
    ("fake", "fake"): "supports",
    ("real", "real"): "supports",
    ("uncertain", "uncertain"): "inconclusive",
    ("fake", "uncertain"): "inconclusive",
    ("real", "uncertain"): "inconclusive",
    ("uncertain", "fake"): "inconclusive",
    ("uncertain", "real"): "inconclusive",
}


def _infer_relation(query_label: Optional[str], evidence_label: Optional[str]) -> str:
    if not query_label or not evidence_label:
        return "inconclusive"
    return RELATION_LABELS.get((query_label, evidence_label), "inconclusive")


class EvidenceRetriever:
    """
    Semantic search over the article corpus.

    Steps:
      1. Embed query text → dense vector
      2. FAISS top-k search → candidate articles
      3. Enrich metadata with relation labels
    """

    def __init__(
        self,
        faiss_store: Optional[FAISSStore] = None,
        embedder: Optional[ArticleEmbedder] = None,
    ) -> None:
        self.faiss_store = faiss_store or FAISSStore()
        self.embedder = embedder or ArticleEmbedder()

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        query_label: Optional[str] = None,
        min_score: float = 0.5,
    ) -> List[dict]:
        """
        Find the top-k most semantically similar articles.

        Args:
            query_text: Article text to search against.
            top_k: Number of results.
            query_label: Predicted label of the query article (for relation tagging).
            min_score: Minimum cosine similarity to include in results.

        Returns:
            List of evidence dicts, each with keys:
            {faiss_id, article_id, title, text_snippet, source_url,
             score, relation}
        """
        logger.info(
            "Evidence retrieval",
            text_len=len(query_text),
            top_k=top_k,
            query_label=query_label,
        )

        # Embed query
        query_vec = self.embedder.embed_single(query_text)

        # FAISS search
        raw_results = self.faiss_store.search(
            query_vec, top_k=top_k, min_score=min_score
        )

        if not raw_results:
            logger.info("No evidence found above similarity threshold")
            return []

        # Tag with relation (requires article labels stored in metadata)
        enriched = []
        for r in raw_results:
            evidence_label = r.get("label")  # populated when articles are indexed
            relation = _infer_relation(query_label, evidence_label)
            enriched.append({**r, "relation": relation})

        logger.info(
            "Evidence retrieved",
            n_results=len(enriched),
            top_score=enriched[0]["score"] if enriched else 0,
        )
        return enriched

    def index_article(
        self,
        article_id: str,
        text: str,
        title: Optional[str] = None,
        source_url: Optional[str] = None,
        source_domain: Optional[str] = None,
        label: Optional[str] = None,
    ) -> int:
        """
        Embed and index a new article.

        Returns:
            Assigned FAISS id.
        """
        embedding = self.embedder.embed_single(text)
        import numpy as np
        faiss_ids = self.faiss_store.add(
            embeddings=embedding.reshape(1, -1),
            article_metas=[{
                "article_id": article_id,
                "title": title or "",
                "text": text,
                "source_url": source_url or "",
                "source_domain": source_domain or "",
                "label": label or "",
            }],
        )
        self.faiss_store.persist()
        return faiss_ids[0]
