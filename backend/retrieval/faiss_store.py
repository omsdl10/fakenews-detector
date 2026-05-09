"""
FAISS-based vector store for article embeddings.

Index type: IndexFlatIP (Inner Product on normalised vectors = cosine similarity)
  - No quantisation: exact nearest-neighbour search
  - Appropriate for < 1M articles; switch to IndexIVFFlat for larger corpora

Metadata is stored alongside the FAISS index in a JSON file:
  {faiss_id (int) → {article_id, title, text_snippet, source_url}}

Thread safety: FAISS C++ library is not thread-safe for writes.
Use a threading.Lock() when building the index concurrently.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class FAISSStore:
    """
    Manages a persistent FAISS index + associated metadata.

    On startup: load index and metadata from disk if they exist.
    At ingestion: add embeddings + metadata, persist immediately.
    At search: return top-k article metadata sorted by cosine similarity.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        embedding_dim: int = 768,
    ) -> None:
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.metadata_path = Path(metadata_path or settings.FAISS_METADATA_PATH)
        self.embedding_dim = embedding_dim
        self._lock = threading.Lock()

        self.index: faiss.Index
        self.metadata: Dict[int, dict]  # {faiss_id: {...}}

        self._load_or_create()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            logger.info("Loading FAISS index", path=str(self.index_path))
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r") as f:
                raw = json.load(f)
            # JSON keys are strings; cast back to int
            self.metadata = {int(k): v for k, v in raw.items()}
            logger.info(
                "FAISS index loaded",
                n_vectors=self.index.ntotal,
                n_metadata=len(self.metadata),
            )
        else:
            logger.info(
                "Creating new FAISS index",
                dim=self.embedding_dim,
            )
            # IndexFlatIP: exact search over inner products (cosine on normalised vecs)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = {}

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray,
        article_metas: List[Dict],
    ) -> List[int]:
        """
        Add a batch of embeddings to the index.

        Args:
            embeddings: float32 array (n, embedding_dim). Must be L2-normalised.
            article_metas: list of dicts with at least keys:
                {article_id, title, text_snippet, source_url}

        Returns:
            list of assigned FAISS ids (sequential integers).
        """
        assert embeddings.ndim == 2, "Embeddings must be 2-D"
        assert embeddings.shape[0] == len(article_metas), "Mismatch: embeddings vs metas"
        assert embeddings.dtype == np.float32, "FAISS requires float32"

        with self._lock:
            start_id = self.index.ntotal
            self.index.add(embeddings)

            assigned_ids = list(range(start_id, start_id + len(article_metas)))
            for faiss_id, meta in zip(assigned_ids, article_metas):
                self.metadata[faiss_id] = {
                    "article_id": str(meta.get("article_id", "")),
                    "title": meta.get("title", ""),
                    "text_snippet": (meta.get("text", "") or "")[:300],
                    "source_url": meta.get("source_url", ""),
                    "source_domain": meta.get("source_domain", ""),
                }

        logger.info(
            "Added vectors to FAISS",
            n_added=len(article_metas),
            total=self.index.ntotal,
        )
        return assigned_ids

    def persist(self) -> None:
        """Write index and metadata to disk atomically."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)
        logger.info(
            "FAISS index persisted",
            path=str(self.index_path),
            n_vectors=self.index.ntotal,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> List[Dict]:
        """
        Find top-k most similar articles.

        Args:
            query_embedding: float32 array (dim,) or (1, dim). Must be L2-normalised.
            top_k: number of results to return.
            min_score: minimum cosine similarity threshold (0–1).

        Returns:
            List of dicts: {faiss_id, article_id, title, text_snippet, score}
            sorted by score descending.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty — no evidence available")
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            score = float(dist)  # Inner product on normalised = cosine similarity
            if score < min_score:
                continue
            meta = self.metadata.get(int(idx), {})
            results.append({
                "faiss_id": int(idx),
                "score": round(score, 4),
                **meta,
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self.index.ntotal

    def delete_by_faiss_id(self, faiss_id: int) -> bool:
        """
        Remove a single vector.
        Note: IndexFlatIP doesn't support direct removal — rebuild the index
        in production or use IndexIDMap for O(1) deletes.
        """
        # Remove metadata
        removed = self.metadata.pop(faiss_id, None)
        if removed is None:
            return False
        logger.warning(
            "Vector metadata removed but FAISS flat index cannot delete vectors. "
            "Rebuild the index for full removal.",
            faiss_id=faiss_id,
        )
        return True
