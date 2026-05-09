"""
Integration + unit tests.

Run with:
    pytest tests/ -v --cov=backend --cov-report=term-missing

Requires a running PostgreSQL and Redis (or use pytest-mock to stub them).
For CI without a DB, set DATABASE_URL to SQLite:
    DATABASE_URL=sqlite+aiosqlite:///./test.db pytest tests/
"""

import json
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_model():
    """Return a model that always predicts 'fake' with 0.9 confidence."""
    model = MagicMock()
    model.predict.return_value = {
        "label": "fake",
        "confidence": 0.9,
        "credibility_score": 8.5,
        "probabilities": {"fake": 0.9, "real": 0.07, "uncertain": 0.03},
        "latency_ms": 42.0,
    }
    model.model = MagicMock()
    model.tokenizer = MagicMock()
    model.device = MagicMock()
    return model


@pytest.fixture(scope="session")
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "faiss_id": 0,
            "article_id": str(uuid.uuid4()),
            "title": "Scientists debunk vaccine claims",
            "text_snippet": "A new study contradicts the claim made in this article.",
            "source_url": "https://example.com/article/1",
            "score": 0.85,
            "relation": "contradicts",
        }
    ]
    retriever.index_article.return_value = 1
    return retriever


@pytest.fixture(scope="session")
def mock_explainer_output():
    return {
        "method": "attention",
        "token_importance": [
            {"token": "misleading", "score": 0.92, "char_start": 0, "char_end": 10},
            {"token": "false", "score": 0.78, "char_start": 11, "char_end": 16},
        ],
        "reasoning_summary": "The article shows characteristics of fake news (confidence: 90%).",
    }


@pytest_asyncio.fixture
async def async_client(mock_model, mock_retriever, mock_explainer_output) -> AsyncGenerator:
    """
    FastAPI test client with mocked ML dependencies.
    Avoids loading real model weights in CI.
    """
    with (
        patch("backend.api.dependencies.get_inference_model", return_value=mock_model),
        patch("backend.api.dependencies.get_retriever", return_value=mock_retriever),
        patch("backend.ml.explainer.build_explanation", return_value=mock_explainer_output),
        patch("backend.api.dependencies.get_cache", return_value=None),
        patch("backend.db.session.init_db", new_callable=AsyncMock),
    ):
        from backend.api.main import create_app
        app = create_app()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(async_client):
    resp = await async_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ── Predict endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_with_text(async_client):
    payload = {
        "text": "Scientists have discovered that the moon is made of cheese. "
                "This has been confirmed by multiple sources with exclusive evidence.",
        "retrieve_evidence": False,
        "deep_explain": False,
    }
    resp = await async_client.post("/api/v1/predict/", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["label"] in ("fake", "real", "uncertain")
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["credibility_score"] <= 100.0
    assert "probabilities" in data
    assert "explanation" in data
    assert "article_id" in data


@pytest.mark.asyncio
async def test_predict_returns_explanation(async_client):
    payload = {
        "text": "A" * 100,  # 100-char dummy text
        "retrieve_evidence": False,
    }
    resp = await async_client.post("/api/v1/predict/", json=payload)
    assert resp.status_code == 200
    expl = resp.json()["explanation"]
    assert "token_importance" in expl
    assert "reasoning_summary" in expl
    assert expl["method"] in ("attention", "shap")


@pytest.mark.asyncio
async def test_predict_requires_text_or_url(async_client):
    resp = await async_client.post("/api/v1/predict/", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_text_too_short(async_client):
    resp = await async_client.post(
        "/api/v1/predict/", json={"text": "Too short"}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_with_evidence(async_client):
    payload = {
        "text": "B" * 100,
        "retrieve_evidence": True,
    }
    resp = await async_client.post("/api/v1/predict/", json=payload)
    assert resp.status_code == 200
    evidence = resp.json()["evidence"]
    assert isinstance(evidence, list)
    if evidence:
        item = evidence[0]
        assert "score" in item
        assert item["relation"] in ("supports", "contradicts", "inconclusive")


# ── Evidence search endpoint ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_evidence(async_client):
    payload = {"query": "Vaccine side effects causing major health problems globally", "top_k": 3}
    resp = await async_client.post("/api/v1/search-evidence/", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "total_found" in data
    assert isinstance(data["results"], list)


@pytest.mark.asyncio
async def test_search_evidence_query_too_short(async_client):
    resp = await async_client.post(
        "/api/v1/search-evidence/", json={"query": "short"}
    )
    assert resp.status_code == 422


# ── Upload endpoint (requires auth) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_requires_auth(async_client):
    payload = {
        "text": "C" * 100,
        "title": "Test Article",
    }
    resp = await async_client.post("/api/v1/upload-article/", json=payload)
    assert resp.status_code == 401


# ── ML unit tests ─────────────────────────────────────────────────────────────

class TestDatasetPreprocessing:
    def test_clean_text_normalises_whitespace(self):
        from backend.ml.dataset import clean_text
        assert clean_text("hello   world\t\n") == "hello world"

    def test_clean_text_removes_html(self):
        from backend.ml.dataset import clean_text
        assert clean_text("<b>Breaking</b> news") == "Breaking news"

    def test_clean_text_truncates(self):
        from backend.ml.dataset import clean_text
        long_text = "a" * 5000
        result = clean_text(long_text, max_chars=100)
        assert len(result) == 100

    def test_liar_label_map_covers_all_labels(self):
        from backend.ml.dataset import LIAR_LABEL_MAP
        expected = {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}
        assert set(LIAR_LABEL_MAP.keys()) == expected

    def test_compute_class_weights_inverse_frequency(self):
        from datasets import Dataset
        from backend.ml.dataset import compute_class_weights
        # Balanced dataset → all weights equal
        ds = Dataset.from_dict({"label": [0, 1, 2, 0, 1, 2]})
        weights = compute_class_weights(ds)
        assert len(weights) == 3
        assert abs(weights[0] - weights[1]) < 0.01  # balanced → equal weights


class TestFAISSStore:
    def test_add_and_search(self, tmp_path):
        from backend.retrieval.faiss_store import FAISSStore
        store = FAISSStore(
            index_path=str(tmp_path / "test.index"),
            metadata_path=str(tmp_path / "meta.json"),
            embedding_dim=8,
        )
        # Add 3 normalised random vectors
        vecs = np.random.randn(3, 8).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        metas = [
            {"article_id": str(i), "title": f"Article {i}", "text": "x", "source_url": ""}
            for i in range(3)
        ]
        ids = store.add(vecs, metas)
        assert len(ids) == 3
        assert store.size == 3

        # Query with the first vector — should find itself with score ≈ 1.0
        results = store.search(vecs[0], top_k=1, min_score=0.0)
        assert len(results) == 1
        assert results[0]["score"] > 0.99

    def test_persist_and_reload(self, tmp_path):
        from backend.retrieval.faiss_store import FAISSStore
        idx_path = str(tmp_path / "p.index")
        meta_path = str(tmp_path / "p.json")

        store = FAISSStore(index_path=idx_path, metadata_path=meta_path, embedding_dim=8)
        vecs = np.random.randn(2, 8).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.add(vecs, [{"article_id": "a", "title": "t", "text": "x", "source_url": ""}] * 2)
        store.persist()

        store2 = FAISSStore(index_path=idx_path, metadata_path=meta_path, embedding_dim=8)
        assert store2.size == 2
        assert 0 in store2.metadata

    def test_search_empty_index(self, tmp_path):
        from backend.retrieval.faiss_store import FAISSStore
        store = FAISSStore(
            index_path=str(tmp_path / "e.index"),
            metadata_path=str(tmp_path / "e.json"),
            embedding_dim=8,
        )
        results = store.search(np.zeros(8, dtype=np.float32), top_k=5)
        assert results == []


class TestExplainability:
    def test_clean_token(self):
        from backend.ml.explainer import _clean_token
        assert _clean_token("Ġhello") == "hello"
        assert _clean_token("world") == "world"

    def test_infer_relation(self):
        from backend.retrieval.search import _infer_relation
        assert _infer_relation("fake", "real") == "contradicts"
        assert _infer_relation("real", "real") == "supports"
        assert _infer_relation("fake", "uncertain") == "inconclusive"
        assert _infer_relation(None, "real") == "inconclusive"


class TestSchemas:
    def test_predict_request_requires_text_or_url(self):
        from pydantic import ValidationError
        from backend.schemas.request_response import PredictRequest
        with pytest.raises(ValidationError):
            PredictRequest()  # neither text nor url

    def test_predict_request_valid_text(self):
        from backend.schemas.request_response import PredictRequest
        req = PredictRequest(text="A" * 50)
        assert req.text == "A" * 50

    def test_register_request_validates_email(self):
        from pydantic import ValidationError
        from backend.schemas.request_response import RegisterRequest
        with pytest.raises(ValidationError):
            RegisterRequest(email="notanemail", password="secure123")
