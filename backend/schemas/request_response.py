"""
Pydantic v2 schemas for all API request/response bodies.
Strict validation with clear error messages.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_loaded: bool
    index_size: int


# ── Auth ──────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Predict ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: Optional[str] = Field(
        default=None,
        min_length=30,
        max_length=10000,
        description="Full article text to classify",
        examples=["Scientists discover that vaccines cause 5G towers to grow..."],
    )
    url: Optional[HttpUrl] = Field(
        default=None,
        description="URL of a news article to scrape and classify",
        examples=["https://example.com/news/article"],
    )
    deep_explain: bool = Field(
        default=False,
        description="Run SHAP for deeper (slower) explanation",
    )
    retrieve_evidence: bool = Field(
        default=True,
        description="Retrieve semantically similar articles as evidence",
    )

    @model_validator(mode="after")
    def require_text_or_url(self) -> "PredictRequest":
        if not self.text and not self.url:
            raise ValueError("Either 'text' or 'url' must be provided")
        return self


class TokenImportance(BaseModel):
    token: str
    score: float = Field(..., ge=0.0, le=1.0)
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class ExplanationResponse(BaseModel):
    method: str  # "attention" | "shap"
    token_importance: List[TokenImportance]
    reasoning_summary: str


class EvidenceItem(BaseModel):
    faiss_id: int
    article_id: str
    title: str
    text_snippet: str
    source_url: Optional[str]
    score: float = Field(..., description="Cosine similarity (0–1)")
    relation: str = Field(..., description="supports | contradicts | inconclusive")


class PredictResponse(BaseModel):
    article_id: UUID
    prediction_id: UUID
    label: str = Field(..., description="fake | real | uncertain")
    confidence: float = Field(..., ge=0.0, le=1.0)
    credibility_score: float = Field(..., ge=0.0, le=100.0)
    probabilities: Dict[str, float]
    explanation: ExplanationResponse
    evidence: List[EvidenceItem]
    latency_ms: float


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadArticleRequest(BaseModel):
    text: str = Field(..., min_length=30, max_length=50000)
    title: Optional[str] = Field(default=None, max_length=512)
    source_url: Optional[HttpUrl] = None
    label_ground_truth: Optional[str] = Field(
        default=None,
        description="Known ground truth label for corpus building",
        pattern="^(fake|real|uncertain)$",
    )


class UploadArticleResponse(BaseModel):
    article_id: UUID
    faiss_id: int
    message: str = "Article indexed successfully"


# ── Evidence search ───────────────────────────────────────────────────────────

class SearchEvidenceRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Article text or claim to search evidence for",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchEvidenceResponse(BaseModel):
    query_length: int
    results: List[EvidenceItem]
    total_found: int


# ── Register ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address")
        return v.lower().strip()


class RegisterResponse(BaseModel):
    user_id: UUID
    email: str
    message: str = "Registration successful"
