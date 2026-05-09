# FakeNews Detector 🔍

An explainable, production-grade fake news detection system powered by RoBERTa,
FAISS, and FastAPI. Classifies articles as **fake / real / uncertain**, retrieves
semantically similar evidence articles, and explains *why* a prediction was made
using attention weights or SHAP values.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLIENT LAYER                                                        │
│  React SPA  ─────  URL Scraper  ─────  REST clients (curl/Postman)  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTPS
┌────────────────────────────▼────────────────────────────────────────┐
│  GATEWAY (FastAPI)                                                   │
│  /predict  /upload-article  /search-evidence  /auth/token           │
│  JWT Auth ── Rate Limiting ── Request Logging ── Redis Cache        │
└──────────────┬───────────────────┬───────────────────┬─────────────┘
               │                   │                   │
┌──────────────▼──────┐  ┌────────▼──────────┐  ┌────▼──────────────┐
│  ML CLASSIFIER      │  │  RAG RETRIEVAL    │  │  EXPLAINABILITY   │
│  RoBERTa fine-tuned │  │  FAISS + sentence │  │  Attention weights│
│  fake/real/uncertain│  │  transformers     │  │  + SHAP values    │
└──────────────┬──────┘  └────────┬──────────┘  └────┬──────────────┘
               │                   │                   │
┌──────────────▼───────────────────▼───────────────────▼─────────────┐
│  STORAGE                                                             │
│  PostgreSQL (articles, predictions, users)                           │
│  FAISS Index (768-dim embeddings)                                    │
│  Model Registry (HuggingFace checkpoints)                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **Classification** — fake / real / uncertain with calibrated confidence
- **Credibility score** — 0–100 composite credibility rating
- **Explainability** — token-level importance via attention or SHAP
- **Evidence retrieval** — FAISS semantic search finds supporting/contradicting articles
- **URL scraping** — submit a URL instead of pasting text
- **JWT authentication** — secure upload and management endpoints
- **Redis caching** — repeated predictions return instantly
- **Structured logging** — JSON logs for production observability
- **Rate limiting** — 60 req/min per IP

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 20+ |
| PostgreSQL | 15+ |
| Redis | 7+ |
| Docker + Compose | 24+ (optional) |

---

## Quick Start (Docker — Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your-org/fakenews-detector.git
cd fakenews-detector

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env — at minimum change SECRET_KEY

# 3. Start all services (DB, Redis, backend, frontend)
docker compose up --build

# 4. Open the UI
open http://localhost:3000

# 5. Open API docs
open http://localhost:8000/api/v1/docs
```

---

## Local Development Setup

### Backend

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Copy and configure .env
cp .env.example .env

# Start PostgreSQL and Redis (using Docker for convenience)
docker run -d --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15-alpine
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run the API server
uvicorn backend.api.main:app --reload --port 8000

# API docs available at:
# http://localhost:8000/api/v1/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI available at http://localhost:5173
```

---

## Phase 1: Train the Model

```bash
# Step 1: Download the LIAR dataset
python data/scripts/download_dataset.py --dataset liar --output data/raw/liar

# Step 2: Train RoBERTa classifier
#   Trains for 5 epochs with early stopping
#   Best checkpoint saved to models/fakenews_roberta/
python -m backend.ml.trainer \
    --dataset liar \
    --data_dir data/raw/liar \
    --output_dir models/fakenews_roberta \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5

# Expected results after training (LIAR test set):
#   Accuracy:  ~64-68%
#   Macro F1:  ~62-66%
#   ROC-AUC:   ~0.78-0.82
```

> **GPU Training:** Set `DEVICE=cuda` in `.env` and install `torch` with CUDA support.
> Training time: ~2h on CPU, ~15min on a T4 GPU.

---

## Phase 2: Build the Evidence Index

```bash
# First, upload some articles to populate the database:
# (see API examples below, or use the seed script)
python scripts/seed_db.py

# Then build the FAISS index from all articles in the DB:
python scripts/build_faiss_index.py --batch_size 64
```

---

## API Reference

### Authentication

```bash
# Register a new user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123"}'

# Get access token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123"}'
# → {"access_token": "eyJ...", "token_type": "bearer"}

export TOKEN="eyJ..."
```

### POST /api/v1/predict — Classify an article

```bash
# With text
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Scientists have confirmed that drinking bleach cures all diseases. Multiple studies have shown remarkable results with zero side effects. The government is suppressing this information to protect pharmaceutical profits.",
    "retrieve_evidence": true,
    "deep_explain": false
  }'

# With URL
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/news/article"}'
```

**Response:**
```json
{
  "article_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "prediction_id": "7b8a9c10-...",
  "label": "fake",
  "confidence": 0.9231,
  "credibility_score": 11.2,
  "probabilities": {
    "fake": 0.9231,
    "real": 0.0512,
    "uncertain": 0.0257
  },
  "explanation": {
    "method": "attention",
    "token_importance": [
      {"token": "bleach", "score": 0.94, "char_start": 54, "char_end": 60},
      {"token": "cures", "score": 0.87, "char_start": 61, "char_end": 66},
      {"token": "suppressing", "score": 0.83, "char_start": 178, "char_end": 189}
    ],
    "reasoning_summary": "The article shows characteristics of fake news (confidence: 92%). Key influencing terms: 'bleach', 'cures', 'suppressing', 'pharmaceutical', 'confirmed'."
  },
  "evidence": [
    {
      "faiss_id": 42,
      "article_id": "abc123...",
      "title": "FDA warns against bleach consumption",
      "text_snippet": "Health authorities have repeatedly warned...",
      "source_url": "https://fda.gov/...",
      "score": 0.812,
      "relation": "contradicts"
    }
  ],
  "latency_ms": 284.5
}
```

### POST /api/v1/upload-article — Index an article (requires auth)

```bash
curl -X POST http://localhost:8000/api/v1/upload-article/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "According to peer-reviewed research published in the New England Journal of Medicine, the COVID-19 vaccine has a 95% efficacy rate with mild, transient side effects in the majority of participants.",
    "title": "COVID-19 Vaccine Efficacy Study",
    "source_url": "https://nejm.org/article/example",
    "label_ground_truth": "real"
  }'
```

### POST /api/v1/search-evidence — Semantic search

```bash
curl -X POST http://localhost:8000/api/v1/search-evidence/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Vaccines are dangerous and cause autism in children",
    "top_k": 5,
    "min_score": 0.5
  }'
```

---

## Running Tests

```bash
# Unit + integration tests (with mocked ML dependencies)
pytest tests/ -v --cov=backend --cov-report=term-missing

# Run only fast unit tests
pytest tests/ -v -m "not slow"

# Run with real model (slow — requires trained weights)
pytest tests/ -v -m slow
```

---

## Project Structure

```
fakenews-detector/
├── backend/
│   ├── api/            FastAPI routes, middleware, dependency injection
│   ├── ml/             RoBERTa model, trainer, embedder, explainer
│   ├── retrieval/      FAISS store, semantic search
│   ├── db/             SQLAlchemy models, CRUD, session
│   ├── core/           Config, logging, security
│   ├── scraper/        URL → article text
│   └── schemas/        Pydantic I/O schemas
├── data/
│   ├── raw/            Downloaded datasets (git-ignored)
│   ├── processed/      Tokenised splits (git-ignored)
│   └── scripts/        Download helpers
├── frontend/
│   └── src/
│       ├── components/ ArticleInput, ResultCard, EvidencePanel, ExplanationHighlight
│       ├── hooks/      usePredict
│       └── api/        Axios client
├── models/             Trained checkpoints + FAISS index (git-ignored)
├── scripts/            build_faiss_index.py, seed_db.py
├── tests/              pytest suite
├── docker/             Dockerfiles + nginx.conf
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Production Deployment Checklist

- [ ] Change `SECRET_KEY` to a cryptographically random 32+ char string
- [ ] Set `DEBUG=false` and `LOG_JSON=true`
- [ ] Use a managed PostgreSQL instance (RDS, Cloud SQL, Supabase)
- [ ] Use a managed Redis instance (ElastiCache, Upstash)
- [ ] Mount `./models` as a persistent volume or use S3 for model artefacts
- [ ] Set `DEVICE=cuda` and use a GPU instance for production inference
- [ ] Configure CORS `ALLOWED_ORIGINS` to your production domain only
- [ ] Add HTTPS via a load balancer or Caddy reverse proxy
- [ ] Use Alembic for database migrations instead of `create_all()`
- [ ] Set up log aggregation (Datadog, Grafana Loki)
- [ ] Configure horizontal pod autoscaling (the model is stateless)

---

## Railway Deployment

This repo includes `railway.json` and `docker/Dockerfile.railway` so Railway can build one service that serves both the React frontend and the FastAPI backend.

1. Push the repo to GitHub.
2. In Railway, create a new project and choose **Deploy from GitHub repo**.
3. Select `omsdl10/fakenews-detector`.
4. Add a PostgreSQL service in the same Railway project.
5. In the app service variables, set:

```
DATABASE_URL=${{Postgres.DATABASE_URL}}
SECRET_KEY=<a long random value>
DEBUG=false
LOG_JSON=true
LIGHTWEIGHT_MODE=true
```

Redis is optional. If you add a Redis service, also set `REDIS_URL` to that service URL.

After the deployment succeeds, open the service settings, go to Networking, and generate a public Railway domain. The frontend will be available at that domain, and the backend health check will be at `/api/v1/health`.

---

## Performance Benchmarks

| Scenario | Latency (CPU) | Latency (GPU T4) |
|---|---|---|
| Predict (cached) | ~5ms | ~5ms |
| Predict (attention) | ~280–450ms | ~60–90ms |
| Predict (SHAP deep) | ~8–15s | ~1–3s |
| Evidence search (FAISS 10k) | ~12ms | ~12ms |
| Article embedding | ~120ms | ~25ms |

---

## Licence

MIT
