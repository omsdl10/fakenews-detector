"""
FastAPI application factory.

Startup sequence:
  1. Initialise structured logging
  2. Connect to PostgreSQL and create tables
  3. Load ML model + embedder into memory
  4. Load / create FAISS index
  5. Register all API routers

All heavy I/O happens inside the lifespan context so FastAPI can
report readiness before accepting traffic.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.dependencies import get_embedder, get_faiss_store, get_inference_model
from backend.api.middleware import LoggingMiddleware, RateLimitMiddleware
from backend.api.routes.predict import router as predict_router
from backend.api.routes.upload import evidence_router, upload_router
from backend.core.config import get_settings
from backend.core.logging import get_logger, setup_logging
from backend.db.session import init_db

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ── Auth router (inline — small enough to not warrant its own file) ───────────

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import create_access_token, hash_password, verify_password
from backend.db.crud import create_user, get_user_by_email
from backend.db.session import get_db
from backend.schemas.request_response import (
    RegisterRequest,
    RegisterResponse,
    TokenRequest,
    TokenResponse,
)

auth_router = APIRouter(prefix="/auth", tags=["Auth"])


@auth_router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await get_user_by_email(db, req.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = await create_user(db, req.email, hash_password(req.password))
    return RegisterResponse(user_id=user.id, email=user.email)


@auth_router.post("/token", response_model=TokenResponse)
async def login(req: TokenRequest, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_email(db, req.email)
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all resources on startup; release on shutdown."""
    logger.info("Starting up", app=settings.APP_NAME, version=settings.APP_VERSION)

    try:
        await init_db()
    except Exception as exc:
        if not settings.LIGHTWEIGHT_MODE:
            raise
        logger.warning(
            "Database unavailable in lightweight mode; continuing without DB",
            error=str(exc),
        )

    # ML model (loads weights from disk or uses lightweight fallback mode)
    model = get_inference_model()
    logger.info("Classifier loaded", device=settings.DEVICE)

    store = None
    if not settings.LIGHTWEIGHT_MODE:
        # Embedder
        embedder = get_embedder()
        logger.info("Embedder loaded")

        # FAISS
        store = get_faiss_store()
        logger.info("FAISS index ready", n_vectors=store.size)
    else:
        logger.info("Lightweight mode enabled; skipping embedder and FAISS startup")

    yield  # ← application serves requests here

    logger.info("Shutting down — persisting FAISS index")
    if store is not None:
        store.persist()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Explainable fake news detection API. "
            "Classifies articles, retrieves evidence, and explains predictions."
        ),
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost first) ───────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

    # ── Routers ───────────────────────────────────────────────────────────
    prefix = settings.API_PREFIX
    app.include_router(auth_router, prefix=prefix)
    app.include_router(predict_router, prefix=prefix)
    app.include_router(upload_router, prefix=prefix)
    app.include_router(evidence_router, prefix=prefix)

    # ── Health check ──────────────────────────────────────────────────────
    @app.get(f"{prefix}/health", tags=["Meta"])
    async def health():
        from backend.api.dependencies import _faiss_store, _inference_model
        return {
            "status": "ok",
            "version": settings.APP_VERSION,
            "model_loaded": _inference_model is not None,
            "lightweight_mode": settings.LIGHTWEIGHT_MODE,
            "index_size": _faiss_store.size if _faiss_store else 0,
        }

    # ── Global error handlers ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error. Please try again later."},
        )

    # ── Production SPA hosting ────────────────────────────────────────────
    static_dir_candidates = [
        Path("frontend/dist"),
        Path("/app/frontend/dist"),
    ]
    static_dir = next((p for p in static_dir_candidates if p.exists()), None)
    if static_dir:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


app = create_app()
