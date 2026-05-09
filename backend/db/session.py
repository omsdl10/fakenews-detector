"""
Async SQLAlchemy engine and session factory.
Uses asyncpg driver for PostgreSQL.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,         # SQL query logging in debug mode
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,          # validate connections before use
    pool_recycle=3600,           # recycle stale connections after 1h
)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,      # keep objects usable after commit
    autoflush=False,
)


async def init_db() -> None:
    """Create all tables on startup (use Alembic migrations in production)."""
    from backend.db.models import Base  # local import to avoid circular deps

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context-manager style session for use outside of FastAPI DI."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session; roll back on error, commit on success."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
