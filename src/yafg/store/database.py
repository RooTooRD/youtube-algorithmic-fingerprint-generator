"""Async database helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from yafg.settings import settings
from yafg.store.models import Base


def make_engine(url: str | None = None) -> AsyncEngine:
    database_url = url or settings.db_url
    kwargs: dict[str, object] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        # SQLite's local single-agent path does not benefit from connection probing.
        kwargs["pool_pre_ping"] = False
    return create_async_engine(database_url, **kwargs)


async def ensure_schema(engine: AsyncEngine) -> None:
    """Create missing tables for a fresh local database.

    Existing databases must be upgraded with Alembic. ``create_all`` is intentionally
    retained so a clean clone can still be exercised without a migration bootstrap.
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def session_scope(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.commit()
