"""Async database helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from yafg.settings import settings
from yafg.store.models import Base


def make_engine(url: str | None = None) -> AsyncEngine:
    return create_async_engine(url or settings.db_url)


async def ensure_schema(engine: AsyncEngine) -> None:
    """Create missing tables for local/P1 use.

    Production/history upgrades belong to Alembic; this bootstrap only makes a fresh
    clone usable without a separate migration command.
    """
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def session_scope(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.commit()
