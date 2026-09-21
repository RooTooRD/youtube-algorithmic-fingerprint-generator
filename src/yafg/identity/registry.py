"""Persistence boundary for manually provisioned accounts."""

from __future__ import annotations

import builtins
import datetime as dt
from pathlib import Path
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from yafg.identity.schema import Account, AccountStatus, ProxyConfig
from yafg.store.models import AccountRecord


def _to_account(row: AccountRecord) -> Account:
    proxy = ProxyConfig.model_validate(row.proxy_config) if row.proxy_config else None
    geolocation = None
    if row.geolocation is not None:
        geolocation = (float(row.geolocation[0]), float(row.geolocation[1]))
    viewport = (int(row.viewport[0]), int(row.viewport[1]))
    return Account(
        label=row.label,
        persona_id=row.persona_id,
        status=cast(AccountStatus, row.status),
        profile_dir=Path(row.profile_dir),
        locale=row.locale,
        timezone=row.timezone,
        country=row.country,
        geolocation=geolocation,
        viewport=viewport,
        proxy=proxy,
        notes=row.notes,
    )


class AccountRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def list(self) -> builtins.list[Account]:
        async with AsyncSession(self.engine) as session:
            rows = (await session.scalars(select(AccountRecord).order_by(AccountRecord.label))).all()
            return [_to_account(row) for row in rows]

    async def get(self, label: str) -> Account | None:
        async with AsyncSession(self.engine) as session:
            row = await session.get(AccountRecord, label)
            return _to_account(row) if row else None

    async def save(self, account: Account, *, checked: bool = False) -> None:
        async with AsyncSession(self.engine) as session:
            row = await session.get(AccountRecord, account.label)
            payload = {
                "persona_id": account.persona_id,
                "status": account.status,
                "profile_dir": str(account.profile_dir),
                "locale": account.locale,
                "timezone": account.timezone,
                "country": account.country,
                "geolocation": list(account.geolocation) if account.geolocation else None,
                "viewport": list(account.viewport),
                "proxy_config": account.proxy.model_dump(mode="json") if account.proxy else None,
                "notes": account.notes,
            }
            if row is None:
                row = AccountRecord(label=account.label, **payload)
                session.add(row)
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
            if checked:
                row.last_checked_at = dt.datetime.now(dt.UTC)
            await session.commit()

    async def set_status(self, label: str, status: AccountStatus) -> Account:
        async with AsyncSession(self.engine) as session:
            row = await session.get(AccountRecord, label)
            if row is None:
                raise KeyError(label)
            row.status = status
            row.last_checked_at = dt.datetime.now(dt.UTC)
            await session.commit()
            await session.refresh(row)
            return _to_account(row)
