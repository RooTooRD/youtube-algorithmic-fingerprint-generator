"""Persistence boundary for manually provisioned accounts."""

from __future__ import annotations

import builtins
import datetime as dt
from pathlib import Path
from typing import cast

from sqlalchemy import select, update
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


class AccountInUse(RuntimeError):
    pass


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
                "profile_dir": str(account.profile_dir.resolve()),
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

    async def lease_owner(self, label: str) -> str | None:
        async with AsyncSession(self.engine) as session:
            return await session.scalar(select(AccountRecord.lease_id).where(AccountRecord.label == label))

    async def acquire_lease(
        self,
        label: str,
        lease_id: str,
        *,
        required_status: AccountStatus | None = None,
    ) -> None:
        """Atomically reserve a persistent profile for one execution attempt."""
        conditions = [AccountRecord.label == label, AccountRecord.lease_id.is_(None)]
        if required_status is not None:
            conditions.append(AccountRecord.status == required_status)
        statement = (
            update(AccountRecord)
            .where(*conditions)
            .values(lease_id=lease_id, lease_acquired_at=dt.datetime.now(dt.UTC))
        )
        async with AsyncSession(self.engine) as session:
            acquired = await session.scalar(statement.returning(AccountRecord.label))
            await session.commit()
            if acquired is not None:
                return
            row = await session.get(AccountRecord, label)
            if row is None:
                raise KeyError(label)
            if row.lease_id is not None:
                raise AccountInUse(f"account {label!r} is already leased by attempt {row.lease_id!r}")
            raise ValueError(f"account {label!r} is {row.status!r}, expected {required_status!r}")

    async def release_lease(self, label: str, lease_id: str) -> None:
        """Release only the matching attempt lease; never clear another process's lease."""
        statement = (
            update(AccountRecord)
            .where(AccountRecord.label == label, AccountRecord.lease_id == lease_id)
            .values(lease_id=None, lease_acquired_at=None)
        )
        async with AsyncSession(self.engine) as session:
            await session.execute(statement)
            await session.commit()
