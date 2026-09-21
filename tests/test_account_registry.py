from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("aiosqlite")

from sqlalchemy.ext.asyncio import create_async_engine

from yafg.identity.registry import AccountRegistry
from yafg.identity.schema import Account, ProxyConfig
from yafg.store.database import ensure_schema


@pytest.mark.asyncio
async def test_account_registry_round_trip(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await ensure_schema(engine)
    registry = AccountRegistry(engine)
    account = Account(
        label="amina-01",
        persona_id="amina-22-dz-student",
        status="active",
        profile_dir=tmp_path / "profile",
        locale="fr-FR",
        timezone="Africa/Algiers",
        country="DZ",
        geolocation=(36.7538, 3.0588),
        proxy=ProxyConfig(server="http://proxy.test:8000", username="research", password_env="PROXY_PASS"),
    )
    await registry.save(account, checked=True)

    loaded = await registry.get("amina-01")
    assert loaded == account
    assert [item.label for item in await registry.list()] == ["amina-01"]

    changed = await registry.set_status("amina-01", "logged_out")
    assert changed.status == "logged_out"
    await engine.dispose()
