"""Process-wide settings. Env-driven, `.env` supported, no secrets in YAML."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YAFG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    profile_dir: Path = Path("data/profiles")
    config_dir: Path = Path("configs")
    db_url: str = "sqlite+aiosqlite:///data/yafg.db"

    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    youtube_api_key: str | None = Field(default=None, validation_alias="YOUTUBE_API_KEY")
    ollama_base_url: str = "http://localhost:11434"

    headless: bool = True
    # Global kill-switch for platform side effects. Even if an experiment YAML enables
    # likes or subscribes, this must also be true. Defaults to observation-only.
    allow_interactions: bool = False

    log_level: str = "INFO"
    log_json: bool = False

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
