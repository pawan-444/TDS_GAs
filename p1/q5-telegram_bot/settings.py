"""Application configuration."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ai_pipe_token: SecretStr
    telegram_bot_token: SecretStr
    public_url: HttpUrl
    model: str = "openai/gpt-4.1-nano"
    log_directory: Path = Field(default=Path("logs"))
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_download_bytes: int = Field(default=50_000_000, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_settings() -> Settings:
    try:
        settings = get_settings()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid configuration: {exc}") from exc
    settings.log_directory.mkdir(parents=True, exist_ok=True)
    return settings
