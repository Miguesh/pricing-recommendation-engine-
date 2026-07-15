"""Application configuration loaded exclusively from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Secrets are never logged. Deployment-specific configuration belongs in the
    environment, while model behavior belongs in versioned training configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRICING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_title: str = "Pricing Recommendation Engine"
    api_version: str = "0.1.0"
    api_key: SecretStr | None = None
    cors_origins: list[AnyHttpUrl] = Field(default_factory=list)
    model_uri: str | None = None
    mlflow_tracking_uri: str = "http://localhost:5000"
    recommendation_max_candidates: int = Field(default=500, ge=10, le=2_000)
    request_timeout_seconds: float = Field(default=5.0, gt=0, le=30)


@lru_cache
def get_settings() -> Settings:
    """Return a singleton settings instance for the process."""

    return Settings()
