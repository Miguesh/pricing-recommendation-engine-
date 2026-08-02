"""Application configuration loaded exclusively from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pricing_engine.domain.statistical import (
    MAX_STATISTICAL_IDENTITY_LENGTH,
    MAX_STATISTICAL_REQUEST_BYTES,
)


class Settings(BaseSettings):
    """Runtime settings.

    Secrets are never logged. Deployment-specific configuration belongs in the
    environment, while model behavior belongs in versioned training configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRICING_",
        extra="ignore",
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_title: str = "Pricing Recommendation Engine"
    api_version: str = "0.1.0"
    api_key: SecretStr | None = None
    api_key_tenant_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_STATISTICAL_IDENTITY_LENGTH,
    )
    serving_tenant_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_STATISTICAL_IDENTITY_LENGTH,
    )
    trusted_hosts: list[str] = Field(default_factory=list)
    trust_proxy_identity: bool = False
    trusted_tenant_header: str | None = Field(default=None, min_length=1, max_length=100)
    cors_origins: list[AnyHttpUrl] = Field(default_factory=list)
    model_uri: str | None = None
    mlflow_tracking_uri: str = "http://localhost:5000"
    dependency_project_path: Path | None = None
    recommendation_max_candidates: int = Field(default=500, ge=10, le=2_000)
    recommendation_max_concurrency: int = Field(default=4, ge=1, le=64)
    request_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    request_body_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    max_request_body_bytes: int = Field(
        default=MAX_STATISTICAL_REQUEST_BYTES,
        ge=1_024,
        le=MAX_STATISTICAL_REQUEST_BYTES,
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_blank_api_key(cls, value: object) -> object:
        """Treat a blank local `.env` placeholder as an unconfigured secret."""

        if isinstance(value, SecretStr):
            return None if not value.get_secret_value().strip() else value
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "api_key_tenant_id",
        "serving_tenant_id",
        "trusted_tenant_header",
        mode="before",
    )
    @classmethod
    def normalize_optional_identity_value(cls, value: object) -> object:
        """Normalize security identifiers and reject whitespace-only values."""

        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("Security identity values cannot be blank.")
        return normalized

    @field_validator("trusted_tenant_header")
    @classmethod
    def validate_tenant_header_name(cls, value: str | None) -> str | None:
        """Accept only a conventional HTTP field name, never raw header syntax."""

        if value is not None and not all(
            character.isalnum() or character == "-" for character in value
        ):
            raise ValueError("trusted_tenant_header must be a valid HTTP header name.")
        return value

    @field_validator("trusted_hosts")
    @classmethod
    def validate_trusted_hosts(cls, values: list[str]) -> list[str]:
        """Normalize hosts and reject permissive or URL-shaped entries."""

        normalized: list[str] = []
        for value in values:
            host = value.strip().lower()
            if not host:
                raise ValueError("trusted_hosts cannot contain blank entries.")
            if host == "*":
                raise ValueError("trusted_hosts cannot contain the wildcard '*'.")
            if "://" in host or "/" in host or any(character.isspace() for character in host):
                raise ValueError("trusted_hosts entries must be host patterns, not URLs.")
            if host not in normalized:
                normalized.append(host)
        return normalized

    @model_validator(mode="after")
    def validate_security_posture(self) -> Settings:
        """Fail closed when production identity or host controls are incomplete."""

        if (
            self.api_key_tenant_id is not None or self.trust_proxy_identity
        ) and self.api_key is None:
            raise ValueError("Tenant identity binding requires a configured api_key.")
        if self.trust_proxy_identity and self.trusted_tenant_header is None:
            raise ValueError(
                "trusted_tenant_header is required when trust_proxy_identity is enabled."
            )
        if self.trust_proxy_identity and self.api_key_tenant_id is not None:
            raise ValueError(
                "Configure either api_key_tenant_id or trusted proxy identity, not both."
            )
        if self.environment == "production":
            if self.api_key is None:
                raise ValueError("api_key is required in production.")
            if len(self.api_key.get_secret_value()) < 32:
                raise ValueError("api_key must contain at least 32 characters in production.")
            if not self.trusted_hosts:
                raise ValueError("trusted_hosts must be explicitly configured in production.")
            if self.model_uri is not None and not (
                self.model_uri.startswith("models:/")
                and self.model_uri.endswith("@champion")
                and len(self.model_uri.removeprefix("models:/").removesuffix("@champion")) > 0
            ):
                raise ValueError(
                    "When configured, production model_uri must reference a governed "
                    "MLflow @champion alias."
                )
            if self.serving_tenant_id is None:
                raise ValueError("serving_tenant_id is required in production.")
            if self.api_key_tenant_id is None and not self.trust_proxy_identity:
                raise ValueError(
                    "Production requires api_key_tenant_id or explicit trusted proxy identity."
                )
            if (
                self.api_key_tenant_id is not None
                and self.api_key_tenant_id != self.serving_tenant_id
            ):
                raise ValueError("api_key_tenant_id must match serving_tenant_id.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return process settings, loading the optional local dotenv at composition time."""

    return Settings(_env_file=".env", _env_file_encoding="utf-8")
