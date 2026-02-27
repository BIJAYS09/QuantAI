"""
Application Configuration
===========================
Single source of truth for all config values.
All secrets are pulled from the SecretsManager — never from os.environ directly.

Usage:
    from core.config import settings
    llm = ChatOpenAI(api_key=settings.openai_api_key)
"""

import os
import logging
from functools import cached_property
from core.secrets import secrets, SecretNotFoundError

logger = logging.getLogger(__name__)


class Settings:
    """
    Centralized app settings.
    Secrets are lazy-loaded on first access and cached for the process lifetime.
    Non-secret config (timeouts, feature flags, etc.) reads from env directly.
    """

    # ── Environment ──────────────────────────────────────────────────────────
    @property
    def app_env(self) -> str:
        return os.environ.get("APP_ENV", "development")

    @property
    def is_production(self) -> bool:
        return self.app_env in ("production", "prod")

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def debug(self) -> bool:
        return os.environ.get("DEBUG", "false").lower() == "true" and not self.is_production

    # ── API keys (always via secrets manager) ────────────────────────────────
    @cached_property
    def openai_api_key(self) -> str:
        return secrets.get("OPENAI_API_KEY")

    @cached_property
    def openai_model(self) -> str:
        # Model name is not a secret — config only
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # ── Optional third-party keys (non-fatal if missing) ─────────────────────
    @cached_property
    def news_api_key(self) -> str | None:
        try:
            return secrets.get("NEWS_API_KEY")
        except SecretNotFoundError:
            logger.info("[Config] NEWS_API_KEY not set — news features disabled")
            return None

    @cached_property
    def finnhub_api_key(self) -> str | None:
        try:
            return secrets.get("FINNHUB_API_KEY")
        except SecretNotFoundError:
            return None

    # ── Database (secret) ────────────────────────────────────────────────────
    @cached_property
    def database_url(self) -> str | None:
        try:
            return secrets.get("DATABASE_URL")
        except SecretNotFoundError:
            logger.info("[Config] DATABASE_URL not set — using in-memory only")
            return None

    # ── Redis (non-secret, just host config) ─────────────────────────────────
    @property
    def redis_url(self) -> str:
        return os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    @property
    def cache_ttl_stock(self) -> int:
        """Seconds to cache stock data."""
        return int(os.environ.get("CACHE_TTL_STOCK", "300"))  # 5 min

    @property
    def cache_ttl_crypto(self) -> int:
        """Seconds to cache crypto data."""
        return int(os.environ.get("CACHE_TTL_CRYPTO", "120"))  # 2 min

    # ── CORS ─────────────────────────────────────────────────────────────────
    @property
    def allowed_origins(self) -> list[str]:
        raw = os.environ.get("ALLOWED_ORIGINS", "*")
        if raw == "*":
            if self.is_production:
                logger.warning(
                    "[Config] ALLOWED_ORIGINS=* in production — set this to your domain!"
                )
            return ["*"]
        return [o.strip() for o in raw.split(",")]

    # ── Server ────────────────────────────────────────────────────────────────
    @property
    def host(self) -> str:
        return os.environ.get("HOST", "0.0.0.0")

    @property
    def port(self) -> int:
        return int(os.environ.get("PORT", "8000"))

    # ── JWT ───────────────────────────────────────────────────────────────────
    @cached_property
    def jwt_secret_key(self) -> str:
        """
        Secret key for signing JWTs.
        Must be stored in secrets manager (not hardcoded, not in .env in prod).
        Generate a strong key: python -c "import secrets; print(secrets.token_hex(64))"
        """
        return secrets.get("JWT_SECRET_KEY")

    # ── Database path ─────────────────────────────────────────────────────────
    @property
    def db_path(self) -> str:
        return os.environ.get("DB_PATH", "quantai.db")

    # ── Rate limiting ─────────────────────────────────────────────────────────
    @property
    def rate_limit_per_minute(self) -> int:
        return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))

    @property
    def rate_limit_enabled(self) -> bool:
        return os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"

    # ── Summary (safe to log — no secrets) ───────────────────────────────────
    def summary(self) -> dict:
        return {
            "app_env": self.app_env,
            "debug": self.debug,
            "openai_model": self.openai_model,
            "secrets_provider": os.environ.get("SECRETS_PROVIDER", "env"),
            "allowed_origins": self.allowed_origins,
            "rate_limit_enabled": self.rate_limit_enabled,
            "features": {
                "news_api": self.news_api_key is not None,
                "finnhub": self.finnhub_api_key is not None,
                "database": self.database_url is not None,
            },
        }


# Singleton
settings = Settings()