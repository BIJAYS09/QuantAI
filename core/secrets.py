"""
Secrets Manager Abstraction Layer
===================================
Supports three providers in priority order:
  1. AWS Secrets Manager   (production)
  2. HashiCorp Vault       (on-prem / self-hosted production)
  3. Environment Variables (local development only)

Usage:
    from core.secrets import secrets
    api_key = secrets.get("OPENAI_API_KEY")

The provider is selected via the SECRETS_PROVIDER env var:
  SECRETS_PROVIDER=aws      → AWS Secrets Manager
  SECRETS_PROVIDER=vault    → HashiCorp Vault
  SECRETS_PROVIDER=env      → Local .env (default, dev only)
"""

import os
import json
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class SecretNotFoundError(Exception):
    """Raised when a secret cannot be found in the provider."""
    pass


class SecretProviderError(Exception):
    """Raised when the secrets provider itself fails."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY CACHE  (TTL-based, never persisted to disk)
# ─────────────────────────────────────────────────────────────────────────────

class _SecretCache:
    """
    Thread-safe in-memory cache for secrets.
    Prevents hammering the secrets provider on every request.
    Secrets are evicted after `ttl_seconds` (default: 5 minutes).
    Keys are stored as hashes so secret names aren't in memory as plaintext.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def _key(self, name: str) -> str:
        return hashlib.sha256(name.encode()).hexdigest()

    def get(self, name: str) -> Optional[Any]:
        k = self._key(name)
        if k in self._store:
            value, expires_at = self._store[k]
            if time.monotonic() < expires_at:
                return value
            del self._store[k]
        return None

    def set(self, name: str, value: Any) -> None:
        k = self._key(name)
        self._store[k] = (value, time.monotonic() + self._ttl)

    def invalidate(self, name: str) -> None:
        k = self._key(name)
        self._store.pop(k, None)

    def clear(self) -> None:
        self._store.clear()


# ─────────────────────────────────────────────────────────────────────────────
# BASE PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

class BaseSecretProvider(ABC):
    """All providers implement this interface."""

    @abstractmethod
    def get_secret(self, secret_name: str) -> str:
        """Retrieve a secret value by name. Raises SecretNotFoundError if missing."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Returns True if the provider is reachable and healthy."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 1: AWS SECRETS MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class AWSSecretsManagerProvider(BaseSecretProvider):
    """
    Retrieves secrets from AWS Secrets Manager.

    Required environment variables (set these on your EC2/ECS/Lambda, NOT in .env):
        AWS_REGION              e.g. us-east-1
        AWS_SECRET_NAME         e.g. quantai/production/secrets
                                OR a comma-separated list of secret names

    Authentication (pick ONE — never hardcode credentials):
        Option A — EC2/ECS/Lambda: attach an IAM Role (recommended)
        Option B — Local testing:  set AWS_PROFILE in your shell
        Option C — CI/CD:          set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
                                   as encrypted pipeline secrets

    The secret in AWS should be stored as a JSON object:
        {
          "OPENAI_API_KEY": "sk-...",
          "ANOTHER_SECRET": "value"
        }
    """

    def __init__(self):
        self._region = os.environ.get("AWS_REGION", "us-east-1")
        self._secret_name = os.environ.get("AWS_SECRET_NAME", "quantai/production/secrets")
        self._client = None
        self._secret_cache: Optional[dict] = None  # full JSON blob from AWS

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                from botocore.exceptions import NoCredentialsError, ClientError
                self._client = boto3.client("secretsmanager", region_name=self._region)
            except ImportError:
                raise SecretProviderError(
                    "boto3 is not installed. Run: pip install boto3"
                )
        return self._client

    def _fetch_secret_blob(self) -> dict:
        """Fetch the full secret JSON blob from AWS (cached per process lifetime)."""
        if self._secret_cache is not None:
            return self._secret_cache

        try:
            from botocore.exceptions import ClientError
            client = self._get_client()
            response = client.get_secret_value(SecretId=self._secret_name)

            raw = response.get("SecretString", "{}")
            blob = json.loads(raw)
            self._secret_cache = blob

            logger.info(
                f"[Secrets] Loaded {len(blob)} secrets from AWS Secrets Manager "
                f"(secret: {self._secret_name}, region: {self._region})"
            )
            return blob

        except Exception as e:
            from botocore.exceptions import ClientError
            if isinstance(e, ClientError):
                code = e.response["Error"]["Code"]
                if code == "ResourceNotFoundException":
                    raise SecretNotFoundError(
                        f"Secret '{self._secret_name}' not found in AWS "
                        f"Secrets Manager (region: {self._region})"
                    )
                elif code == "AccessDeniedException":
                    raise SecretProviderError(
                        f"Access denied to secret '{self._secret_name}'. "
                        "Check your IAM role permissions (see deploy/aws/iam-policy.json)."
                    )
            raise SecretProviderError(f"AWS Secrets Manager error: {e}")

    def get_secret(self, secret_name: str) -> str:
        blob = self._fetch_secret_blob()
        if secret_name not in blob:
            raise SecretNotFoundError(
                f"Key '{secret_name}' not found in AWS secret '{self._secret_name}'"
            )
        return str(blob[secret_name])

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.describe_secret(SecretId=self._secret_name)
            return True
        except Exception as e:
            logger.warning(f"[Secrets] AWS health check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        return f"aws-secrets-manager:{self._region}/{self._secret_name}"


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 2: HASHICORP VAULT
# ─────────────────────────────────────────────────────────────────────────────

class VaultProvider(BaseSecretProvider):
    """
    Retrieves secrets from HashiCorp Vault (KV v2 secrets engine).

    Required environment variables:
        VAULT_ADDR          e.g. https://vault.internal.mycompany.com:8200
        VAULT_TOKEN         Vault token (use AppRole or Kubernetes auth in prod)
        VAULT_SECRET_PATH   e.g. secret/data/quantai/production

    In Vault, store secrets as:
        vault kv put secret/quantai/production \
            OPENAI_API_KEY="sk-..." \
            ANOTHER_SECRET="value"
    """

    def __init__(self):
        self._addr = os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
        self._token = os.environ.get("VAULT_TOKEN", "")
        self._path = os.environ.get("VAULT_SECRET_PATH", "secret/data/quantai/production")
        self._secret_cache: Optional[dict] = None

    def _fetch_secret_blob(self) -> dict:
        if self._secret_cache is not None:
            return self._secret_cache

        if not self._token:
            raise SecretProviderError(
                "VAULT_TOKEN is not set. Set it as an environment variable "
                "or use AppRole/Kubernetes auth."
            )

        try:
            import requests as req
            url = f"{self._addr}/v1/{self._path}"
            response = req.get(
                url,
                headers={"X-Vault-Token": self._token},
                timeout=5,
            )

            if response.status_code == 403:
                raise SecretProviderError(
                    f"Vault access denied at path '{self._path}'. "
                    "Check token policies."
                )
            if response.status_code == 404:
                raise SecretNotFoundError(f"Vault path '{self._path}' not found.")

            response.raise_for_status()
            data = response.json()

            # KV v2 nests data under data.data
            blob = data.get("data", {}).get("data", data.get("data", {}))
            self._secret_cache = blob

            logger.info(
                f"[Secrets] Loaded {len(blob)} secrets from HashiCorp Vault "
                f"(path: {self._path})"
            )
            return blob

        except (SecretNotFoundError, SecretProviderError):
            raise
        except Exception as e:
            raise SecretProviderError(f"Vault connection error: {e}")

    def get_secret(self, secret_name: str) -> str:
        blob = self._fetch_secret_blob()
        if secret_name not in blob:
            raise SecretNotFoundError(
                f"Key '{secret_name}' not found in Vault path '{self._path}'"
            )
        return str(blob[secret_name])

    def health_check(self) -> bool:
        try:
            import requests as req
            url = f"{self._addr}/v1/sys/health"
            response = req.get(url, timeout=3)
            return response.status_code in (200, 429, 472, 473)
        except Exception as e:
            logger.warning(f"[Secrets] Vault health check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        return f"vault:{self._addr}/{self._path}"


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER 3: ENVIRONMENT VARIABLES (local dev fallback only)
# ─────────────────────────────────────────────────────────────────────────────

class EnvProvider(BaseSecretProvider):
    """
    Reads secrets from environment variables / .env file.

    ⚠️  FOR LOCAL DEVELOPMENT ONLY.
    Never use this provider in production — .env files are not secret storage.
    """

    def __init__(self):
        # Load .env if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.debug("[Secrets] Loaded .env file (dev mode)")
        except ImportError:
            pass

        env = os.environ.get("APP_ENV", "development")
        if env in ("production", "staging", "prod"):
            logger.warning(
                "⚠️  WARNING: Using EnvProvider in a non-development environment! "
                "Set SECRETS_PROVIDER=aws or SECRETS_PROVIDER=vault in production."
            )

    def get_secret(self, secret_name: str) -> str:
        value = os.environ.get(secret_name)
        if not value:
            raise SecretNotFoundError(
                f"Secret '{secret_name}' not found in environment variables. "
                f"Add it to your .env file (dev) or set SECRETS_PROVIDER=aws/vault (prod)."
            )
        return value

    def health_check(self) -> bool:
        return True  # env vars are always available

    @property
    def provider_name(self) -> str:
        return "env-variables (⚠️ dev only)"


# ─────────────────────────────────────────────────────────────────────────────
# SECRETS MANAGER  (main interface used by the app)
# ─────────────────────────────────────────────────────────────────────────────

class SecretsManager:
    """
    Single access point for all secrets in the application.

    Selects the provider based on SECRETS_PROVIDER env var:
        aws   → AWSSecretsManagerProvider
        vault → VaultProvider
        env   → EnvProvider (default, dev only)

    Example:
        from core.secrets import secrets

        openai_key = secrets.get("OPENAI_API_KEY")
        db_url     = secrets.get("DATABASE_URL")
    """

    def __init__(self):
        self._provider: Optional[BaseSecretProvider] = None
        self._cache = _SecretCache(ttl_seconds=300)  # 5-minute TTL
        self._initialized = False
        self._audit_log: list[dict] = []  # in-memory audit trail

    def _build_provider(self) -> BaseSecretProvider:
        provider_name = os.environ.get("SECRETS_PROVIDER", "env").lower().strip()

        providers = {
            "aws": AWSSecretsManagerProvider,
            "aws-secrets-manager": AWSSecretsManagerProvider,
            "vault": VaultProvider,
            "hashicorp-vault": VaultProvider,
            "env": EnvProvider,
        }

        if provider_name not in providers:
            raise SecretProviderError(
                f"Unknown SECRETS_PROVIDER='{provider_name}'. "
                f"Valid options: {', '.join(providers.keys())}"
            )

        provider_cls = providers[provider_name]
        provider = provider_cls()

        logger.info(f"[Secrets] Provider initialized: {provider.provider_name}")
        return provider

    def initialize(self) -> None:
        """
        Initialize the provider and validate that required secrets exist.
        Call this once at app startup (in FastAPI lifespan).
        Fails fast if any required secret is missing.
        """
        if self._initialized:
            return

        self._provider = self._build_provider()

        # Validate required secrets exist at startup
        required = [
            "OPENAI_API_KEY",
            "JWT_SECRET_KEY",
        ]

        missing = []
        for name in required:
            try:
                self.get(name)
            except SecretNotFoundError:
                missing.append(name)

        if missing:
            raise SecretProviderError(
                f"[Secrets] STARTUP FAILED — missing required secrets: {missing}\n"
                f"Provider: {self._provider.provider_name}"
            )

        self._initialized = True
        logger.info(
            f"[Secrets] Startup validation passed. "
            f"All required secrets present via {self._provider.provider_name}"
        )

    def get(self, secret_name: str, default: Optional[str] = None) -> str:
        """
        Retrieve a secret by name.

        Args:
            secret_name: The secret key to look up.
            default:     If provided, return this instead of raising on missing.

        Returns:
            The secret value as a string.

        Raises:
            SecretNotFoundError: If secret is missing and no default given.
            SecretProviderError: If the provider is unavailable.
        """
        if self._provider is None:
            self._provider = self._build_provider()

        # Check cache first
        cached = self._cache.get(secret_name)
        if cached is not None:
            self._audit(secret_name, "cache_hit")
            return cached

        try:
            value = self._provider.get_secret(secret_name)
            self._cache.set(secret_name, value)
            self._audit(secret_name, "fetched")
            return value

        except SecretNotFoundError:
            if default is not None:
                self._audit(secret_name, "default_used")
                return default
            self._audit(secret_name, "not_found", error=True)
            raise

        except SecretProviderError as e:
            self._audit(secret_name, "provider_error", error=True)
            if default is not None:
                logger.error(f"[Secrets] Provider error for '{secret_name}', using default: {e}")
                return default
            raise

    def rotate(self, secret_name: str) -> str:
        """
        Force-refresh a secret from the provider (bypass cache).
        Use after rotating credentials.
        """
        self._cache.invalidate(secret_name)
        logger.info(f"[Secrets] Cache invalidated for '{secret_name}', fetching fresh value.")
        return self.get(secret_name)

    def health_check(self) -> dict:
        """Returns provider health status. Exposed via /health endpoint."""
        if self._provider is None:
            return {"status": "uninitialized", "provider": "none"}

        is_healthy = self._provider.health_check()
        return {
            "status": "healthy" if is_healthy else "degraded",
            "provider": self._provider.provider_name,
            "cache_entries": len(self._cache._store),
        }

    def _audit(self, secret_name: str, event: str, error: bool = False) -> None:
        """
        Write an audit log entry.
        Note: logs the SECRET NAME but NEVER the value.
        In production, ship these to CloudWatch / Datadog / your SIEM.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "secret": secret_name,   # name only — never the value
            "provider": self._provider.provider_name if self._provider else "none",
            "error": error,
        }
        self._audit_log.append(entry)

        if error:
            logger.error(f"[Secrets][AUDIT] {entry}")
        else:
            logger.debug(f"[Secrets][AUDIT] {entry}")

    def get_audit_log(self) -> list[dict]:
        """Returns the in-memory audit log (last 1000 entries)."""
        return self._audit_log[-1000:]


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON — import this everywhere in the app
# ─────────────────────────────────────────────────────────────────────────────

secrets = SecretsManager()