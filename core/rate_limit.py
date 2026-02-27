"""
Rate Limiter
============
Multi-tier rate limiting using slowapi (built on limits library).

Tiers (requests per minute unless noted):
  - Global IP:       100/min  — all endpoints, unauthenticated
  - Authenticated:    60/min  — per user ID
  - Auth endpoints:   10/min  — /auth/login, /auth/register  (brute force protection)
  - AI Chat:          20/min  — /api/chat  (expensive, LLM costs real money)
  - Market data:      60/min  — /api/quick-analyze, /api/market-overview

Storage backend:
  - Redis (if REDIS_URL set) — shared across multiple workers/pods, persistent
  - In-memory             — dev/single-worker fallback, resets on restart

Usage:
    from core.rate_limit import limiter, RateLimits

    @app.post("/api/chat")
    @limiter.limit(RateLimits.AI_CHAT)
    async def chat(request: Request, ...):
        ...

    # Per-user limiting (requires auth):
    @app.post("/api/chat")
    @limiter.limit(RateLimits.AI_CHAT, key_func=user_key)
    async def chat(request: Request, user = Depends(get_current_user)):
        ...
"""

import logging
import os
from typing import Callable, Optional

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KEY FUNCTIONS  (who gets their own bucket?)
# ─────────────────────────────────────────────────────────────────────────────

def ip_key(request: Request) -> str:
    """
    Rate limit key: client IP address.
    Handles X-Forwarded-For for requests behind a reverse proxy/load balancer.

    ⚠️  In production behind a load balancer, make sure your proxy is trusted.
        Set TRUSTED_PROXY_IPS env var to your LB's IP range so spoofing is prevented.
    """
    # Check for forwarded IP (behind nginx/AWS ALB)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list — take the first (original client)
        client_ip = forwarded_for.split(",")[0].strip()
        return client_ip
    return get_remote_address(request)


def user_key(request: Request) -> str:
    """
    Rate limit key: authenticated user ID from the JWT.
    Falls back to IP if the user isn't authenticated (for public endpoints).

    This is attached to request.state.user_id by the auth middleware.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    return f"ip:{ip_key(request)}"


def auth_endpoint_key(request: Request) -> str:
    """
    Stricter key for auth endpoints — always by IP.
    Prevents an attacker from bypassing brute-force limits by rotating IPs.
    """
    return f"auth:{ip_key(request)}"


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMIT STRINGS
# ─────────────────────────────────────────────────────────────────────────────

class RateLimits:
    """
    Centralized rate limit definitions.
    Format: "N/period" where period is second, minute, hour, day.
    """
    GLOBAL       = "100/minute"    # all endpoints, anonymous
    AUTHENTICATED= "200/minute"    # all endpoints, logged-in users
    AUTH_LOGIN   = "10/minute"     # login — brute force protection
    AUTH_REGISTER= "5/minute"      # register — prevent account farming
    AI_CHAT      = "20/minute"     # LLM calls are expensive
    MARKET_DATA  = "60/minute"     # yfinance / CoinGecko calls


# ─────────────────────────────────────────────────────────────────────────────
# LIMITER SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _build_storage_uri() -> str:
    """
    Select the rate limit storage backend.
    Redis preferred (shared across workers), memory fallback for dev.
    """
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        logger.info(f"[RateLimit] Using Redis storage: {redis_url}")
        return redis_url

    logger.warning(
        "[RateLimit] REDIS_URL not set — using in-memory storage. "
        "This resets on restart and is NOT shared across workers. "
        "Set REDIS_URL in production."
    )
    return "memory://"


limiter = Limiter(
    key_func=ip_key,                   # default key: IP address
    storage_uri=_build_storage_uri(),
    default_limits=[RateLimits.GLOBAL],
    enabled=os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true",
)


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMIT EXCEEDED HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Returns a clean JSON 429 response with Retry-After header.
    Never exposes internal limit details to the client.
    """
    # Log for monitoring — include IP and path
    client = ip_key(request)
    logger.warning(f"[RateLimit] 429 — {client} exceeded limit on {request.url.path}")

    retry_after = getattr(exc, "retry_after", 60)

    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Please slow down.",
            "retry_after_seconds": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(exc.limit.limit) if hasattr(exc, "limit") else "unknown",
        },
    )
