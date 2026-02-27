"""
Auth Utilities
==============
JWT access + refresh tokens, bcrypt password hashing,
FastAPI dependency injection for protected routes.

Token strategy:
  - Access token:  short-lived (15 min), stateless JWT
  - Refresh token: long-lived (7 days), stored in DB so it can be revoked
  - Rotation:      each /auth/refresh issues a new refresh token and
                   invalidates the old one (prevents replay attacks)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

# FastAPI bearer scheme — reads the Authorization: Bearer <token> header
_bearer = HTTPBearer(auto_error=False)

# ─────────────────────────────────────────────────────────────────────────────
# PASSWORD HASHING
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """
    Hash a plaintext password with bcrypt (work factor 12).
    The salt is embedded in the returned string — no need to store it separately.
    """
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash. Timing-safe."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN CREATION
# ─────────────────────────────────────────────────────────────────────────────

def _make_token(
    subject: str,
    token_type: str,
    expires_delta: timedelta,
    extra_claims: Optional[dict] = None,
) -> tuple[str, datetime]:
    """
    Internal helper — build and sign a JWT.
    Returns (encoded_token, expiry_datetime).
    """
    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    payload = {
        "sub": subject,           # user ID (string)
        "type": token_type,       # "access" | "refresh"
        "iat": now,               # issued at
        "exp": expire,            # expiry
        "jti": str(uuid4()),      # unique token ID — for revocation
    }
    if extra_claims:
        payload.update(extra_claims)

    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)
    return token, expire


def create_access_token(user_id: str, email: str, role: str = "user") -> tuple[str, datetime]:
    """
    Create a short-lived JWT access token.
    Returns (token_string, expiry_datetime).
    """
    return _make_token(
        subject=user_id,
        token_type=TOKEN_TYPE_ACCESS,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        extra_claims={"email": email, "role": role},
    )


def create_refresh_token(user_id: str) -> tuple[str, datetime]:
    """
    Create a long-lived refresh token.
    This MUST be stored in the database so it can be revoked.
    Returns (token_string, expiry_datetime).
    """
    return _make_token(
        subject=user_id,
        token_type=TOKEN_TYPE_REFRESH,
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT. Returns the payload dict.
    Raises HTTPException 401 on any failure.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"[Auth] Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def decode_access_token(token: str) -> dict:
    """Decode and validate an access token specifically."""
    payload = decode_token(token)
    if payload.get("type") != TOKEN_TYPE_ACCESS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expected an access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def decode_refresh_token(token: str) -> dict:
    """Decode and validate a refresh token specifically."""
    payload = decode_token(token)
    if payload.get("type") != TOKEN_TYPE_REFRESH:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expected a refresh token.",
        )
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

class CurrentUser:
    """Represents the authenticated user extracted from the JWT."""
    def __init__(self, user_id: str, email: str, role: str):
        self.user_id = user_id
        self.email = email
        self.role = role

    def __repr__(self):
        return f"<CurrentUser id={self.user_id} email={self.email} role={self.role}>"


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> CurrentUser:
    """
    FastAPI dependency — extract and validate the Bearer token.

    Usage:
        @app.get("/protected")
        async def protected(user: CurrentUser = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)

    user_id = payload.get("sub")
    email = payload.get("email", "")
    role = payload.get("role", "user")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is malformed.",
        )

    return CurrentUser(user_id=user_id, email=email, role=role)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[CurrentUser]:
    """
    Like get_current_user but returns None instead of raising if no token.
    Use for endpoints that work both authenticated and unauthenticated,
    but give extra features when authenticated.
    """
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.

    Usage:
        @app.delete("/admin/user/{id}")
        async def delete_user(user: CurrentUser = Depends(require_role("admin"))):
            ...
    """
    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role != required_role and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the '{required_role}' role.",
            )
        return user
    return _check
