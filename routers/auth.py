"""
Auth Router
===========
Endpoints:
  POST /auth/register  — create a new account
  POST /auth/login     — get access + refresh token pair
  POST /auth/refresh   — rotate refresh token, get new access token
  POST /auth/logout    — revoke the current refresh token
  POST /auth/logout-all — revoke ALL sessions (force re-login everywhere)
  GET  /auth/me        — get the current user's profile

Token flow:
  1. Client calls /auth/login → gets {access_token, refresh_token}
  2. Client stores both (access in memory, refresh in httpOnly cookie or secure storage)
  3. Client sends: Authorization: Bearer <access_token> on protected requests
  4. When access token expires → call /auth/refresh with refresh_token
  5. /auth/refresh returns a NEW pair and invalidates the old refresh token
  6. On logout → call /auth/logout → old refresh token is revoked
"""

import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, field_validator

from core.auth import (
    CurrentUser,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_current_user,
    hash_password,
    verify_password,
)
from core.database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    is_refresh_token_valid,
    revoke_all_user_tokens,
    revoke_refresh_token,
    store_refresh_token,
    update_last_login,
)
from core.rate_limit import RateLimits, limiter, auth_endpoint_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(v) > 32:
            raise ValueError("Username must be 32 characters or fewer.")
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens.")
        return v

    @field_validator("password")
    @classmethod
    def password_strong(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one number.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes in seconds


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    role: str
    created_at: str
    last_login: str | None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def _issue_token_pair(user_id: str, email: str, role: str) -> TokenResponse:
    """Create and persist a new access + refresh token pair."""
    access_token, _access_exp = create_access_token(user_id, email, role)
    refresh_token, refresh_exp = create_refresh_token(user_id)

    # Decode to get the JTI so we can store it for revocation
    from core.auth import decode_refresh_token as _decode
    rt_payload = _decode(refresh_token)
    jti = rt_payload["jti"]

    await store_refresh_token(jti=jti, user_id=user_id, expires_at=refresh_exp)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(RateLimits.AUTH_REGISTER, key_func=auth_endpoint_key)
async def register(request: Request, body: RegisterRequest):
    """
    Create a new user account and return a token pair.
    Rate limited to 5 requests/minute per IP to prevent account farming.
    """
    password_hash = hash_password(body.password)

    try:
        user = await create_user(
            email=body.email,
            username=body.username,
            password_hash=password_hash,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    logger.info(f"[Auth] New registration: {user['id']} ({body.email})")
    return await _issue_token_pair(user["id"], user["email"], user["role"])


@router.post("/login", response_model=TokenResponse)
@limiter.limit(RateLimits.AUTH_LOGIN, key_func=auth_endpoint_key)
async def login(request: Request, body: LoginRequest):
    """
    Authenticate with email + password. Returns a token pair.
    Rate limited to 10 requests/minute per IP — brute force protection.

    Note: The error message is intentionally generic ("invalid credentials")
    regardless of whether the email doesn't exist or the password is wrong.
    This prevents user enumeration attacks.
    """
    user = await get_user_by_email(body.email)

    # Constant-time check — always run verify_password even if user not found
    # to prevent timing-based user enumeration
    dummy_hash = "$2b$12$invalidhashfortimingprotectionxxxxxxxxxxxxxxxxxxxxxxxx"
    password_hash = user["password_hash"] if user else dummy_hash
    password_ok = verify_password(body.password, password_hash)

    if not user or not password_ok:
        logger.warning(f"[Auth] Failed login attempt for email: {body.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    await update_last_login(user["id"])
    logger.info(f"[Auth] Successful login: {user['id']} ({user['email']})")

    return await _issue_token_pair(user["id"], user["email"], user["role"])


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest):
    """
    Exchange a valid refresh token for a new access + refresh token pair.

    Token rotation: the old refresh token is revoked and a new one is issued.
    If the refresh token has already been used (revoked), ALL sessions are
    invalidated — this indicates a possible token theft.
    """
    payload = decode_refresh_token(body.refresh_token)
    jti = payload["jti"]
    user_id = payload["sub"]

    # Check if this refresh token is still valid in the DB
    is_valid = await is_refresh_token_valid(jti)

    if not is_valid:
        # Token was already used or doesn't exist — possible replay attack
        # Revoke ALL sessions for this user as a safety measure
        logger.warning(
            f"[Auth] Refresh token reuse detected for user {user_id}. "
            "Revoking all sessions."
        )
        await revoke_all_user_tokens(user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has already been used or revoked. Please log in again.",
        )

    # Fetch the user to make sure they still exist and are active
    user = await get_user_by_id(user_id)
    if not user or not user.get("is_active"):
        await revoke_refresh_token(jti)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or deactivated.",
        )

    # Rotate: revoke the old token and issue a new pair
    await revoke_refresh_token(jti)
    logger.info(f"[Auth] Token rotated for user {user_id}")

    return await _issue_token_pair(user["id"], user["email"], user["role"])


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, user: CurrentUser = Depends(get_current_user)):
    """
    Revoke the provided refresh token (logout from current session only).
    The access token remains valid until it expires naturally (max 15 min).
    To immediately invalidate access, use a token denylist (Redis) — add later.
    """
    payload = decode_refresh_token(body.refresh_token)
    jti = payload["jti"]

    # Security: make sure the token belongs to the authenticated user
    if payload.get("sub") != user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token mismatch.")

    await revoke_refresh_token(jti)
    logger.info(f"[Auth] Logout: user {user.user_id} (revoked token {jti[:8]}...)")


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
async def logout_all(user: CurrentUser = Depends(get_current_user)):
    """
    Revoke ALL refresh tokens for the current user.
    Forces re-login on all devices. Use after a password change or compromise.
    """
    count = await revoke_all_user_tokens(user.user_id)
    logger.info(f"[Auth] Full logout: user {user.user_id} ({count} sessions revoked)")


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser = Depends(get_current_user)):
    """
    Return the current authenticated user's profile.
    Password hash is never included in the response.
    """
    db_user = await get_user_by_id(user.user_id)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    return UserResponse(
        id=db_user["id"],
        email=db_user["email"],
        username=db_user["username"],
        role=db_user["role"],
        created_at=db_user["created_at"],
        last_login=db_user.get("last_login"),
    )
