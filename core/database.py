"""
User Database
=============
Async SQLite via aiosqlite. Simple and zero-dependency for local/small prod.
Swap the connection string for asyncpg (PostgreSQL) when you scale.

Tables:
  users          — account records
  refresh_tokens — stored refresh tokens (enables revocation)

All queries use parameterized statements — no SQL injection possible.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import aiosqlite

logger = logging.getLogger(__name__)

# Default DB path — override with DATABASE_URL env var
DB_PATH = "quantai.db"


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    jti        TEXT PRIMARY KEY,          -- JWT token ID (from the 'jti' claim)
    user_id    TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
"""


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────

async def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist. Call once at app startup."""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_SCHEMA)
        await db.commit()
    logger.info(f"[DB] Initialized at {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# USER CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_user(
    email: str,
    username: str,
    password_hash: str,
    role: str = "user",
    db_path: str = DB_PATH,
) -> dict:
    """
    Insert a new user. Raises ValueError on duplicate email/username.
    Returns the created user record (without password_hash).
    """
    user_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                INSERT INTO users (id, email, username, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (user_id, email.lower().strip(), username.strip(), password_hash, role, now),
            )
            await db.commit()
    except sqlite3.IntegrityError as e:
        if "email" in str(e):
            raise ValueError(f"Email '{email}' is already registered.")
        if "username" in str(e):
            raise ValueError(f"Username '{username}' is already taken.")
        raise ValueError(str(e))

    logger.info(f"[DB] Created user: {user_id} ({email})")
    return {"id": user_id, "email": email, "username": username, "role": role, "created_at": now}


async def get_user_by_email(email: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Fetch a user by email. Returns None if not found."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE email = ? AND is_active = 1",
            (email.lower().strip(),),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_user_by_id(user_id: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Fetch a user by ID. Returns None if not found."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, email, username, role, is_active, created_at, last_login FROM users WHERE id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def update_last_login(user_id: str, db_path: str = DB_PATH) -> None:
    """Update the last_login timestamp after a successful login."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE users SET last_login = ? WHERE id = ?", (now, user_id)
        )
        await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# REFRESH TOKEN CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def store_refresh_token(
    jti: str,
    user_id: str,
    expires_at: datetime,
    db_path: str = DB_PATH,
) -> None:
    """
    Persist a refresh token's JTI so we can later verify and revoke it.
    The full token string is NEVER stored — only its ID (jti claim).
    """
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO refresh_tokens (jti, user_id, expires_at, revoked, created_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (jti, user_id, expires_at.isoformat(), now),
        )
        await db.commit()


async def is_refresh_token_valid(jti: str, db_path: str = DB_PATH) -> bool:
    """
    Return True only if the refresh token exists, is not revoked,
    and has not expired.
    """
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """
            SELECT 1 FROM refresh_tokens
            WHERE jti = ? AND revoked = 0 AND expires_at > ?
            """,
            (jti, now),
        ) as cursor:
            return (await cursor.fetchone()) is not None


async def revoke_refresh_token(jti: str, db_path: str = DB_PATH) -> None:
    """Revoke a specific refresh token (used during rotation and logout)."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE jti = ?", (jti,)
        )
        await db.commit()


async def revoke_all_user_tokens(user_id: str, db_path: str = DB_PATH) -> int:
    """
    Revoke ALL refresh tokens for a user.
    Use for: password change, account compromise, force-logout-everywhere.
    Returns the number of tokens revoked.
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ? AND revoked = 0",
            (user_id,),
        )
        await db.commit()
        count = cursor.rowcount
    logger.info(f"[DB] Revoked {count} refresh token(s) for user {user_id}")
    return count


async def cleanup_expired_tokens(db_path: str = DB_PATH) -> int:
    """
    Delete expired refresh tokens from the DB.
    Run this periodically (e.g. daily) to keep the table small.
    Returns the number of rows deleted.
    """
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?", (now,)
        )
        await db.commit()
        return cursor.rowcount
