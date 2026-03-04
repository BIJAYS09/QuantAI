import logging
import redis
from core.config import settings

logger = logging.getLogger(__name__)

redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis | None:
    """Lazily initialize and return a redis client, or None on failure."""
    global redis_client
    if redis_client is not None:
        return redis_client
    try:
        url = settings.redis_url
        if not url:
            return None
        redis_client = redis.from_url(url, decode_responses=True)
        logger.info(f"[Cache] connected to Redis: {url}")
    except Exception as e:
        logger.warning(f"[Cache] unable to connect to Redis: {e}")
        redis_client = None
    return redis_client


def cache_get(key: str) -> str | None:
    r = get_redis()
    if not r:
        return None
    try:
        return r.get(key)
    except Exception:
        return None


def cache_set(key: str, value: str, ex: int = 60) -> None:
    r = get_redis()
    if not r:
        return
    try:
        r.set(key, value, ex=ex)
    except Exception:
        pass
