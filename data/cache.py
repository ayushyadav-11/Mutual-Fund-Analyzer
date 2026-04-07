import json
import logging
import os
import time
import redis
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Fallback in-memory dict if Redis is completely unavailable
_memory_cache = {}

# By default expect a local Redis on 6379, configurable via env
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

try:
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # Ping to check if actually alive immediately
    _redis_client.ping()
    _USE_REDIS = True
    logger.info("Connected to Redis at %s", REDIS_URL)
except Exception as e:
    _redis_client = None
    _USE_REDIS = False
    logger.warning("Redis not available at %s, falling back to in-memory dict cache. Error: %s", REDIS_URL, e)


def get_cached(key: str) -> dict | None:
    """Retrieve string payload from cache and parse back to dict."""
    try:
        if _USE_REDIS and _redis_client:
            val = _redis_client.get(key)
            if val:
                return json.loads(val)
        else:
            entry = _memory_cache.get(key)
            if not entry:
                return None
            expires_at = entry.get("expires_at")
            if expires_at is not None and expires_at <= time.time():
                _memory_cache.pop(key, None)
                return None
            return entry.get("data")
    except Exception as e:
        logger.error(f"Cache GET error for key {key}: {e}")
    return None


def set_cached(key: str, data: dict, ttl_seconds: int = 86400):
    """Store dict payload as string in cache with TTL (default 24h)."""
    try:
        val = json.dumps(data)
        if _USE_REDIS and _redis_client:
            _redis_client.setex(key, ttl_seconds, val)
        else:
            _memory_cache[key] = {
                "data": data,
                "expires_at": time.time() + ttl_seconds,
            }
    except Exception as e:
        logger.error(f"Cache SET error for key {key}: {e}")


def flush_cache():
    """Clear all keys in the cache (for admin/debug)."""
    if _USE_REDIS and _redis_client:
        try:
            _redis_client.flushdb()
            logger.info("Redis cache flushed.")
        except:
            pass
    else:
        _memory_cache.clear()
        logger.info("Memory cache flushed.")

def is_redis_active() -> bool:
    return _USE_REDIS
