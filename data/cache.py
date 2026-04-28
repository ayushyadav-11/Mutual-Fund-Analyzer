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
    """Retrieve payload from L1 memory cache or L2 database/Redis cache."""
    # L1: Fast In-Memory Cache
    if key in _memory_cache:
        data, expires_at = _memory_cache[key]
        if expires_at is None or expires_at > time.time():
            return data
        else:
            _memory_cache.pop(key, None)

    try:
        # L2: Redis Cache
        if _USE_REDIS and _redis_client:
            val = _redis_client.get(key)
            if val:
                data = json.loads(val)
                # Populate L1 cache for next time
                _memory_cache[key] = (data, time.time() + 86400)
                return data
        else:
            # L2: Database Cache (PostgreSQL/SQLite)
            from data.database import get_connection
            conn = get_connection()
            c = conn.cursor()
            c.execute('SELECT value_json, expires_at FROM kv_cache WHERE key = ?', (key,))
            row = c.fetchone()
            conn.close()
            
            if row:
                expires_at = row['expires_at']
                if expires_at is not None and expires_at <= time.time():
                    # Delete stale cache natively
                    conn = get_connection()
                    c = conn.cursor()
                    c.execute('DELETE FROM kv_cache WHERE key = ?', (key,))
                    conn.commit()
                    conn.close()
                    return None
                
                data = json.loads(row['value_json'])
                # Populate L1 cache for next time
                _memory_cache[key] = (data, expires_at if expires_at else time.time() + 86400)
                return data
            
            return None
    except Exception as e:
        logger.error(f"Cache GET error for key {key}: {e}")
    return None


def set_cached(key: str, data: dict, ttl_seconds: int = 86400):
    """Store dict payload as string in L2 cache with TTL, and populate L1 memory cache."""
    try:
        # Populate L1 Memory Cache immediately
        _memory_cache[key] = (data, time.time() + ttl_seconds)
        
        val = json.dumps(data)
        if _USE_REDIS and _redis_client:
            _redis_client.setex(key, ttl_seconds, val)
        else:
            # Fallback to database cache
            from data.database import get_connection
            conn = get_connection()
            c = conn.cursor()
            expires_at = time.time() + ttl_seconds
            
            c.execute('''
                INSERT INTO kv_cache (key, value_json, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    expires_at=excluded.expires_at
            ''', (key, val, expires_at))
            conn.commit()
            conn.close()
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
        try:
            from data.database import get_connection
            conn = get_connection()
            c = conn.cursor()
            c.execute('DELETE FROM kv_cache')
            conn.commit()
            conn.close()
            logger.info("Database kv_cache flushed.")
        except Exception as e:
            logger.error("Failed to flush database kv_cache: %s", e)


def delete_cached(key: str):
    """Delete a single key from Redis or database cache."""
    if _USE_REDIS and _redis_client:
        try:
            _redis_client.delete(key)
            logger.info("Deleted Redis key: %s", key)
        except Exception as e:
            logger.warning("Redis DELETE error for key %s: %s", key, e)
    else:
        _memory_cache.pop(key, None)
        try:
            from data.database import get_connection
            conn = get_connection()
            c = conn.cursor()
            c.execute('DELETE FROM kv_cache WHERE key = ?', (key,))
            conn.commit()
            conn.close()
            logger.info("Deleted Database cache key: %s", key)
        except Exception as e:
            logger.warning("Database cache DELETE error for key %s: %s", key, e)


def is_redis_active() -> bool:
    return _USE_REDIS
