"""Thin async Redis wrapper with TTLs tuned to how often each kind of data
actually changes. Avoids hammering the FPL/Understat APIs on every request.
"""
import json
from functools import wraps

import redis.asyncio as redis

from app.config import get_settings

settings = get_settings()
_client: redis.Redis | None = None

TTL_BOOTSTRAP_STATIC = 60 * 60 * 6     # players/teams: refreshes a few times/day
TTL_FIXTURES = 60 * 60 * 12            # fixture list: rarely changes mid-season
TTL_PREDICTIONS = 60 * 30              # predictions: recompute at most every 30 min
TTL_LIVE_GAMEWEEK = 60                 # during live matches, data moves fast


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


async def cache_get(key: str):
    raw = await get_client().get(key)
    return json.loads(raw) if raw else None


async def cache_set(key: str, value, ttl: int) -> None:
    await get_client().set(key, json.dumps(value), ex=ttl)


def cached(key_fn, ttl: int):
    """Decorator for async functions: caches the JSON-serialisable return
    value under a key built from key_fn(*args, **kwargs).
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            hit = await cache_get(key)
            if hit is not None:
                return hit
            result = await fn(*args, **kwargs)
            await cache_set(key, result, ttl)
            return result
        return wrapper
    return decorator
