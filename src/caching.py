import hashlib
import json
import time
import asyncio
from typing import Optional, Any
from collections import OrderedDict
import redis.asyncio as aioredis
from src.config import settings


def make_cache_key(prompt: str, model_name: str, max_new_tokens: int) -> str:
    """
    Privacy-preserving cache key: SHA-256 hash of prompt + model params only.
    NO user identifiers are stored.
    """
    payload = json.dumps({
        "prompt": prompt,
        "model": model_name,
        "max_new_tokens": max_new_tokens,
    }, sort_keys=True)
    return "cache:" + hashlib.sha256(payload.encode()).hexdigest()


class InMemoryCache:
    """LRU + TTL in-process cache with asyncio lock."""

    def __init__(self, max_entries: int, ttl_seconds: int):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            expires_at = time.monotonic() + self.ttl_seconds
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def flush(self) -> None:
        async with self._lock:
            self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    async def stats(self) -> dict:
        async with self._lock:
            return {
                "backend": "inmemory",
                "size": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hit_rate,
            }


class RedisCache:
    """Redis-backed cache with TTL and max-entry limits."""

    def __init__(self, host: str, port: int, ttl_seconds: int, max_entries: int):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._client = aioredis.Redis(host=host, port=port, decode_responses=True)
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[str]:
        value = await self._client.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    async def set(self, key: str, value: str) -> None:
        await self._client.setex(key, self.ttl_seconds, value)
        await self._client.sadd("cache:keys", key)
        count = await self._client.scard("cache:keys")
        if count > self.max_entries:
            old = await self._client.spop("cache:keys")
            if old:
                await self._client.delete(old)

    async def invalidate(self, key: str) -> None:
        await self._client.delete(key)
        await self._client.srem("cache:keys", key)

    async def flush(self) -> None:
        await self._client.flushdb()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    async def stats(self) -> dict:
        size = await self._client.scard("cache:keys")
        return {
            "backend": "redis",
            "size": size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }


def get_cache():
    """Factory: returns configured cache backend."""
    if settings.cache_backend == "redis":
        return RedisCache(
            host=settings.redis_host,
            port=settings.redis_port,
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
        )
    return InMemoryCache(
        max_entries=settings.cache_max_entries,
        ttl_seconds=settings.cache_ttl_seconds,
    )
