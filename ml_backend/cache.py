"""
Disk-backed query cache using diskcache (SQLite under the hood).
No Redis, no separate service — cache lives in a Docker volume.
"""

import hashlib
import logging
import os

import diskcache as dc

logger = logging.getLogger(__name__)

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", str(6 * 60)))
CACHE_SIZE_LIMIT = int(os.getenv("CACHE_SIZE_BYTES", str(2 * 10**9)))  # 2 GB

_cache: dc.Cache | None = None


def get_cache() -> dc.Cache:
    """Returns the cache instance, initializing it on first call."""
    global _cache
    if _cache is None:
        cache_dir = os.getenv("CACHE_DIR", "./data/query_cache")
        os.makedirs(cache_dir, exist_ok=True)
        _cache = dc.Cache(cache_dir, size_limit=CACHE_SIZE_LIMIT)
        logger.info(f"[CACHE] Initialized at {cache_dir}")
    return _cache


def _key(query: str, top_k: int, do_rerank: bool) -> str:
    raw = f"{query.lower().strip()}|{top_k}|{do_rerank}"
    return "rag:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached(query: str, top_k: int, do_rerank: bool):
    hit = get_cache().get(_key(query, top_k, do_rerank))
    if hit is not None:
        logger.info(f"[CACHE HIT]  '{query[:60]}'")
    return hit


def set_cached(query: str, top_k: int, do_rerank: bool, results: list) -> None:
    get_cache().set(_key(query, top_k, do_rerank), results, expire=CACHE_TTL)
    logger.debug(
        f"[CACHE SET]  '{query[:60]}'  ({len(results)} results, ttl={CACHE_TTL}s)"
    )


def cache_stats() -> dict:
    c = get_cache()
    return {
        "entries": len(c),
        "volume_bytes": c.volume(),
        "size_limit": CACHE_SIZE_LIMIT,
        "directory": os.getenv("CACHE_DIR", "./data/query_cache"),
        "ttl_seconds": CACHE_TTL,
    }


def evict_all() -> int:
    c = get_cache()
    n = len(c)
    c.clear()
    logger.warning(f"[CACHE] Cleared {n} entries.")
    return n
