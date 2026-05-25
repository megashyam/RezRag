"""
Disk-backed query cache using diskcache (SQLite under the hood).
No Redis, no separate service — cache lives in a Docker volume.
"""

import hashlib
import logging
import os

import diskcache as dc

logger = logging.getLogger(__name__)

CACHE_DIR = os.getenv("CACHE_DIR", "./data/query_cache")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", str(6 * 60 * 60)))  # 6 h default
CACHE_SIZE_LIMIT = int(os.getenv("CACHE_SIZE_BYTES", str(2 * 10**9)))  # 2 GB

# Single module-level cache instance — diskcache is process-safe via file locks
_cache = dc.Cache(CACHE_DIR, size_limit=CACHE_SIZE_LIMIT)


def _key(query: str, top_k: int, do_rerank: bool) -> str:
    raw = f"{query.lower().strip()}|{top_k}|{do_rerank}"
    return "rag:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached(query: str, top_k: int, do_rerank: bool):
    hit = _cache.get(_key(query, top_k, do_rerank))
    if hit is not None:
        logger.info(f"[CACHE HIT]  '{query[:60]}'")
    return hit


def set_cached(query: str, top_k: int, do_rerank: bool, results: list) -> None:
    _cache.set(_key(query, top_k, do_rerank), results, expire=CACHE_TTL)
    logger.debug(
        f"[CACHE SET]  '{query[:60]}'  ({len(results)} results, ttl={CACHE_TTL}s)"
    )


def cache_stats() -> dict:
    return {
        "entries": len(_cache),
        "volume_bytes": _cache.volume(),
        "size_limit": CACHE_SIZE_LIMIT,
        "directory": CACHE_DIR,
        "ttl_seconds": CACHE_TTL,
    }


def evict_all() -> int:
    n = len(_cache)
    _cache.clear()
    logger.warning(f"[CACHE] Cleared {n} entries.")
    return n
