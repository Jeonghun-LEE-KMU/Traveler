"""
검색 결과 캐싱 유틸.

REDIS_URL이 설정되어 있으면 Redis TTL 캐시를 우선 사용한다.
Redis 설정이 없거나 연결에 실패하면 프로세스 메모리 TTL 캐시로 fallback한다.
수정 : REDIS_URL 기본값 비우기, 메모리 캐싱 상한 추가(#35)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "300"))
SEARCH_CACHE_MEMORY_MAX_SIZE = int(
    os.getenv("SEARCH_CACHE_MEMORY_MAX_SIZE", "256")
)
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_KEY_PREFIX = os.getenv("SEARCH_CACHE_KEY_PREFIX", "traveler:search")

# value 타입은 tuple[expires_at, cached_results]
# cached_results는 HybridSearchResult 리스트지만 순환 import를 피하기 위해 Any로 둔다.
_memory_cache: dict[str, tuple[float, tuple[Any, ...]]] = {}

_redis_client: Any | None = None
_redis_disabled = False


def get_redis_client() -> Any | None:
    """Redis client를 lazy singleton으로 생성한다. 실패하면 메모리 캐시로 fallback한다."""
    global _redis_client, _redis_disabled

    if _redis_disabled:
        return None

    if not REDIS_URL:
        logger.info("[search cache] REDIS_URL not set; using memory cache")
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis

        client = redis.Redis.from_url(
            REDIS_URL,
            socket_connect_timeout=1,
            socket_timeout=1,
            decode_responses=False,
        )
        client.ping()
        _redis_client = client
        logger.info("[search cache] redis enabled")
        return _redis_client
    except Exception as exc:
        _redis_disabled = True
        logger.warning("[search cache] redis unavailable; fallback to memory: %s", exc)
        return None


def make_search_cache_key(
    *,
    user_query: str,
    preferences: dict | None,
    visited_ids: set[str] | None,
    region: str | None,
    content_types: set[str],
    top_k: int,
    db_path: Path,
    score_threshold: float | None = None,
    score_weights: dict | None = None,
) -> str:
    """동일 검색 요청을 재사용하기 위한 안정적인 cache key를 만든다."""
    stat = db_path.stat()

    payload = {
        "version": 1,
        "query": user_query.strip(),
        "preferences": preferences or {},
        "visited_ids": sorted(str(content_id) for content_id in (visited_ids or set())),
        "region": region or "",
        "content_types": sorted(str(content_type) for content_type in content_types),
        "top_k": top_k,
        "score_threshold": score_threshold,
        "score_weights": score_weights or {},
        "db_path": str(db_path.resolve()),
        "db_mtime_ns": stat.st_mtime_ns,
        "db_size": stat.st_size,
    }

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{REDIS_KEY_PREFIX}:{digest}"


def get_cached_search_results(cache_key: str) -> list[Any] | None:
    """TTL이 살아 있는 검색 결과를 반환한다. 없거나 만료되면 None."""
    if SEARCH_CACHE_TTL_SECONDS <= 0:
        return None

    redis_client = get_redis_client()
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached is None:
                logger.info("[search cache] redis miss")
                return None

            logger.info("[search cache] redis hit")
            return list(pickle.loads(cached))
        except Exception as exc:
            logger.warning("[search cache] redis get failed; fallback to memory: %s", exc)

    cached = _memory_cache.get(cache_key)
    if cached is None:
        logger.info("[search cache] memory miss")
        return None

    expires_at, results = cached
    if expires_at <= time.time():
        _memory_cache.pop(cache_key, None)
        logger.info("[search cache] memory expired")
        return None

    logger.info("[search cache] memory hit")
    return list(results)


def set_cached_search_results(cache_key: str, results: list[Any]) -> None:
    """검색 결과를 TTL 캐시에 저장한다."""
    if SEARCH_CACHE_TTL_SECONDS <= 0:
        return

    payload = tuple(results)

    redis_client = get_redis_client()
    if redis_client is not None:
        try:
            redis_client.setex(
                cache_key,
                SEARCH_CACHE_TTL_SECONDS,
                pickle.dumps(payload),
            )
            logger.info("[search cache] redis set")
            return
        except Exception as exc:
            logger.warning("[search cache] redis set failed; fallback to memory: %s", exc)

    expires_at = time.time() + SEARCH_CACHE_TTL_SECONDS

    if SEARCH_CACHE_MEMORY_MAX_SIZE <= 0:
        logger.info("[search cache] memory disabled")
        return

    while len(_memory_cache) >= SEARCH_CACHE_MEMORY_MAX_SIZE:
        _memory_cache.pop(next(iter(_memory_cache)))

    _memory_cache[cache_key] = (expires_at, payload)
    logger.info("[search cache] memory set")
    

def clear_search_cache() -> None:
    """테스트나 수동 초기화를 위한 캐시 비우기 함수."""
    redis_client = get_redis_client()
    if redis_client is not None:
        for key in redis_client.scan_iter(f"{REDIS_KEY_PREFIX}:*"):
            redis_client.delete(key)

    _memory_cache.clear()