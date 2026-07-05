"""
Redis trip_session 관리 모듈

Redis 키 구조:
    trip:{session_id}:visited  -> SET   방문한 place_id 목록
    trip:{session_id}:prefs    -> HASH  누적 선호도 (companion, mood, travel_mode)

TTL: 24시간 (write 시마다 갱신)
Redis 장애 시 빈 값 반환으로 Graceful Degradation
"""

import logging
import os
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL_SECONDS = 60 * 60 * 24

_redis: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
        )
    return _redis


async def add_visited(session_id: str, place_ids: list[str]) -> None:
    if not place_ids:
        return
    try:
        r = get_redis()
        key = f"trip:{session_id}:visited"
        await r.sadd(key, *place_ids)
        await r.expire(key, TTL_SECONDS)
    except Exception as e:
        logger.warning("[Redis] visited 추가 실패: %s", e)


async def get_visited(session_id: str) -> set[str]:
    try:
        r = get_redis()
        key = f"trip:{session_id}:visited"
        return await r.smembers(key)
    except Exception as e:
        logger.warning("[Redis] visited 조회 실패: %s", e)
        return set()


async def update_prefs(session_id: str, prefs: dict) -> None:
    filtered = {k: v for k, v in prefs.items() if v}
    if not filtered:
        return
    try:
        r = get_redis()
        key = f"trip:{session_id}:prefs"
        await r.hset(key, mapping=filtered)
        await r.expire(key, TTL_SECONDS)
    except Exception as e:
        logger.warning("[Redis] prefs 업데이트 실패: %s", e)


async def get_prefs(session_id: str) -> dict:
    try:
        r = get_redis()
        key = f"trip:{session_id}:prefs"
        return await r.hgetall(key)
    except Exception as e:
        logger.warning("[Redis] prefs 조회 실패: %s", e)
        return {}


async def clear_session(session_id: str) -> None:
    try:
        r = get_redis()
        await r.delete(
            f"trip:{session_id}:visited",
            f"trip:{session_id}:prefs",
        )
        logger.info("[Redis] 세션 초기화: session=%s", session_id)
    except Exception as e:
        logger.warning("[Redis] 세션 초기화 실패: %s", e)
