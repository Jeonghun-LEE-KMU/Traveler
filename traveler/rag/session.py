"""
Redis trip_session 관리 모듈 — Sprint 2 PoC

파이프라인 위치:
  사용자 요청 → ★ [session.py] → visited 필터 주입 → GPT-4o → 장소 추천
                               → prefs 누적 저장 → 다음 추천에 반영

[Redis 키 구조]
  trip:{session_id}:visited  → SET  방문한 place_id 목록
                                    ex) {"1", "3", "7"}
  trip:{session_id}:prefs    → HASH 누적 선호도 (companion, mood, travel_mode)
                                    ex) {companion: "커플", mood: "감성, 힐링"}

[TTL 정책]
  모든 키: 24시간 (write 시마다 갱신)
  WHY 24h: 하루 드라이브 단위가 trip_session의 자연스러운 수명.
           24h 지나면 초기화 → 다음날 새 세션으로 시작.

[Graceful Degradation]
  Redis 연결 실패 시 → 빈 값 반환 (앱이 중단되지 않음)
  WHY: Redis는 보조 기능 (방문 필터 + 선호도). 없어도 추천은 동작해야 함.
"""

import logging
import os
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL_SECONDS = 60 * 60 * 24  # 24시간

# ── 싱글턴 클라이언트 ─────────────────────────────────────────────────────────
# WHY 싱글턴: aioredis.Redis는 내부에 커넥션 풀을 가짐.
# 요청마다 새로 만들면 연결 낭비 → 한 번 만들어서 전역 재사용.
_redis: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    """Redis 클라이언트 싱글턴 반환 (Lazy Init)"""
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,   # bytes 대신 str 반환
            socket_connect_timeout=2,  # 연결 대기 2초 (Graceful Degradation)
        )
    return _redis


# ── visited SET 관리 ──────────────────────────────────────────────────────────

async def add_visited(session_id: str, place_ids: list[str]) -> None:
    """
    방문 확정된 place_id 목록을 visited SET에 추가하고 TTL 갱신.

    호출 시점: 프론트엔드에서 사용자가 코스를 확정했을 때
    (POST /api/session/{session_id}/visited 엔드포인트)

    INPUT: session_id, place_ids (ex: ["1", "3", "7"])
    SIDE EFFECT: Redis SET에 추가 + TTL 24h 갱신
    """
    if not place_ids:
        return
    try:
        r = get_redis()
        key = f"trip:{session_id}:visited"
        await r.sadd(key, *place_ids)
        await r.expire(key, TTL_SECONDS)
        logger.debug("[Redis] visited 추가: session=%s, places=%s", session_id, place_ids)
    except Exception as e:
        # Redis 장애 시 무시 — visited 필터가 없어도 추천은 동작함
        logger.warning("[Redis] visited 추가 실패 (무시): %s", e)


async def get_visited(session_id: str) -> set[str]:
    """
    세션의 방문 장소 SET 반환.

    호출 시점: 매 추천 요청 시, visited 필터 컨텍스트 구성에 사용.

    OUTPUT: place_id 문자열 SET (ex: {"1", "3"})
            Redis 실패 시 → 빈 set (Graceful Degradation)
    """
    try:
        r = get_redis()
        key = f"trip:{session_id}:visited"
        return await r.smembers(key)   # decode_responses=True라 str set 반환
    except Exception as e:
        logger.warning("[Redis] visited 조회 실패 (빈 set 반환): %s", e)
        return set()


# ── prefs HASH 관리 ───────────────────────────────────────────────────────────

async def update_prefs(session_id: str, prefs: dict) -> None:
    """
    누적 선호도 HASH에 새 선호도를 병합(upsert)하고 TTL 갱신.

    호출 시점: 매 사용자 메시지 처리 시 (preference.py가 추출한 결과)
    WHY HSET: 기존 값을 지우지 않고 새 값만 업데이트 (누적 효과)
              ex) 1회차: {companion: "혼자"}
                  2회차: {mood: "감성"} → 최종: {companion: "혼자", mood: "감성"}

    INPUT: prefs dict (빈 값 포함 가능 → 내부에서 필터링)
    """
    # 빈 문자열/None 값은 Redis에 저장하지 않음
    filtered = {k: v for k, v in prefs.items() if v}
    if not filtered:
        return
    try:
        r = get_redis()
        key = f"trip:{session_id}:prefs"
        await r.hset(key, mapping=filtered)
        await r.expire(key, TTL_SECONDS)
        logger.debug("[Redis] prefs 업데이트: session=%s, prefs=%s", session_id, filtered)
    except Exception as e:
        logger.warning("[Redis] prefs 업데이트 실패 (무시): %s", e)


async def get_prefs(session_id: str) -> dict:
    """
    세션의 누적 선호도 HASH 반환.

    호출 시점: 매 추천 요청 시, 프롬프트 컨텍스트 주입에 사용.

    OUTPUT: 선호도 dict (ex: {"companion": "커플", "mood": "감성, 힐링"})
            Redis 실패 시 → 빈 dict (Graceful Degradation)
    """
    try:
        r = get_redis()
        key = f"trip:{session_id}:prefs"
        return await r.hgetall(key)
    except Exception as e:
        logger.warning("[Redis] prefs 조회 실패 (빈 dict 반환): %s", e)
        return {}


# ── 세션 초기화 ───────────────────────────────────────────────────────────────

async def clear_session(session_id: str) -> None:
    """
    세션 데이터 전체 삭제 (visited + prefs).

    호출 시점: 사용자가 "새 여행 시작" 요청 시
    (POST /api/session/{session_id}/clear 엔드포인트에서 호출)
    """
    try:
        r = get_redis()
        await r.delete(
            f"trip:{session_id}:visited",
            f"trip:{session_id}:prefs",
        )
        logger.info("[Redis] 세션 초기화: session=%s", session_id)
    except Exception as e:
        logger.warning("[Redis] 세션 초기화 실패: %s", e)
