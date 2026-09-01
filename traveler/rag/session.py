"""
세션 관리 모듈 - Sprint 5

Redis(캐시) + SQLite(영속화) 이중 저장 구조.

[저장 전략]
- Redis: 빠른 읽기/쓰기, TTL 24시간
- SQLite trip_session: 서버 재시작 후에도 세션 복원 가능
- Redis miss -> SQLite fallback으로 Graceful Degradation

[Redis 키 구조]
  trip:{session_id}:visited  -> SET   방문한 place_id 목록
  trip:{session_id}:prefs    -> HASH  누적 선호도 (companion, mood, travel_mode)

[SQLite trip_session 컬럼]
  session_id, status, context(JSON), recommended_places(JSON), turn_count, created_at, updated_at
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379")
from traveler.rag.hybrid_search import DEFAULT_DB_PATH
DB_PATH     = os.getenv("PLACES_DB_PATH", str(DEFAULT_DB_PATH))
TTL_SECONDS = 60 * 60 * 24  # 24시간

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


# ──────────────────────────────────────────────
# SQLite helpers (동기 — asyncio.to_thread로 감쌈)
# ──────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_upsert_session(session_id: str) -> None:
    """세션 레코드가 없으면 생성, 있으면 updated_at만 갱신."""
    now = _now_iso()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO trip_session (session_id, created_at, updated_at, status)
            VALUES (?, ?, ?, 'active')
            ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
        """, (session_id, now, now))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[SQLite] upsert_session 실패: %s", e)


def _sqlite_save_visited(session_id: str, place_ids: list) -> None:
    """recommended_places 컬럼에 place_id 목록을 JSON으로 저장."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT recommended_places FROM trip_session WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        existing = []
        if row and row[0]:
            try:
                existing = json.loads(row[0])
            except json.JSONDecodeError:
                existing = []

        merged = list(dict.fromkeys(existing + place_ids))  # 중복 제거, 순서 유지
        conn.execute("""
            UPDATE trip_session
            SET recommended_places = ?, updated_at = ?
            WHERE session_id = ?
        """, (json.dumps(merged, ensure_ascii=False), _now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[SQLite] save_visited 실패: %s", e)


def _sqlite_load_visited(session_id: str) -> set:
    """SQLite에서 recommended_places를 복원."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT recommended_places FROM trip_session WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return {str(v) for v in json.loads(row[0])}
    except Exception as e:
        logger.warning("[SQLite] load_visited 실패: %s", e)
    return set()


def _sqlite_save_prefs(session_id: str, prefs: dict) -> None:
    """context 컬럼에 선호도를 JSON으로 저장."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT context FROM trip_session WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        existing = {}
        if row and row[0]:
            try:
                existing = json.loads(row[0])
            except json.JSONDecodeError:
                existing = {}

        merged = {**existing, **prefs}
        conn.execute("""
            UPDATE trip_session
            SET context = ?, updated_at = ?
            WHERE session_id = ?
        """, (json.dumps(merged, ensure_ascii=False), _now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[SQLite] save_prefs 실패: %s", e)


def _sqlite_load_prefs(session_id: str) -> dict:
    """SQLite에서 context(선호도)를 복원."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT context FROM trip_session WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        logger.warning("[SQLite] load_prefs 실패: %s", e)
    return {}


def _sqlite_increment_turn(session_id: str) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            UPDATE trip_session
            SET turn_count = turn_count + 1, updated_at = ?
            WHERE session_id = ?
        """, (_now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[SQLite] increment_turn 실패: %s", e)


def _sqlite_close_session(session_id: str) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            UPDATE trip_session
            SET status = 'completed', recommended_places = NULL, context = NULL, updated_at = ?
            WHERE session_id = ?
        """, (_now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[SQLite] close_session 실패: %s", e)


# ──────────────────────────────────────────────
# Public async API
# ──────────────────────────────────────────────

async def add_visited(session_id: str, place_ids: list) -> None:
    """방문 장소 추가 — Redis SET + SQLite recommended_places."""
    if not place_ids:
        return

    # Redis
    try:
        r = get_redis()
        key = f"trip:{session_id}:visited"
        await r.sadd(key, *place_ids)
        await r.expire(key, TTL_SECONDS)
    except Exception as e:
        logger.warning("[Redis] visited 추가 실패: %s", e)

    # SQLite
    await asyncio.to_thread(_sqlite_upsert_session, session_id)
    await asyncio.to_thread(_sqlite_save_visited, session_id, place_ids)
    await asyncio.to_thread(_sqlite_increment_turn, session_id)


async def get_visited(session_id: str) -> set:
    """방문 장소 조회 — Redis 우선, miss 시 SQLite 복원."""
    redis_ok = False
    try:
        r = get_redis()
        key = f"trip:{session_id}:visited"
        members = await r.smembers(key)
        redis_ok = True
        if members:
            return members
    except Exception as e:
        logger.warning("[Redis] visited 조회 실패: %s", e)

    if redis_ok and not members:
        # Redis 정상 접속됐고 실제로 데이터 없는 세션 → SQLite 왕복 불필요
        return set()

    # Redis miss(장애) -> SQLite fallback
    logger.info("[Session] Redis miss, SQLite에서 visited 복원: session=%s", session_id)
    visited = await asyncio.to_thread(_sqlite_load_visited, session_id)

    # SQLite에서 복원한 데이터를 Redis에 재캐싱
    if visited:
        try:
            r = get_redis()
            key = f"trip:{session_id}:visited"
            await r.sadd(key, *visited)
            await r.expire(key, TTL_SECONDS)
        except Exception:
            pass

    return visited


async def update_prefs(session_id: str, prefs: dict) -> None:
    """선호도 업데이트 — Redis HASH + SQLite context."""
    filtered = {k: v for k, v in prefs.items() if v}
    if not filtered:
        return

    # Redis
    try:
        r = get_redis()
        key = f"trip:{session_id}:prefs"
        await r.hset(key, mapping=filtered)
        await r.expire(key, TTL_SECONDS)
    except Exception as e:
        logger.warning("[Redis] prefs 업데이트 실패: %s", e)

    # SQLite
    await asyncio.to_thread(_sqlite_upsert_session, session_id)
    await asyncio.to_thread(_sqlite_save_prefs, session_id, filtered)


async def get_prefs(session_id: str) -> dict:
    """선호도 조회 — Redis 우선, miss 시 SQLite 복원."""
    try:
        r = get_redis()
        key = f"trip:{session_id}:prefs"
        prefs = await r.hgetall(key)
        if prefs:
            return prefs
    except Exception as e:
        logger.warning("[Redis] prefs 조회 실패: %s", e)

    # Redis miss -> SQLite fallback
    logger.info("[Session] Redis miss, SQLite에서 prefs 복원: session=%s", session_id)
    prefs = await asyncio.to_thread(_sqlite_load_prefs, session_id)

    # Redis 재캐싱
    if prefs:
        try:
            r = get_redis()
            key = f"trip:{session_id}:prefs"
            await r.hset(key, mapping={k: v for k, v in prefs.items() if isinstance(v, str)})
            await r.expire(key, TTL_SECONDS)
        except Exception:
            pass

    return prefs


async def clear_session(session_id: str) -> None:
    """세션 초기화 — Redis 삭제 + SQLite status='completed'."""
    try:
        r = get_redis()
        await r.delete(
            f"trip:{session_id}:visited",
            f"trip:{session_id}:prefs",
        )
        logger.info("[Redis] 세션 초기화: session=%s", session_id)
    except Exception as e:
        logger.warning("[Redis] 세션 초기화 실패: %s", e)

    await asyncio.to_thread(_sqlite_close_session, session_id)
