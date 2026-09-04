"""
세션 관리 모듈 - Sprint 5 (배포 버전)

Redis 제거 — ElastiCache 미지원으로 PostgreSQL(RDS) 단일 저장으로 전환.

[저장 전략]
- DATABASE_URL 환경변수로 PostgreSQL 연결
- trip_session 테이블에 visited(JSON), prefs(JSON) 저장
- 모든 DB 작업은 asyncio.to_thread로 비동기 처리

[trip_session 컬럼]
  session_id, status, context(JSON), recommended_places(JSON), turn_count, created_at, updated_at
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────
# PostgreSQL helpers (동기 — asyncio.to_thread로 감쌈)
# ──────────────────────────────────────────────

def _upsert_session(session_id: str) -> None:
    now = _now_iso()
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trip_session (session_id, created_at, updated_at, status)
                VALUES (%s, %s, %s, 'active')
                ON CONFLICT (session_id) DO UPDATE SET updated_at = EXCLUDED.updated_at
            """, (session_id, now, now))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[PG] upsert_session 실패: %s", e)


def _save_visited(session_id: str, place_ids: list) -> None:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT recommended_places FROM trip_session WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            existing = []
            if row and row["recommended_places"]:
                data = row["recommended_places"]
                existing = json.loads(data) if isinstance(data, str) else data
            merged = list(dict.fromkeys(existing + [str(p) for p in place_ids]))
            cur.execute("""
                UPDATE trip_session
                SET recommended_places = %s, updated_at = %s
                WHERE session_id = %s
            """, (json.dumps(merged), _now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[PG] save_visited 실패: %s", e)


def _load_visited(session_id: str) -> set:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT recommended_places FROM trip_session WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
        conn.close()
        if row and row["recommended_places"]:
            data = row["recommended_places"]
            if isinstance(data, str):
                data = json.loads(data)
            return {str(v) for v in data}
    except Exception as e:
        logger.warning("[PG] load_visited 실패: %s", e)
    return set()


def _save_prefs(session_id: str, prefs: dict) -> None:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT context FROM trip_session WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
            existing = {}
            if row and row["context"]:
                data = row["context"]
                existing = json.loads(data) if isinstance(data, str) else data
            merged = {**existing, **prefs}
            cur.execute("""
                UPDATE trip_session
                SET context = %s, updated_at = %s
                WHERE session_id = %s
            """, (json.dumps(merged), _now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[PG] save_prefs 실패: %s", e)


def _load_prefs(session_id: str) -> dict:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT context FROM trip_session WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
        conn.close()
        if row and row["context"]:
            data = row["context"]
            return json.loads(data) if isinstance(data, str) else data
    except Exception as e:
        logger.warning("[PG] load_prefs 실패: %s", e)
    return {}


def _increment_turn(session_id: str) -> None:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trip_session
                SET turn_count = turn_count + 1, updated_at = %s
                WHERE session_id = %s
            """, (_now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[PG] increment_turn 실패: %s", e)


def _close_session(session_id: str) -> None:
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE trip_session
                SET status = 'completed', recommended_places = NULL, context = NULL, updated_at = %s
                WHERE session_id = %s
            """, (_now_iso(), session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[PG] close_session 실패: %s", e)


def _check_db() -> bool:
    """PostgreSQL 연결 확인."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────
# Public async API (인터페이스 기존과 동일 유지)
# ──────────────────────────────────────────────

async def add_visited(session_id: str, place_ids: list) -> None:
    """방문 장소 추가."""
    if not place_ids:
        return
    await asyncio.to_thread(_upsert_session, session_id)
    await asyncio.to_thread(_save_visited, session_id, place_ids)
    await asyncio.to_thread(_increment_turn, session_id)


async def get_visited(session_id: str) -> set:
    """방문 장소 조회."""
    return await asyncio.to_thread(_load_visited, session_id)


async def update_prefs(session_id: str, prefs: dict) -> None:
    """선호도 업데이트."""
    filtered = {k: v for k, v in prefs.items() if v}
    if not filtered:
        return
    await asyncio.to_thread(_upsert_session, session_id)
    await asyncio.to_thread(_save_prefs, session_id, filtered)


async def get_prefs(session_id: str) -> dict:
    """선호도 조회."""
    return await asyncio.to_thread(_load_prefs, session_id)


async def clear_session(session_id: str) -> None:
    """세션 초기화."""
    await asyncio.to_thread(_close_session, session_id)
