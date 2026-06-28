# ============================================================
# SQLite 필터 쿼리 함수
# 프로젝트: LLM-RAG 트래블 큐레이터
# 작성자: 송가영
# ============================================================
# 역할: 유저 조건을 받아서 SQLite places 테이블에서
#       조건에 맞는 장소를 동적으로 필터링해서 반환
# ============================================================

import sqlite3
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "places.db")


def get_connection() -> sqlite3.Connection:
    """DB 연결 반환"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 결과를 dict처럼 사용 가능
    return conn


# ── 메인 필터 쿼리 함수 ───────────────────────────────────────
def filter_places(
    content_type_id: Optional[str] = None,   # 관광 타입 (12=관광지, 14=문화시설, 15=축제, 39=음식점)
    area_code: Optional[str] = None,          # 도 단위 (1=서울, 6=부산 등)
    sigungu_code: Optional[str] = None,       # 시/군/구 단위
    parking: Optional[bool] = None,           # 주차 가능 여부
    is_pet_allowed: Optional[bool] = None,    # 반려동물 동반 가능 여부
    is_nokidszone: Optional[bool] = None,     # 노키즈존 여부
    exclude_rest_day: Optional[str] = None,   # 특정 요일 휴무 제외 (예: "월요일")
    limit: int = 20,                          # 최대 반환 건수
) -> list[dict]:
    """
    유저 조건을 받아서 places 테이블에서 조건에 맞는 장소 반환

    Parameters:
        content_type_id : 관광 타입 코드 (None이면 전체)
        area_code       : 도 단위 지역 코드 (None이면 전체)
        sigungu_code    : 시/군/구 단위 코드 (None이면 전체)
        parking         : True면 주차 가능한 곳만, False면 조건 없음
        is_pet_allowed  : True면 반려동물 가능한 곳만
        is_nokidszone   : True면 노키즈존만, False면 노키즈존 제외
        exclude_rest_day: 해당 요일 휴무인 곳 제외 (예: "월요일")
        limit           : 최대 반환 건수 (기본 20)

    Returns:
        조건에 맞는 장소 목록 (list of dict)
    """
    query  = "SELECT * FROM places WHERE 1=1"
    params = []

    # 관광 타입 필터
    if content_type_id is not None:
        query += " AND content_type_id = ?"
        params.append(content_type_id)

    # 지역 필터 (도 단위)
    if area_code is not None:
        query += " AND area_code = ?"
        params.append(area_code)

    # 지역 필터 (시/군/구 단위) — area_code랑 같이 써야 의미 있음
    if sigungu_code is not None:
        query += " AND sigungu_code = ?"
        params.append(sigungu_code)

    # 주차 필터 — parking 컬럼이 비어있지 않고 "불가능" 아닌 곳
    if parking is True:
        query += " AND parking != '' AND parking NOT LIKE '%불가능%'"

    # 반려동물 필터
    if is_pet_allowed is True:
        query += " AND is_pet_allowed = 1"

    # 노키즈존 필터
    if is_nokidszone is True:
        query += " AND is_nokidszone = 1"
    elif is_nokidszone is False:
        query += " AND is_nokidszone = 0"

    # 휴무일 필터 — 특정 요일 휴무인 곳 제외
    if exclude_rest_day is not None:
        query += " AND (rest_date NOT LIKE ? AND rest_date_food NOT LIKE ? AND rest_date_culture NOT LIKE ?)"
        like = f"%{exclude_rest_day}%"
        params.extend([like, like, like])

    # iconic_score 높은 순으로 정렬
    query += " ORDER BY iconic_score DESC"

    # 반환 건수 제한
    query += f" LIMIT {limit}"

    conn = get_connection()
    try:
        cursor = conn.execute(query, params)
        rows   = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ── 단건 조회 함수 ────────────────────────────────────────────
def get_place_by_id(content_id: str) -> Optional[dict]:
    """
    content_id로 단건 조회
    """
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT * FROM places WHERE content_id = ?", [content_id])
        row    = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── 테스트 실행 ───────────────────────────────────────────────
if __name__ == "__main__":

    # 테스트 1: 서울 관광지 중 주차 가능한 곳
    print("=== 테스트 1: 서울 관광지 + 주차 가능 ===")
    results = filter_places(content_type_id="12", area_code="1", parking=True, limit=5)
    for r in results:
        print(f"  {r['name']} | 주차: {r['parking']}")

    # 테스트 2: 반려동물 동반 가능한 곳
    print("\n=== 테스트 2: 반려동물 동반 가능 ===")
    results = filter_places(is_pet_allowed=True, limit=5)
    for r in results:
        print(f"  {r['name']} | 반려동물: {r['is_pet_allowed']}")

    # 테스트 3: 노키즈존 제외한 음식점
    print("\n=== 테스트 3: 노키즈존 제외 음식점 ===")
    results = filter_places(content_type_id="39", is_nokidszone=False, limit=5)
    for r in results:
        print(f"  {r['name']} | 노키즈존: {r['is_nokidszone']}")

    # 테스트 4: 월요일 휴무 제외
    print("\n=== 테스트 4: 월요일 휴무 제외 ===")
    results = filter_places(exclude_rest_day="월요일", limit=5)
    for r in results:
        print(f"  {r['name']} | 휴무일: {r.get('rest_date') or r.get('rest_date_food')}")
