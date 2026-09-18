"""
RAG Retriever - Sprint 4
hybrid_search.py(Pinecone + SQLite) + GPS 반경 필터 + RRF 순위 결합 + SpotCandidate 변환
"""

import logging
import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

from traveler.core.rrf import reciprocal_rank_fusion
from traveler.rag.types import ContextData, SpotCandidate

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_KM = 100.0
DEFAULT_TOP_K = 10


def _today_kst() -> date:
    """today 미지정 시 기본값. 테스트에서는 today= 인자로 주입해 고정한다."""
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _to_place_id(content_id: str) -> str:
    """
    content_id → place_id 변환.
    PR #17(place_id 문자열 전환) 이후: content_id 문자열을 그대로 사용.
    hash() 기반 임시 변환 제거 — 서버 재시작에도 안전하고 SQLite PK와 1:1 매핑.
    """
    return str(content_id) if content_id else ""


def _apply_diversity(spots: list[SpotCandidate], max_per_category: int = 3) -> list[SpotCandidate]:
    """카테고리 다양성 보장. 같은 카테고리가 max_per_category 초과하면 뒤로 밀고 나머지로 채움."""
    category_count: dict[str, int] = {}
    primary: list[SpotCandidate] = []
    overflow: list[SpotCandidate] = []
    for spot in spots:
        count = category_count.get(spot.category, 0)
        if count < max_per_category:
            primary.append(spot)
            category_count[spot.category] = count + 1
        else:
            overflow.append(spot)
    return primary + overflow


# ── 경주 축제/쇼핑 필터 키워드 (2026-09-15) ──
# WHY: 사용자가 명시적으로 찾는 것까지 무조건 빼면 과한 필터링이다.
_SHOPPING_KEYWORDS = ("쇼핑", "기념품", "특산품", "시장", "아울렛", "면세")
_FESTIVAL_KEYWORDS = ("축제", "행사", "페스티벌", "공연", "야행", "카니발")


def _is_festival_ongoing(start: str | None, end: str | None, today: date) -> bool:
    """event_start_date/event_end_date('YYYYMMDD')와 today를 문자열로 비교한다.
    YYYYMMDD는 자릿수가 고정이라 문자열 비교 순서가 날짜 순서와 같다."""
    if not start or not end:
        return False
    today_str = today.strftime("%Y%m%d")
    return start <= today_str <= end


def _lookup_festival_dates(festival_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """
    # WHY: HybridSearchResult에는 event_start_date/event_end_date가 없다.
    #      축제(content_type_id="15") 후보만 골라 한 번에 조회한다.
    # ⚠️ 주의: 실패해도 예외를 던지지 않는다 — 호출부가 빈 dict를 "진행 중 아님"으로
    #      해석해 키워드 매칭 없는 축제만 걸러지게 한다(과한 fallback 방지).
    """
    if not festival_ids:
        return {}

    import sqlite3
    from contextlib import closing
    from pathlib import Path

    try:
        # ⚠️ 주의: PLACES_DB_PATH 환경변수를 안 봄(#28). 경로가 어긋나면 빈 dict로 필터만 무효화된다.
        from traveler.rag.hybrid_search import DEFAULT_DB_PATH

        db_path = Path(DEFAULT_DB_PATH)
        if not db_path.exists():
            raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

        placeholders = ", ".join("?" for _ in festival_ids)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT content_id, event_start_date, event_end_date
                FROM places
                WHERE content_id IN ({placeholders})
                """,
                festival_ids,
            ).fetchall()
        return {
            str(row["content_id"]): (row["event_start_date"], row["event_end_date"])
            for row in rows
        }
    except Exception as e:
        logger.warning("[Retriever] 축제 진행 기간 조회 실패, 진행중 아님으로 처리: %s", e)
        return {}


# ── 요일 휴무(매주 X요일) 판정 (2026-09-15) ──
# WHY: 프롬프트가 "오늘 휴무인 곳은 빼라"고 지시해도 LLM이 놓칠 수 있다 —
#      실제 A/B에서 화요일에 "매주 월요일, 화요일" 휴무인 박물관을 GPT-4o가
#      추천한 사례가 있었다. 그래서 코드에서 결정적으로 한 번 더 걸러낸다.
_WEEKDAY_CHARS = "월화수목금토일"  # index == date.weekday() (월=0 ... 일=6)
_WEEKLY_SEGMENT_CUT_CHARS = "([※/"


def _expand_weekday_range(start_ch: str, end_ch: str) -> set[str]:
    start = _WEEKDAY_CHARS.index(start_ch)
    end = _WEEKDAY_CHARS.index(end_ch)
    if start <= end:
        return set(_WEEKDAY_CHARS[start:end + 1])
    # ⚠️ 주의: "일~월"처럼 주 경계를 넘는 범위 — 뒤로 감아 이어붙인다.
    return set(_WEEKDAY_CHARS[start:] + _WEEKDAY_CHARS[:end + 1])


def _parse_weekday_set(segment: str) -> set[str]:
    """
    세그먼트(예: " 월요일, 화요일 ")에서 요일 문자를 뽑는다.
    - "X요일"(전체 표기)은 항상 인정한다.
    - 맨 글자만 있는 "화,수" 같은 표기는 뒤에 "," "~" "요일" / 세그먼트 끝 /
      "정기" "휴무"가 바로 이어질 때만 요일로 인정한다 — 다른 단어 속 글자와
      구분하기 위함이다.
    """
    days: set[str] = set()
    i, n = 0, len(segment)
    while i < n:
        ch = segment[i]
        if ch not in _WEEKDAY_CHARS:
            i += 1
            continue

        has_yoil = segment[i + 1:i + 3] == "요일"
        after = i + (3 if has_yoil else 1)
        k = after
        while k < n and segment[k] == " ":
            k += 1
        rest = segment[k:]

        if has_yoil:
            valid = True
        else:
            valid = (
                rest == ""
                or rest.startswith((",", "~", "요일", "정기", "휴무"))
            )

        if not valid:
            i = after
            continue

        if rest.startswith("~"):
            k2 = k + 1
            while k2 < n and segment[k2] == " ":
                k2 += 1
            if k2 < n and segment[k2] in _WEEKDAY_CHARS:
                end_ch = segment[k2]
                days |= _expand_weekday_range(ch, end_ch)
                end_has_yoil = segment[k2 + 1:k2 + 3] == "요일"
                i = k2 + (3 if end_has_yoil else 1)
                continue

        days.add(ch)
        i = after
    return days


def _is_closed_weekly(rest_date: str | None, weekday: int) -> bool:
    """
    휴무일 텍스트가 오늘 요일에 "매주 정기 휴무"인지 판정한다.
    보수적으로 판단한다 — 여러 시설이 섞였거나 연중무휴가 낀 텍스트는
    "판단 불가"로 보고 제외하지 않는다(과한 제외로 실제 방문 가능한 곳을
    빼는 것보다, 못 거른 케이스가 프롬프트 지시로 한 번 더 걸러지는 게 낫다).
    """
    if not rest_date or not rest_date.strip():
        return False
    if "매주" not in rest_date:
        return False
    # ⚠️ 주의: "연중무휴"/콜론(여러 시설 목록)/"-"로 시작(목록 항목)은
    # 판단 불가로 취급한다.
    if "연중" in rest_date or ":" in rest_date or rest_date.lstrip().startswith("-"):
        return False

    segment = rest_date[rest_date.index("매주") + len("매주"):]
    cut_positions = [p for p in (segment.find(c) for c in _WEEKLY_SEGMENT_CUT_CHARS) if p != -1]
    if cut_positions:
        segment = segment[:min(cut_positions)]

    closed_days = _parse_weekday_set(segment)
    return _WEEKDAY_CHARS[weekday] in closed_days


def _rest_date_column_expr() -> str:
    """_REST_DATE_COLUMNS COALESCE SQL 조각. _enrich_hours와 _lookup_rest_dates가 공유한다."""
    return "COALESCE(" + ", ".join(f"NULLIF({col}, '')" for col in _REST_DATE_COLUMNS) + ")"


def _lookup_rest_dates(place_ids: list[str]) -> dict[str, str | None]:
    """
    # WHY: 경주 후보의 "매주 X요일 휴무" 판정에 필요한 휴무일 텍스트를 후보
    #      전체에 대해 한 번에 조회한다. _enrich_hours는 최종 top_k에서만
    #      도는 별개 단계라 여기서 재사용하지 않는다(조회 시점이 다름).
    # ⚠️ 주의: 실패해도 예외를 던지지 않는다 — 빈 dict를 "휴무 정보 없음"으로
    #      해석해 아무것도 제외하지 않는다(fallback을 유발하면 안 됨).
    """
    if not place_ids:
        return {}

    import sqlite3
    from contextlib import closing
    from pathlib import Path

    try:
        # ⚠️ 주의: PLACES_DB_PATH 환경변수를 안 봄(#28). 경로가 어긋나면 빈 dict로 필터만 무효화된다.
        from traveler.rag.hybrid_search import DEFAULT_DB_PATH

        db_path = Path(DEFAULT_DB_PATH)
        if not db_path.exists():
            raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

        placeholders = ", ".join("?" for _ in place_ids)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT content_id, {_rest_date_column_expr()} AS rest_date
                FROM places
                WHERE content_id IN ({placeholders})
                """,
                place_ids,
            ).fetchall()
        return {str(row["content_id"]): row["rest_date"] for row in rows}
    except Exception as e:
        logger.warning("[Retriever] 경주 요일 휴무 조회 실패, 제외 없이 진행: %s", e)
        return {}


def _apply_gyeongju_filter(
    raw_results: list,
    user_query: str,
    today: date,
) -> tuple[list, dict[str, tuple[str | None, str | None]]]:
    """
    경주 권역 전용 후보 필터.
      - 쇼핑(38): 쇼핑 키워드가 없으면 제외.
      - 축제(15): 축제 키워드가 있거나, 오늘 진행 중이면 유지. 아니면 제외.
      - 그 외: 그대로 유지.
      - (공통) 위 규칙을 통과해도 오늘이 "매주 X요일" 정기 휴무면 제외.
    """
    has_shopping_keyword = any(k in user_query for k in _SHOPPING_KEYWORDS)
    has_festival_keyword = any(k in user_query for k in _FESTIVAL_KEYWORDS)

    festival_ids = [r.content_id for r in raw_results if r.content_type_id == "15"]
    event_dates = _lookup_festival_dates(festival_ids)

    # ⚠️ 주의: RRF 랭킹 전 단계라 후보 전체(id) 기준으로 한 번에 조회한다 —
    # 장소 하나씩 쿼리하면 N+1이 된다.
    rest_dates = _lookup_rest_dates([r.content_id for r in raw_results])
    weekday = today.weekday()

    kept = []
    excluded_weekly = 0
    for r in raw_results:
        if r.content_type_id == "38":
            if not has_shopping_keyword:
                continue
        elif r.content_type_id == "15":
            start, end = event_dates.get(r.content_id, (None, None))
            if not (has_festival_keyword or _is_festival_ongoing(start, end, today)):
                continue
        if _is_closed_weekly(rest_dates.get(r.content_id), weekday):
            excluded_weekly += 1
            continue
        kept.append(r)

    if excluded_weekly:
        logger.info("[Retriever] 경주 요일 휴무 제외: %d건 (weekday=%d)", excluded_weekly, weekday)

    return kept, event_dates


def search_places(
    user_query: str,
    lat: float,
    lon: float,
    visited: set,
    prefs: dict,
    top_k: int = DEFAULT_TOP_K,
    radius_km: float = DEFAULT_RADIUS_KM,
    dest_lat: float | None = None,
    dest_lon: float | None = None,
    *,
    today: date | None = None,
    fallback_region: str | None = None,
) -> list[SpotCandidate]:
    """
    장소 검색 메인 함수.

    Args:
        dest_lat, dest_lon: 목적지 좌표 (장거리 모드에서만 사용).
            설정 시 Tmap searchPoiAlong으로 경로 상 POI를 추가 탐색한다.
        today: 축제 진행 여부 판정 기준 날짜. 생략하면 KST 오늘 날짜(테스트에서는 주입해 고정).
        fallback_region: user_query에서 지역을 못 찾았을 때만 쓰는 대체 지역(예: 이전 세션의
            지역). user_query 자체의 지역이 항상 우선한다 — "경주 코스"라고 다시 말하면
            그게 최신 의도이므로 이전 지역보다 앞선다.
    """
    from traveler.core.region import is_gyeongju
    from traveler.rag.hybrid_search import search_places as hybrid_search_places
    from traveler.rag.preference import extract_region

    try:
        # WHY: "다른 코스 추천해줘"처럼 후속 발화엔 지역명이 없다. 그럴 때만
        # fallback_region(호출부가 넘긴 이전 지역)을 쓰고, query 자체에 지역이 있으면 그게 이긴다.
        region = extract_region(user_query) or fallback_region
        raw_results = hybrid_search_places(
            user_query=user_query,
            preferences=prefs or None,
            visited_ids={str(v) for v in visited} if visited else None,
            top_k=top_k * 5,
            region=region,
        )
        logger.info("[Retriever] hybrid_search 결과: %d건 (region=%s)", len(raw_results), region)

        # ⚠️ 주의: RRF 랭킹 전에 걸러야 한다 — 랭킹 이후에 빼면 순위가 이미
        # 축제/쇼핑 포함 기준으로 정해진 뒤라 결과 개수·순서가 실행마다 달라질 수 있다.
        event_dates_by_id: dict[str, tuple[str | None, str | None]] = {}
        if is_gyeongju(region):
            raw_results, event_dates_by_id = _apply_gyeongju_filter(
                raw_results, user_query, today or _today_kst()
            )

        # GPS 반경 필터 + (SpotCandidate, raw_result) 튜플 생성
        # main.py의 detect_region_coords가 이미 lat/lon을 지역 좌표로 갱신하므로
        # 반경은 그대로 유지 — region 감지 여부와 무관하게 radius_km 적용.
        filtered: list[tuple[SpotCandidate, object]] = []
        for r in raw_results:
            if r.lat == 0.0 and r.lng == 0.0:
                continue
            dist_km = _haversine(lat, lon, r.lat, r.lng)
            if dist_km > radius_km:
                continue
            distance_min = round(dist_km, 1)  # 60km/h 기준 1km = 1분, dist_km == distance_min
            spot = SpotCandidate(
                place_id=_to_place_id(r.content_id),
                place_name=r.name,
                category=r.category,
                distance_min=distance_min,
                indoor=r.indoor,
                lat=r.lat,
                lng=r.lng,
            )
            if r.content_type_id == "15":
                spot.event_start_date, spot.event_end_date = event_dates_by_id.get(
                    r.content_id, (None, None)
                )
            filtered.append((spot, r))

        # 장거리 모드 Tmap 병합: /api/chat에 dest_lat/dest_lon 파라미터 추가 후 활성화 예정
        # if dest_lat is not None and dest_lon is not None:
        #     filtered = _merge_tmap_results(
        #         filtered, lat, lon, dest_lat, dest_lon, visited
        #     )

        if not filtered:
            logger.warning("[FALLBACK][Retriever] GPS 필터 후 결과 없음 — 하드코딩 3곳(경주 불국사·경주 첨성대·황남빵) 반환")
            return _fallback_spots()

        spots_by_id = {spot.place_id: spot for spot, _ in filtered}

        # RRF 세 신호 준비
        # semantic: Pinecone 의미 유사도 순위
        semantic_ranking = [
            _to_place_id(r.content_id)
            for _, r in sorted(filtered, key=lambda x: x[1].semantic_score, reverse=True)
        ]
        # iconic: 대표 명소 점수 순위
        iconic_ranking = [
            _to_place_id(r.content_id)
            for _, r in sorted(filtered, key=lambda x: x[1].iconic_score, reverse=True)
        ]
        # diversity: 카테고리 다양성 기반 순위
        diversity_ranking = [
            spot.place_id for spot in _apply_diversity([s for s, _ in filtered])
        ]

        rrf_result = reciprocal_rank_fusion({
            "semantic": semantic_ranking,
            "iconic": iconic_ranking,
            "diversity": diversity_ranking,
        })

        final_spots = [spots_by_id[pid] for pid in rrf_result if pid in spots_by_id][:top_k]
        logger.info("[Retriever] 최종 결과: %d건 (반경 %.0fkm)", len(final_spots), radius_km)
        _enrich_hours(final_spots)
        return final_spots

    except Exception as e:
        logger.error("[FALLBACK][Retriever] hybrid_search 실패 — 하드코딩 3곳(경주 불국사·경주 첨성대·황남빵) 반환: %s", e)
        return _fallback_spots()


# TourAPI는 콘텐츠 유형(관광/문화/음식/레포츠/쇼핑/축제)마다 영업시간·휴무일을
# 서로 다른 컬럼에 담는다. 유형을 가리지 않고 한 번에 채우기 위해 COALESCE로
# 값이 있는 첫 컬럼을 쓴다 — 실제로 축제는 휴무일 컬럼 자체가 없다.
_USE_TIME_COLUMNS = (
    "use_time", "use_time_culture", "open_time_food",
    "use_time_leports", "open_time", "use_time_festival",
)
_REST_DATE_COLUMNS = (
    "rest_date", "rest_date_culture", "rest_date_food", "rest_date_leports", "rest_date_shopping",
)


def _enrich_hours(spots: list[SpotCandidate]) -> None:
    """
    # WHY: 영업시간/휴무일은 최종 top_k 장소에만 필요하다(모든 후보에 채우면 낭비).
    #      place_id(content_id) IN (...) 단일 쿼리로 한 번에 조회해 SRP + 성능을 지킨다.
    # ⚠️ 주의: 여기서 예외가 새 나가면 search_places()의 바깥 try/except가 잡아
    #      이미 랭킹까지 끝난 좋은 후보를 통째로 버리고 fallback으로 빠진다.
    #      그건 과한 반응이므로 이 함수 안에서만 예외를 잡고 필드는 None으로 둔다.
    """
    if not spots:
        return

    import sqlite3
    from contextlib import closing
    from pathlib import Path

    try:
        # ⚠️ 주의: PLACES_DB_PATH 환경변수를 안 봄(#28). 경로가 어긋나면 빈 dict로 필터만 무효화된다.
        from traveler.rag.hybrid_search import DEFAULT_DB_PATH

        db_path = Path(DEFAULT_DB_PATH)
        if not db_path.exists():
            raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

        use_time_expr = "COALESCE(" + ", ".join(
            f"NULLIF({col}, '')" for col in _USE_TIME_COLUMNS
        ) + ")"
        rest_date_expr = _rest_date_column_expr()
        placeholders = ", ".join("?" for _ in spots)

        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT content_id, {use_time_expr} AS use_time, {rest_date_expr} AS rest_date
                FROM places
                WHERE content_id IN ({placeholders})
                """,
                [spot.place_id for spot in spots],
            ).fetchall()

        hours_by_id = {
            str(row["content_id"]): (row["use_time"], row["rest_date"]) for row in rows
        }
        for spot in spots:
            spot.use_time, spot.rest_date = hours_by_id.get(spot.place_id, (None, None))

    except Exception as e:
        logger.warning("[Retriever] 영업시간/휴무일 조회 실패, use_time/rest_date는 None 유지: %s", e)


def _merge_tmap_results(
    filtered: list[tuple[SpotCandidate, object]],
    lat: float,
    lon: float,
    dest_lat: float,
    dest_lon: float,
    visited: set,
) -> list[tuple[SpotCandidate, object]]:
    """
    Tmap searchPoiAlong 결과를 hybrid_search 결과에 추가.
    이미 있는 place_id는 중복 추가하지 않는다.
    """
    import asyncio
    from traveler.api.tmap import search_poi_along_route

    try:
        loop = asyncio.get_event_loop()
        tmap_results = loop.run_until_complete(
            search_poi_along_route(lat, lon, dest_lat, dest_lon)
        )
    except Exception as e:
        logger.warning("[Retriever] Tmap 호출 실패: %s", e)
        return filtered

    existing_ids = {spot.place_id for spot, _ in filtered}
    visited_str = {str(v) for v in visited}

    for t in tmap_results:
        place_id = f"tmap_{t.lat}_{t.lon}"
        if place_id in existing_ids:
            continue
        if str(place_id) in visited_str:
            continue
        dist_km = _haversine(lat, lon, t.lat, t.lon)
        distance_min = round((dist_km / 60.0) * 60, 1)
        spot = SpotCandidate(
            place_id=place_id,
            place_name=t.name,
            category=t.category,
            distance_min=distance_min,
            indoor=False,
        )
        # Tmap 결과는 semantic_score/iconic_score가 없으므로
        # dummy 객체로 감싸서 필터 리스트에 추가
        filtered.append((spot, _TmapDummy(t.lat, t.lon)))
        existing_ids.add(place_id)

    logger.info("[Retriever] Tmap 병합 후 총 %d건", len(filtered))
    return filtered


class _TmapDummy:
    """Tmap POI용 dummy raw_result (RRF 신호 최솟값 부여)."""
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lng = lon
        self.semantic_score = 0.0
        self.iconic_score = 0.0


def _fallback_spots() -> list[SpotCandidate]:
    return [
        SpotCandidate(place_id="126166",  place_name="경주 불국사", category="관광", distance_min=5.0,  indoor=False, lat=35.7923, lng=129.3317),
        SpotCandidate(place_id="126207",  place_name="경주 첨성대", category="관광", distance_min=12.0, indoor=False, lat=35.8343, lng=129.2185),
        SpotCandidate(place_id="2765063", place_name="황남빵",      category="음식", distance_min=15.0, indoor=True, lat=35.8409, lng=129.2137),
    ]


def build_spots_context(spots: list[SpotCandidate]) -> str:
    if not spots:
        return ""
    lines = ["[실시간 검색된 주변 장소 후보 - 이 목록에서만 추천할 것]"]
    for i, s in enumerate(spots, 1):
        indoor_str = "실내" if s.indoor else "실외"
        lines.append(
            f"{i}. {s.place_name} | 카테고리: {s.category} | "
            f"이동 {s.distance_min:.0f}분 | {indoor_str} | place_id: {s.place_id}"
        )
    return "\n".join(lines)


def get_context(
    user_query: str = "경주에서 반나절 자차로 여행하고 싶어요",
    lat: float = 35.8562,
    lon: float = 129.2247,
    weather: str = "맑음",
    visited: set = None,
    prefs: dict = None,
    radius_km: float = DEFAULT_RADIUS_KM,
    top_k: int = DEFAULT_TOP_K,
    dest_lat: float | None = None,
    dest_lon: float | None = None,
    duration_hours: int = 5,
    *,
    today: date | None = None,
    fallback_region: str | None = None,
) -> tuple:
    from traveler.rag.preference import extract_region

    if visited is None:
        visited = set()
    if prefs is None:
        prefs = {}

    # WHY: region을 여기서 한 번만 계산해 search_places()에 그대로 넘긴다(fallback_region
    # 자리에). search_places()가 내부에서 다시 extract_region(user_query) or ... 를
    # 계산해도 같은 값이 나오므로, 여기 ContextData.region과 절대 어긋나지 않는다.
    region = extract_region(user_query) or fallback_region

    spots = search_places(
        user_query=user_query,
        lat=lat,
        lon=lon,
        visited=visited,
        prefs=prefs,
        radius_km=radius_km,
        top_k=top_k,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        today=today,
        fallback_region=region,
    )

    context_data = ContextData(
        user_query=user_query,
        current_lat=lat,
        current_lng=lon,
        weather=weather,
        candidate_spots=spots,
        companion=prefs.get("companion", "혼자"),
        mood=prefs.get("mood", ""),
        duration_hours=duration_hours,
        region=region,
    )

    return context_data, build_spots_context(spots)
