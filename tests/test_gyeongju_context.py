"""
경주 특화 데이터 플러밍 테스트 (2026-09-15).

WHY 이 테스트가 필요한가:
  프롬프트(별도 작업자 담당)가 쓸 수 있으려면 ContextData/SpotCandidate에
  경주 관련 데이터(region, use_time, rest_date)가 먼저 채워져야 한다.
  이 파일은 그 데이터 플러밍만 검증한다 — 프롬프트 문구는 검증 대상이 아니다.

파이프라인 위치:
  사용자 query → extract_region() → is_gyeongju() 판별
    → search_places() 축제/쇼핑 제외 + 영업시간 보강 → ContextData.region
"""

import sqlite3
from contextlib import closing
from datetime import date
from unittest.mock import patch

import pytest

from traveler.core.region import is_gyeongju
from traveler.rag.hybrid_search import HybridSearchResult
from traveler.rag.retriever import get_context, search_places


# ── 1. is_gyeongju 진리표 ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "region, expected",
    [
        ("경주", True),
        ("경주시", True),
        ("보문단지", True),
        ("감포", True),
        ("문무대왕릉", True),
        ("불국사", True),
        ("석굴암", True),
        ("황리단길", True),
        ("대릉원", True),
        ("첨성대", True),
        ("부산", False),
        ("전주", False),
        (None, False),
        ("", False),
    ],
)
def test_is_gyeongju_truth_table(region, expected):
    assert is_gyeongju(region) is expected


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def make_result(content_id: str, content_type_id: str, name: str = "장소") -> HybridSearchResult:
    """테스트용 HybridSearchResult. GPS 반경(100km) 안에 들어오도록 경주 좌표 근처로 고정."""
    return HybridSearchResult(
        content_id=content_id,
        name=name,
        content_type_id=content_type_id,
        category="기타",
        address="경상북도 경주시",
        lat=35.8562,
        lng=129.2247,
        indoor=False,
        semantic_score=0.5,
        keyword_score=0.5,
        preference_score=0.5,
        iconic_score=0.5,
        final_score=0.5,
    )


# ── 2. get_context()가 ContextData.region을 채우는지 ─────────────────────────

def test_get_context_sets_region_for_gyeongju_query():
    ctx, _ = get_context(user_query="불국사 근처 뭐 있어?")
    assert ctx.region == "경주"


def test_get_context_sets_region_none_when_no_region_detected():
    ctx, _ = get_context(user_query="맛집 추천해줘")
    assert ctx.region is None


# ── 3. 경주 축제(15)/쇼핑(38) 필터 — 키워드 예외 + 축제 진행 여부 ────────────
# WHY: 무조건 제외하면 "경주 기념품 사고 싶어", "경주 축제 뭐해" 같은 명시적
#      요청까지 막는다. 쇼핑은 키워드로만, 축제는 키워드 또는 "오늘 진행 중"이면 유지한다.

FESTIVAL_DATE_SCHEMA = """
CREATE TABLE places (
    content_id TEXT,
    content_type_id TEXT,
    event_start_date TEXT,
    event_end_date TEXT
)
"""


def make_festival_db(tmp_path, rows):
    """rows: [(content_id, content_type_id, event_start_date, event_end_date), ...]"""
    db_path = tmp_path / "festival_places.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(FESTIVAL_DATE_SCHEMA)
        conn.executemany(
            "INSERT INTO places VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    return db_path


def test_shopping_excluded_by_default(tmp_path):
    raw_results = [make_result("1", "12", "불국사"), make_result("2", "38", "쇼핑몰")]
    db_path = make_festival_db(tmp_path, [])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
        )
    assert {s.place_id for s in spots} == {"1"}


def test_shopping_kept_with_keyword(tmp_path):
    raw_results = [make_result("1", "12", "불국사"), make_result("2", "38", "쇼핑몰")]
    db_path = make_festival_db(tmp_path, [])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 기념품 살 곳", lat=35.8562, lon=129.2247, visited=set(), prefs={},
        )
    assert {s.place_id for s in spots} == {"1", "2"}


def test_festival_ongoing_today_is_kept(tmp_path):
    raw_results = [make_result("10", "15", "국가유산 미디어아트")]
    db_path = make_festival_db(tmp_path, [("10", "15", "20260901", "20260920")])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),
        )
    assert {s.place_id for s in spots} == {"10"}
    assert spots[0].event_start_date == "20260901"
    assert spots[0].event_end_date == "20260920"


def test_festival_ended_is_excluded(tmp_path):
    # 후보가 이 축제 하나뿐이라 걸러지면 결과가 비고, search_places()는 빈 필터
    # 결과에서 _fallback_spots()로 빠진다(기존 동작 — #31). 여기서 검증할 것은
    # "축제가 살아남지 않았다"이지 fallback 자체는 아니므로 place_id로 확인한다.
    raw_results = [make_result("11", "15", "EX HORROR 시즌 6")]
    db_path = make_festival_db(tmp_path, [("11", "15", "20260801", "20260830")])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),
        )
    assert "11" not in {s.place_id for s in spots}


def test_festival_upcoming_is_excluded(tmp_path):
    raw_results = [make_result("12", "15", "2026 경주 황금카니발")]
    db_path = make_festival_db(tmp_path, [("12", "15", "20260919", "20260919")])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),
        )
    assert "12" not in {s.place_id for s in spots}


def test_festival_keyword_keeps_ended_festival(tmp_path):
    raw_results = [make_result("11", "15", "EX HORROR 시즌 6")]
    db_path = make_festival_db(tmp_path, [("11", "15", "20260801", "20260830")])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 축제 뭐해", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),
        )
    assert {s.place_id for s in spots} == {"11"}


def test_non_gyeongju_keeps_festival_and_shopping_regardless_of_date(tmp_path):
    raw_results = [
        make_result("1", "12", "장소1"),
        make_result("2", "15", "부산축제"),
        make_result("3", "38", "쇼핑몰"),
    ]
    db_path = make_festival_db(tmp_path, [("2", "15", "20260101", "20260102")])  # 이미 종료
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="부산 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),
        )
    assert {s.place_id for s in spots} == {"1", "2", "3"}


# ── 4. 영업시간/휴무일 보강 (타입별 컬럼 → COALESCE) ──────────────────────────

PLACES_SCHEMA = """
CREATE TABLE places (
    content_id TEXT,
    content_type_id TEXT,
    name TEXT,
    use_time TEXT,
    rest_date TEXT,
    use_time_culture TEXT,
    rest_date_culture TEXT,
    open_time_food TEXT,
    rest_date_food TEXT,
    use_time_leports TEXT,
    rest_date_leports TEXT,
    open_time TEXT,
    rest_date_shopping TEXT,
    use_time_festival TEXT
)
"""


def make_temp_db(tmp_path, rows):
    db_path = tmp_path / "places.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(PLACES_SCHEMA)
        conn.executemany(
            """
            INSERT INTO places (
                content_id, content_type_id, name,
                use_time, rest_date, use_time_culture, rest_date_culture,
                open_time_food, rest_date_food, use_time_leports, rest_date_leports,
                open_time, rest_date_shopping, use_time_festival
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return db_path


def test_enrichment_fills_use_time_and_rest_date_by_content_type(tmp_path):
    from traveler.rag.retriever import _enrich_hours
    from traveler.rag.types import SpotCandidate

    # 관광지(12)는 use_time/rest_date, 음식점(39)은 open_time_food/rest_date_food에 실제 값이 있다.
    rows = [
        ("1", "12", "불국사", "09:00~18:00", "연중무휴", None, None, None, None, None, None, None, None, None),
        ("2", "39", "황남빵",  None, None, None, None, "08:00~22:00", "매주 월요일", None, None, None, None, None),
    ]
    db_path = make_temp_db(tmp_path, rows)

    spots = [
        SpotCandidate(place_id="1", place_name="불국사", category="관광", distance_min=5.0, indoor=False),
        SpotCandidate(place_id="2", place_name="황남빵", category="음식", distance_min=10.0, indoor=True),
    ]

    with patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        _enrich_hours(spots)

    assert spots[0].use_time == "09:00~18:00"
    assert spots[0].rest_date == "연중무휴"
    assert spots[1].use_time == "08:00~22:00"
    assert spots[1].rest_date == "매주 월요일"


def test_enrichment_db_error_leaves_fields_none_without_raising(tmp_path):
    from traveler.rag.retriever import _enrich_hours
    from traveler.rag.types import SpotCandidate

    missing_db_path = tmp_path / "does_not_exist.db"
    spots = [
        SpotCandidate(place_id="1", place_name="불국사", category="관광", distance_min=5.0, indoor=False),
    ]

    with patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", missing_db_path):
        _enrich_hours(spots)  # 예외를 던지면 안 된다.

    assert spots[0].use_time is None
    assert spots[0].rest_date is None
    assert not missing_db_path.exists()  # ⚠️ sqlite3.connect()가 빈 파일을 만들면 안 된다.


def test_search_places_enrichment_failure_does_not_trigger_fallback(tmp_path):
    """DB 조회가 실패해도 search_places()가 fallback(_fallback_spots)으로 빠지면 안 된다."""
    raw_results = [make_result("999", "12", "실제후보")]
    missing_db_path = tmp_path / "does_not_exist.db"

    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", missing_db_path):
        spots = search_places(
            user_query="경주 여행",
            lat=35.8562,
            lon=129.2247,
            visited=set(),
            prefs={},
        )

    place_ids = {s.place_id for s in spots}
    assert place_ids == {"999"}
    assert spots[0].use_time is None


# ── 5. 요일 휴무 제외 (_is_closed_weekly + _apply_gyeongju_filter 확장) ──────
# WHY: 프롬프트가 "오늘 휴무인 곳은 빼라"고 지시해도 LLM이 놓칠 수 있다 —
#      실제 A/B에서 화요일에 "매주 월요일, 화요일" 휴무인 박물관을 GPT-4o가
#      추천한 사례가 있었다. 그래서 코드에서 결정적으로 한 번 더 걸러낸다.
# ⚠️ 주의: 시설이 섞였거나(예: "- A: 매주 월 / B: 연중무휴") 연중무휴가 낀
#      텍스트는 보수적으로 "판단 불가 → 제외 안 함"으로 처리한다(과한 제외 방지).

_WEEKDAY_INDEX = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}

_WEEKLY_CLOSURE_TRUTH_TABLE = [
    ("연중무휴", set()),
    ("매주 화요일", {"화"}),
    ("매주 수요일", {"수"}),
    ("매주 월요일", {"월"}),
    ("매주 목요일", {"목"}),
    ("매주 일요일", {"일"}),
    ("점포 별로 상이함", set()),
    ("설·추석 당일", set()),
    (
        "- 식물원 : 매주 월요일 (단, 월요일이 공휴일인 경우 그 다음 평일) / "
        "1월 1일 / 설·추석 당일- 버드파크 : 연중무휴",
        set(),
    ),
    ("- 마을 연중무휴 - 문화관 매주 월요일", set()),
    ("매달 마지막 월요일", set()),
    ("매월 첫 번째, 두 번째, 세 번째, 네 번째 화요일", set()),
    ("매주 월요일 ※ 단, 월요일이 공휴일인 경우 정상 개관", {"월"}),
    ("매주 일요일 / 법정공휴일", {"일"}),
    ("매주 월요일, 화요일 (단, 월요일과 화요일이 법정 휴무일 경우 정상 개관)", {"월", "화"}),
    ("매주 월요일, 목요일 ※ 변동될 경우 홈페이지 공지", {"월", "목"}),
    ("매주 월요일 / 법정공휴일", {"월"}),
    ("매주 월요일 (단, 월요일이 대체휴일 및 법정공휴일일 시 다음날 휴무)", {"월"}),
    ("매주 목요일 (단, 목요일이 법정공휴일인 경우 정상 영업)", {"목"}),
    ("매주 일요일~월요일", {"일", "월"}),
    ("매주 화요일~수요일", {"화", "수"}),
    ("매주 월요일 정기휴무", {"월"}),
    ("매주 화,수 정기휴무", {"화", "수"}),
    ("매주 수요일 (단, 공휴일 제외) / 명절 연휴", {"수"}),
    ("매주 월요일, 금요일", {"월", "금"}),
    ("매주 화요일 (단, 화요일이 공휴일인 경우 다음날 휴무)", {"화"}),
    ("연중무휴※ 단, 우천 시 제외", set()),
    (None, set()),
]


@pytest.mark.parametrize("rest_date, closed_days", _WEEKLY_CLOSURE_TRUTH_TABLE)
def test_is_closed_weekly_truth_table(rest_date, closed_days):
    from traveler.rag.retriever import _is_closed_weekly

    for day_char, weekday in _WEEKDAY_INDEX.items():
        expected = day_char in closed_days
        assert _is_closed_weekly(rest_date, weekday) is expected, (
            f"{rest_date!r} weekday={weekday}({day_char}) expected {expected}"
        )


REST_DATE_SCHEMA = """
CREATE TABLE places (
    content_id TEXT,
    rest_date TEXT,
    rest_date_culture TEXT,
    rest_date_food TEXT,
    rest_date_leports TEXT,
    rest_date_shopping TEXT
)
"""


def make_rest_date_db(tmp_path, rows):
    """rows: [(content_id, rest_date_culture), ...] — 문화시설(14) 컬럼만 채운다."""
    db_path = tmp_path / "rest_date_places.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(REST_DATE_SCHEMA)
        conn.executemany(
            "INSERT INTO places (content_id, rest_date_culture) VALUES (?, ?)",
            rows,
        )
        conn.commit()
    return db_path


def test_gyeongju_weekly_closure_excludes_on_matching_weekday(tmp_path):
    raw_results = [make_result("20", "14", "박물관")]
    db_path = make_rest_date_db(tmp_path, [("20", "매주 월요일, 화요일")])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),  # 화요일
        )
    assert "20" not in {s.place_id for s in spots}


def test_gyeongju_weekly_closure_keeps_on_non_matching_weekday(tmp_path):
    raw_results = [make_result("20", "14", "박물관")]
    db_path = make_rest_date_db(tmp_path, [("20", "매주 월요일, 화요일")])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="경주 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 16),  # 수요일
        )
    assert {s.place_id for s in spots} == {"20"}


def test_non_gyeongju_ignores_weekly_closure(tmp_path):
    raw_results = [make_result("20", "14", "박물관")]
    db_path = make_rest_date_db(tmp_path, [("20", "매주 월요일, 화요일")])
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", db_path):
        spots = search_places(
            user_query="부산 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),  # 화요일
        )
    assert {s.place_id for s in spots} == {"20"}


def test_gyeongju_weekly_closure_lookup_failure_keeps_candidate(tmp_path):
    """조회 실패 시 아무것도 제외하지 않는다 — fallback을 유발해서도 안 된다."""
    raw_results = [make_result("20", "14", "박물관")]
    missing_db_path = tmp_path / "does_not_exist.db"
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results), \
         patch("traveler.rag.hybrid_search.DEFAULT_DB_PATH", missing_db_path):
        spots = search_places(
            user_query="경주 여행", lat=35.8562, lon=129.2247, visited=set(), prefs={},
            today=date(2026, 9, 15),
        )
    assert {s.place_id for s in spots} == {"20"}
