"""
traveler/core/season_context.py 단위 테스트 (Sprint 5 Task 5 — 계절 L3 축제 통합).

검증 대상:
  - build_season_context(): festivals=None/빈 리스트 → 기존 "현재 계절: X" 한 줄과 완전히 동일(회귀 0)
  - filter_active_festivals(): 날짜 경계 포함/제외, 상한(limit) 적용, 잘못된 날짜 문자열 스킵
  - build_context_message(): festivals=None(인자 생략)일 때 이번 변경 이전과 동일한 문자열을 반환

API·DB 의존 없음 — 전부 순수 함수 테스트.
"""

from datetime import date

from traveler.core.prompt import build_context_message
from traveler.core.season import get_season
from traveler.core.season_context import (
    FestivalInfo,
    build_season_context,
    filter_active_festivals,
)
from traveler.rag.types import ContextData, SpotCandidate


def _make_context(user_query: str = "분위기 좋은 곳 추천") -> ContextData:
    return ContextData(
        user_query=user_query,
        current_lat=37.5665,
        current_lng=126.9780,
        weather="맑음",
        candidate_spots=[
            SpotCandidate(
                place_id="1",
                place_name="불국사",
                category="관광",
                distance_min=15.0,
                indoor=False,
            )
        ],
    )


def _festival(name: str, start: str, end: str, place: str = "") -> FestivalInfo:
    return FestivalInfo(
        name=name, event_start_date=start, event_end_date=end, event_place=place
    )


# ── 가장 중요한 테스트: 회귀 0 보장 ──────────────────────────────────────────

def test_build_context_message_with_no_festivals_matches_pre_change_output():
    """
    festivals=None(인자 생략)으로 build_context_message()를 호출하면,
    이번 변경 이전에 "현재 계절: {get_season(today)}" 한 줄로 직접 만들던 결과와
    완전히 동일한 문자열이 나와야 한다 (D4/§6.3 회귀 0 보장).
    """
    ctx = _make_context()
    today = date(2026, 1, 15)

    message = build_context_message(ctx, today=today)

    # 변경 전 build_context_message()가 만들던 것과 동일한 라인이 그대로 포함되어야 함
    assert f"현재 계절: {get_season(today)}" in message
    assert "현재 계절: 겨울" in message
    # 축제 블록은 전혀 추가되지 않아야 함
    assert "[진행 중인 축제·행사]" not in message


def test_build_context_message_omitting_festivals_arg_still_works():
    """festivals 인자를 아예 생략해도(하위 호환) 정상 동작해야 한다."""
    ctx = _make_context()
    message = build_context_message(ctx, today=date(2026, 1, 15))
    assert "현재 계절: 겨울" in message


def test_build_season_context_none_festivals_returns_season_line_only():
    today = date(2026, 8, 7)
    result = build_season_context(today, None)
    assert result == f"현재 계절: {get_season(today)}"


def test_build_season_context_empty_list_returns_season_line_only():
    today = date(2026, 8, 7)
    result = build_season_context(today, [])
    assert result == f"현재 계절: {get_season(today)}"


# ── 축제 개수별 컨텍스트 블록 포함 검증 ──────────────────────────────────────

def test_build_season_context_includes_single_active_festival():
    today = date(2026, 8, 7)
    festivals = [_festival("가을 단풍 축제", "20260805", "20260810", "남산")]

    result = build_season_context(today, festivals)

    assert "[진행 중인 축제·행사]" in result
    assert "가을 단풍 축제" in result
    assert "남산" in result


def test_build_season_context_includes_all_three_when_exactly_three():
    today = date(2026, 8, 7)
    festivals = [
        _festival("축제A", "20260805", "20260810"),
        _festival("축제B", "20260806", "20260809"),
        _festival("축제C", "20260807", "20260807"),
    ]

    result = build_season_context(today, festivals)

    for name in ("축제A", "축제B", "축제C"):
        assert name in result


def test_build_season_context_caps_at_three_when_five_active():
    today = date(2026, 8, 7)
    festivals = [
        _festival("축제1", "20260801", "20260810"),
        _festival("축제2", "20260802", "20260810"),
        _festival("축제3", "20260803", "20260810"),
        _festival("축제4", "20260804", "20260810"),
        _festival("축제5", "20260805", "20260810"),
    ]

    result = build_season_context(today, festivals)

    # 시작일 오름차순 상위 3개(축제1~3)만 포함, 나머지는 제외
    assert "축제1" in result
    assert "축제2" in result
    assert "축제3" in result
    assert "축제4" not in result
    assert "축제5" not in result


# ── filter_active_festivals: 날짜 경계 ───────────────────────────────────────

def test_filter_active_festivals_includes_start_date_boundary():
    today = date(2026, 8, 5)
    festivals = [_festival("경계축제", "20260805", "20260810")]
    result = filter_active_festivals(festivals, today)
    assert len(result) == 1
    assert result[0].name == "경계축제"


def test_filter_active_festivals_includes_end_date_boundary():
    today = date(2026, 8, 10)
    festivals = [_festival("경계축제", "20260805", "20260810")]
    result = filter_active_festivals(festivals, today)
    assert len(result) == 1
    assert result[0].name == "경계축제"


def test_filter_active_festivals_excludes_day_before_start():
    today = date(2026, 8, 4)
    festivals = [_festival("경계축제", "20260805", "20260810")]
    result = filter_active_festivals(festivals, today)
    assert result == []


def test_filter_active_festivals_excludes_day_after_end():
    today = date(2026, 8, 11)
    festivals = [_festival("경계축제", "20260805", "20260810")]
    result = filter_active_festivals(festivals, today)
    assert result == []


def test_filter_active_festivals_respects_limit_parameter():
    today = date(2026, 8, 7)
    festivals = [
        _festival("축제1", "20260801", "20260810"),
        _festival("축제2", "20260802", "20260810"),
    ]
    result = filter_active_festivals(festivals, today, limit=1)
    assert len(result) == 1
    assert result[0].name == "축제1"


# ── 잘못된 날짜 문자열 Graceful Degradation ─────────────────────────────────

def test_filter_active_festivals_skips_invalid_date_string_without_raising():
    today = date(2026, 8, 7)
    festivals = [
        FestivalInfo(name="깨진축제", event_start_date="not-a-date", event_end_date="20260810"),
        _festival("정상축제", "20260805", "20260810"),
    ]

    result = filter_active_festivals(festivals, today)

    names = [f.name for f in result]
    assert "깨진축제" not in names
    assert "정상축제" in names


def test_build_season_context_skips_invalid_date_and_keeps_valid_ones():
    today = date(2026, 8, 7)
    festivals = [
        FestivalInfo(name="깨진축제", event_start_date="not-a-date", event_end_date="20260810"),
        _festival("정상축제", "20260805", "20260810"),
    ]

    result = build_season_context(today, festivals)

    assert "깨진축제" not in result
    assert "정상축제" in result
