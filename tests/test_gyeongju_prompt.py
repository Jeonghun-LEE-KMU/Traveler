"""
경주 특화 프롬프트 블록 테스트 (2026-09-15).

WHY 이 테스트가 필요한가:
  build_context_message()가 ctx.region이 경주(is_gyeongju)일 때만 운영시간·휴무·
  축제기간과 권역 동선 규칙 블록을 추가해야 한다. 그 외 지역은 하드 제약 —
  오늘 출력과 완전히 동일(byte-for-byte)해야 한다(회귀 0).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from traveler.core.prompt import _clean_raw_text, build_context_message
from traveler.rag.types import ContextData, SpotCandidate

KST = ZoneInfo("Asia/Seoul")


def minimal_ctx(**overrides) -> ContextData:
    defaults = dict(
        user_query="경주 여행 추천해줘",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[],
    )
    defaults.update(overrides)
    return ContextData(**defaults)


def sample_spot(**overrides) -> SpotCandidate:
    defaults = dict(
        place_id="1",
        place_name="불국사",
        category="관광",
        distance_min=10.0,
        indoor=False,
    )
    defaults.update(overrides)
    return SpotCandidate(**defaults)


# ── 1. _clean_raw_text() 단위 테스트 ─────────────────────────────────────────

def test_clean_raw_text_none_and_blank_map_to_default():
    assert _clean_raw_text(None) == "정보 없음"
    assert _clean_raw_text("   ") == "정보 없음"


def test_clean_raw_text_replaces_br_variants_and_collapses_whitespace():
    assert _clean_raw_text("[평일]<br>11:00~18:00<br>[주말]<br>11:00~19:00") == (
        "[평일] / 11:00~18:00 / [주말] / 11:00~19:00"
    )
    assert _clean_raw_text("a<BR/>b<br />c") == "a / b / c"
    assert _clean_raw_text("a   b\n\tc") == "a b c"


def test_clean_raw_text_truncates_over_80_chars():
    long_text = "가" * 100
    cleaned = _clean_raw_text(long_text)
    assert cleaned == "가" * 80 + "…"
    assert len(cleaned) == 81


def test_clean_raw_text_exactly_80_chars_not_truncated():
    text = "나" * 80
    assert _clean_raw_text(text) == text


# ── 2. 비경주 지역 — 출력 완전 동일(하드 제약) ───────────────────────────────

@pytest.mark.parametrize("region", [None, "부산"])
def test_non_gyeongju_output_ignores_hours_fields(region):
    spot_with_hours = sample_spot(use_time="09:00~18:00", rest_date="매주 월요일")
    spot_without_hours = sample_spot(use_time=None, rest_date=None)

    ctx_with = minimal_ctx(region=region, candidate_spots=[spot_with_hours])
    ctx_without = minimal_ctx(region=region, candidate_spots=[spot_without_hours])

    msg_with = build_context_message(ctx_with)
    msg_without = build_context_message(ctx_without)

    assert "운영:" not in msg_with
    assert "경주 특화" not in msg_with
    assert msg_with == msg_without


# ── 3. 경주 — 현재 시각·규칙 블록·운영시간 표기 ──────────────────────────────

def test_gyeongju_message_includes_current_time_and_rules_block():
    now = datetime(2026, 9, 15, 14, 30, tzinfo=KST)  # 화요일 (Python으로 검증됨)
    spot = sample_spot(
        use_time="[평일]<br>11:00~18:00<br>[주말]<br>11:00~19:00",
        rest_date=None,
    )
    ctx = minimal_ctx(region="경주", candidate_spots=[spot])

    msg = build_context_message(ctx, now=now)

    assert "현재 시각: 2026-09-15(화요일) 14:30 (한국 시간)" in msg
    assert "[경주 특화 — 운영시간·권역 동선 규칙]" in msg
    assert "운영: [평일] / 11:00~18:00 / [주말] / 11:00~19:00" in msg
    assert "휴무: 정보 없음" in msg
    # 규칙 6줄 정도가 실제로 들어있는지 대표로 확인
    assert "오늘(화요일)이 휴무일로 명시된 장소는 코스에서 제외하십시오." in msg
    assert "감포·동해안권은 가용 시간이 6시간 이상일 때만 포함하십시오." in msg


def test_gyeongju_festival_line_shows_period_without_zero_padding():
    now = datetime(2026, 9, 15, 14, 30, tzinfo=KST)
    spot = sample_spot(
        place_id="99",
        place_name="국가유산 미디어아트",
        category="축제",
        event_start_date="20260814",
        event_end_date="20261108",
    )
    ctx = minimal_ctx(region="경주", candidate_spots=[spot])

    msg = build_context_message(ctx, now=now)

    assert "기간: 8/14~11/8" in msg


def test_gyeongju_landmark_region_also_triggers_block():
    now = datetime(2026, 9, 15, 14, 30, tzinfo=KST)
    ctx = minimal_ctx(region="불국사", candidate_spots=[])
    msg = build_context_message(ctx, now=now)
    assert "[경주 특화 — 운영시간·권역 동선 규칙]" in msg


def test_gyeongju_empty_candidates_still_appends_block():
    now = datetime(2026, 9, 15, 14, 30, tzinfo=KST)
    ctx = minimal_ctx(region="경주", candidate_spots=[])

    msg = build_context_message(ctx, now=now)

    assert "[경주 특화 — 운영시간·권역 동선 규칙]" in msg


def test_gyeongju_default_now_used_when_not_injected():
    """now를 안 넘기면 예외 없이 실제 KST now로 동작해야 한다."""
    ctx = minimal_ctx(region="경주", candidate_spots=[])
    msg = build_context_message(ctx)
    assert "[경주 특화 — 운영시간·권역 동선 규칙]" in msg
    assert "현재 시각: " in msg
