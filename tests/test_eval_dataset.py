"""
eval/dataset.py + eval/cases.jsonl 골든셋 검증 — Sprint 5 Task 1.

[설계 근거] docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §2.7
설계 문서의 수용 기준을 그대로 테스트로 옮긴다. 전부 API 키 없이 통과해야
한다 — 이 테스트는 "골든셋 파일 자체가 깨지지 않았는가"만 확인하고 LLM은
전혀 호출하지 않는다.
"""

from collections import Counter
import json
from pathlib import Path

import pytest

from eval.dataset import GoldenCase, load_cases
from eval.judge_prompt import build_judge_user_message
from traveler.core.prompt import build_context_message
from traveler.rag.types import ContextData
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel

CASES_PATH = Path(__file__).resolve().parent.parent / "eval" / "cases.jsonl"

MODES = ("근교", "장거리", "타지")

# expectations가 가질 수 있는 키 화이트리스트 — 오타로 새 키가 몰래 섞이는 걸 막는다.
EXPECTATION_KEYS = {
    "min_indoor_places",
    "forbidden_place_ids",
    "max_total_minutes",
    "expect_places_range",
}

# 각 축의 수준(level) — 모드별로 최소 2회씩 등장해야 하는 태그들.
WEATHER_TAGS = {"sunny", "rain", "cloudy", "snow"}
COMPANION_TAGS = {"solo", "couple", "family_infant", "family_elderly", "friends"}
DURATION_TAGS = {"short", "medium", "long"}


@pytest.fixture(scope="module")
def all_cases() -> list[GoldenCase]:
    return load_cases(CASES_PATH)


def test_sixty_cases_total(all_cases: list[GoldenCase]):
    """[Sprint 7 — 30→60 확장] 모드별 10→20개로 늘어 총 60개."""
    assert len(all_cases) == 60


def test_exactly_twenty_cases_per_mode(all_cases: list[GoldenCase]):
    counts = Counter(c.mode for c in all_cases)
    for mode in MODES:
        assert counts[mode] == 20, f"{mode} 케이스가 20개가 아님: {counts[mode]}"


def test_no_duplicate_case_ids(all_cases: list[GoldenCase]):
    ids = [c.case_id for c in all_cases]
    assert len(ids) == len(set(ids))


def test_all_cases_build_context_data(all_cases: list[GoldenCase]):
    """60개 전부 ContextData 복원에 성공했는지(로더가 이미 이걸 하지만,
    복원된 타입이 실제로 ContextData인지 명시적으로 재확인)."""
    assert len(all_cases) == 60
    for case in all_cases:
        assert isinstance(case.ctx, ContextData)
        # 필수 필드가 비어있지 않은지도 함께 확인(로더가 필드명을 잘못
        # 매핑해도 dataclass 생성 자체는 성공할 수 있는 경우를 대비).
        assert case.ctx.user_query
        assert case.ctx.weather
        assert case.ctx.candidate_spots


def test_expectations_keys_are_whitelisted(all_cases: list[GoldenCase]):
    for case in all_cases:
        extra_keys = set(case.expectations.keys()) - EXPECTATION_KEYS
        assert not extra_keys, f"{case.case_id}에 화이트리스트 밖 키: {extra_keys}"
        # expect_places_range는 다른 세 키와 달리 반드시 채워져 있어야 한다
        # (metrics.py가 지금 당장 소비하는 유일한 키이기 때문).
        assert case.expectations.get("expect_places_range")


@pytest.mark.parametrize("mode", MODES)
def test_weather_levels_appear_at_least_twice_per_mode(all_cases, mode):
    tags_in_mode = [tag for c in all_cases if c.mode == mode for tag in c.tags]
    counts = Counter(tags_in_mode)
    for weather_tag in WEATHER_TAGS:
        assert counts[weather_tag] >= 2, (
            f"{mode} 모드에서 날씨 태그 '{weather_tag}'가 2회 미만 등장"
        )


@pytest.mark.parametrize("mode", MODES)
def test_companion_levels_appear_at_least_twice_per_mode(all_cases, mode):
    tags_in_mode = [tag for c in all_cases if c.mode == mode for tag in c.tags]
    counts = Counter(tags_in_mode)
    for companion_tag in COMPANION_TAGS:
        assert counts[companion_tag] >= 2, (
            f"{mode} 모드에서 동행자 태그 '{companion_tag}'가 2회 미만 등장"
        )


@pytest.mark.parametrize("mode", MODES)
def test_duration_levels_appear_at_least_twice_per_mode(all_cases, mode):
    tags_in_mode = [tag for c in all_cases if c.mode == mode for tag in c.tags]
    counts = Counter(tags_in_mode)
    for duration_tag in DURATION_TAGS:
        assert counts[duration_tag] >= 2, (
            f"{mode} 모드에서 가용시간 태그 '{duration_tag}'가 2회 미만 등장"
        )


@pytest.mark.parametrize("mode", MODES)
def test_rain_or_snow_with_family_covered_at_least_twice_per_mode(all_cases, mode):
    """Graceful Degradation 경로(비/눈 + 실내 제약 필요 동행자) 커버리지 확인."""
    matches = [
        c
        for c in all_cases
        if c.mode == mode
        and {"rain", "snow"} & set(c.tags)
        and {"family_infant", "family_elderly"} & set(c.tags)
    ]
    assert len(matches) >= 2, f"{mode} 모드에 비/눈+가족 조합이 2개 미만"


@pytest.mark.parametrize("mode", MODES)
def test_mode_filter_returns_exactly_twenty(mode):
    filtered = load_cases(CASES_PATH, mode=mode)
    assert len(filtered) == 20
    assert all(c.mode == mode for c in filtered)


def _load_single_case(tmp_path: Path, duration_fields: dict) -> GoldenCase:
    ctx = {
        "user_query": "경주 여행 추천해줘",
        "current_lat": 35.8,
        "current_lng": 129.2,
        "weather": "맑음",
        "candidate_spots": [
            {
                "place_id": "1",
                "place_name": "첨성대",
                "category": "관광",
                "distance_min": 1.0,
                "indoor": False,
            }
        ],
        **duration_fields,
    }
    row = {
        "case_id": "duration-contract",
        "mode": "타지",
        "today": "2026-09-19",
        "ctx": ctx,
        "expectations": {"expect_places_range": [1, 3]},
        "tags": ["sunny", "solo", "short"],
    }
    path = tmp_path / "case.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return load_cases(path)[0]


@pytest.mark.parametrize("hours", (2, 4, 6))
def test_loader_marks_omitted_duration_hours_min_as_fixed_explicit_hours(tmp_path, hours):
    case = _load_single_case(tmp_path, {"duration_hours": hours})

    assert case.ctx.duration_hours == hours
    assert case.ctx.duration_hours_min is None


def test_loader_marks_explicit_five_hour_value_as_fixed_hours(tmp_path):
    case = _load_single_case(tmp_path, {"duration_hours": 5})

    assert case.ctx.duration_hours == 5
    assert case.ctx.duration_hours_min is None


def test_loader_keeps_default_three_to_five_hours_when_duration_is_missing(tmp_path):
    case = _load_single_case(tmp_path, {})

    assert case.ctx.duration_minutes is None
    assert case.ctx.duration_hours == 5
    assert case.ctx.duration_hours_min == 3


@pytest.mark.parametrize("duration_hours_min", (None, 2))
def test_loader_preserves_explicit_duration_hours_min_range(tmp_path, duration_hours_min):
    case = _load_single_case(
        tmp_path,
        {"duration_hours": 5, "duration_hours_min": duration_hours_min},
    )

    assert case.ctx.duration_hours_min == duration_hours_min


def test_loader_preserves_duration_minutes_as_ssot(tmp_path):
    case = _load_single_case(
        tmp_path,
        {"duration_minutes": 90, "duration_hours": 4, "duration_hours_min": None},
    )

    assert case.ctx.duration_minutes == 90
    assert case.ctx.duration_hours == 4
    assert case.ctx.duration_hours_min is None
    expected = "가용 시간: 90분 (1.5시간)"
    assert expected in build_context_message(case.ctx, today=case.today)
    course = StreamingCourseModel(
        bot_message="테스트 코스",
        recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1)],
    )
    assert expected in build_judge_user_message(case.ctx, case.mode, course)


def test_shipped_cases_keep_fixed_hours_in_generator_and_judge_messages(all_cases):
    for case in all_cases:
        assert case.ctx.duration_minutes is None
        assert case.ctx.duration_hours_min is None
        expected = f"가용 시간: {case.ctx.duration_hours}시간"
        generator_message = build_context_message(case.ctx, today=case.today)
        course = StreamingCourseModel(
            bot_message="테스트 코스",
            recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1)],
        )
        judge_message = build_judge_user_message(case.ctx, case.mode, course)
        assert expected in generator_message
        assert expected in judge_message
