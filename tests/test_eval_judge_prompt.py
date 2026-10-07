"""PR47 judge input contract tests for coordinate and schedule metadata."""

from eval.judge_prompt import JUDGE_PROMPT_VERSION, build_judge_user_message
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import (
    RecommendedPlaceModel,
    StreamingCourseModel,
    TimeBudgetSummary,
)


def make_ctx(**overrides) -> ContextData:
    defaults = dict(
        user_query="경주 여행 추천해줘",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[],
    )
    defaults.update(overrides)
    return ContextData(**defaults)


def make_course(**overrides) -> StreamingCourseModel:
    defaults = dict(
        bot_message="경주의 고즈넉한 코스를 추천해요.",
        recommended_places=[
            RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=60)
        ],
    )
    defaults.update(overrides)
    return StreamingCourseModel(**defaults)


def test_judge_prompt_uses_only_valid_candidate_coordinates_not_legacy_distance():
    ctx = make_ctx(
        candidate_spots=[
            SpotCandidate("1", "유효 장소", "관광", 17.0, False, 35.8, 129.2),
            SpotCandidate("2", "좌표 없는 장소", "음식", 8.0, True, 0.0, 129.2),
        ]
    )

    message = build_judge_user_message(ctx, "타지", make_course())

    assert "유효 장소 | 관광 | 좌표 35.8, 129.2 | 실외" in message
    assert "좌표 없는 장소 | 음식 | 좌표 미확인 | 실내" in message
    assert "이동 17분" not in message
    assert "이동 8분" not in message


def test_judge_prompt_rejects_nonfinite_and_out_of_range_candidate_coordinates():
    ctx = make_ctx(
        candidate_spots=[
            SpotCandidate("1", "유효 장소", "관광", 1.0, False, 35.8, 129.2),
            SpotCandidate("2", "NaN 장소", "관광", 1.0, False, float("nan"), 129.2),
            SpotCandidate("3", "범위 밖 장소", "관광", 1.0, False, 91.0, 129.2),
        ]
    )

    message = build_judge_user_message(ctx, "타지", make_course())

    assert "유효 장소 | 관광 | 좌표 35.8, 129.2 | 실외" in message
    assert "NaN 장소 | 관광 | 좌표 미확인 | 실외" in message
    assert "범위 밖 장소 | 관광 | 좌표 미확인 | 실외" in message
    assert "nan, 129.2" not in message
    assert "91.0, 129.2" not in message


def test_judge_prompt_keeps_actual_schedule_origin_separate_and_duration_minutes_is_ssot():
    ctx = make_ctx(
        duration_minutes=90,
        duration_hours=4,
        duration_hours_min=None,
        schedule_origin_lat=35.7,
        schedule_origin_lng=129.1,
    )

    message = build_judge_user_message(ctx, "타지", make_course())

    assert "검색 중심 위치: 위도 35.8562, 경도 129.2247" in message
    assert "일정 출발지: 위도 35.7, 경도 129.1" in message
    assert "가용 시간: 90분 (1.5시간)" in message
    assert "가용 시간: 4시간" not in message


def test_judge_prompt_does_not_present_search_center_as_missing_actual_gps_origin():
    message = build_judge_user_message(make_ctx(), "타지", make_course())

    assert "검색 중심 위치: 위도 35.8562, 경도 129.2247" in message
    assert "일정 출발지: 실제 GPS 미확인" in message
    assert "레거시 계산은 검색 중심을 사용할 수 있음" in message


def test_judge_prompt_marks_missing_schedule_origin_and_includes_server_time_budget_limits():
    ctx = make_ctx(schedule_origin_lat=0.0, schedule_origin_lng=0.0)
    course = make_course(
        time_budget=TimeBudgetSummary(
            budget_min=240,
            travel_min=75.5,
            stay_min=120,
            total_min=195.5,
            status="within_budget",
            note="좌표 기반 예상치입니다.",
            includes_initial_travel=True,
            includes_return_travel=False,
        )
    )

    message = build_judge_user_message(ctx, "타지", course)

    assert "가용 시간: 3~5시간" in message
    assert "일정 출발지: 미확인" in message
    assert "시간 예산 검증: within_budget" in message
    assert "예산 240분 | 예상 이동 75.5분 | 체류 120분 | 예상 합계 195.5분" in message
    assert "좌표 기반 예상치이며 실제 도로·교통 시간은 아님" in message
    assert "첫 이동 포함, 마지막 장소에서의 복귀 이동 제외" in message


def test_judge_prompt_keeps_unverified_missing_times_and_server_reason():
    course = make_course(
        time_budget=TimeBudgetSummary(
            budget_min=90,
            travel_min=None,
            stay_min=60,
            total_min=None,
            status="unverified",
            note="후보 또는 출발지 좌표가 없어 검증하지 못했습니다.",
        )
    )

    message = build_judge_user_message(make_ctx(), "타지", course)

    assert "시간 예산 검증: unverified" in message
    assert "예산 90분 | 예상 이동 미확인 | 체류 60분 | 예상 합계 미확인" in message
    assert "예상 이동 0분" not in message
    assert "예상 합계 0분" not in message
    assert "서버 메모: 후보 또는 출발지 좌표가 없어 검증하지 못했습니다." in message


def test_judge_prompt_keeps_no_feasible_status_and_reason_for_empty_course():
    course = make_course(
        recommended_places=[],
        time_budget=TimeBudgetSummary(
            budget_min=60,
            travel_min=None,
            stay_min=0,
            total_min=None,
            status="no_feasible",
            note="가용 시간 안에 가능한 코스가 없습니다.",
        ),
    )

    message = build_judge_user_message(make_ctx(), "타지", course)

    assert "시간 예산 검증: no_feasible" in message
    assert "서버 메모: 가용 시간 안에 가능한 코스가 없습니다." in message
    assert message.endswith("recommended_places:")
    assert "visit_sequence=" not in message


def test_judge_prompt_version_is_v11():
    assert JUDGE_PROMPT_VERSION == "v1.1"
