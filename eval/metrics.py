"""
코스 출력에 대한 결정론적 구조 메트릭 — Sprint 5 Task 3.

[설계 근거] docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §4
LLM 없이 규칙으로 100% 판정 가능한 항목만 다룬다("LLM as tiebreaker" 원칙).
이 메트릭이 FAIL이면 judge(Task 2)는 호출하지 않는다 — 구조가 깨진 코스의
주관적 품질을 채점하는 건 의미가 없고 비용만 든다.
"""

from pydantic import BaseModel, ValidationError

from traveler.rag.types import ContextData
from traveler.validation.schema import StreamingCourseModel

# course_generator._build_fallback()의 bot_message와 정확히 동일해야 한다.
# WHY: _build_fallback()도 스키마 검증을 통과하므로, 이 문자열 비교 없이는
# API가 전부 죽어도 "구조 검증 통과"로 오판된다.
FALLBACK_BOT_MESSAGE = "죄송해요, 지금 코스를 만드는 데 문제가 있었어요. 잠시 후 다시 시도해주세요."

# WHY 20% 여유: duration_hours는 사용자가 말한 가용 시간이고, 이동시간은
# distance_min(현재 위치 기준) * 2로 근사한 추정치라 약간의 초과는 정상 범위다.
TIME_BUDGET_SLACK = 1.2


class StructuralMetrics(BaseModel):
    json_valid: bool
    is_fallback: bool
    place_count: int
    place_count_ok: bool
    grounded: bool
    hallucinated_ids: list[int]
    estimated_total_min: int
    time_budget_ok: bool
    passed: bool


def _failed(**overrides: object) -> StructuralMetrics:
    base = dict(
        json_valid=False,
        is_fallback=False,
        place_count=0,
        place_count_ok=False,
        grounded=False,
        hallucinated_ids=[],
        estimated_total_min=0,
        time_budget_ok=False,
        passed=False,
    )
    base.update(overrides)
    return StructuralMetrics(**base)


def measure(ctx: ContextData, expectations: dict, raw_output: str) -> StructuralMetrics:
    """
    구조 메트릭을 측정한다. judge(Task 2) 호출 전 게이트로 쓰인다 — `passed`가
    False면 러너는 judge를 호출하지 않고 이 케이스를 채점 대상에서 제외한다.

    raw_output은 파싱된 객체가 아니라 GPT가 뱉은 원문 문자열이어야 한다.
    JSON 파싱/스키마 검증 실패 자체가 측정 대상이기 때문이다.
    """
    try:
        course = StreamingCourseModel.model_validate_json(raw_output)
    except (ValidationError, ValueError):
        return _failed()

    is_fallback = course.bot_message == FALLBACK_BOT_MESSAGE

    place_count = len(course.recommended_places)
    lo, hi = expectations.get("expect_places_range", [1, 10])
    place_count_ok = lo <= place_count <= hi

    valid_ids = {spot.place_id for spot in ctx.candidate_spots}
    hallucinated_ids = sorted(
        {p.place_id for p in course.recommended_places} - valid_ids
    )
    grounded = not hallucinated_ids

    distance_by_id = {spot.place_id: spot.distance_min for spot in ctx.candidate_spots}
    max_distance = max(
        (distance_by_id.get(p.place_id, 0.0) for p in course.recommended_places),
        default=0.0,
    )
    duration_sum = sum(p.duration_min or 0 for p in course.recommended_places)
    estimated_total_min = round(duration_sum + max_distance * 2)
    time_budget_min = ctx.duration_hours * 60 * TIME_BUDGET_SLACK
    time_budget_ok = estimated_total_min <= time_budget_min

    passed = not is_fallback and place_count_ok and grounded and time_budget_ok

    return StructuralMetrics(
        json_valid=True,
        is_fallback=is_fallback,
        place_count=place_count,
        place_count_ok=place_count_ok,
        grounded=grounded,
        hallucinated_ids=hallucinated_ids,
        estimated_total_min=estimated_total_min,
        time_budget_ok=time_budget_ok,
        passed=passed,
    )
