"""
eval/judge.py 단위 테스트 — Sprint 5 Task 2.

WHY 이 테스트가 필요한가:
  judge_course()는 실 API 키 없이도 파싱/검증/예외 처리 경로가 전부 검증돼야
  CI에서 안전하게 돈다. tests/test_course_generator.py의
  `@patch("...get_client")` + AsyncMock 모킹 패턴을 그대로 따르되, judge는
  스트리밍이 아닌 단발 호출이라 `_fake_stream`류는 필요 없다.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from eval.judge import JudgeResult, judge_course
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

def minimal_ctx(**overrides) -> ContextData:
    """테스트용 최소 ContextData. 필수 필드만 채운 기본값."""
    defaults = dict(
        user_query="경주 여행 추천해줘",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[
            SpotCandidate(
                place_id=1,
                place_name="불국사",
                category="관광",
                distance_min=20.0,
                indoor=False,
            )
        ],
    )
    defaults.update(overrides)
    return ContextData(**defaults)


def sample_course() -> StreamingCourseModel:
    """테스트용 StreamingCourseModel 하나."""
    return StreamingCourseModel(
        bot_message="경주의 고즈넉한 코스를 추천해요.",
        recommended_places=[
            RecommendedPlaceModel(place_id=1, visit_sequence=1, duration_min=60)
        ],
    )


def _mock_response(content: str) -> SimpleNamespace:
    """
    OpenAI chat.completions.create()의 단발(비스트리밍) 응답을 흉내내는 가짜 객체.

    WHY: judge_course()는 스트리밍이 아니라 단발 호출이라 course_generator
    테스트의 `_fake_stream`류가 필요 없다. response.choices[0].message.content
    형태만 흉내내면 충분하다.
    """
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


VALID_JUDGE_JSON = """\
{
  "condition_fit": {"reason": "날씨와 시간 제약을 모두 반영함", "score": 5},
  "mood_match": {"reason": "고즈넉한 분위기를 잘 살림", "score": 4},
  "route_sanity": {"reason": "동선이 자연스러움", "score": 4},
  "diversity": {"reason": "관광 위주로 다양성은 보통", "score": 3},
  "completeness": {"reason": "체류시간까지 명시됨", "score": 5}
}
"""


# ── Test (a): 정상 JSON 응답 → JudgeResult 파싱 + mean 계산 ──────────────────

@patch("eval.judge.get_client")
async def test_judge_course_valid_json_parses_successfully(mock_get_client):
    """
    정상 JSON 응답이면 JudgeResult가 5축 전부 1~5 정수로 파싱되고,
    mean 프로퍼티가 올바르게 계산되는지 확인.
    """
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_mock_response(VALID_JUDGE_JSON))
            )
        )
    )
    mock_get_client.return_value = mock_client

    result = await judge_course(minimal_ctx(), "타지", sample_course())

    assert isinstance(result, JudgeResult)
    assert result.condition_fit.score == 5
    assert result.mood_match.score == 4
    assert result.route_sanity.score == 4
    assert result.diversity.score == 3
    assert result.completeness.score == 5
    # mean = (5+4+4+3+5) / 5 = 4.2
    assert result.mean == 4.2


# ── Test (b): 깨진 JSON → None 반환(예외 안 던짐) ─────────────────────────────

@patch("eval.judge.get_client")
async def test_judge_course_malformed_json_returns_none(mock_get_client):
    """
    GPT가 깨진(문법 오류) JSON을 주면 예외를 던지지 않고 None을 반환하는지 확인.

    WHY: judge_course()는 채점 실패를 예외가 아니라 None으로 표현해야
    러너가 이 케이스를 skip 처리하고 계속 진행할 수 있다(Graceful Degradation).
    """
    broken_json = '{"condition_fit": {"reason": "..."'  # 닫는 괄호 없이 잘림
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_mock_response(broken_json))
            )
        )
    )
    mock_get_client.return_value = mock_client

    result = await judge_course(minimal_ctx(), "타지", sample_course())

    assert result is None


# ── Test (c): API 호출 자체가 예외를 던지는 상황 → None 반환 ─────────────────

@patch("eval.judge.get_client")
async def test_judge_course_api_error_returns_none(mock_get_client):
    """
    OpenAI API 호출 자체가 예외(타임아웃·네트워크 끊김·rate limit 등)를 던져도
    예외를 전파하지 않고 None을 반환하는지 확인.
    """
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=Exception("simulated API timeout"))
            )
        )
    )
    mock_get_client.return_value = mock_client

    result = await judge_course(minimal_ctx(), "타지", sample_course())

    assert result is None
