"""
course_generator 모듈 단위 테스트.

WHY 이 테스트가 필요한가:
  build_context_message()는 이수연↔정훈 팀 인터페이스의 핵심 변환 함수.
  RAG 출력(ContextData)이 GPT 프롬프트에 올바르게 주입되는지 보장해야 한다.
  특히 candidate_spots가 place_id와 함께 포함되는지가 핵심
  — 없으면 GPT가 존재하지 않는 place_id를 생성해 SQLite 조회 실패.

파이프라인 위치:
  ContextData → ★ build_context_message() → GPT user 메시지 → 코스 생성
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from traveler.core.course_generator import generate_course
from traveler.core.prompt import build_context_message
from traveler.rag.types import ContextData, SpotCandidate


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

def minimal_ctx(**overrides) -> ContextData:
    """테스트용 최소 ContextData. 필수 필드만 채운 기본값."""
    defaults = dict(
        user_query="경주 여행 추천해줘",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[],
    )
    defaults.update(overrides)
    return ContextData(**defaults)


def sample_spot(place_id: str = "1") -> SpotCandidate:
    """테스트용 SpotCandidate 하나."""
    return SpotCandidate(
        place_id=place_id,
        place_name="불국사",
        category="관광",
        distance_min=20.0,
        indoor=False,
    )


def _chunk(content: str | None = None, usage=None) -> SimpleNamespace:
    """
    OpenAI 스트리밍 청크를 흉내내는 가짜 객체.

    WHY: 실제 OpenAI API를 호출하지 않고 스트리밍 루프(course_generator.py)를
    테스트하기 위함. content가 있으면 choices[0].delta.content로,
    usage 청크(마지막)는 choices=[]로 — 실제 API 동작과 동일한 모양.
    """
    if content is not None:
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
            usage=usage,
        )
    return SimpleNamespace(choices=[], usage=usage)


async def _fake_stream(chunks: list[SimpleNamespace]):
    """가짜 비동기 스트림 — course_generator.py의 `async for chunk in stream`이 그대로 동작."""
    for chunk in chunks:
        yield chunk


def _mock_client_with_chunks(mock_get_client, chunks: list[SimpleNamespace]) -> None:
    """get_client() 패치 대상에 가짜 chat.completions.create를 연결."""
    fake_usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks = chunks + [_chunk(usage=fake_usage)]   # 실제 API의 마지막 usage 청크
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_fake_stream(chunks))
            )
        )
    )
    mock_get_client.return_value = mock_client


# ── Test 1: 필수 필드가 메시지에 포함되는지 ───────────────────────────────────

def test_required_fields_included():
    """
    user_query, weather, duration_hours, transport, companion이
    build_context_message() 출력에 반드시 포함되는지 확인.

    WHY: 이 필드들이 누락되면 GPT가 날씨 제약이나 동행자 특성을 반영하지 못함.
    """
    ctx = minimal_ctx(
        user_query="조용한 여행지 추천",
        weather="비",
        duration_hours=4,
        transport="대중교통",
        companion="혼자",
    )
    msg = build_context_message(ctx)

    assert "조용한 여행지 추천" in msg
    assert "비" in msg
    assert "4시간" in msg
    assert "대중교통" in msg
    assert "혼자" in msg


# ── Test 2: candidate_spots가 place_id와 함께 포함되는지 ─────────────────────

def test_candidate_spots_with_place_id_included():
    """
    candidate_spots가 있으면 place_id와 장소명이 메시지에 포함되는지 확인.

    WHY: GPT가 "이 목록에서 골라라"는 제약을 받아야
    SQLite에 존재하는 place_id만 반환한다.
    place_id가 메시지에 없으면 GPT가 임의 번호를 생성해 DB 조회 실패.
    """
    spot = sample_spot(place_id="42")
    ctx = minimal_ctx(candidate_spots=[spot])
    msg = build_context_message(ctx)

    assert "place_id=42" in msg
    assert "불국사" in msg


# ── Test 3: candidate_spots 없어도 에러 없이 동작 (Graceful Degradation) ──────

def test_empty_candidate_spots_no_error():
    """
    candidate_spots=[]이면 에러 없이 메시지가 생성되는지 확인.

    WHY: Graceful Degradation — RAG가 후보를 못 찾아도 시스템이 멈추면 안 됨.
    후보 없이 GPT에게 자유 생성을 맡기는 게 서비스 중단보다 낫다.
    단, 이때 GPT가 임의 place_id를 만들 수 있으므로 품질은 낮아질 수 있음.
    """
    ctx = minimal_ctx(candidate_spots=[])
    msg = build_context_message(ctx)   # 에러 없이 반환되어야 함
    assert isinstance(msg, str)
    assert len(msg) > 0


# ── Test 4: optional 필드(mood, route_summary)는 값 있을 때만 포함 ────────────

def test_optional_fields_included_only_when_set():
    """
    mood, route_summary는 값이 있을 때만 메시지에 포함되는지 확인.

    WHY: Graceful Degradation — Optional 필드가 빈 문자열이면 GPT에게 불필요한
    "분위기: " 같은 빈 줄이 전달되어 프롬프트 품질이 낮아짐.
    """
    # mood가 없는 경우
    ctx_no_mood = minimal_ctx(mood="")
    msg_no_mood = build_context_message(ctx_no_mood)
    assert "분위기" not in msg_no_mood

    # mood가 있는 경우
    ctx_with_mood = minimal_ctx(mood="조용하고 힐링되는")
    msg_with_mood = build_context_message(ctx_with_mood)
    assert "조용하고 힐링되는" in msg_with_mood


# ── Test 5: 실내 여부(indoor)가 spot 정보에 포함되는지 ───────────────────────

def test_indoor_flag_included_in_spot_info():
    """
    SpotCandidate.indoor 값이 "실내"/"실외"로 변환되어 메시지에 포함되는지 확인.

    WHY: 날씨가 "비"일 때 GPT가 실외 장소를 피하려면
    각 후보의 실내 여부를 알아야 한다. 이 정보가 없으면 날씨 제약 반영 불가.
    """
    indoor_spot = SpotCandidate("1", "박물관", "관광", 10.0, True)
    outdoor_spot = SpotCandidate("2", "해수욕장", "관광", 15.0, False)
    ctx = minimal_ctx(candidate_spots=[indoor_spot, outdoor_spot])
    msg = build_context_message(ctx)

    assert "실내" in msg
    assert "실외" in msg


# ── Test 6: 정상 JSON 스트리밍 → 검증 통과 (회귀 방지용 happy path) ───────────

@patch("traveler.core.course_generator.get_client")
async def test_generate_course_valid_json_parses_successfully(mock_get_client):
    """
    GPT가 스키마에 맞는 JSON을 스트리밍하면 그대로 파싱되는지 확인.

    WHY: fallback 로직을 추가하면서 정상 흐름을 깨뜨리지 않았는지 보장하는
    회귀 테스트. generate_course() 전체를 처음으로 테스트하는 케이스.
    """
    valid_json = (
        '{"bot_message": "경주 코스를 추천해요", '
        '"recommended_places": [{"place_id": "1", "visit_sequence": 1}]}'
    )
    chunks = [_chunk(valid_json[:20]), _chunk(valid_json[20:])]
    _mock_client_with_chunks(mock_get_client, chunks)

    ctx = minimal_ctx()
    course = await generate_course(ctx)

    assert course.bot_message == "경주 코스를 추천해요"
    assert course.recommended_places[0].place_id == "1"


# ── Test 7: 깨진 JSON → ValidationError → fallback 반환 ──────────────────────

@patch("traveler.core.course_generator.get_client")
async def test_generate_course_malformed_json_returns_fallback(mock_get_client):
    """
    GPT가 깨진(잘리거나 스키마에 안 맞는) JSON을 주면 예외를 던지지 않고
    안전한 fallback StreamingCourseModel을 반환하는지 확인.

    WHY: "API 오류나 JSON 파싱 실패 시 데모가 멈추지 않아야 한다"는 요구사항.
    이 테스트가 없으면 fallback 코드가 조용히 죽은 코드가 될 위험이 있음.
    """
    broken_json = '{"bot_message": "경주 코스를 추천'   # 닫는 괄호 없이 잘림
    chunks = [_chunk(broken_json)]
    _mock_client_with_chunks(mock_get_client, chunks)

    ctx = minimal_ctx()   # candidate_spots=[] → fallback place_id는 1
    course = await generate_course(ctx)

    assert "문제가 있었" in course.bot_message
    assert course.recommended_places[0].place_id == "1"


# ── Test 8: 계절(L2)이 today 주입값 기준으로 메시지에 포함되는지 ──────────────

def test_season_included_with_injected_today():
    """
    today를 주입하면 그 날짜 기준 계절이 메시지에 포함되는지 확인.

    WHY: 계절은 ContextData가 아니라 build_context_message()가 직접 계산한다
    (get_season() 참조). today를 안 고정하면 테스트가 실행 시점의 실제 날짜에
    좌우되므로, 여기서는 1월로 고정해 "겨울"이 나오는지를 검증한다.
    """
    ctx = minimal_ctx()
    msg = build_context_message(ctx, today=date(2026, 1, 15))

    assert "현재 계절: 겨울" in msg


# ── Test 9: API 호출 자체 실패(타임아웃/네트워크) → fallback 반환 ────────────────

@patch("traveler.core.course_generator.get_client")
async def test_generate_course_api_error_returns_fallback(mock_get_client):
    """
    OpenAI API 호출 자체가 예외(타임아웃·네트워크 끊김·rate limit 등)를 던져도
    예외를 전파하지 않고 안전한 fallback StreamingCourseModel을 반환하는지 확인.

    WHY: "API 오류 시 graceful degradation"(MVP1 P0 요구사항)은 JSON 파싱뿐 아니라
    API 호출 자체가 실패한 경우에도 보장되어야 한다. ValidationError만 잡으면
    타임아웃 같은 비(非)-ValidationError 예외가 generate_course()를 그대로 터뜨려
    데모가 중단된다. 이 테스트는 API 호출 경로도 fallback으로 흡수되는지 검증한다.
    """
    # create()가 호출되는 순간 예외를 던지도록 side_effect 설정
    # (malformed-JSON 테스트는 return_value로 정상 스트림을 주지만, 여기선 호출 자체가 실패)
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=Exception("simulated API timeout"))
            )
        )
    )
    mock_get_client.return_value = mock_client

    ctx = minimal_ctx()   # candidate_spots=[] → fallback place_id는 1
    course = await generate_course(ctx)

    assert "문제가 있었" in course.bot_message
    assert course.recommended_places[0].place_id == "1"


# ── Test 10: companion이 있으면 "동행자:" 줄이 메시지에 포함 ─────────────────

def test_companion_line_included_when_set():
    """
    companion="혼자" → "동행자: 혼자" 줄이 메시지에 포함되는지 확인.

    WHY: 개정된 정책에서 "혼자"는 빈 문자열이 아니라 가드(if ctx.companion:)를
    통과하는 실제 라벨이다. 명시적으로 혼자라고 말한 정보를 GPT에 전달해야 한다.
    """
    ctx = minimal_ctx(companion="혼자")
    msg = build_context_message(ctx)

    assert "동행자" in msg
    assert "혼자" in msg


# ── Test 11: companion이 빈 문자열이면 "동행자" 줄이 생략 ────────────────────

def test_companion_line_omitted_when_empty():
    """
    companion="" (동승자 미언급) → "동행자" 줄이 메시지에서 생략되는지 확인.

    WHY: 미언급("")은 "혼자라고 단정"하면 안 되는 정보 없음 상태다. 가드가
    빈 문자열을 걸러 "동행자" 줄 자체를 빼야 GPT에 솔로 편향을 주지 않는다
    (mood/route_summary의 기존 Graceful Degradation 패턴과 동일).
    """
    ctx = minimal_ctx(companion="")
    msg = build_context_message(ctx)

    assert "동행자" not in msg
