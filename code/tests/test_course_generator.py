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


def sample_spot(place_id: int = 1) -> SpotCandidate:
    """테스트용 SpotCandidate 하나."""
    return SpotCandidate(
        place_id=place_id,
        place_name="불국사",
        category="관광",
        distance_min=20.0,
        indoor=False,
    )


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
    spot = sample_spot(place_id=42)
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
    indoor_spot = SpotCandidate(1, "박물관", "관광", 10.0, True)
    outdoor_spot = SpotCandidate(2, "해수욕장", "관광", 15.0, False)
    ctx = minimal_ctx(candidate_spots=[indoor_spot, outdoor_spot])
    msg = build_context_message(ctx)

    assert "실내" in msg
    assert "실외" in msg
