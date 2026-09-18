"""
/api/chat: 지역명 없는 후속 요청에 이전 세션 지역을 이어받는지 확인 (2026-09-15).

[문제] "경주 코스 추천해줘" → 코스 생성 → "다른 코스 추천해줘"(지역명 없음)로 이어지면
detect_region_coords()가 None을 반환해 좌표만 이전 코스에서 재사용하고, retriever는
extract_region(새 메시지)가 None이라 지역 필터 없이 전국 풀에서 검색한다 — 경주 특화
필터/프롬프트가 날아간다. 또한 "다른 코스"를 원하는데 이전과 겹치는 장소가 다시 나올
수 있다.

[검증 전략] rag_get_context를 monkeypatch로 대체해 호출 인자를 캡처한다 — 이 테스트가
보는 건 "엔드포인트가 fallback_region/visited를 올바르게 계산해서 넘기는가"이지
retriever 내부 동작(별도 tests/test_region_carry.py에서 검증)이 아니다.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

import main
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel

_CHUNKS = ["{}"]  # LLM 응답 본문은 이 테스트의 관심사가 아니다 — 짧게 둔다


def _patch_llm_layer(monkeypatch):
    """ChatOpenAI(네트워크·API키 필요)와 체인 실행부를 가짜로 교체한다."""

    class _FakeEngine:
        async def astream(self, _inputs, config=None):
            for chunk in _CHUNKS:
                yield chunk

    monkeypatch.setattr(main, "ChatOpenAI", lambda **kwargs: RunnableLambda(lambda x: ""))
    monkeypatch.setattr(main, "RunnableWithMessageHistory", lambda *a, **k: _FakeEngine())


def _patch_common(monkeypatch, captured):
    """weather/visited/prefs/prefs추출을 고정값으로 대체하고, rag_get_context 호출을 캡처한다."""

    async def fake_get_weather(lat, lon):
        return "맑음"

    async def fake_get_visited(session_id):
        return set()

    async def fake_get_prefs(session_id):
        return {}

    def fake_rag_get_context(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        ctx = ContextData(
            user_query="맛집 추천해줘",
            current_lat=35.8562,
            current_lng=129.2247,
            weather="맑음",
            candidate_spots=[],
        )
        return ctx, None

    monkeypatch.setattr(main, "get_weather", fake_get_weather)
    monkeypatch.setattr(main, "get_visited", fake_get_visited)
    monkeypatch.setattr(main, "get_prefs", fake_get_prefs)
    monkeypatch.setattr(main, "extract_prefs_from_message", lambda msg: {})
    monkeypatch.setattr(main, "rag_get_context", fake_rag_get_context)


def _seed_prev_course(session_id: str) -> None:
    """place_id A/B를 담은 이전 코스 + region="경주" ContextData를 세션에 미리 채워둔다."""
    course = StreamingCourseModel(
        bot_message="이전 코스",
        recommended_places=[
            RecommendedPlaceModel(place_id="A", visit_sequence=1, place_name="장소A"),
            RecommendedPlaceModel(place_id="B", visit_sequence=2, place_name="장소B"),
        ],
    )
    ctx = ContextData(
        user_query="경주 코스 추천해줘",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[SpotCandidate("A", "장소A", "관광", 5.0, False)],
        region="경주",
    )
    main._session_courses[session_id] = (course, ctx, "근교")


@pytest.fixture
def session_id():
    sid = "test-region-carry"
    main._session_courses.pop(sid, None)
    yield sid
    main._session_courses.pop(sid, None)


def test_different_course_request_carries_region_and_excludes_previous(monkeypatch, session_id):
    """'다른 코스 추천해줘' → fallback_region="경주", visited에 이전 장소(A,B) 포함,
    add_visited는 호출되지 않는다(세션에 영구 기록하지 않음)."""
    _patch_llm_layer(monkeypatch)
    captured = {}
    _patch_common(monkeypatch, captured)
    add_visited_mock = None

    async def fail_if_called(*a, **k):
        raise AssertionError("add_visited가 호출되면 안 된다 — 이 제외는 이번 검색 한정")

    monkeypatch.setattr(main, "add_visited", fail_if_called)
    _seed_prev_course(session_id)

    with TestClient(main.app) as client:
        res = client.get(
            "/api/chat",
            params={"user_message": "다른 코스 추천해줘", "session_id": session_id},
        )

    assert res.status_code == 200
    assert captured["kwargs"]["fallback_region"] == "경주"
    visited_arg = captured["args"][4]
    assert {"A", "B"} <= visited_arg


def test_refine_request_carries_region_without_excluding_previous(monkeypatch, session_id):
    """'다른/새로운/말고' 없는 후속 요청(같은 코스를 다듬는 요청) → fallback_region은
    이어받지만 이전 장소(A,B)는 제외하지 않는다."""
    _patch_llm_layer(monkeypatch)
    captured = {}
    _patch_common(monkeypatch, captured)
    _seed_prev_course(session_id)

    with TestClient(main.app) as client:
        res = client.get(
            "/api/chat",
            params={"user_message": "비 오면 실내로 추천해줘", "session_id": session_id},
        )

    assert res.status_code == 200
    assert captured["kwargs"]["fallback_region"] == "경주"
    visited_arg = captured["args"][4]
    assert "A" not in visited_arg
    assert "B" not in visited_arg


def test_region_in_message_does_not_carry_previous_course(monkeypatch, session_id):
    """메시지에 지역명("부산")이 있으면 이전 코스에서 아무것도 이어받지 않는다."""
    _patch_llm_layer(monkeypatch)
    captured = {}
    _patch_common(monkeypatch, captured)
    _seed_prev_course(session_id)

    with TestClient(main.app) as client:
        res = client.get(
            "/api/chat",
            params={"user_message": "부산 코스 추천해줘", "session_id": session_id},
        )

    assert res.status_code == 200
    assert captured["kwargs"]["fallback_region"] is None
    visited_arg = captured["args"][4]
    assert "A" not in visited_arg
    assert "B" not in visited_arg


def test_no_previous_course_means_no_fallback_region(monkeypatch, session_id):
    """세션에 이전 코스가 없으면(첫 요청) fallback_region은 None이다."""
    _patch_llm_layer(monkeypatch)
    captured = {}
    _patch_common(monkeypatch, captured)
    # session_id fixture가 이미 _session_courses에서 지워둔 상태 — 이전 코스 없음

    with TestClient(main.app) as client:
        res = client.get(
            "/api/chat",
            params={"user_message": "맛집 추천해줘", "session_id": session_id},
        )

    assert res.status_code == 200
    assert captured["kwargs"]["fallback_region"] is None
