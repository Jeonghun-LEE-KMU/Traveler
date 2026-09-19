"""생성·SSE·수정이 동일한 가용시간 검증을 거치는지 확인한다 (외부 I/O 없음)."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from tests.test_main_chat_persist import _patch_llm_layer
from traveler.core.course_generator import parse_course
from traveler.core.prompt import get_system_prompt
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import StreamingCourseModel


def _context(*, hours=4, count=4, step_km=20):
    # WHY: 북쪽으로 20km씩 이동하면 기존 30km/h 추정에서 구간당 약 40분이다.
    return ContextData(
        user_query="여행 코스 추천해줘", current_lat=35.0, current_lng=129.0,
        weather="맑음", duration_hours=hours, duration_hours_min=None,
        candidate_spots=[
            SpotCandidate(str(i), f"장소 {i}", "관광", 0, False,
                          lat=35.0 + i * step_km / 111.195, lng=129.0)
            for i in range(1, count + 1)
        ],
    )


def _raw(count=4, stay=30):
    return json.dumps({
        "bot_message": "여행 코스를 추천합니다.",
        "recommended_places": [
            {"place_id": str(i), "visit_sequence": i, "duration_min": stay}
            for i in range(1, count + 1)
        ],
    }, ensure_ascii=False)


def test_parse_enforces_travel_plus_stay_budget():
    course, fallback = parse_course(_raw(), _context())
    assert not fallback
    assert len(course.recommended_places) == 3
    assert course.time_budget.status == "within_budget"
    assert course.time_budget.total_min == pytest.approx(210, abs=2)
    assert course.time_budget.total_min <= course.time_budget.budget_min


def test_parse_fills_minimum_only_from_feasible_candidates():
    course, _ = parse_course(_raw(count=1), _context(step_km=1))
    assert len(course.recommended_places) == 3
    assert {p.place_id for p in course.recommended_places} <= {"1", "2", "3", "4"}


def test_no_feasible_course_is_explicit_not_over_budget():
    course, _ = parse_course(_raw(count=1, stay=120), _context(hours=1, count=1))
    assert course.recommended_places == []
    assert course.time_budget.status == "no_feasible"
    assert "시간" in course.bot_message


def test_empty_model_requires_explicit_server_budget_result():
    with pytest.raises(ValidationError):
        StreamingCourseModel(bot_message="빈 모델", recommended_places=[])
    result = StreamingCourseModel(
        bot_message="가용시간에 맞는 장소가 없습니다.", recommended_places=[],
        time_budget={"budget_min": 60, "travel_min": 0, "stay_min": 0,
                     "total_min": 0, "status": "no_feasible", "note": "시간 부족"},
    )
    assert result.recommended_places == []


@pytest.mark.asyncio
async def test_sse_final_payload_and_saved_course_use_same_schedule(monkeypatch):
    _patch_llm_layer(monkeypatch, [_raw()])
    queue = asyncio.Queue()
    sid = "schedule-integration-stream"
    try:
        await main._stream_and_persist(queue, sid, "타지", "문맥", _context())
        payloads = []
        while not queue.empty():
            chunk = queue.get_nowait()
            if chunk is not None:
                payloads.append(json.loads(chunk.removeprefix("data: ")))
        assert len(payloads[-1]["recommended_places"]) == 3
        assert payloads[-1]["time_budget"]["total_min"] <= 240
        assert "예상" in "".join(p["bot_message"] for p in payloads)
        assert len(main._session_courses[sid][0].recommended_places) == 3
    finally:
        main._session_courses.pop(sid, None)


@pytest.mark.asyncio
async def test_legacy_candidate_id_keeps_coordinates_in_final_sse(monkeypatch):
    ctx = _context(count=1, step_km=1)
    ctx.candidate_spots[0].place_id = 1
    _patch_llm_layer(monkeypatch, [_raw(count=1)])
    queue = asyncio.Queue()
    sid = "schedule-legacy-id"
    try:
        await main._stream_and_persist(queue, sid, "타지", "문맥", ctx)
        chunks = [queue.get_nowait() for _ in range(queue.qsize())]
        final = json.loads(chunks[-2].removeprefix("data: "))
        assert final["recommended_places"][0]["lat"] == ctx.candidate_spots[0].lat
        assert final["recommended_places"][0]["lng"] == ctx.candidate_spots[0].lng
    finally:
        main._session_courses.pop(sid, None)


def test_revise_cannot_bypass_budget(monkeypatch):
    sid = "schedule-integration-revise"
    original = StreamingCourseModel.model_validate_json(_raw(count=1))
    main._session_courses[sid] = (original, _context(), "타지")

    async def fake_revise(*args, **kwargs):
        return StreamingCourseModel.model_validate_json(_raw())

    async def fake_visited(*args):
        return set()

    monkeypatch.setattr(main, "revise_course", fake_revise)
    monkeypatch.setattr(main, "get_visited", fake_visited)
    try:
        with TestClient(main.app) as client:
            response = client.post("/api/chat/revise", json={
                "user_message": "장소 추가해줘", "session_id": sid,
            })
        assert response.status_code == 200
        assert len(response.json()["recommended_places"]) == 3
        assert response.json()["time_budget"]["total_min"] <= 240
    finally:
        main._session_courses.pop(sid, None)
        main.memory_store.pop(sid, None)


def test_prompt_explains_time_priority_and_estimate_scope():
    prompt = get_system_prompt("타지")
    assert "출발지→첫 장소" in prompt
    assert "복귀" in prompt
    assert "예상 이동시간 + 체류시간" in prompt
    assert "3곳보다" in prompt


def test_revise_new_available_time_overrides_saved_budget(monkeypatch):
    sid = "schedule-revise-new-budget"
    ctx = _context(step_km=1)
    original = StreamingCourseModel.model_validate_json(_raw(count=3, stay=30))
    main._session_courses[sid] = (original, ctx, "타지")

    async def fake_revise(*args, **kwargs):
        return original

    async def fake_visited(*args):
        return set()

    monkeypatch.setattr(main, "revise_course", fake_revise)
    monkeypatch.setattr(main, "get_visited", fake_visited)
    try:
        with TestClient(main.app) as client:
            response = client.post("/api/chat/revise", json={
                "user_message": "이제 1시간만 가능해", "session_id": sid,
            })
        assert response.status_code == 200
        assert response.json()["time_budget"]["budget_min"] == 60
        assert response.json()["time_budget"]["total_min"] <= 60
        assert main._session_courses[sid][1].duration_hours == 1
        assert ctx.duration_hours == 4  # 기존 객체를 직접 변형하지 않는다.
    finally:
        main._session_courses.pop(sid, None)
        main.memory_store.pop(sid, None)


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/revise"])
def test_invalid_time_is_explained_not_silently_replaced_by_old_budget(endpoint, monkeypatch):
    sid = "schedule-fractional-budget"
    original = StreamingCourseModel.model_validate_json(_raw(count=1))
    main._session_courses[sid] = (original, _context(), "타지")
    _patch_llm_layer(monkeypatch, [_raw(count=1)])

    async def fake_weather(*args):
        return "맑음"

    async def fake_visited(*args):
        return set()

    async def fake_prefs(*args):
        return {}

    async def fake_revise(*args, **kwargs):
        return original

    monkeypatch.setattr(main, "get_weather", fake_weather)
    monkeypatch.setattr(main, "get_visited", fake_visited)
    monkeypatch.setattr(main, "get_prefs", fake_prefs)
    monkeypatch.setattr(main, "revise_course", fake_revise)
    monkeypatch.setattr(main, "rag_get_context", lambda *a, **k: (_context(), None))
    try:
        with TestClient(main.app) as client:
            payload = {"user_message": "0분만 가능해", "session_id": sid}
            response = (client.get(endpoint, params=payload) if endpoint == "/api/chat"
                        else client.post(endpoint, json=payload))
        if endpoint == '/api/chat':
            assert response.status_code == 200
            assert 'event: application_error' in response.text
            assert 'INVALID_REQUEST' in response.text
        else:
            assert response.status_code == 422
            assert '가용 시간' in response.json()['detail']
        assert main._session_courses[sid][1].duration_hours == 4
    finally:
        main._session_courses.pop(sid, None)
        main.memory_store.pop(sid, None)


@pytest.mark.parametrize("gps", [True, False])
def test_chat_keeps_real_origin_separate_from_region_center(monkeypatch, gps):
    ctx = _context()
    _patch_llm_layer(monkeypatch, [_raw()])

    async def weather(*args):
        return "맑음"

    async def visited(*args):
        return set()

    async def prefs(*args):
        return {}

    monkeypatch.setattr(main, "get_weather", weather)
    monkeypatch.setattr(main, "get_visited", visited)
    monkeypatch.setattr(main, "get_prefs", prefs)
    monkeypatch.setattr(main, "extract_prefs_from_message", lambda value: {})
    monkeypatch.setattr(main, "rag_get_context", lambda *a, **k: (ctx, None))
    params = {"user_message": "경주 여행", "session_id": "schedule-origin"}
    if gps:
        params.update(lat=37.5, lon=127.0)
    try:
        with TestClient(main.app) as client:
            response = client.get("/api/chat", params=params)
        assert response.status_code == 200
        assert ctx.schedule_origin_lat == (37.5 if gps else 0.0)
        assert ctx.schedule_origin_lng == (127.0 if gps else 0.0)
        assert ctx.current_lat == 35.0  # 검색 좌표를 다시 바꾸지 않는다.
    finally:
        main._session_courses.pop("schedule-origin", None)
