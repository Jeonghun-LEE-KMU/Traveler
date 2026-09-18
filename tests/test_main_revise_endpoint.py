"""
/api/chat/revise 엔드포인트 회귀 테스트 — #31 PM 리뷰 5·6·9·11번 대응.

[5번] GET → POST + JSON body로 전환: _session_courses를 바꾸는 부수효과가 있는
호출이라 GET(안전·재시도 가능 전제)에 두면 프리페치·재시도로 의도치 않게 코스가
바뀔 수 있다.
[6번] 응답에 place_name 포함: RecommendedPlaceModel에 이미 있는 값을 안 보내면
프론트가 place_id로 다시 조회해야 한다.
[9번] revise 후 memory_store(대화 이력)에 사용자 발화·응답을 남긴다: 안 남기면
그 다음 /api/chat 호출에서 LLM이 코스가 이미 수정된 사실을 모른 채 답한다.
[11번] 엔드포인트가 revise_course()에 최신 visited 집합을 넘기는지 확인: ctx는
/api/chat 시점에 고정되므로, 그 뒤 방문 처리된 장소를 걸러내려면 세션의 최신
visited를 매 revise 호출마다 다시 조회해서 넘겨야 한다.

TestClient로 실제 HTTP 계층(메서드·body 파싱·상태 코드)까지 검증한다.
revise_course()는 LLM을 호출하므로 monkeypatch로 대체 — 이 테스트가 확인하려는
건 "엔드포인트가 revise_course()의 결과를 올바른 HTTP 모양으로 감싸는가"이지
LLM 응답 품질이 아니다.
"""

import pytest
from fastapi.testclient import TestClient

import main
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel

client = TestClient(main.app)


def _dummy_ctx() -> ContextData:
    return ContextData(
        user_query="경주 여행",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[SpotCandidate("1", "불국사", "관광", 20.0, False)],
    )


def test_revise_post_returns_lat_lng_from_candidate_spots(monkeypatch):
    """#27 나머지: 응답의 lat/lng이 context_data.candidate_spots에서 조회된 값인지 확인."""
    session_id = "test-revise-latlng"
    ctx = ContextData(
        user_query="경주 여행",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[SpotCandidate("1", "불국사", "관광", 20.0, False, lat=35.79, lng=129.33)],
    )
    course = StreamingCourseModel(
        bot_message="코스",
        recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사")],
    )
    main._session_courses[session_id] = (course, ctx, "근교")

    async def fake_revise_course(user_message, course, context_data, travel_mode, visited=None):
        return StreamingCourseModel(
            bot_message="수정했어요.",
            recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사")],
        )

    monkeypatch.setattr(main, "revise_course", fake_revise_course)

    res = client.post("/api/chat/revise", json={"user_message": "바꿔줘", "session_id": session_id})

    assert res.status_code == 200
    place = res.json()["recommended_places"][0]
    assert place["lat"] == 35.79
    assert place["lng"] == 129.33


def test_revise_post_returns_place_name(monkeypatch):
    session_id = "test-revise-post"
    course = StreamingCourseModel(
        bot_message="코스",
        recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사")],
    )
    main._session_courses[session_id] = (course, _dummy_ctx(), "근교")

    async def fake_revise_course(user_message, course, context_data, travel_mode, visited=None):
        return StreamingCourseModel(
            bot_message="수정했어요.",
            recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="첨성대")],
        )

    monkeypatch.setattr(main, "revise_course", fake_revise_course)

    res = client.post("/api/chat/revise", json={"user_message": "바꿔줘", "session_id": session_id})

    assert res.status_code == 200
    body = res.json()
    assert body["recommended_places"][0]["place_name"] == "첨성대"


def test_revise_post_appends_to_chat_history(monkeypatch):
    """9번: revise 이후 memory_store에 사용자 발화·bot_message가 쌓이는지 확인."""
    session_id = "test-revise-history"
    course = StreamingCourseModel(
        bot_message="코스",
        recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사")],
    )
    main._session_courses[session_id] = (course, _dummy_ctx(), "근교")
    main.memory_store.pop(session_id, None)  # 이전 테스트 잔여 이력 제거

    async def fake_revise_course(user_message, course, context_data, travel_mode, visited=None):
        return StreamingCourseModel(
            bot_message="불국사를 뺐어요.",
            recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사")],
        )

    monkeypatch.setattr(main, "revise_course", fake_revise_course)

    client.post("/api/chat/revise", json={"user_message": "불국사 빼줘", "session_id": session_id})

    history_messages = main.memory_store[session_id].messages
    assert len(history_messages) == 2
    assert history_messages[0].content == "불국사 빼줘"
    assert history_messages[1].content == "불국사를 뺐어요."


def test_revise_post_passes_latest_visited_to_revise_course(monkeypatch):
    """11번: 엔드포인트가 세션의 최신 visited를 조회해 revise_course에 넘기는지 확인."""
    session_id = "test-revise-visited"
    course = StreamingCourseModel(
        bot_message="코스",
        recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사")],
    )
    main._session_courses[session_id] = (course, _dummy_ctx(), "근교")

    async def fake_get_visited(sid):
        assert sid == session_id
        return {"2", "3"}

    captured = {}

    async def fake_revise_course(user_message, course, context_data, travel_mode, visited=None):
        captured["visited"] = visited
        return StreamingCourseModel(
            bot_message="수정했어요.",
            recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사")],
        )

    monkeypatch.setattr(main, "get_visited", fake_get_visited)
    monkeypatch.setattr(main, "revise_course", fake_revise_course)

    client.post("/api/chat/revise", json={"user_message": "바꿔줘", "session_id": session_id})

    assert captured["visited"] == {"2", "3"}


def test_revise_get_no_longer_allowed():
    """5번: GET이 더 이상 이 경로를 처리하지 않는지(405) 확인."""
    res = client.get("/api/chat/revise", params={"user_message": "바꿔줘"})
    assert res.status_code == 405


def test_revise_post_unknown_session_404():
    res = client.post(
        "/api/chat/revise",
        json={"user_message": "바꿔줘", "session_id": "no-such-session"},
    )
    assert res.status_code == 404
