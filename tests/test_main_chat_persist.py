"""
/api/chat 스트리밍-저장 분리 회귀 테스트 — #31 PM 리뷰 10번 대응.

[문제] _session_courses 저장이 event_generator() 안에 있었다. 제너레이터는 소비자가
다음 값을 요청해야만 다음 줄이 실행되므로, 프론트가 스트림을 중간에 끊거나 사용자가
새로고침하면 저장 코드에 도달하지 못했다 → 그 세션은 /api/chat/revise가 계속 404.

[검증 전략] TestClient로 실제 TCP 연결 끊김을 재현하는 건 불안정하다. 대신 이 버그의
본질("소비자가 안 읽어도 생산자는 끝까지 돈다")을 직접 재현한다 — 생산자
_stream_and_persist()를 호출하면서 큐를 단 한 번도 읽지 않고, 그래도 _session_courses가
채워지는지 본다. 큐를 안 읽는 것 = 클라이언트가 스트림을 안 받는 것과 같은 상황이다.

LLM 계층(ChatOpenAI · RunnableWithMessageHistory)은 monkeypatch로 대체한다 — 여기서
확인하려는 건 "저장이 스트림 소비와 무관하게 끝나는가"이지 LLM 응답 품질이 아니다.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

import main
from traveler.rag.types import ContextData, SpotCandidate

# 스트리밍으로 흘려보낼 정상 코스 JSON을 청크로 쪼갠 것 (실제 gpt-4o가 보내는 모양)
_COURSE_JSON = json.dumps(
    {
        "bot_message": "불국사부터 보시죠.",
        "recommended_places": [
            {"place_id": "126508", "visit_sequence": 1, "place_name": "불국사"},
            {"place_id": "126509", "visit_sequence": 2, "place_name": "첨성대"},
        ],
    },
    ensure_ascii=False,
)
_CHUNKS = [_COURSE_JSON[i:i + 20] for i in range(0, len(_COURSE_JSON), 20)]


def _dummy_ctx() -> ContextData:
    return ContextData(
        user_query="경주 여행",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[
            SpotCandidate("126508", "불국사", "관광", 20.0, False),
            SpotCandidate("126509", "첨성대", "관광", 22.0, False),
        ],
    )


def _patch_llm_layer(monkeypatch, chunks):
    """ChatOpenAI(네트워크·API키 필요)와 체인 실행부를 가짜로 교체한다."""

    class _FakeEngine:
        async def astream(self, _inputs, config=None):
            for chunk in chunks:
                yield chunk

    # rag_prompt | llm | StrOutputParser() 조립만 통과하면 되므로 Runnable이기만 하면 된다.
    monkeypatch.setattr(main, "ChatOpenAI", lambda **kwargs: RunnableLambda(lambda x: ""))
    monkeypatch.setattr(main, "RunnableWithMessageHistory", lambda *a, **k: _FakeEngine())


@pytest.mark.asyncio
async def test_persist_runs_even_if_queue_never_consumed(monkeypatch):
    """10번 핵심: 큐를 한 번도 읽지 않아도 _session_courses가 채워져야 한다."""
    _patch_llm_layer(monkeypatch, _CHUNKS)

    session_id = "test-chat-persist-no-consumer"
    main._session_courses.pop(session_id, None)
    ctx = _dummy_ctx()
    queue: asyncio.Queue = asyncio.Queue()

    # ★ 큐를 소비하는 코드가 어디에도 없다 — 클라이언트가 스트림을 끊은 상황과 동일
    await main._stream_and_persist(queue, session_id, "근교", "컨텍스트", ctx)

    assert session_id in main._session_courses, "스트림 미소비 시 코스가 저장되지 않았다 (#31 10번 회귀)"
    course, saved_ctx, mode_kr = main._session_courses[session_id]
    assert [p.place_id for p in course.recommended_places] == ["126508", "126509"]
    assert saved_ctx is ctx
    assert mode_kr == "근교"

    main._session_courses.pop(session_id, None)


@pytest.mark.asyncio
async def test_producer_finishes_with_sentinel(monkeypatch):
    """소비자가 영원히 대기하지 않도록 마지막에 종료 신호(None)가 들어가야 한다."""
    _patch_llm_layer(monkeypatch, _CHUNKS)

    session_id = "test-chat-persist-sentinel"
    ctx = _dummy_ctx()
    queue: asyncio.Queue = asyncio.Queue()

    await main._stream_and_persist(queue, session_id, "근교", "컨텍스트", ctx)

    items = [queue.get_nowait() for _ in range(queue.qsize())]
    assert items[-1] is None
    # #13 변경: 이제 원시 JSON 청크가 아니라 BotMessageExtractor가 걸러낸
    # bot_message "디코딩된 텍스트"만 새로 드러날 때마다 큐에 들어간다. 이 고정
    # JSON(_COURSE_JSON)을 20자 단위로 잘라 넣으면 bot_message 값이 초반 청크
    # 안에서 다 드러나 텍스트 청크 2개만 생긴다(그 뒤는 JSON 문법뿐이라 내보낼 게
    # 없음) — 텍스트 청크 2개 + 최종 장소 청크 + 종료 신호.
    assert len(items) == 4

    main._session_courses.pop(session_id, None)


@pytest.mark.asyncio
async def test_llm_failure_still_emits_sentinel(monkeypatch):
    """LLM이 도중에 터져도 종료 신호는 들어가야 한다(소비자 무한 대기 방지)."""

    class _BoomEngine:
        async def astream(self, _inputs, config=None):
            yield "{"
            raise RuntimeError("LLM 연결 끊김")

    monkeypatch.setattr(main, "ChatOpenAI", lambda **kwargs: RunnableLambda(lambda x: ""))
    monkeypatch.setattr(main, "RunnableWithMessageHistory", lambda *a, **k: _BoomEngine())

    session_id = "test-chat-persist-boom"
    main._session_courses.pop(session_id, None)
    ctx = _dummy_ctx()
    queue: asyncio.Queue = asyncio.Queue()

    await main._stream_and_persist(queue, session_id, "근교", "컨텍스트", ctx)

    items = [queue.get_nowait() for _ in range(queue.qsize())]
    assert items[-1] is None


def test_chat_endpoint_streams_chunks_in_order(monkeypatch):
    """정상 경로 회귀: SSE 청크가 순서대로 오고, 마지막 청크가 추천 장소를 담는다."""
    _patch_llm_layer(monkeypatch, _CHUNKS)

    ctx = _dummy_ctx()

    async def fake_get_weather(lat, lon):
        return "맑음"

    async def fake_get_visited(session_id):
        return set()

    async def fake_get_prefs(session_id):
        return {}

    monkeypatch.setattr(main, "get_weather", fake_get_weather)
    monkeypatch.setattr(main, "get_visited", fake_get_visited)
    monkeypatch.setattr(main, "get_prefs", fake_get_prefs)
    monkeypatch.setattr(main, "extract_prefs_from_message", lambda msg: {})
    monkeypatch.setattr(main, "rag_get_context", lambda *a, **k: (ctx, None))

    session_id = "test-chat-stream-order"
    main._session_courses.pop(session_id, None)

    with TestClient(main.app) as client:
        res = client.get("/api/chat", params={"user_message": "경주 여행", "session_id": session_id})

    assert res.status_code == 200
    payloads = [
        json.loads(line[len("data: "):])
        for line in res.text.splitlines()
        if line.startswith("data: ")
    ]
    # #13: 텍스트 청크들이 이어 붙으면 원본 JSON이 아니라 bot_message
    # 필드의 "디코딩된 텍스트"만 복원돼야 한다 (JSON 문법 조각이 새어나가면 안 됨).
    assert "".join(p["bot_message"] for p in payloads) == json.loads(_COURSE_JSON)["bot_message"]
    assert all("{" not in p["bot_message"] for p in payloads)
    # 마지막 청크만 recommended_places를 담는다 (프론트가 의존하는 계약 — 변경 금지)
    assert payloads[-1]["bot_message"] == ""
    assert payloads[-1]["recommended_places"] == [
        {"place_id": "126508", "visit_sequence": 1, "place_name": "불국사", "duration_min": None, "lat": 0.0, "lng": 0.0},
        {"place_id": "126509", "visit_sequence": 2, "place_name": "첨성대", "duration_min": None, "lat": 0.0, "lng": 0.0},
    ]
    assert session_id in main._session_courses

    main._session_courses.pop(session_id, None)
