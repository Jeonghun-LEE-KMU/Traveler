"""
TRAVELER FastAPI 서버 - Sprint 3
"""

import asyncio
import json
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import (
    OPENAI_API_KEY,
    KORSERVICE_API_KEY,
    KMA_API_KEY,
    KORWITHSERVICE_API_KEY,
    TARRITETARSERVICE_API_KEY,
    MDCLTURSMSERVICE_API_KEY,
)
from traveler.api.weather import get_weather
from traveler.api.proxy import router as proxy_router
from traveler.core.edit_intent import revise_course
from traveler.core.prompt import get_system_prompt, build_context_message
from traveler.rag.retriever import get_context as rag_get_context
from traveler.rag.session import (
    add_visited,
    get_visited,
    get_prefs,
    update_prefs,
    clear_session,
)
from traveler.rag.preference import (
    extract_prefs_from_message,
    build_visited_context,
    extract_duration,
)
from traveler.rag.types import ContextData
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="TRAVELER_Mobility_RAG_Hub_Production")
app.include_router(proxy_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://traveler-map.kr",
        "https://www.traveler-map.kr",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 대화 히스토리 in-memory (Sprint 5에서 SQLite로 전환 예정)
memory_store: dict = {}

# WHY 메모리 딕셔너리인가(D11, Sprint 7 — 데모 전용 스캐폴딩): /api/chat/revise가
# "지금 코스"를 수정하려면 그 코스와 그걸 만든 candidate_spots가 필요한데, 지금까지는
# 세션에 visited·prefs만 저장했다(memory_store는 LangChain 대화 이력일 뿐 코스가
# 아니다). session.py(Redis+SQLite)로 정식 이관하는 대신 이 딕셔너리를 먼저 쓰는
# 이유: session.py는 이번 스프린트에 PR #30으로 개편됐다(이미 develop 머지) —
# 이 딕셔너리를 쓸 당시엔 지금 필드를 더하면 그쪽 머지 일정과 부딪혔다.
# 발표는 일단 작동하는 게 최우선이라 가장 작고
# 되돌리기 쉬운 방법을 먼저 쓴다(설계 spec §9). 서버 재시작하면 사라지고 워커
# 프로세스 간 공유도 안 된다 — 정식 영속화는 Sprint 8 이후 별도 과제.
_session_courses: dict[str, tuple[StreamingCourseModel, ContextData, str]] = {}

# #31 PM 리뷰 10번: 아래 _stream_and_persist()를 asyncio.create_task()로 띄우는데,
# 이벤트 루프는 Task를 약한 참조로만 들고 있다 — 어디서도 참조하지 않으면 실행 도중
# GC가 수거해 조용히 사라질 수 있다(asyncio 공식 문서에 명시된 함정).
# ⚠️ 주의: 이 set에 담아 강한 참조를 유지하고, 완료 콜백으로 다시 빼야 누수가 없다.
_persist_tasks: set[asyncio.Task] = set()


def get_session_history(s_id: str) -> ChatMessageHistory:
    if s_id not in memory_store:
        memory_store[s_id] = ChatMessageHistory()
    return memory_store[s_id]


RADIUS_BY_MODE = {
    "nearby":   30.0,
    "longdist": 150.0,
    "remote":   300.0,
}

MODE_TO_KR = {
    "nearby":   "근교",
    "longdist": "장거리",
    "remote":   "타지",
}

# #31 PM 리뷰 8번: revise(바꿔줘/추가)가 고를 수 있는 미사용 후보가 코스 크기(최대 10)
# 만큼만 있으면 금방 바닥난다. 후보 풀은 넉넉히 뽑아 _session_courses에 전부 저장하고,
# 초기 생성 프롬프트에는 CANDIDATE_PROMPT_LIMIT개만 보여준다(상위 랭킹순이라 슬라이싱이
# 곧 "최선 N개" — 토큰 비용과 GPT 선택 부담을 늘리지 않으면서 revise 풀만 넓힌다).
CANDIDATE_POOL_SIZE = 20
CANDIDATE_PROMPT_LIMIT = 10


async def _stream_and_persist(
        queue: asyncio.Queue,
        session_id: str,
        mode_kr: str,
        context_message: str,
        context_data: ContextData,
        prompt_spots: list,
) -> None:
    """
    LLM 스트리밍 + Pydantic 검증 + _session_courses 저장을 담당하는 "생산자".

    # WHY (#31 PM 리뷰 10번): 이 로직이 예전엔 event_generator() 안에 있었다. 제너레이터는
    #   소비자가 next()를 호출해야만 다음 줄이 실행되므로, 프론트가 스트림을 중간에 끊거나
    #   사용자가 새로고침하면 저장 코드까지 도달하지 못했다 → 그 세션은 /api/chat/revise가
    #   계속 404. 저장을 스트리밍 소비와 분리해, HTTP 연결과 무관하게 끝까지 실행되게 한다.
    # INPUT: 엔드포인트가 미리 계산한 RAG 컨텍스트 (LLM 호출 외 IO는 여기서 하지 않는다)
    # OUTPUT: queue에 SSE 문자열들 → 마지막에 None(종료 신호). 부수효과로 _session_courses 저장
    # COST: ~2,000 tokens · gpt-4o · 코스 1건당

    # ★ 핵심: put_nowait만 쓴다. queue는 무제한이라 절대 막히지 않으므로, 소비자
    #   (event_generator)가 클라이언트 연결 끊김으로 사라져도 이 함수는 끝까지 돈다.
    #   maxsize를 주면 소비자가 죽는 순간 여기서 영원히 대기하게 되어 버그가 되살아난다.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            streaming=True,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        # 정훈 system prompt (모드별 JSON 출력 강제)
        # ⚠️ 주의: 시스템 프롬프트를 ("system", 문자열) 튜플로 넣으면 LangChain이
        # f-string 템플릿으로 파싱을 시도한다. 프롬프트 안 JSON 예시의 중괄호
        # ({"bot_message": ...})가 변수 자리로 오인돼 항상
        # "Invalid format specifier in f-string template"로 죽었다(main.py
        # 자체 버그, Sprint 7 이전부터 존재 — git show 612074c:main.py로 확인).
        # SystemMessage로 감싸면 템플릿 파싱을 건너뛰고 문자열 그대로 쓴다.
        rag_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=get_system_prompt(mode_kr)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_message}"),
        ])

        rag_chain = rag_prompt | llm | StrOutputParser()

        final_rag_engine = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="user_message",
            history_messages_key="chat_history",
        )

        # 스트리밍하면서 full_text 누적
        # 대안: 동기 .stream()을 스레드에서 돌리고 call_soon_threadsafe로 큐에 밀어넣는
        # 방법도 있지만, 이 함수가 이미 코루틴이므로 .astream()이 그냥 더 짧다.
        # RunnableWithMessageHistory의 대화 이력 기록은 astream도 동일하게 수행한다.
        full_text = ""
        async for chunk_text in final_rag_engine.astream(
                {"user_message": context_message},
                config={"configurable": {"session_id": session_id}},
        ):
            full_text += chunk_text
            queue.put_nowait("data: {}\n\n".format(json.dumps({
                "bot_message": chunk_text,
                "recommended_places": [],
            }, ensure_ascii=False)))

        # 스트리밍 완료 후 Pydantic 검증
        try:
            validated = StreamingCourseModel.model_validate_json(full_text)
            spot_map = {s.place_id: s for s in context_data.candidate_spots}
            recommended = [
                {
                    "place_id": p.place_id,
                    "visit_sequence": p.visit_sequence,
                    "lat": spot_map[p.place_id].lat if p.place_id in spot_map else 0.0,
                    "lng": spot_map[p.place_id].lng if p.place_id in spot_map else 0.0,
                }
                for p in validated.recommended_places
            ]
            logger.info("[Chat] 검증 성공: %d개 장소", len(recommended))
            # D11: 후속 /api/chat/revise가 "지금 코스"를 찾을 수 있도록 저장.
            _session_courses[session_id] = (validated, context_data, mode_kr)
        except Exception as e:
            logger.warning("[Chat] 검증 실패, Pinecone fallback 사용: %s | full_text[:200]=%s", e, full_text[:200])
            recommended = [
                {"place_id": s.place_id, "visit_sequence": i + 1, "lat": s.lat, "lng": s.lng}
                for i, s in enumerate(prompt_spots)
            ]
            # D11: 폴백 응답도 형태만 맞으면 저장해 후속 수정 요청이 걸 게 있게
            # 한다. candidate_spots가 비어있거나 place_id 타입이 안 맞으면(#18)
            # 이 구성 자체가 실패할 수 있어 별도로 감싼다 — 실패해도 /api/chat의
            # 응답 자체(위 recommended)에는 영향 없음.
            try:
                fallback_course = StreamingCourseModel(
                    bot_message="",
                    recommended_places=[
                        RecommendedPlaceModel(
                            place_id=s.place_id,  # #30 이후 이미 str — str() 불필요
                            visit_sequence=i + 1,
                            place_name=s.place_name or None,  # #31 PM 리뷰 4번
                        )
                        for i, s in enumerate(prompt_spots)
                    ],
                )
                _session_courses[session_id] = (fallback_course, context_data, mode_kr)
            except Exception:
                logger.warning("[Chat] 폴백 코스를 세션에 저장하지 못함 — /api/chat/revise 불가")

        queue.put_nowait("data: {}\n\n".format(json.dumps({
            "bot_message": "",
            "recommended_places": recommended,
        }, ensure_ascii=False)))

    except Exception as e:
        # ⚠️ 주의: 백그라운드 Task라 예외가 밖으로 전파되지 않는다 — 여기서 안 남기면
        # 조용히 사라진다. 소비자는 아래 finally의 종료 신호로 정상 종료된다(스트림이
        # 최종 청크 없이 끝나므로 프론트는 코스를 못 받는다 — 예외 전 동작과 동일하게
        # "실패는 실패로" 두고, 있지도 않은 코스를 지어내지 않는다).
        logger.exception("[Chat] 스트리밍/저장 실패: %s", e)
    finally:
        # ★ 핵심: 어떤 경로로 끝나든 종료 신호를 넣어야 소비자가 영원히 대기하지 않는다.
        queue.put_nowait(None)


@app.get("/api/chat")
async def chat_endpoint(
        user_message: str,
        lat: float = Query(35.8347, description="차량 현재 위도"),
        lon: float = Query(129.2189, description="차량 현재 경도"),
        session_id: str = Query("default_session", description="세션 ID"),
        mode: Literal["nearby", "longdist", "remote"] = Query("nearby", description="여행 모드 (nearby=근교, longdist=장거리, remote=타지)"),
):
    """
    여행 코스 추천 스트리밍 엔드포인트 (SSE)

    응답 형식:
        data: {"bot_message": "텍스트 청크...", "recommended_places": []}
        ...
        data: {"bot_message": "", "recommended_places": [{"place_id": 1, "visit_sequence": 1}, ...]}
    """
    try:
        from traveler.core.region import detect_region_coords
        region_name, _rlat, _rlon = detect_region_coords(user_message)
        if region_name:
            logger.info("[Chat] region=%s lat=%.4f lon=%.4f", region_name, _rlat, _rlon)
            lat, lon = _rlat, _rlon
        elif session_id in _session_courses:
            # "다른 코스 추천" 등 지역명 없이 재호출 시 이전 코스 좌표 유지
            _, _prev_ctx, _ = _session_courses[session_id]
            lat, lon = _prev_ctx.current_lat, _prev_ctx.current_lng
            logger.info("[Chat] region 미감지 → 이전 세션 좌표 재사용 lat=%.4f lon=%.4f", lat, lon)
        weather, visited, prefs = await asyncio.gather(
            get_weather(lat, lon),
            get_visited(session_id),
            get_prefs(session_id),
        )
        logger.info("[Chat] session=%s | weather=%s | visited=%d", session_id, weather, len(visited))

        # "요청:" 이후 텍스트만 LLM user_query로 사용 (UI 선택값 "동행자: 친구" 등이 LLM을 혼란시키지 않도록)
        llm_query = user_message.split("요청:", 1)[-1].strip() if "요청:" in user_message else user_message

        new_prefs = extract_prefs_from_message(user_message)
        if new_prefs:
            await update_prefs(session_id, new_prefs)
            prefs = {**prefs, **new_prefs}

        duration_hours = extract_duration(llm_query) or 5
        logger.info("[Chat] companion=%s | duration=%dh | llm_query=%s", prefs.get("companion"), duration_hours, llm_query[:80])
        visited_context = build_visited_context(visited)

        mode_kr = MODE_TO_KR.get(mode, "타지")
        radius_km = RADIUS_BY_MODE[mode]
        context_data, _ = await asyncio.to_thread(
            rag_get_context,
            llm_query,
            lat,
            lon,
            weather,
            visited,
            prefs,
            radius_km,
            top_k=CANDIDATE_POOL_SIZE,
            duration_hours=duration_hours,
        )
        # #31 PM 리뷰 8번: candidate_spots는 20개(넓은 풀) — revise용으로 context_data에
        # 그대로 저장한다. 폴백 응답에 노출하는 건 상위 10개로 제한(prompt_spots와 동일
        # 기준) — 20개를 그대로 "추천"으로 보여주면 오히려 폴백 UX가 나빠진다.
        candidate_spots = context_data.candidate_spots
        prompt_spots = candidate_spots[:CANDIDATE_PROMPT_LIMIT]

        # build_context_message로 user 메시지 조립, 방문 이력 추가
        # #31 PM 리뷰 8번: 프롬프트에는 상위 10곳만(spot_limit) — context_data 자체는
        # 20곳을 그대로 들고 있어 _session_courses에 저장될 때 revise가 넓은 풀을 쓴다.
        context_message = build_context_message(context_data, spot_limit=CANDIDATE_PROMPT_LIMIT)
        if visited_context:
            context_message += "\n\n" + visited_context

        # #31 PM 리뷰 10번: 생산자(LLM 호출+저장)를 별도 Task로 띄우고, 응답 제너레이터는
        # 큐를 읽어 흘려보내기만 하는 얇은 소비자로 둔다.
        # ★ 핵심: create_task는 부모 Task의 자식이 아니라 독립 Task다 — 클라이언트가
        #   끊겨 요청 Task가 취소돼도 여기로 취소가 전파되지 않는다. 그래서 스트림을
        #   끝까지 안 읽어도 _session_courses 저장은 완료된다(= PM 리뷰 10번의 요구사항).
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(_stream_and_persist(
            queue, session_id, mode_kr, context_message, context_data, prompt_spots,
        ))
        _persist_tasks.add(task)
        task.add_done_callback(_persist_tasks.discard)

        async def event_generator():
            # 큐에 들어오는 즉시 내보낸다 — 버퍼링하지 않으므로 체감 지연은 기존과 동일.
            while True:
                chunk = await queue.get()
                if chunk is None:  # 생산자가 넣은 종료 신호
                    break
                yield chunk

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("[Chat] 요청 처리 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/visited")
async def add_visited_places(
        session_id: str,
        place_ids: list[str] = Body(..., example=["1", "3", "7"]),
):
    """코스 확정 시 방문 장소 등록 (이후 추천에서 제외)"""
    await add_visited(session_id, place_ids)
    return {"ok": True, "session_id": session_id, "added": place_ids}


@app.get("/api/session/{session_id}/visited")
async def get_visited_places(session_id: str):
    """방문 장소 목록 조회"""
    visited = await get_visited(session_id)
    return {"session_id": session_id, "visited": sorted(list(visited))}


@app.get("/api/session/{session_id}/prefs")
async def get_session_prefs(session_id: str):
    """누적 선호도 조회"""
    prefs = await get_prefs(session_id)
    return {"session_id": session_id, "prefs": prefs}


@app.post("/api/session/{session_id}/clear")
async def clear_session_data(session_id: str):
    """새 여행 시작 시 세션 초기화 (visited + prefs + 대화 기록 + 코스)"""
    await clear_session(session_id)
    if session_id in memory_store:
        del memory_store[session_id]
    # D11 후속(#31 PM 리뷰): 여기서 안 지우면 "새 여행 시작" 이후에도
    # /api/chat/revise가 이전 여행의 코스를 수정 대상으로 삼는다(session_id가
    # 그대로 재사용되기 때문).
    _session_courses.pop(session_id, None)
    return {
        "ok": True,
        "session_id": session_id,
        "cleared": ["visited", "prefs", "chat_history", "course"],
    }


class ReviseRequest(BaseModel):
    """#31 PM 리뷰 5번: GET → POST 전환에 맞춘 JSON body 스키마.
    쿼리스트링 대신 body로 받아 user_message가 로그·히스토리에 평문 노출되지 않게 한다."""
    user_message: str
    session_id: str = "default_session"


@app.post("/api/chat/revise")
async def revise_chat_endpoint(req: ReviseRequest):
    """
    이미 만들어진 코스를 후속 발화로 수정한다 — 바꿔줘/추가/빼줘/순서변경/전체재생성
    5종(Sprint 7, docs/superpowers/specs/2026-08-30-sprint7-multiturn-edit-intent-design.md).

    ⚠️ #31 PM 리뷰 5번: revise_course()는 세션 상태(_session_courses)를 바꾸는 부수효과가
    있는 호출이라, GET(안전·재시도 가능 전제)이 아니라 POST + JSON body로 받는다.

    응답 형식(스트리밍 아님 — revise_course()는 단발 호출):
        {"bot_message": "...",
         "recommended_places": [{"place_id": "1", "visit_sequence": 1, "place_name": "..."}, ...]}

    ⚠️ D11: "지금 코스"는 메모리 딕셔너리(_session_courses)에만 있다 — 이 session_id로
    /api/chat을 먼저 호출한 적이 없거나 서버가 재시작됐으면 404.
    """
    stored = _session_courses.get(req.session_id)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail="수정할 코스를 찾을 수 없습니다. 먼저 /api/chat으로 코스를 생성하세요.",
        )
    course, context_data, mode_kr = stored

    # #31 PM 리뷰 11번: ctx.candidate_spots는 /api/chat 최초 호출 시점에 고정된다.
    # 그 뒤 POST /api/session/{id}/visited로 방문 처리된 장소가 있으면 최신 visited를
    # 조회해서 넘겨야, replace/add가 이미 방문한 곳을 다시 추천하지 않는다.
    visited = await get_visited(req.session_id)
    revised = await revise_course(
        req.user_message, course, context_data, travel_mode=mode_kr, visited=visited,
    )
    _session_courses[req.session_id] = (revised, context_data, mode_kr)

    # #31 PM 리뷰 9번: /api/chat은 RunnableWithMessageHistory가 자동으로
    # memory_store에 발화를 쌓지만, 이 엔드포인트는 그 체인을 안 거쳐서 아무것도
    # 안 쌓였다. 그대로 두면 revise 이후 사용자가 다시 /api/chat으로 말을 걸 때
    # LLM이 "코스가 이미 수정된" 사실을 모른 채 이어간다 — 여기서 수동으로 같은
    # 히스토리에 추가해 맞춘다.
    history = get_session_history(req.session_id)
    history.add_user_message(req.user_message)
    history.add_ai_message(revised.bot_message)

    return {
        "bot_message": revised.bot_message,
        "recommended_places": [
            # #31 PM 리뷰 6번: place_name은 이미 RecommendedPlaceModel에 있으니
            # 여기서 빼면 프론트가 place_id로 이름을 다시 조회해야 한다.
            {"place_id": p.place_id, "visit_sequence": p.visit_sequence, "place_name": p.place_name}
            for p in revised.recommended_places
        ],
    }


@app.get("/health")
async def health_check():
    """서버 + DB 연결 상태 확인"""
    from traveler.rag.session import _check_db
    db_status = "connected" if await asyncio.to_thread(_check_db) else "disconnected"
    return {
        "status": "ok",
        "db": db_status,
    }
