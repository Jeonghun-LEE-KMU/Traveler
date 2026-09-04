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
)
from traveler.validation.schema import StreamingCourseModel

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
        weather, visited, prefs = await asyncio.gather(
            get_weather(lat, lon),
            get_visited(session_id),
            get_prefs(session_id),
        )
        logger.info("[Chat] session=%s | weather=%s | visited=%d", session_id, weather, len(visited))

        new_prefs = extract_prefs_from_message(user_message)
        if new_prefs:
            await update_prefs(session_id, new_prefs)
            prefs = {**prefs, **new_prefs}

        visited_context = build_visited_context(visited)

        mode_kr = MODE_TO_KR.get(mode, "타지")
        radius_km = RADIUS_BY_MODE[mode]
        context_data, _ = await asyncio.to_thread(
            rag_get_context,
            user_message,
            lat,
            lon,
            weather,
            visited,
            prefs,
            radius_km,
        )
        candidate_spots = context_data.candidate_spots

        def event_generator():
            llm = ChatOpenAI(
                model="gpt-4o",
                api_key=OPENAI_API_KEY,
                streaming=True,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

            # 정훈 system prompt (모드별 JSON 출력 강제)
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", get_system_prompt(mode_kr)),
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

            # build_context_message로 user 메시지 조립, 방문 이력 추가
            context_message = build_context_message(context_data)
            if visited_context:
                context_message += "\n\n" + visited_context

            # 스트리밍하면서 full_text 누적
            full_text = ""
            for chunk_text in final_rag_engine.stream(
                    {"user_message": context_message},
                    config={"configurable": {"session_id": session_id}},
            ):
                full_text += chunk_text
                yield "data: {}\n\n".format(json.dumps({
                    "bot_message": chunk_text,
                    "recommended_places": [],
                }, ensure_ascii=False))

            # 스트리밍 완료 후 Pydantic 검증
            try:
                validated = StreamingCourseModel.model_validate_json(full_text)
                spot_map = {s.place_id: s for s in candidate_spots}
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
            except Exception as e:
                logger.warning("[Chat] 검증 실패, Pinecone fallback 사용: %s | full_text[:200]=%s", e, full_text[:200])
                recommended = [
                    {"place_id": s.place_id, "visit_sequence": i + 1, "lat": s.lat, "lng": s.lng}
                    for i, s in enumerate(candidate_spots)
                ]

            yield "data: {}\n\n".format(json.dumps({
                "bot_message": "",
                "recommended_places": recommended,
            }, ensure_ascii=False))

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
    """새 여행 시작 시 세션 초기화 (visited + prefs + 대화 기록)"""
    await clear_session(session_id)
    if session_id in memory_store:
        del memory_store[session_id]
    return {
        "ok": True,
        "session_id": session_id,
        "cleared": ["visited", "prefs", "chat_history"],
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
