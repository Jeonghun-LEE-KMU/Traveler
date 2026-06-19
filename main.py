"""
TRAVELER FastAPI 메인 서버 - Sprint 2

[Sprint 2 변경사항]
  1. 날씨 실시간 주입: get_weather() -> 기상청 API 실제 날씨를 컨텍스트에 반영
  2. 누적 선호도: 사용자 메시지에서 키워드 추출 -> Redis prefs HASH 누적 -> 프롬프트 주입
  3. Redis trip_session: 방문 장소 visited SET 관리 + TTL 24h

[파이프라인 흐름]
  사용자 요청 (user_message + lat/lon + session_id)
      ↓
  asyncio.gather -> 날씨 + visited + prefs 병렬 조회
      ↓
  선호도 추출 -> Redis 누적 업데이트
      ↓
  integrated_context 조립 (날씨 + visited 필터 + 누적 선호도 + DB 팩트)
      ↓
  LangChain + ChatMessageHistory -> GPT-4o 스트리밍
      ↓
  StreamingResponse (SSE) -> 프론트

[Sprint 3 예정]
  integrated_context 더미 데이터 -> 가영 언니 filter_query.py 실데이터로 교체
  retriever.py -> Pinecone + SQLite 실제 연결
"""

import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 지금 무료 사용중 이거 서버 나오면 챗오픈에이아이로변경
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
from traveler.rag.session import (
    add_visited,
    get_visited,
    get_prefs,
    update_prefs,
    clear_session,
)
from traveler.rag.preference import (
    extract_prefs_from_message,
    build_prefs_context,
    build_visited_context,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="TRAVELER_Mobility_RAG_Hub_Production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain 대화 메모리 (in-memory)
# WHY in-memory: Sprint 2는 PoC. Sprint 5에서 SQLite 영속화로 전환 예정.
memory_store: dict = {}


def get_session_history(s_id: str) -> ChatMessageHistory:
    if s_id not in memory_store:
        memory_store[s_id] = ChatMessageHistory()
    return memory_store[s_id]


# ==============================================================================
# 메인 추천 엔드포인트
# ==============================================================================

@app.get("/api/chat")
async def chat_endpoint(
        user_message: str,
        lat: float = Query(35.8347, description="차량 현재 위도 (기본값: 경주 보문)"),
        lon: float = Query(129.2189, description="차량 현재 경도"),
        session_id: str = Query("default_session", description="사용자 세션 ID"),
):
    """
    여행 코스 추천 스트리밍 엔드포인트.

    응답 형식 (SSE):
        data: {"bot_message": "텍스트 청크...", "recommended_places": []}
        ...
        data: {"bot_message": "", "recommended_places": [{"place_id": 1, "visit_sequence": 1}, ...]}
    """
    try:
        # 1. 실시간 데이터 병렬 조회
        # asyncio.gather: 세 I/O 작업 동시 실행 -> 순차 대비 지연 감소
        # get_weather: 기상청 API (실패 시 "정보없음" 반환 - Graceful Degradation)
        # get_visited: Redis SET (실패 시 빈 set 반환 - Graceful Degradation)
        # get_prefs:   Redis HASH (실패 시 빈 dict 반환 - Graceful Degradation)
        weather, visited, prefs = await asyncio.gather(
            get_weather(lat, lon),
            get_visited(session_id),
            get_prefs(session_id),
        )
        logger.info(
            "[Chat] session=%s | weather=%s | visited=%d | prefs=%s",
            session_id, weather, len(visited), prefs,
        )

        # 2. 이번 메시지에서 선호도 추출 -> Redis 누적
        # LLM 호출 없이 키워드 매칭으로 빠르게 처리 (동기, 즉시 완료)
        new_prefs = extract_prefs_from_message(user_message)
        if new_prefs:
            await update_prefs(session_id, new_prefs)
            # Redis 재조회 없이 바로 이번 요청에 반영
            prefs = {**prefs, **new_prefs}

        # 3. 프롬프트 컨텍스트 조각 생성
        visited_context = build_visited_context(visited)
        prefs_context = build_prefs_context(prefs)

        # 4. 통합 컨텍스트 조립
        # 가영 언니가 filter_query.py 완성되면 여기에 진짜 sqlite3 쿼리로
        # "lat, lon 주변 데이터 필터링"을 구현할 자리! (Sprint 3)
        context_sections = [
            "[실시간 환경 데이터]\n"
            "- 현재 차량 GPS: 위도 {lat}, 경도 {lon}\n"
            "- 현재 날씨: {weather}".format(lat=lat, lon=lon, weather=weather),
        ]

        if visited_context:
            context_sections.append(visited_context)

        if prefs_context:
            context_sections.append(prefs_context)

        # 더미 DB 컨텍스트 (Sprint 3에서 실데이터로 교체)
        context_sections.append(
            "[가영이가 정리해준 API 및 SQLite 기반 필터링 팩트 데이터]\n"
            "1. 국문 관광정보 서비스 (Key: {k1}...): '경주 보문단지 웰니스 힐링파크'\n"
            "2. 무장애 여행 정보 (Key: {k2}...): 유모차 대여 가능, 경사로 완비\n"
            "3. 티맵 연관 관광지 (Key: {k3}...): 보문정 산책로 연계율 89%\n"
            "4. 의료 관광 정보 (Key: {k4}...): 반경 2km 내 한방 테라피 센터 위치함\n"
            "5. 노키즈존 XML 데이터: 해당 장소는 'YES 키즈존'으로 영유아 동반 입장 적극 권장함".format(
                k1=KORSERVICE_API_KEY[:5],
                k2=KORWITHSERVICE_API_KEY[:5],
                k3=TARRITETARSERVICE_API_KEY[:5],
                k4=MDCLTURSMSERVICE_API_KEY[:5],
            )
        )

        integrated_context = "\n\n".join(context_sections)

        # 5. 스트리밍 제너레이터
        # WHY sync generator: RunnableWithMessageHistory.stream()이 동기 이터레이터.
        # integrated_context는 클로저로 캡처되어 내부에서 사용됨.
        def event_generator():
            # 얘도 진짜 서버 나오면 변경!!!!
            llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, streaming=True)

            rag_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "당신은 모빌리티 전문 트래블 큐레이터 [TRAVELER]입니다.\n"
                 "아래 [통합 관광지 DB 컨텍스트]의 정보만을 '절대적 팩트'로 바탕하여 "
                 "유저에게 완벽한 드라이브 코스를 제안하세요.\n\n"
                 "[날씨 규칙]\n"
                 "- 날씨가 '비' 또는 '흐림'이면 실내 위주 코스로 구성하세요.\n"
                 "- 날씨가 '맑음'이면 실외 명소를 적극 포함하세요.\n"
                 "- 날씨가 '정보없음'이면 날씨 언급 없이 일반 코스를 제안하세요.\n\n"
                 "[방문 이력 규칙]\n"
                 "- '이미 방문한 장소' 목록의 place_id는 절대 추천하지 마세요.\n\n"
                 "[개인화 규칙]\n"
                 "- '누적 여행 선호도'가 있으면 해당 취향을 반영하되, 이번 요청 내용이 최우선입니다.\n\n"
                 "[통합 관광지 DB 컨텍스트]\n"
                 "{db_context}\n"),
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

            # 텍스트 스트리밍 - 대답이 쏟아지는 중에는 장소 배열을 비워서 전송
            for chunk_text in final_rag_engine.stream(
                    {"db_context": integrated_context, "user_message": user_message},
                    config={"configurable": {"session_id": session_id}},
            ):
                chunk = {
                    "bot_message": chunk_text,
                    "recommended_places": [],
                }
                yield "data: {}\n\n".format(json.dumps(chunk, ensure_ascii=False))

            # AI 멘트 스트리밍이 끝나면, 맨 마지막 조각에 장소 JSON 데이터를 박아서 마무리!
            # [Sprint 3 교체 예정]: place_id를 GPT 출력 파싱 or SQLite 조회로 채움
            final_route_data = {
                "bot_message": "",
                "recommended_places": [
                    {"place_id": 1, "visit_sequence": 1},
                    {"place_id": 2, "visit_sequence": 2},
                ],
            }
            yield "data: {}\n\n".format(json.dumps(final_route_data, ensure_ascii=False))

        # 최종 전송: 실시간 전송 통로인 StreamingResponse로 발송!
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("[Chat] 요청 처리 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# trip_session 관리 엔드포인트 (Sprint 2 신규)
# ==============================================================================

@app.post("/api/session/{session_id}/visited")
async def add_visited_places(
        session_id: str,
        place_ids: list = Body(..., example=["1", "3", "7"]),
):
    """
    사용자가 코스를 확정했을 때 프론트엔드가 호출.
    place_id들을 visited SET에 추가 -> 이후 추천에서 제외됨. TTL 24시간.
    """
    await add_visited(session_id, place_ids)
    return {"ok": True, "session_id": session_id, "added": place_ids}


@app.get("/api/session/{session_id}/visited")
async def get_visited_places(session_id: str):
    """디버깅용: 세션의 방문 장소 목록 조회"""
    visited = await get_visited(session_id)
    return {"session_id": session_id, "visited": sorted(list(visited))}


@app.get("/api/session/{session_id}/prefs")
async def get_session_prefs(session_id: str):
    """디버깅용: 세션의 누적 선호도 조회"""
    prefs = await get_prefs(session_id)
    return {"session_id": session_id, "prefs": prefs}


@app.post("/api/session/{session_id}/clear")
async def clear_session_data(session_id: str):
    """
    새 여행 시작 시 세션 전체 초기화.
    Redis (visited + prefs) + LangChain 대화 메모리 모두 삭제.
    """
    await clear_session(session_id)
    if session_id in memory_store:
        del memory_store[session_id]
    return {
        "ok": True,
        "session_id": session_id,
        "cleared": ["visited", "prefs", "chat_history"],
    }


# ==============================================================================
# 헬스체크
# ==============================================================================

@app.get("/health")
async def health_check():
    """서버 + Redis 연결 상태 확인"""
    from traveler.rag.session import get_redis
    redis_status = "disconnected"
    try:
        await get_redis().ping()
        redis_status = "connected"
    except Exception:
        pass
    return {
        "status": "ok",
        "redis": redis_status,
        "note": "Redis disconnected여도 Graceful Degradation으로 서비스 동작함",
    }
