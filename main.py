import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

#지금 무료 사용중 이거 서버 나오면 챗오픈에이아이로변경
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# config에서 키 가져옴
from config import (
    OPENAI_API_KEY,
    KORSERVICE_API_KEY,
    KMA_API_KEY,
    KORWITHSERVICE_API_KEY,
    TARRITETARSERVICE_API_KEY,
    MDCLTURSMSERVICE_API_KEY
)

app = FastAPI(title="TRAVELER_Mobility_RAG_Hub_Production")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# 세션 메모리
memory_store = {}


def get_session_history(s_id: str):
    if s_id not in memory_store:
        memory_store[s_id] = ChatMessageHistory()
    return memory_store[s_id]


# 실시간 스트리밍 + 데이터 분리 전송
@app.get("/api/chat")
async def chat_endpoint(
        user_message: str,
        lat: float = Query(35.8347),  # 프론트가 줄 실시간 차량 위도 (기본값: 경주 보문)
        lon: float = Query(129.2189),  # 프론트가 줄 실시간 차량 경도
        session_id: str = "suyeon_session"
):
    try:
        # 1. 데이터 통합 및 필터링 로직
        # 💡 가영 언니가 DB 속성 명세 주면 여기에 진짜 sqlite3 쿼리로 "lat, lon 주변 데이터 필터링"을 구현할 자리!
        integrated_context = f"""
        [가영이가 정리해준 API 및 SQLite 기반 필터링 팩트 데이터]
        - 현재 차량 GPS 좌표: 위도 {lat}, 경도 {lon}
        1. 국문 관광정보 서비스 (Key: {KORSERVICE_API_KEY[:5]}...): '경주 보문단지 웰니스 힐링파크'
        2. 무장애 여행 정보 (Key: {KORWITHSERVICE_API_KEY[:5]}...): 유모차 대여 가능, 경사로 완비 ♿
        3. 티맵 연관 관광지 (Key: {TARRITETARSERVICE_API_KEY[:5]}...): 연관 드라이브 코스 '보문정 산책로' 연계율 89%
        4. 의료 관광 정보 (Key: {MDCLTURSMSERVICE_API_KEY[:5]}...): 반경 2km 내 한방 테라피 센터 위치함
        5. 기상청 API 허브 (Key: {KMA_API_KEY[:5]}...): 현재 날씨 실시간 관측 결과 '비가 내림 🌧️'
        6. 노키즈존 XML 데이터: 해당 장소는 'YES 키즈존'으로 영유아 동반 입장 적극 권장함
        """

        # 2️. 실시간 쪼개기 전송을 위한 제너레이터 함수 가동
        #  파이썬 3.13 안정성을 위해 일반 생성자(def)로 변경
        def event_generator():

            # 스트리밍 옵션,
            #얘도 진짜 서버 나오면 변경!!!!
            llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, streaming=True)
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 모빌리티 전문 트래블 큐레이터 [TRAVELER]입니다.
        밑에 입력된 [통합 관광지 DB 컨텍스트]의 정보만을 '절대적 팩트'로 바탕하여 유저에게 완벽한 드라이브 코스를 제안하세요.

        [통합 관광지 DB 컨텍스트]
        {db_context}
        """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_message}")
            ])

            rag_chain = rag_prompt | llm | StrOutputParser()

            final_rag_engine = RunnableWithMessageHistory(
                rag_chain, get_session_history, input_messages_key="user_message", history_messages_key="chat_history"
            )

            for chunk_text in final_rag_engine.stream(
                    {"db_context": integrated_context, "user_message": user_message},
                    config={"configurable": {"session_id": session_id}}
            ):
                chunk = {
                    "bot_message": chunk_text,
                    "recommended_places": []  # 대답이 쏟아지는 중에는 장소 배열을 비워서 전송
                }
                # 가독성을 위해 yield 앞에 바이트로 인코딩을 명시
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            #3. AI 멘트 스트리밍이 끝나면, 맨 마지막 조각에 장소 JSON 데이터를 쾅 박아서 마무리!
            final_route_data = {
                "bot_message": "",
                "recommended_places": [
                    {"place_id": 1, "visit_sequence": 1},
                    {"place_id": 2, "visit_sequence": 2}
                ]
            }
            yield f"data: {json.dumps(final_route_data, ensure_ascii=False)}\n\n"


        # 4. 최종 전송:실시간 전송 통로인 StreamingResponse로 발송!
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))