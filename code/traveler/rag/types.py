"""
RAG 엔지니어(이수연) ↔ LLM 엔지니어(정훈) 인터페이스 계약.

[계약 내용 — 2026-05-26 확정]
- 이수연이 ContextData를 채워서 정훈의 generate_course()에 넘긴다
- 정훈은 ContextData를 프롬프트에 주입해 GPT-4o를 호출한다
- 출력 스키마: StreamingCourseModel (traveler/validation/schema.py 참조)
- 스트리밍 방식: A (bot_message 텍스트 먼저, recommended_places JSON 나중)

[변경 시 규칙]
이 파일을 수정하면 반드시 양측(이수연, 정훈) 모두에게 알려야 한다.
"""

from dataclasses import dataclass, field


@dataclass
class SpotCandidate:
    """
    이수연의 SQLite에서 필터링된 장소 후보 하나.

    이수연 담당:
      - SQLite에서 GPS 반경 + 카테고리 필터링
      - Pinecone 벡터 검색으로 분위기 유사 장소 추가
      - 두 결과를 합쳐서 이 형식으로 변환

    정훈 담당:
      - 이 후보 목록을 프롬프트에 주입
      - GPT가 후보 중에서 골라 visit_sequence 부여
    """
    place_id: int        # SQLite PK — 정훈 출력의 place_id와 1:1 매핑
    place_name: str      # 장소명 (예: "불국사")
    category: str        # 카테고리 (예: "관광", "음식", "체험", "휴식")
    distance_min: float  # 현재 GPS 기준 Tmap 이동 시간 (분)
    indoor: bool         # 실내 여부 (날씨 제약 판단에 사용)


@dataclass
class ContextData:
    """
    이수연이 채워서 정훈에게 넘기는 컨텍스트 데이터.

    정훈은 이 객체를 받아 build_context_message()로 프롬프트 문자열로 변환,
    GPT-4o user 메시지에 주입한다.

    [필수 필드] — 없으면 LLM 호출 불가
    [선택 필드] — 없어도 Graceful Degradation으로 기본값 사용
    """
    # --- 필수 ---
    user_query: str                      # 사용자 자연어 요청 (예: "분위기 좋은 곳 추천")
    current_lat: float                   # 현재 차량 위도
    current_lng: float                   # 현재 차량 경도
    weather: str                         # 기상청 API 날씨 (예: "맑음", "비", "흐림")
    candidate_spots: list[SpotCandidate] # SQLite + Pinecone 필터링 결과 (3~10개 권장)

    # --- 선택 ---
    duration_hours: int = 3              # 가용 시간 (기본 3시간)
    transport: str = "자차"             # 이동 수단
    companion: str = "혼자"             # 동행자
    mood: str = ""                       # 분위기 힌트 (Pinecone 쿼리에서 추출)
    route_summary: str = ""             # Tmap 경로 요약 (선택)
