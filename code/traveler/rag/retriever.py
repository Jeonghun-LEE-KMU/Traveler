"""
RAG retriever 더미 스텁 — 이수연 RAG 파이프라인 연결 전 개발/테스트용.

파이프라인 위치:
  사용자 입력 → ★ [retriever.py → ContextData] → generate_course() → GPT-4o → 코스 JSON

WHY 이 파일이 필요한가:
  이수연의 실제 RAG 파이프라인(Pinecone + SQLite)이 완성되기 전에도
  generate_course()를 단독으로 테스트할 수 있어야 한다.
  → "스텁이 실제와 동일한 인터페이스를 가져야 교체 비용이 0에 가까워진다"
  → Sprint 3에서 이수연 실제 구현으로 교체할 때 이 파일 내부만 바꾸면 됨.
  → 인터페이스(함수 시그니처 + 반환 타입)는 절대 바꾸지 않는다.

[LLM 엔지니어 관점]
  실제 서비스에서 Retriever 계층을 stub으로 분리하는 이유:
  1. 팀 간 의존성 차단: RAG 팀이 작업 중이어도 LLM 팀이 독립적으로 테스트 가능
  2. 테스트 격리: 단위 테스트에서 실제 DB/API 호출 없이 인터페이스만 검증
  3. Graceful Degradation 설계: 실제 retriever가 오류 나도 stub으로 폴백 가능

[교체 타임라인]
  Sprint 1 (현재): 하드코딩된 경주 5개 장소 반환
  Sprint 3: 이수연 실제 Pinecone + SQLite 연결로 교체

[주의] 이 파일을 실제 구현으로 교체할 때:
  - 함수 이름을 get_context_stub → get_context 로 변경
  - 반환 타입(ContextData)은 유지
"""

from traveler.rag.types import ContextData, SpotCandidate


# WHY 기본 인자(경주 시나리오)를 넣은 이유:
#   main.py에서 인자 없이 호출해도 바로 실행되도록 (개발 편의성)
#   실제 구현에서는 user_query + 위치 정보가 외부에서 들어온다
def get_context_stub(
    user_query: str = "경주에서 반나절 자차로 여행하고 싶어요",
    weather: str = "맑음",
) -> ContextData:
    """
    개발/테스트용 더미 ContextData 반환. 경주 시나리오 5개 장소 하드코딩.

    WHY 경주 시나리오:
      팀 내 기본 테스트 케이스로 합의된 지역.
      place_id 1~5는 이수연 SQLite의 실제 PK라고 가정.
      나중에 실제 연결하면 place_id가 실제 DB 값으로 대체됨.

    INPUT: 사용자 자연어 쿼리, 더미 날씨 (실제는 기상청 API가 채움)
    OUTPUT: ContextData → generate_course()에 바로 전달 가능

    대안: fixtures YAML 파일로 관리하는 방법도 있음
      → 지금은 Sprint 1 PoC라 코드 내 하드코딩이 간단하고 충분함
    """
    # ★ 핵심: place_id가 이수연 SQLite PK와 1:1 매핑되어야 한다
    #   → 실제 DB 확정 전까지 1~5로 임시 부여
    spots = [
        SpotCandidate(
            place_id=1,
            place_name="불국사",
            category="관광",
            distance_min=5.0,
            indoor=False,  # 실외 → 비 올 때 코스에서 후순위
        ),
        SpotCandidate(
            place_id=2,
            place_name="첨성대",
            category="관광",
            distance_min=12.0,
            indoor=False,
        ),
        SpotCandidate(
            place_id=3,
            place_name="교리김밥",
            category="음식",
            distance_min=15.0,
            indoor=True,   # 실내 → 날씨 무관하게 포함 가능
        ),
        SpotCandidate(
            place_id=4,
            place_name="경주 황리단길",
            category="체험",
            distance_min=18.0,
            indoor=False,
        ),
        SpotCandidate(
            place_id=5,
            place_name="석굴암",
            category="관광",
            distance_min=25.0,
            indoor=False,
        ),
    ]

    return ContextData(
        user_query=user_query,
        current_lat=35.8562,    # 경주역 기준 좌표 (더미)
        current_lng=129.2247,
        weather=weather,        # 파라미터로 받아서 테스트 시 다양한 날씨 시나리오 실험 가능
        candidate_spots=spots,
        duration_hours=4,
        transport="자차",
        companion="친구",
        mood="역사적인 분위기, 여유로운",
        route_summary="",       # 더미: 실제는 Tmap API가 채움 (Sprint 2)
    )
