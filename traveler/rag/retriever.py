"""
RAG retriever — 이수연 RAG 파이프라인 연결 전 개발/테스트용 더미 구현.

파이프라인 위치:
  사용자 입력 → ★ [retriever.py → ContextData] → generate_course() → GPT-4o → 코스 JSON

[현재 상태: DUMMY]
  Sprint 1 — 경주 시나리오 5개 장소 하드코딩.
  Sprint 3에서 이수연 실제 Pinecone + SQLite 연결로 내부 교체 예정.
  함수 시그니처와 반환 타입(ContextData)은 변경하지 않는다.
"""

from traveler.rag.types import ContextData, SpotCandidate


# WHY 기본 인자(경주 시나리오)를 넣은 이유:
#   main.py에서 인자 없이 호출해도 바로 실행되도록 (개발 편의성)
#   실제 구현에서는 user_query + 위치 정보가 외부에서 들어온다
def get_context(
    user_query: str = "경주에서 반나절 자차로 여행하고 싶어요",
    weather: str = "맑음",
) -> ContextData:
    """
    [DUMMY] 개발/테스트용 더미 ContextData 반환. 경주 시나리오 5개 장소 하드코딩.

    Sprint 3에서 이수연의 실제 Pinecone + SQLite 구현으로 교체 예정.
    place_id 1~5는 이수연 SQLite의 실제 PK라고 가정.
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
