"""
RAG 엔지니어(이수연) ↔ LLM 엔지니어(정훈) 인터페이스 계약.

[계약 내용 — 2026-05-26 확정]
- 이수연이 ContextData를 채워서 정훈의 generate_course()에 넘긴다
- 정훈은 ContextData를 프롬프트에 주입해 GPT-4o를 호출한다
- 출력 스키마: StreamingCourseModel (traveler/validation/schema.py 참조)
- 스트리밍 방식: A (bot_message 텍스트 먼저, recommended_places JSON 나중)

[변경 시 규칙]
이 파일을 수정하면 반드시 양측(이수연, 정훈) 모두에게 알려야 한다.

[고지 — 2026-08-04]
generate_course()에 temperature 키워드 인자 추가됨 — eval 파이프라인 재현성용,
기본값 0.3 유지, 프로덕션 영향 없음.
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
    place_id: str        # content_id 문자열 — 정훈 출력의 place_id와 1:1 매핑
                          # (2026-08-09 변경: SQLite PK int는 재시작마다 hash()
                          #  값이 바뀌는 문제가 있어 content_id 문자열로 통일 — Issue #15, 이수연 합의)
    place_name: str      # 장소명 (예: "불국사")
    category: str        # 카테고리 (예: "관광", "음식", "체험", "휴식")
    distance_min: float  # 레거시 검색 거리 값(현 retriever는 km). 일정 시간 계산에는 쓰지 않음.
    indoor: bool         # 실내 여부 (날씨 제약 판단에 사용)
    lat: float = 0.0     # 장소 위도 (Pinecone 메타데이터 재사용 — #27)
    lng: float = 0.0     # 장소 경도 (Pinecone 메타데이터 재사용 — #27)
    # ── 경주 특화 (2026-09-15) ──
    # WHY: TourAPI 원문 그대로 보관 — <br> 정리 등 가공은 프롬프트 계층 책임(SRP).
    use_time: str | None = None   # 영업/이용 시간 원문 (콘텐츠 유형별 컬럼 COALESCE 결과)
    rest_date: str | None = None  # 휴무일 원문
    event_start_date: str | None = None  # 축제 시작일 'YYYYMMDD' (content_type_id "15"만 채움)
    event_end_date: str | None = None    # 축제 종료일 'YYYYMMDD' (content_type_id "15"만 채움)


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
    current_lat: float                   # 검색 중심 위도 (지역명 감지 시 바뀔 수 있음)
    current_lng: float                   # 검색 중심 경도
    weather: str                         # 기상청 API 날씨 (예: "맑음", "비", "흐림")
    candidate_spots: list[SpotCandidate] # SQLite + Pinecone 필터링 결과 (3~10개 권장)

    # --- 선택 ---
    # 가용 시간. 사용자가 "5시간"처럼 말하면 preference.extract_duration_hours가
    # 뽑아 이 값을 덮고, duration_hours_min은 None이 된다.
    duration_hours: int | float = 5      # 레거시 호환 시간 표기 (분 예산이 있으면 소수 유지)
    duration_minutes: int | None = None  # 명시된 가용 시간의 SSOT. None이면 기존 3~5시간 기본값.
    # 사용자가 시간을 말하지 않았을 때만 채운다 → 프롬프트에 "가용 시간: 3~5시간"으로
    # 나간다. 3시간 단일값이던 시절엔 GPT가 2~3곳에서 코스를 끊어 빈약했고(2026-09-16
    # 실측), 5시간 단일값은 짧게 다녀오려는 사용자에게 과하다. 범위로 여지를 준다.
    # ⚠️ 주의: eval/metrics.py는 duration_hours(상한)만 예산 계산에 쓴다.
    duration_hours_min: int | None = 3   # 미지정일 때의 범위 하한 (지정되면 None)
    transport: str = "자차"             # 이동 수단
    transport_assumed: bool = False      # 명시/저장된 수단이 없어 기본 자차를 가정했는지 여부
    companion: str = "혼자"             # 동행자
    mood: str = ""                       # 분위기 힌트 (Pinecone 쿼리에서 추출)
    route_summary: str = ""             # Tmap 경로 요약 (선택)
    region: str | None = None            # extract_region()이 감지한 지역명 (2026-09-15, 경주 특화)
    # WHY: 지역 검색 중심점과 실제 출발지를 섞으면 첫 이동시간이 사라진다.
    # None은 기존 직접 호출의 current_lat/lng 사용, 0은 API에서 GPS 미제공을 뜻한다.
    schedule_origin_lat: float | None = None
    schedule_origin_lng: float | None = None
