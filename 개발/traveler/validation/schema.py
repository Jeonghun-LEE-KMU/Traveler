"""
Pydantic v2 기반 스트리밍 출력 스키마 — Guardrails 1단계.

[설계 결정 — 2026-05-26]
A 방식 스트리밍 채택:
  - bot_message가 JSON의 첫 번째 필드 → 텍스트 스트리밍이 즉시 시작됨
  - recommended_places가 두 번째 필드 → 텍스트 완료 후 지도 핀 일괄 업데이트
  - 필드 순서가 UX에 직접 영향을 미치므로 절대 변경 금지

이수연(RAG 엔지니어) 협의 완료 — 2026-05-27:
  - place_id: 이수연의 SQLite PK와 매핑 (1:1)
  - lat/lng: GPT 출력에서 제거 (환각 위험 + SQLite 내부 조회로 대체)
    → GPT는 place_id만 반환 → 백엔드가 SQLite에서 정확한 좌표 조회
  - SQLite는 도커로 동일 인스턴스에 탑재 → 내부 조회, 속도 지연 없음
  - 스트리밍 파서는 이수연의 프론트 담당
"""

from pydantic import BaseModel, Field, field_validator


class RecommendedPlaceModel(BaseModel):
    """
    경유지 하나.

    [이수연 협의 반영 — 2026-05-27]
    lat/lng를 GPT 출력에서 제거:
      - GPT가 GPS 좌표(소수점 6자리)를 생성하면 환각 위험이 높음
      - SQLite는 도커 동일 인스턴스 내부 조회 → 속도 지연 없음
      → GPT는 place_id만 반환 → 백엔드가 SQLite에서 정확한 lat/lng 조회

    [좌표 제공 흐름]
    GPT 출력 → place_id → 백엔드 SQLite 조회 → 정확한 lat/lng → 프론트 지도 핀
    """
    place_id: int = Field(ge=1)               # SQLite 장소 ID (PK)
    place_name: str                           # 장소명 (예: "불국사")
    visit_sequence: int = Field(ge=1)         # 방문 순서 (1부터 시작)
    duration_min: int = Field(ge=10, le=480)  # 체류 시간 (10분~8시간)

    @field_validator("place_name")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """GPT가 빈 문자열을 반환할 때 방어"""
        if not v.strip():
            raise ValueError("place_name이 비어있습니다.")
        return v


class StreamingCourseModel(BaseModel):
    """
    GPT-4o 스트리밍 응답의 최상위 스키마.

    [A 방식 스트리밍 — 2026-05-26 확정]
    - bot_message: 텍스트를 글자 단위로 스트리밍 → 프론트가 즉시 표시
    - recommended_places: JSON 완성 후 한 번에 파싱 → 지도 핀 일괄 표시
    - 프론트와 백엔드 코드는 동일. 렌더링 방식만 프론트에서 결정.

    ⚠️ bot_message가 반드시 첫 번째 필드여야 텍스트가 먼저 스트리밍됨.
    파싱 실패 시: 재프롬프트 또는 부분 재생성 트리거.
    """
    bot_message: str                          # ← 첫 번째: 텍스트 스트리밍용
    recommended_places: list[RecommendedPlaceModel] = Field(
        min_length=1,
        max_length=10,
    )                                         # ← 두 번째: 지도 핀용

    @field_validator("recommended_places")
    @classmethod
    def check_sequence_sequential(
        cls, places: list[RecommendedPlaceModel]
    ) -> list[RecommendedPlaceModel]:
        """visit_sequence가 1, 2, 3... 순서인지 확인"""
        for i, place in enumerate(places, start=1):
            if place.visit_sequence != i:
                raise ValueError(
                    f"visit_sequence가 연속적이지 않습니다: "
                    f"예상 {i}, 실제 {place.visit_sequence} ({place.place_name})"
                )
        return places
