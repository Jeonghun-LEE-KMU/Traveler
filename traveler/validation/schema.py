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

[v1.1 스키마 변경 — 2026-06-01]
  이수연 /api/chat 출력 포맷에 맞춰 place_name, duration_min을 Optional로 변경.
  이수연 출력: {"place_id": 1, "visit_sequence": 1}  ← place_name, duration_min 없음
  → 백엔드가 place_id로 SQLite 조회해 place_name 채움 (lat/lng와 동일 패턴)
  → duration_min은 TourAPI 평균 체류시간 데이터로 채울 예정 (Sprint 2)
  GPT가 두 필드를 생성하면 그대로 사용, 없으면 None → 백엔드 보강 흐름으로 처리.

[v1.2 스키마 변경 — 2026-08-09, Issue #15 · 이수연 합의]
  place_id를 SQLite PK(int)에서 content_id(문자열)로 변경.
  PK는 DB 스키마가 바뀌면 값이 깨질 수 있어, 재조회 없이도 안정적인
  content_id 문자열을 단일 출처로 쓰기로 함.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RecommendedPlaceModel(BaseModel):
    """
    경유지 하나.

    [필수 필드] place_id, visit_sequence — 이수연 /api/chat 출력에 항상 존재
    [Optional 필드] place_name, duration_min — 없으면 백엔드가 SQLite/TourAPI로 채움

    [데이터 흐름]
    GPT/이수연 출력 → place_id → 백엔드 SQLite 조회 → place_name, lat/lng
                                                      → TourAPI → duration_min
    """
    place_id: str = Field(min_length=1)                 # content_id 문자열 — 필수
    visit_sequence: int = Field(ge=1)                  # 방문 순서 (1부터) — 필수
    place_name: Optional[str] = None                   # 장소명 — 백엔드가 SQLite로 채움
    duration_min: Optional[int] = None                 # 체류 시간 — TourAPI로 채울 예정. 범위는 _clamp_duration이 흡수

    @field_validator("place_name")
    @classmethod
    def not_empty(cls, v: Optional[str]) -> Optional[str]:
        """
        place_name이 제공됐는데 빈 문자열이면 방어.
        None이면 스킵 (백엔드가 나중에 채워주므로).
        """
        if v is not None and not v.strip():
            raise ValueError("place_name이 비어있습니다.")
        return v

    @field_validator("duration_min", mode="before")
    @classmethod
    def _clamp_duration(cls, v):
        # WHY: 범위 밖 체류시간 하나로 코스 전체가 폴백되던 것을 clamp로 흡수
        # (Guardrails ③ 부분 재생성 철학). 10곳 중 1곳만 값이 이상해도
        # 예전엔 course_generator가 ValidationError를 잡아 전체를 "죄송해요" 1줄로 강등했다.
        if v is None:
            return v
        try:
            v = int(v)
        except (TypeError, ValueError):
            return v   # 잘못된 타입은 pydantic 기본 검증에 맡김
        return max(10, min(480, v))   # 범위 밖은 clamp (600→480, 5→10)


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
    bot_message: str                           # ← 첫 번째: 텍스트 스트리밍용
    recommended_places: list[RecommendedPlaceModel] = Field(
        min_length=1,
        max_length=10,
    )                                          # ← 두 번째: 지도 핀용

    @field_validator("recommended_places")
    @classmethod
    def check_sequence_sequential(
        cls, places: list[RecommendedPlaceModel]
    ) -> list[RecommendedPlaceModel]:
        """visit_sequence 값 기준 1, 2, 3... 연속성 확인 (배열 순서는 무관)"""
        # WHY: 배열 순서가 아니라 visit_sequence 값 기준 정렬 검증 —
        # 순서 맞는데 배열만 섞인 정상 출력이 폴백되지 않게.
        # 단 중복(1,2,2)·gap(1,2,4)은 여전히 거부 (진짜 malformation).
        ordered = sorted(places, key=lambda p: p.visit_sequence)
        for i, place in enumerate(ordered, start=1):
            if place.visit_sequence != i:
                # place_name이 None일 수 있으므로 안전하게 처리
                name = place.place_name or f"place_id={place.place_id}"
                raise ValueError(
                    f"visit_sequence가 연속적이지 않습니다: "
                    f"예상 {i}, 실제 {place.visit_sequence} ({name})"
                )
        return ordered   # 정렬된 리스트 반환 → 다운스트림이 순서대로 받음
