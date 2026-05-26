"""
Pydantic v2 기반 CourseJSON 스키마 — Guardrails 1단계.

GPT-4o 출력이 이 스키마를 통과해야만 사용자에게 전달된다.
실패 시: 재프롬프트 또는 부분 재생성.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class CoursePlaceModel(BaseModel):
    """
    여행 코스 내 장소 하나.
    BaseModel을 상속하면 Pydantic이 타입 검증을 자동으로 수행.
    """
    order: int = Field(ge=1)          # ge=1: 1 이상의 정수만 허용
    place_name: str
    category: Literal["음식", "관광", "체험", "휴식"]  # 이 4개 외 값 오면 자동 거부
    address: str
    duration_min: int = Field(ge=10, le=480)  # 10분 이상 8시간 이하
    reason: str
    estimated_cost: str
    indoor_outdoor: Literal["실내", "실외", "실내/실외 혼합"]

    @field_validator("place_name", "address", "reason")
    @classmethod
    def not_empty(cls, v: str) -> str:
        """빈 문자열 거부 — LLM이 간혹 "" 를 반환할 때 방어"""
        if not v.strip():
            raise ValueError("빈 문자열은 허용되지 않습니다.")
        return v


class ConstraintCheckModel(BaseModel):
    """CoT 제약 체크 결과. LLM이 실제로 제약을 확인했는지 검증."""
    time: str
    weather: str
    route: str
    special: str


class CourseJSONModel(BaseModel):
    """
    GPT-4o 전체 응답의 최상위 스키마.
    이 모델 파싱에 실패하면 Guardrails가 재프롬프트를 트리거한다.
    """
    constraint_check: ConstraintCheckModel
    course: list[CoursePlaceModel] = Field(min_length=1, max_length=10)
    total_duration_min: int = Field(ge=30)
    curator_comment: str

    @field_validator("course")
    @classmethod
    def check_order_sequential(cls, places: list[CoursePlaceModel]) -> list[CoursePlaceModel]:
        """order 필드가 1, 2, 3... 순서인지 확인"""
        for i, place in enumerate(places, start=1):
            if place.order != i:
                raise ValueError(f"order가 연속적이지 않습니다: 예상 {i}, 실제 {place.order}")
        return places
