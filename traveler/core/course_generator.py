"""
여행 코스 생성 핵심 모듈.

파이프라인 위치:
  ContextData → ★ generate_course() → StreamingCourseModel → FastAPI

이수연의 RAG 파이프라인이 ContextData를 채워서 이 함수에 전달한다.
"""

import logging
import time

from traveler.core.llm_client import MODEL, get_client
from traveler.core.prompt import build_context_message, get_system_prompt
from traveler.rag.types import ContextData
from traveler.validation.schema import StreamingCourseModel

# WHY __name__: 로그 출처가 "traveler.core.course_generator"로 표시되어 디버깅이 쉬워진다
logger = logging.getLogger(__name__)


async def generate_course(
    ctx: ContextData,
    travel_mode: str = "타지",
) -> StreamingCourseModel:
    """
    여행 컨텍스트를 받아 GPT-4o 스트리밍으로 코스를 생성하고 Pydantic으로 검증한다.

    # INPUT:  ContextData — 이수연의 RAG 파이프라인이 채워서 전달하는 컨텍스트
    #         travel_mode — Router(Sprint 6)가 감지한 여행 유형 ("근교"|"타지"|"장거리")
    # OUTPUT: StreamingCourseModel — Pydantic 검증 완료된 코스 객체
    # COST:   GPT-4o 기준 입력 $2.5/1M, 출력 $10/1M 토큰
    # ⚠️ 주의: travel_mode는 Sprint 6 Router 구현 전까지 호출부에서 직접 전달

    스트리밍 백엔드 구조 (A 방식):
      - stream=True: GPT 토큰을 기다리지 않고 생성 즉시 받음
      - full_text에 누적 → 스트리밍 완료 후 Pydantic 일괄 검증
    """
    # WHY build_context_message? TravelRequest가 아닌 ContextData를 프롬프트로 변환.
    # 이 함수가 두 팀의 경계면(이수연 ↔ 정훈)을 이어주는 브릿지 역할.
    user_message = build_context_message(ctx)
    t_start = time.time()

    # WHY get_system_prompt(travel_mode)?
    # 여행 모드에 따라 다른 SystemPrompt 선택. 호출부는 내용을 몰라도 됨.
    stream = await get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": get_system_prompt(travel_mode)},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},   # JSON만 반환
        temperature=0.3,
        stream=True,
        stream_options={"include_usage": True},    # 마지막 청크에 usage 포함
    )

    # 스트리밍 수신 루프
    full_text = ""   # 토큰 누적 버퍼
    usage = None     # 마지막 청크에서 수집할 토큰 통계

    logger.debug("스트리밍 시작")
    async for chunk in stream:
        # delta.content: 이번 청크에서 새로 생성된 텍스트 조각
        # 마지막 청크는 content가 None → 조건문으로 스킵
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            full_text += delta                 # 버퍼 누적
        if chunk.usage:
            usage = chunk.usage

    latency = time.time() - t_start

    # 토큰 사용량 로그 (비용 추적용)
    if usage:
        cost = usage.prompt_tokens * 2.5e-6 + usage.completion_tokens * 10e-6
        logger.info(
            "[LLM LOG] model=%s | 입력=%s | 출력=%s | 전체=%s | 응답=%.2f초 | 비용=$%.5f(≈%.2f원)",
            MODEL,
            f"{usage.prompt_tokens:,}",
            f"{usage.completion_tokens:,}",
            f"{usage.total_tokens:,}",
            latency,
            cost,
            cost * 1350,
        )

    # 스트리밍 완료 후 full_text는 완전한 JSON 문자열
    # Pydantic 검증 — 스키마 불일치 시 ValidationError 발생
    course = StreamingCourseModel.model_validate_json(full_text)
    return course
