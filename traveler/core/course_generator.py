"""
여행 코스 생성 핵심 모듈.

파이프라인 위치:
  ContextData → ★ generate_course() → StreamingCourseModel → FastAPI

이수연의 RAG 파이프라인이 ContextData를 채워서 이 함수에 전달한다.
"""

import logging
import time

from pydantic import ValidationError

from traveler.core.llm_client import MODEL, get_client
from traveler.core.prompt import build_context_message, get_system_prompt, normalize_travel_mode
from traveler.rag.types import ContextData
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel

# WHY __name__: 로그 출처가 "traveler.core.course_generator"로 표시되어 디버깅이 쉬워진다
logger = logging.getLogger(__name__)

# WHY 상수로 노출하는가(PM 리뷰 반영): eval/metrics.py가 fallback 여부를 판정하려면
# 이 문자열과 정확히 같아야 하는데, 예전엔 두 파일에 리터럴로 복제돼 있었다.
# 한쪽만 고치면 폴백 감지가 조용히 깨지는 위험이 있어 여기 한 곳만 진실로 두고
# metrics.py가 import해서 쓴다.
FALLBACK_BOT_MESSAGE = "죄송해요, 지금 코스를 만드는 데 문제가 있었어요. 잠시 후 다시 시도해주세요."


def _build_fallback(ctx: ContextData) -> StreamingCourseModel:
    """
    GPT 출력이 JSON 파싱/스키마 검증에 실패했을 때 반환하는 안전 응답.

    # WHY: 파싱이 깨져도 generate_course()는 항상 유효한 StreamingCourseModel을
    # 반환해야 한다(Graceful Degradation). candidate_spots가 있으면 그중
    # 첫 번째 place_id를 쓰고, 없으면 "1"을 쓴다(recommended_places는 min_length=1).
    """
    fallback_place_id = (
        ctx.candidate_spots[0].place_id if ctx.candidate_spots else "1"
    )
    return StreamingCourseModel(
        bot_message=FALLBACK_BOT_MESSAGE,
        recommended_places=[
            RecommendedPlaceModel(place_id=fallback_place_id, visit_sequence=1)
        ],
    )


async def generate_course(
    ctx: ContextData,
    travel_mode: str = "타지",
    *,
    temperature: float = 0.3,
) -> StreamingCourseModel:
    """
    여행 컨텍스트를 받아 GPT-4o 스트리밍으로 코스를 생성하고 Pydantic으로 검증한다.

    # INPUT:  ContextData — 이수연의 RAG 파이프라인이 채워서 전달하는 컨텍스트
    #         travel_mode — Router(Sprint 6)가 감지한 여행 유형 ("근교"|"타지"|"장거리")
    #         temperature — keyword-only, 기본값 0.3(프로덕션 동작 무변경).
    #           Sprint 5 eval 파이프라인(eval/run_judge.py)이 재현성 확보를 위해
    #           0을 명시적으로 전달하는 용도로 노출됐다(D3, 설계 spec §5.2).
    # OUTPUT: StreamingCourseModel — Pydantic 검증 완료된 코스 객체
    # COST:   GPT-4o 기준 입력 $2.5/1M, 출력 $10/1M 토큰
    # ⚠️ 주의: travel_mode는 Sprint 6 Router 구현 전까지 호출부에서 직접 전달

    스트리밍 백엔드 구조 (A 방식):
      - stream=True: GPT 토큰을 기다리지 않고 생성 즉시 받음
      - full_text에 누적 → 스트리밍 완료 후 Pydantic 일괄 검증
    """
    travel_mode = normalize_travel_mode(travel_mode)
    # WHY build_context_message? TravelRequest가 아닌 ContextData를 프롬프트로 변환.
    # 이 함수가 두 팀의 경계면(이수연 ↔ 정훈)을 이어주는 브릿지 역할.
    user_message = build_context_message(ctx)
    t_start = time.time()

    # WHY get_system_prompt(travel_mode)?
    # 여행 모드에 따라 다른 SystemPrompt 선택. 호출부는 내용을 몰라도 됨.
    # ⚠️ 주의: API 호출/스트리밍은 타임아웃·네트워크 끊김·rate limit 등
    # ValidationError가 아닌 예외를 던질 수 있다. 이 경우에도 데모가 멈추지 않도록
    # fallback으로 흡수한다(Graceful Degradation, MVP1 P0 요구사항).
    try:
        stream = await get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": get_system_prompt(travel_mode)},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},   # JSON만 반환
            temperature=temperature,
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
    except Exception as e:
        # ⚠️ 주의: 데모 중단 방지 — API 호출/스트리밍 오류는 로그로만 남기고 안전 응답으로 계속
        logger.warning("API 호출 실패, fallback 사용: %s", e)
        return _build_fallback(ctx)

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
    # Pydantic 검증 — JSON 문법 오류와 스키마 불일치 모두 ValidationError로 들어옴
    try:
        course = StreamingCourseModel.model_validate_json(full_text)
    except ValidationError as e:
        # ⚠️ 주의: 데모 중단 방지 — 원인은 로그로만 남기고 안전 응답으로 계속
        logger.warning("코스 생성 실패, fallback 사용: %s", e)
        course = _build_fallback(ctx)
    return course
