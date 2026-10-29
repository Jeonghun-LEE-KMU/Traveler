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
from traveler.core.course_schedule import apply_schedule
from traveler.core.prompt import build_context_message, get_system_prompt, normalize_travel_mode
from traveler.rag.types import ContextData
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel, TimeBudgetSummary

# WHY __name__: 로그 출처가 "traveler.core.course_generator"로 표시되어 디버깅이 쉬워진다
logger = logging.getLogger(__name__)

# WHY 상수로 노출하는가(PM 리뷰 반영): eval/metrics.py가 fallback 여부를 판정하려면
# 이 문자열과 정확히 같아야 하는데, 예전엔 두 파일에 리터럴로 복제돼 있었다.
# 한쪽만 고치면 폴백 감지가 조용히 깨지는 위험이 있어 여기 한 곳만 진실로 두고
# metrics.py가 import해서 쓴다.
FALLBACK_BOT_MESSAGE = "죄송해요, 지금 코스를 만드는 데 문제가 있었어요. 잠시 후 다시 시도해주세요."


def _build_fallback(ctx: ContextData, limit: int = 1) -> StreamingCourseModel:
    """
    GPT 출력이 JSON 파싱/스키마 검증에 실패했을 때 반환하는 안전 응답.

    # WHY: 파싱이 깨져도 generate_course()는 항상 유효한 StreamingCourseModel을
    # 반환해야 한다(Graceful Degradation). candidate_spots가 있으면 그중
    # 앞에서부터 limit개를 순서대로 쓰고, 없으면 "1" 자리표시자 하나를 쓴다
    # (recommended_places는 min_length=1).
    # ⚠️ 주의(PM 리뷰 반영, #29): retriever.py가 아직 place_id를 int로 채우는
    # 경로가 남아있다(#18 — 이수연 진행 중). str()로 감싸지 않으면 이 fallback
    # 자체가 RecommendedPlaceModel(place_id: str) 검증에서 ValidationError로
    # 죽어 Graceful Degradation의 최후 방어선이 뚫린다. #18이 int를 완전히
    # 없애도 이 캐스팅은 무해하게 남는다.
    """
    if not ctx.candidate_spots:
        return StreamingCourseModel(
            bot_message=FALLBACK_BOT_MESSAGE,
            recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1)],
        )
    return StreamingCourseModel(
        bot_message=FALLBACK_BOT_MESSAGE,
        recommended_places=[
            RecommendedPlaceModel(
                place_id=str(s.place_id), visit_sequence=i + 1, place_name=s.place_name or None
            )
            for i, s in enumerate(ctx.candidate_spots[:limit])
        ],
    )


def validate_course_schedule(
    course: StreamingCourseModel, ctx: ContextData, *, fill_minimum: bool = True
) -> StreamingCourseModel:
    """생성·평가·수정에 같은 예상 이동+체류시간 정책을 적용한다 (추가 LLM 호출 없음)."""
    # WHY: JSON 스키마에는 현재 위치·가용시간이 없으므로 문맥 기반 검증은 이 경계에서 한다.
    result = apply_schedule(course, ctx, fill_minimum=fill_minimum)
    timing = TimeBudgetSummary(
        budget_min=result.budget_min, travel_min=result.travel_min,
        stay_min=result.stay_min, total_min=result.total_min,
        status=result.status, note=result.note,
    )
    message = course.bot_message
    if message != FALLBACK_BOT_MESSAGE:
        # 재검증 때 이전 요약이 중복으로 붙지 않게 분리한다. 장소가 달라졌다면
        # 이전 LLM 설명을 사실처럼 유지하지 않고 최종 결과를 명확히 안내한다.
        message = message.split("\n\n[예상 일정]", 1)[0]
        if [p.place_id for p in result.places] != [p.place_id for p in course.recommended_places]:
            names = [p.place_name or p.place_id for p in result.places]
            message = ("가용시간을 고려한 최종 코스: " + " → ".join(names)
                       if names else "가용시간 안에 추천할 수 있는 장소를 찾지 못했습니다.")
        message += "\n\n" + result.note
    return StreamingCourseModel(
        bot_message=message, recommended_places=result.places, time_budget=timing,
    )


def parse_course(
    full_text: str, ctx: ContextData, *, fallback_limit: int = 1
) -> tuple[StreamingCourseModel, bool]:
    """
    GPT 출력 문자열을 검증하고, 실패하면 fallback으로 감싸 반환한다.

    # WHY: generate_course()(eval/run_judge.py가 채점하는 대상)와 main.py의
    # _stream_and_persist()가 각자 검증+fallback 로직을 따로 들고 있었다(#25).
    # 이러면 eval이 채점하는 코스와 실제 서비스가 만드는 코스의 검증 기준이
    # 서로 어긋날 수 있다. 이 함수 하나로 합쳐 둘 다 같은 경로를 타게 한다.
    """
    try:
        course = StreamingCourseModel.model_validate_json(full_text)
        is_fallback = False
    except ValidationError as e:
        logger.warning(
            "[FALLBACK] 코스 검증 실패, fallback 사용: %s | full_text[:200]=%s",
            e,
            full_text[:200],
        )
        course = _build_fallback(ctx, fallback_limit)
        is_fallback = True
    return validate_course_schedule(course, ctx), is_fallback


async def generate_course(
    ctx: ContextData,
    travel_mode: str = "타지",
    *,
    temperature: float = 0.3,
    refinement: str | None = None,
) -> StreamingCourseModel:
    """
    여행 컨텍스트를 받아 GPT-4o 스트리밍으로 코스를 생성하고 Pydantic으로 검증한다.

    # INPUT:  ContextData — 이수연의 RAG 파이프라인이 채워서 전달하는 컨텍스트
    #         travel_mode — Router(Sprint 6)가 감지한 여행 유형 ("근교"|"타지"|"장거리")
    #         temperature — keyword-only, 기본값 0.3(프로덕션 동작 무변경).
    #           Sprint 5 eval 파이프라인(eval/run_judge.py)이 재현성 확보를 위해
    #           0을 명시적으로 전달하는 용도로 노출됐다(D3, 설계 spec §5.2).
    #         refinement — keyword-only, 기본값 None(무변경). Sprint 7
    #           edit_intent.revise_course()가 "전체재생성" 인텐트에서 사용자의
    #           후속 발화("카페 중심으로 다시 짜줘")를 그대로 전달한다(D12,
    #           2026-08-30 설계 spec §6). 그대로 build_context_message로 흘려보낸다.
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
    user_message = build_context_message(ctx, refinement=refinement)
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
        logger.warning("[FALLBACK] API 호출 실패, fallback 사용: %s", e)
        return validate_course_schedule(_build_fallback(ctx), ctx)

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
    # ⚠️ 주의: 데모 중단 방지 — 실패해도 parse_course()가 안전 응답으로 감싸 계속 진행한다.
    # fallback_limit 기본값(1)을 그대로 둬 eval 채점 대상의 동작을 바꾸지 않는다.
    course, _ = parse_course(full_text, ctx)
    return course
