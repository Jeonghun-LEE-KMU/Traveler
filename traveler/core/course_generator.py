"""
여행 코스 생성 핵심 모듈.

파이프라인 위치:
  ContextData (이수연 RAG 출력)
    → ★ [이 모듈: generate_course()]
    → StreamingCourseModel (Pydantic 검증 완료)
    → FastAPI → 지도 시각화

WHY main.py에서 분리했는가? (T2)
  generate_course()는 비즈니스 로직이고, main.py는 CLI 진입점이다.
  FastAPI 서버(Sprint 2)가 생기면 main.py 대신 api/routes.py에서
  이 함수를 import해야 한다. 로직이 main.py에 있으면 그때 다시 옮겨야 함.
  지금 분리해두면 Sprint 2에서 import 경로 하나만 바꾸면 끝.

WHY ContextData를 받는가? (T1)
  이전: generate_course(req: TravelRequest) — LLM 엔지니어가 만든 내부 데이터 클래스
  이후: generate_course(ctx: ContextData) — 팀 인터페이스 계약 (rag/types.py)
  이수연의 RAG 파이프라인이 ContextData를 만들어서 이 함수에 넘긴다.
  TravelRequest는 이수연이 모르는 내부 구조이므로 인터페이스로 쓰면 안 됨.
"""

import time

from traveler.core.llm_client import MODEL, get_client
from traveler.core.prompt import build_context_message, get_system_prompt
from traveler.rag.types import ContextData
from traveler.validation.schema import StreamingCourseModel


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

    print("\n[스트리밍 시작]")
    async for chunk in stream:
        # delta.content: 이번 청크에서 새로 생성된 텍스트 조각
        # 마지막 청크는 content가 None → 조건문으로 스킵
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            print(delta, end="", flush=True)   # 터미널 실시간 출력
            full_text += delta                 # 버퍼 누적
        if chunk.usage:
            usage = chunk.usage

    print()   # 스트리밍 완료 후 줄바꿈
    latency = time.time() - t_start

    # 토큰 사용량 로그 (비용 추적용)
    if usage:
        print(f"\n[LLM LOG] model={MODEL}")
        print(f"  입력 토큰: {usage.prompt_tokens:,}")
        print(f"  출력 토큰: {usage.completion_tokens:,}")
        print(f"  전체 토큰: {usage.total_tokens:,}")
        print(f"  응답 시간: {latency:.2f}초")
        cost = usage.prompt_tokens * 2.5e-6 + usage.completion_tokens * 10e-6
        print(f"  예상 비용: ${cost:.5f} (≈ {cost * 1350:.2f}원)")

    # 스트리밍 완료 후 full_text는 완전한 JSON 문자열
    # Pydantic 검증 — 스키마 불일치 시 ValidationError 발생
    course = StreamingCourseModel.model_validate_json(full_text)
    return course
