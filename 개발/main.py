"""
트래블러 진입점.
Phase 2: A 방식 스트리밍 확인용 스크립트.
Sprint 5 이후: FastAPI 서버로 교체 예정.

[A 방식 스트리밍 동작 원리]
- bot_message(첫 번째 필드): 글자 단위로 스트리밍 → 프론트가 즉시 채팅창에 표시
- recommended_places(두 번째 필드): JSON 완성 후 한 번에 파싱 → 지도 핀 일괄 표시
- 백엔드/LLM 코드는 B 방식과 동일. 프론트가 렌더링 타이밍만 다르게 처리.

터미널 실행 시 글자가 한 글자씩 출력되는 것을 확인할 수 있음.
"""

import asyncio
import json
import time
from dotenv import load_dotenv

from traveler.core.prompt import SYSTEM_PROMPT, TravelRequest, build_user_message
from traveler.validation.schema import StreamingCourseModel
from traveler.core.llm_client import get_client, MODEL

load_dotenv()


async def generate_course(req: TravelRequest) -> StreamingCourseModel:
    """
    여행 요청을 받아 GPT-4o 스트리밍으로 코스를 생성하고 Pydantic으로 검증한다.

    B 방식 스트리밍:
      - stream=True: GPT가 토큰을 만들 때마다 즉시 전달 (기다리지 않음)
      - stream_options: 마지막 청크에 usage(토큰 수) 포함 요청
      - full_text에 토큰을 누적 → 스트리밍 완료 후 Pydantic 검증
    """
    user_message = build_user_message(req)
    t_start = time.time()

    # stream=True: 응답을 한 번에 받지 않고, 토큰이 생성될 때마다 청크로 받음
    # stream_options: 스트리밍 모드에서도 토큰 사용량을 마지막 청크에 포함
    stream = await get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},  # JSON만 반환
        temperature=0.3,
        stream=True,
        stream_options={"include_usage": True},   # 마지막 청크에 usage 포함
    )

    # 스트리밍 수신 루프
    full_text = ""  # 토큰을 누적할 버퍼
    usage = None    # 마지막 청크에서 수집할 토큰 통계

    print("\n[스트리밍 시작]")
    async for chunk in stream:
        # chunk.choices[0].delta.content: 이번 청크에서 새로 생성된 텍스트 조각
        # 마지막 청크는 content가 None → "or """로 빈 문자열 처리
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            print(delta, end="", flush=True)  # 터미널에 실시간 출력
            full_text += delta               # 버퍼에 누적

        # usage는 stream_options로 요청했을 때만 마지막 청크에 포함됨
        if chunk.usage:
            usage = chunk.usage

    print()  # 스트리밍 완료 후 줄바꿈
    latency = time.time() - t_start

    # 토큰 사용량 로그
    if usage:
        print(f"\n[LLM LOG] model={MODEL}")
        print(f"  입력 토큰: {usage.prompt_tokens:,}")
        print(f"  출력 토큰: {usage.completion_tokens:,}")
        print(f"  전체 토큰: {usage.total_tokens:,}")
        print(f"  응답 시간: {latency:.2f}초")
        cost = usage.prompt_tokens * 2.5e-6 + usage.completion_tokens * 10e-6
        print(f"  예상 비용: ${cost:.5f} (≈ {cost * 1350:.2f}원)")

    # 스트리밍이 끝난 뒤 full_text는 완전한 JSON 문자열
    # Pydantic으로 검증 — 스키마 불일치 시 ValidationError 발생
    course = StreamingCourseModel.model_validate_json(full_text)
    return course


async def main():
    req = TravelRequest(
        destination="경주",
        duration_hours=5,
        transport="자차",
        companion="커플",
        mood="조용하고 고즈넉한",
        weather="맑음",
    )

    print("=== 여행 요청 ===")
    print(build_user_message(req))
    print("\n=== GPT-4o 코스 생성 중... ===")

    course = await generate_course(req)

    print("\n=== 파싱된 결과 (Pydantic 검증 완료) ===")
    print(json.dumps(course.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
