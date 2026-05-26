"""
트래블러 진입점.
Sprint 3: LLM 프롬프트 v1 동작 확인용 스크립트.
Sprint 5 이후: FastAPI 서버로 교체 예정.
"""

import asyncio
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

from traveler.core.prompt import SYSTEM_PROMPT, TravelRequest, build_user_message
from traveler.validation.schema import CourseJSONModel
from traveler.core.llm_client import get_client, MODEL

load_dotenv()


async def generate_course(req: TravelRequest) -> CourseJSONModel:
    """
    여행 요청을 받아 GPT-4o로 코스를 생성하고, Pydantic으로 검증한다.

    async def: 비동기 함수. await가 있는 줄에서 다른 작업을 실행할 수 있다.
    """
    user_message = build_user_message(req)

    # GPT-4o 호출
    response = await get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},  # JSON만 반환하도록 강제
        temperature=0.3,  # 코스 생성은 약간의 창의성 허용 (0이면 너무 뻔함)
    )

    raw_json = response.choices[0].message.content

    # Pydantic 검증 — 실패하면 ValidationError 발생
    course = CourseJSONModel.model_validate_json(raw_json)
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

    print("\n=== 생성된 코스 ===")
    print(json.dumps(course.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # asyncio.run(): 비동기 함수를 동기 환경(터미널)에서 실행하는 진입점
    asyncio.run(main())
