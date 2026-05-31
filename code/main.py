"""
트래블러 CLI 데모 러너.

[역할 변경 — 2026-05-27]
이전: generate_course() 로직이 이 파일 안에 있었음 (Phase 2 스트리밍 확인용)
이후: 얇은 진입점(thin runner)으로 변경.
      비즈니스 로직은 traveler/core/course_generator.py로 분리.

WHY 분리했는가?
  Sprint 2에서 FastAPI 서버(traveler/api/routes.py)가 생기면
  generate_course()를 거기서도 import해야 한다.
  로직이 main.py에 있으면 "CLI에서만 쓸 수 있는 함수"가 되어버린다.
  course_generator.py에 있으면 CLI와 FastAPI 둘 다 동일하게 import해서 쓸 수 있다.

[ContextData 사용 — T1]
이전: TravelRequest(destination=...) — LLM 엔지니어 내부 데이터 클래스
이후: ContextData(user_query=..., candidate_spots=[...]) — 팀 인터페이스 계약
      이수연의 RAG 파이프라인이 실제로 만들어줄 형식으로 데모를 작성.
"""

import asyncio
import json
from dotenv import load_dotenv

from traveler.rag.types import ContextData, SpotCandidate
from traveler.core.prompt import build_context_message
from traveler.core.course_generator import generate_course

load_dotenv()


async def main():
    # ─── 데모용 ContextData 직접 생성 ───────────────────────────────────────
    # 실제 서비스에서는 이수연의 RAG 파이프라인이 이 객체를 만들어 전달한다.
    # 지금은 Sprint 1 데모이므로 경주 여행 시나리오를 하드코딩.
    ctx = ContextData(
        user_query="조용하고 고즈넉한 경주 여행 코스 추천해줘",
        current_lat=35.8562,    # 경주시청 근처 좌표
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[
            # SpotCandidate(place_id, place_name, category, distance_min, indoor)
            SpotCandidate(1, "첨성대",          "관광", 10.0, False),
            SpotCandidate(2, "황리단길 카페",   "음식", 12.0, True),
            SpotCandidate(3, "불국사",          "관광", 20.0, False),
            SpotCandidate(4, "국립경주박물관",  "관광", 15.0, True),
            SpotCandidate(5, "동궁과 월지",     "관광",  8.0, False),
        ],
        duration_hours=5,
        transport="자차",
        companion="커플",
        mood="조용하고 고즈넉한",
    )

    print("=== 여행 요청 (ContextData) ===")
    print(build_context_message(ctx))
    print("\n=== GPT-4o 코스 생성 중... ===")

    course = await generate_course(ctx, travel_mode="타지")

    print("\n=== 파싱된 결과 (Pydantic 검증 완료) ===")
    print(json.dumps(course.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
