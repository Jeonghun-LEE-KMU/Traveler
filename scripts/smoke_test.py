"""
터미널 smoke test — 정훈 LLM 코어 파이프라인 E2E 수동 실행 (Sprint 3 P0).

WHY 이 스크립트가 필요한가:
  main.py(이수연 임시 FastAPI 프로토타입)는 정훈의 course_generator/prompt/
  ContextData 파이프라인을 쓰지 않는다. 따라서 "고정 입력 시나리오 터미널
  smoke test"는 main.py를 거치지 않고 정훈 코어를 직접 호출해 확인해야 한다.

검증하는 것:
  고정 시나리오 "경주에서 5시간 자차로 여행하고 싶어요"를
    get_context(user_query=...) → ContextData
    → generate_course(ctx)     → StreamingCourseModel
  순서로 끝까지 흘려보내, 파이프라인이 에러 없이 완주하고 유효한 코스 JSON을
  반환하는지를 사람이 눈으로 확인한다.

  OPENAI_API_KEY가 없으면 generate_course() 내부 except가 _build_fallback(ctx)로
  흡수하므로, 키 없이도 죽지 않고 폴백 응답을 출력한다 — 이것 자체가 P0의
  graceful degradation 증거다.

  # 실행 예시: python scripts/smoke_test.py
  # 실제 GPT-4o 응답을 보려면: export OPENAI_API_KEY=... 후 재실행

주의: retriever.py는 Sprint 1 DUMMY라 duration_hours=4, transport="자차"가 내부에
  하드코딩돼 있다. user_query에 "5시간"이라 써도 ctx.duration_hours는 4로 고정된다
  (Sprint 3에서 이수연 실제 RAG로 교체될 부분 — 이 스크립트의 범위 밖).
"""

import asyncio
import sys
from dataclasses import asdict
from pathlib import Path

# WHY sys.path 삽입: 이 스크립트를 어느 위치에서 실행하든(예: python scripts/smoke_test.py)
#   리포 루트가 import 경로에 들어가야 `from traveler...`가 풀린다.
#   parents[1] = scripts/의 부모 = 리포 루트(mobility_rag/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402  (sys.path 설정 후 import해야 하므로 위치 고정)

from traveler.core.course_generator import generate_course  # noqa: E402
from traveler.rag.retriever import get_context  # noqa: E402


# 고정 시나리오 — 변경하지 말 것(재현 가능한 smoke 입력)
SCENARIO = "경주에서 5시간 자차로 여행하고 싶어요"


async def run() -> None:
    """고정 시나리오를 파이프라인 끝까지 흘려보내고 결과를 JSON으로 출력한다."""
    print("=" * 60)
    print(f"[입력 시나리오] {SCENARIO}")
    print("=" * 60)

    # 1단계: RAG 컨텍스트 (현재 DUMMY — 경주 5개 장소 하드코딩)
    ctx = get_context(user_query=SCENARIO)
    print("\n[1단계] get_context() → ContextData")
    print(json.dumps(asdict(ctx), ensure_ascii=False, indent=2))

    # 2단계: 코스 생성 (키 없으면 내부 fallback으로 흡수 — graceful degradation)
    course = await generate_course(ctx, travel_mode="근교")
    print("\n[2단계] generate_course() → StreamingCourseModel")
    print(course.model_dump_json(indent=2))

    print("\n" + "=" * 60)
    print("[완료] 파이프라인이 에러 없이 끝까지 실행됨.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
