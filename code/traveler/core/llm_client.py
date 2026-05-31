"""
OpenAI 비동기 클라이언트 싱글턴 모듈.

파이프라인 위치: 모든 LLM 호출의 진입점.
  사용자 입력 → 프롬프트 조립 → ★ [이 모듈] → GPT-4o → Guardrails

[T6 — 2026-05-27]
  OPENAI_API_KEY 미설정 시 서버 시작 시점에 즉시 RuntimeError.
  이전: 첫 번째 API 호출 때 OpenAI SDK 내부에서 에러 → 원인 파악이 어려웠음.
  이후: get_client() 첫 호출 시 즉시 RuntimeError → 서버 시작 로그에서 바로 확인 가능.
"""

import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# WHY: 모델명을 상수로 분리 — 코드 어디서도 문자열 "gpt-4o"를 하드코딩하지 않기 위해
# 모델을 교체할 때 이 한 줄만 바꾸면 전체 프로젝트에 반영됨
MODEL = "gpt-4o"        # 코스 생성용 (Phase 2~3)

# LLM-as-Judge 평가용 모델 — Phase 8 구현 예정
# WHY 지금 정의하는가: 나중에 추가하면 어디서 쓸지 흩어져서 추적이 어려움.
# 지금 여기에 선언해두면 Phase 8에서 import만 추가하면 됨.
JUDGE_MODEL = "gpt-4o"  # Phase 8: 골든셋 평가 + 회귀 테스트용

# ⚠️ 주의: 모듈 레벨에서 즉시 초기화하지 않음 (Lazy Init 패턴)
# 즉시 초기화하면 import 시점에 API 키를 읽어 테스트 환경에서 에러 발생
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """
    # WHY: Lazy Initialization 패턴 — import 시점이 아닌 실제 호출 시점에 API 키를 읽음
    # INPUT: 없음 (환경변수 OPENAI_API_KEY를 내부에서 읽음)
    # OUTPUT: AsyncOpenAI 싱글턴 인스턴스 → 모든 LLM 호출 함수에서 공유
    # ⚠️ OPENAI_API_KEY 미설정 시 RuntimeError 즉시 발생 (T6)

    싱글턴 효과: _client가 이미 있으면 새로 만들지 않고 재사용.
    AsyncOpenAI는 내부적으로 HTTP 연결 풀을 관리하므로 매번 새로 만들면 비효율.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")

        # T6: API 키 없으면 즉시 실패 (Fail Fast 패턴)
        # WHY Fail Fast? 키 없이 서버가 돌다가 첫 사용자 요청에서 터지는 것보다
        # 서버 시작 시점에 바로 에러를 내는 게 훨씬 디버깅하기 쉽다.
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
                ".env 파일을 확인하거나 환경변수를 직접 설정하세요. "
                "(.env.example 참고)"
            )

        _client = AsyncOpenAI(api_key=api_key)
    return _client
