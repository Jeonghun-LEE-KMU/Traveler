import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o"

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """
    Lazy Initialization 패턴으로 클라이언트를 반환한다.

    WHY 함수로 감쌌나?
    - import 시점이 아닌 "실제 호출 시점"에 API 키를 읽는다
    - API 키 없이 import만 해야 하는 테스트 환경에서도 에러 안 남
    - _client가 이미 있으면 재사용 (싱글턴 효과 유지)
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client
