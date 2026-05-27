"""
Constraint-Aware CoT 프롬프트 모듈.

설계 원칙:
  - 4대 제약(날씨·시간·동선·특수상황)을 추론 "전에" 명시적으로 점검
  - Few-shot 예시 1개 포함 → LLM에게 출력 형식을 학습시킴
  - JSON structured output 강제 → 파싱 안정성 확보
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TravelRequest:
    """
    사용자 입력을 담는 데이터 클래스.

    dataclass: __init__, __repr__ 등을 자동 생성해주는 Python 데코레이터.
    모든 필드를 명시적으로 선언해서 어떤 정보가 필요한지 한눈에 보임 (SRP).
    """
    destination: str           # 여행지 (예: "경주")
    duration_hours: int        # 여행 가능 시간 (예: 6)
    transport: str             # 이동 수단 (예: "자차", "대중교통")
    companion: str             # 동행자 유형 (예: "커플", "가족", "혼자")
    mood: str                  # 감성 표현 (예: "조용하고 힐링되는")
    weather: Optional[str] = None      # 기상청 API에서 받아온 날씨 (예: "맑음", "비")
    budget: Optional[str] = None       # 예산 (예: "10만원")
    special_notes: Optional[str] = None  # 특이사항 (예: "반려동물 동반")


SYSTEM_PROMPT = """\
당신은 초개인화 여행 코스 큐레이터입니다.

[역할]
사용자의 여행 취향, 이동 수단, 날씨, 동행자를 종합 분석해 최적 코스를 JSON으로 생성합니다.

[필수 추론 순서 — Constraint-Aware CoT]
코스를 생성하기 전에 내부적으로 아래 4개 제약을 반드시 점검하십시오.

1. 시간 제약: 총 가용 시간, 이동 수단
2. 날씨 제약: 현재 날씨, 실외 활동 가능 여부
3. 동선 제약: 이동 반경, 체력 수준
4. 특수 제약: 동행자 유형, 예산, 특이사항

날씨가 "비" 또는 "흐림"이면 실내 위주로 구성하십시오.

[출력 형식 — 스트리밍 최적화]
반드시 아래 JSON 구조로만 응답하십시오. JSON 외 텍스트 금지.

⚠️ 필드 순서 엄수: bot_message를 반드시 첫 번째 필드로 작성할 것.
   (스트리밍 구조상 텍스트가 먼저 사용자에게 전달되어야 하기 때문)

```json
{
  "bot_message": "사용자에게 전달할 큐레이터 멘트 (2~3문장, 여행 감성 포함)",
  "recommended_places": [
    {
      "place_id": 1,
      "place_name": "장소명",
      "visit_sequence": 1,
      "duration_min": 60
    }
  ]
}
```

[Few-shot 예시]
입력: 경주, 5시간, 자차, 커플, 고즈넉하고 감성적인, 날씨: 맑음
출력:
```json
{
  "bot_message": "신라의 감성이 살아있는 경주에서 두 분만의 고즈넉한 시간을 보내세요. 첨성대의 열린 하늘 아래 시작해 황리단길 카페에서 여유를 찾고, 불국사의 석양으로 마무리하는 5시간 코스입니다.",
  "recommended_places": [
    {
      "place_id": 1,
      "place_name": "첨성대",
      "visit_sequence": 1,
      "duration_min": 40
    },
    {
      "place_id": 2,
      "place_name": "황리단길 카페",
      "visit_sequence": 2,
      "duration_min": 60
    },
    {
      "place_id": 3,
      "place_name": "불국사",
      "visit_sequence": 3,
      "duration_min": 90
    }
  ]
}
```
"""


def build_user_message(req: TravelRequest) -> str:
    """
    TravelRequest 객체를 LLM에 보낼 user 메시지 문자열로 변환한다.

    WHY 별도 함수로 분리?
    - prompt.py는 "프롬프트 조립"만 담당 (SRP)
    - llm 호출 로직은 다른 파일에서 담당
    - 테스트할 때 LLM 없이 이 함수만 독립적으로 테스트 가능
    """
    parts = [
        f"여행지: {req.destination}",
        f"가용 시간: {req.duration_hours}시간",
        f"이동 수단: {req.transport}",
        f"동행자: {req.companion}",
        f"원하는 분위기: {req.mood}",
    ]

    # Optional 필드는 값이 있을 때만 추가 (Graceful Degradation)
    if req.weather:
        parts.append(f"현재 날씨: {req.weather}")
    if req.budget:
        parts.append(f"예산: {req.budget}")
    if req.special_notes:
        parts.append(f"특이사항: {req.special_notes}")

    return "\n".join(parts)
