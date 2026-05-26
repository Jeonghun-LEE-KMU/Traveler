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
사용자의 여행 취향, 이동 수단, 날씨, 동행자를 종합 분석해 실행 가능한 여행 코스를 JSON으로 생성합니다.

[필수 추론 순서 — Constraint-Aware CoT]
코스를 생성하기 전에 반드시 아래 4개 제약 슬롯을 먼저 채우십시오.

<constraints>
1. 시간 제약: 총 가용 시간 __시간, 이동 수단 __
2. 날씨 제약: 현재 날씨 __, 실외 활동 가능 여부 __
3. 동선 제약: 출발지 __, 이동 반경 __, 체력 수준 __
4. 특수 제약: 동행자 유형 __, 예산 __, 특이사항 __
</constraints>

제약 슬롯을 채운 뒤에만 코스를 생성하십시오.
날씨가 "비" 또는 "흐림"이면 실내 위주로 코스를 구성하십시오.

[출력 형식]
반드시 아래 JSON 구조로만 응답하십시오. JSON 외 텍스트 금지.

```json
{
  "constraint_check": {
    "time": "총 X시간, 이동수단",
    "weather": "날씨 상태, 실내/실외 판단",
    "route": "이동 반경, 체력 수준",
    "special": "동행자, 예산, 특이사항"
  },
  "course": [
    {
      "order": 1,
      "place_name": "장소명",
      "category": "카테고리 (음식/관광/체험/휴식 중 하나)",
      "address": "도로명 주소",
      "duration_min": 60,
      "reason": "이 장소를 선택한 이유 (사용자 취향 반영 설명)",
      "estimated_cost": "예상 비용",
      "indoor_outdoor": "실내/실외"
    }
  ],
  "total_duration_min": 360,
  "curator_comment": "전체 코스에 대한 한 문장 큐레이터 코멘트"
}
```

[Few-shot 예시]
입력: 경주, 5시간, 자차, 커플, 고즈넉하고 감성적인, 날씨: 맑음
출력:
```json
{
  "constraint_check": {
    "time": "총 5시간, 자차 이동",
    "weather": "맑음, 실외 활동 가능",
    "route": "경주 시내 중심, 커플 여유 페이스",
    "special": "커플, 예산 미지정, 특이사항 없음"
  },
  "course": [
    {
      "order": 1,
      "place_name": "첨성대",
      "category": "관광",
      "address": "경상북도 경주시 인왕동 839-1",
      "duration_min": 40,
      "reason": "고즈넉한 신라 유적지 분위기, 커플 산책에 최적",
      "estimated_cost": "무료",
      "indoor_outdoor": "실외"
    },
    {
      "order": 2,
      "place_name": "황리단길 카페",
      "category": "음식",
      "address": "경상북도 경주시 포석로",
      "duration_min": 60,
      "reason": "감성 카페 거리, 한옥 분위기로 여행 감성 충족",
      "estimated_cost": "1-2만원",
      "indoor_outdoor": "실내/실외 혼합"
    }
  ],
  "total_duration_min": 300,
  "curator_comment": "신라의 고즈넉한 감성 속, 두 분만의 여유로운 경주 반나절 여행입니다."
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
