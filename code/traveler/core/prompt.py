"""
Constraint-Aware CoT 프롬프트 모듈.

설계 원칙:
  - 4대 제약(날씨·시간·동선·특수상황)을 추론 "전에" 명시적으로 점검
  - Triple Mode: 근교(🏡) · 타지(✈️) · 장거리(🚗) 3종 SystemPrompt
  - DRY 설계: 타지·장거리는 목적지 추천 로직 공유 (_DESTINATION_PERSONA)
    → _WAYPOINT_ADDON만 추가하면 장거리 프롬프트 완성
  - Few-shot 예시 3개 포함 — 모드 비종속 일반 케이스 (Sprint 1)
    → 모드별 Few-shot은 Sprint 4에서 3종 각각 보강 예정
  - JSON structured output 강제 → 파싱 안정성 확보

[v1.3 변경사항 — 2026-05-27]
  - build_context_message(ctx: ContextData) 추가
    이수연 RAG 출력(ContextData)을 GPT user 메시지 문자열로 변환하는 브릿지 함수.
    이전 build_user_message(req: TravelRequest)는 하위 호환성을 위해 유지.
"""

from dataclasses import dataclass
from typing import Optional

from traveler.rag.types import ContextData


@dataclass
class TravelRequest:
    """
    사용자 입력을 담는 데이터 클래스.

    [v1.2 변경사항]
    travel_mode 필드 추가 — Triple Mode 지원 ("근교" | "타지" | "장거리")
    기본값 "타지": 처음 방문하는 여행객이 가장 일반적인 케이스이므로 기본값으로 설정.
    """
    destination: str           # 여행지 (예: "경주")
    duration_hours: int        # 여행 가능 시간 (예: 6)
    transport: str             # 이동 수단 (예: "자차", "대중교통")
    companion: str             # 동행자 유형 (예: "커플", "가족", "혼자")
    mood: str                  # 감성 표현 (예: "조용하고 힐링되는")
    travel_mode: str = "타지"  # Triple Mode: "근교" | "타지" | "장거리"
    weather: Optional[str] = None      # 기상청 API 날씨 (예: "맑음", "비")
    budget: Optional[str] = None       # 예산 (예: "10만원")
    special_notes: Optional[str] = None  # 특이사항 (예: "반려동물 동반")


# ─── 버전 헤더 ──────────────────────────────────────────────────────────────
# prompt_id: travel_course_generator
# version: v1.3
# updated: 2026-05-27
# changed: build_context_message(ctx: ContextData) 추가
#          이수연 RAG 출력 → GPT user 메시지 브릿지 함수
#          (이전: v1.2 — Triple Mode SystemPrompt 3종 + travel_mode 필드)
# author: 이정훈
# tested_on: 미실시 (골든셋 구축 예정 — Phase 8)
# ─────────────────────────────────────────────────────────────────────────────


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [BLOCK 1] 공통 기반 — 3종 모드 전체가 공유하는 역할·CoT·출력 형식
#
# WHY 별도 상수로 분리?
#   - 3개 프롬프트가 동일한 80줄을 각각 복사하면 유지보수 악몽
#   - CoT 순서나 JSON 스키마가 바뀌면 이 한 곳만 수정하면 전체 반영
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_COMMON_BASE = """\
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
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [BLOCK 2] 모드별 페르소나 섹션
#
# WHY "_" prefix?
#   Python 컨벤션: 앞에 언더스코어(_) = "이 상수는 이 파일 내부용이야"
#   외부에서 직접 import해서 쓰지 말고, get_system_prompt() 함수를 통해서만 쓸 것
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 🏡 근교 모드 전용 페르소나 ───────────────────────────────────────────────
_NEARBY_PERSONA = """
[여행 유형 — 근교 드라이브 🏡]
사용자는 해당 지역 근처에 거주하거나 자주 방문하는 로컬입니다.
대표 관광지는 이미 알고 있으므로 숨겨진 명소 위주로 추천하십시오.

장소 구성 비율 (엄수):
  - 숨은 명소 (로컬 카페, 잘 알려지지 않은 산책로, 숨겨진 맛집): 70%
  - 대표 명소 (필요 시 1~2곳 포함 가능): 20%
  - 창의적 추천 (계절·날씨 특화 이색 경험): 10%

visited_spots 목록이 제공되면 해당 장소는 반드시 추천에서 제외하십시오.
"""

# ── ✈️🚗 타지·장거리 공통 페르소나 (목적지 추천 로직 동일) ─────────────────
#
# WHY 두 모드가 이 페르소나를 공유하는가?
#   사용자가 "목적지에 도착한 뒤 어디를 갈지" 추천하는 로직은 동일하다.
#   장거리 모드는 "목적지까지 가는 길" 정보가 추가될 뿐,
#   목적지에서의 동선 추천 방식 자체는 타지 여행객과 다르지 않다.
_DESTINATION_PERSONA = """
[여행 유형 — 목적지 여행 (타지 · 장거리 공통) ✈️🚗]
사용자는 해당 지역을 처음 방문하거나 낯선 여행객입니다.
그 지역의 정체성을 느낄 수 있도록 대표 명소와 숨겨진 명소를 균형 있게 추천하십시오.

장소 구성 비율 (엄수):
  - 대표 명소 (그 지역을 대표하는 랜드마크·필수 코스): 45%
  - 숨은 명소 (로컬이 즐겨 찾는 장소, 덜 알려진 명소): 45%
  - 창의적 추천 (계절·날씨 특화 이색 경험): 10%
"""

# ── 🚗 장거리 전용 추가 지침 — _DESTINATION_PERSONA 뒤에 덧붙임 ─────────────
#
# WHY 별도 섹션으로 분리?
#   장거리 = 타지 목적지 로직(_DESTINATION_PERSONA) + 경유지 안내(_WAYPOINT_ADDON)
#   덧셈 구조로 설계했기 때문에 장거리 프롬프트에서만 이 블록이 삽입된다.
#   Phase 5에서 Tmap API가 연동되면 이 블록만 교체하면 된다.
_WAYPOINT_ADDON = """
[추가 지침 — 장거리 경유지 안내 🚗]
사용자는 현재 위치에서 목적지까지 직접 운전 중입니다.
bot_message 안에 아래 정보를 간략히 포함하십시오.
  - 추천 경유 휴게소 또는 드라이브 스팟 1~2곳
  - 대략적인 주유 시점 안내 (장거리일 경우)

⚠️ 경유지 정보는 bot_message 텍스트로만 안내하십시오.
   recommended_places에는 목적지 장소만 포함합니다.
   (Phase 5에서 Tmap searchPoiAlong API 연동 시 경유지 실시간 데이터로 고도화 예정)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [BLOCK 3] Few-shot 예시 섹션 (모드 비종속 — Sprint 1)
#
# WHY 모드 비종속 예시를 먼저 쓰는가?
#   Few-shot의 목적은 LLM에게 "출력 형식 + 제약 처리 방법"을 학습시키는 것.
#   기본적인 날씨·동행자 제약 처리는 모든 모드에서 공통 → 3개 공통 예시로 충분.
#   모드별 세부 뉘앙스(숨은명소 vs 대표명소 비율)는 Sprint 4에서 모드별 추가 예정.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_FEW_SHOT_SECTION = """
[Few-shot 예시 1 — 맑음, 커플, 자차]
입력: 경주, 5시간, 자차, 커플, 고즈넉하고 감성적인, 날씨: 맑음
출력:
```json
{
  "bot_message": "신라의 감성이 살아있는 경주에서 두 분만의 고즈넉한 시간을 보내세요. 첨성대의 열린 하늘 아래 시작해 황리단길 카페에서 여유를 찾고, 불국사의 석양으로 마무리하는 5시간 코스입니다.",
  "recommended_places": [
    {"place_id": 1, "place_name": "첨성대", "visit_sequence": 1, "duration_min": 40},
    {"place_id": 2, "place_name": "황리단길 카페", "visit_sequence": 2, "duration_min": 60},
    {"place_id": 3, "place_name": "불국사", "visit_sequence": 3, "duration_min": 90}
  ]
}
```

[Few-shot 예시 2 — 비, 혼자, 대중교통 → 날씨 제약 반영: 실내 위주]
입력: 제주, 6시간, 대중교통, 혼자, 감각적이고 트렌디한, 날씨: 비
출력:
```json
{
  "bot_message": "비 내리는 제주도, 오히려 실내에서 제주의 감성을 더 깊이 느낄 수 있는 날이에요. 넥슨컴퓨터박물관의 독특한 전시로 시작해 제주 현대미술관에서 감각을 깨우고, 애월 감성 카페에서 빗소리를 들으며 마무리하는 코스입니다.",
  "recommended_places": [
    {"place_id": 4, "place_name": "넥슨컴퓨터박물관", "visit_sequence": 1, "duration_min": 90},
    {"place_id": 5, "place_name": "제주 현대미술관", "visit_sequence": 2, "duration_min": 80},
    {"place_id": 6, "place_name": "애월 감성 카페", "visit_sequence": 3, "duration_min": 60}
  ]
}
```

[Few-shot 예시 3 — 맑음, 가족(유아 동반), 자차 → 동행자 제약 반영: 키즈 친화]
입력: 부산, 4시간, 자차, 가족(유아 동반), 신나고 활동적인, 날씨: 맑음
출력:
```json
{
  "bot_message": "아이와 함께하는 부산 나들이, 온 가족이 신나게 즐길 수 있는 코스예요. 어린이 체험 공간이 가득한 아쿠아리움에서 시작해 광안리 해변에서 바닷바람을 맞으며 뛰어놀고, 마지막은 감천문화마을에서 알록달록한 골목길 산책으로 마무리합니다.",
  "recommended_places": [
    {"place_id": 7, "place_name": "부산 아쿠아리움", "visit_sequence": 1, "duration_min": 90},
    {"place_id": 8, "place_name": "광안리 해수욕장", "visit_sequence": 2, "duration_min": 60},
    {"place_id": 9, "place_name": "감천문화마을", "visit_sequence": 3, "duration_min": 70}
  ]
}
```
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [BLOCK 4] Triple Mode SystemPrompt 3종 조립
#
# 핵심 설계 시각화:
#
#   SYSTEM_PROMPT_TOURIST
#     = _COMMON_BASE + _DESTINATION_PERSONA + _FEW_SHOT_SECTION
#
#   SYSTEM_PROMPT_LONGDRIVE
#     = _COMMON_BASE + _DESTINATION_PERSONA + _WAYPOINT_ADDON + _FEW_SHOT_SECTION
#                      ↑ 타지와 동일 (DRY)   ↑ 장거리만 추가
#
#   SYSTEM_PROMPT_NEARBY
#     = _COMMON_BASE + _NEARBY_PERSONA + _FEW_SHOT_SECTION
#
# WHY Python string concatenation (+)?
#   LLM에게 전달되는 것은 결국 하나의 문자열이다.
#   Python 상수끼리 + 로 연결하는 건 컴파일 타임에 처리되어 런타임 오버헤드 없음.
#   결과는 일반 문자열과 동일하지만, 코드에서 구조가 한눈에 보인다는 것이 장점.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT_TOURIST = (
    _COMMON_BASE
    + _DESTINATION_PERSONA
    + _FEW_SHOT_SECTION
)

SYSTEM_PROMPT_LONGDRIVE = (
    _COMMON_BASE
    + _DESTINATION_PERSONA   # WHY: 목적지 추천 로직은 타지와 동일 — 중복 없이 재사용
    + _WAYPOINT_ADDON        # WHY: 장거리만 필요한 경유지 안내를 덧붙임
    + _FEW_SHOT_SECTION
)

SYSTEM_PROMPT_NEARBY = (
    _COMMON_BASE
    + _NEARBY_PERSONA
    + _FEW_SHOT_SECTION
)

# WHY 하위 호환성 유지?
#   기존에 SYSTEM_PROMPT를 직접 import해 쓰는 코드가 있다면 깨지지 않도록 유지.
#   새 코드는 반드시 get_system_prompt()를 통해 접근할 것.
SYSTEM_PROMPT = SYSTEM_PROMPT_TOURIST


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [BLOCK 5] 모드별 프롬프트 선택 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# WHY 딕셔너리로 매핑하는가?
#   if-elif 체인: 새 모드 추가 시 함수 내부를 수정해야 함 (OCP 위반)
#   dict.get(): 새 모드는 딕셔너리에 한 줄만 추가 — 함수 자체는 건드리지 않아도 됨
#   이 패턴을 "First-class Function 패턴" 또는 "Dispatch Table"이라고 부름
_PROMPT_MAP: dict[str, str] = {
    "근교":  SYSTEM_PROMPT_NEARBY,
    "타지":  SYSTEM_PROMPT_TOURIST,
    "장거리": SYSTEM_PROMPT_LONGDRIVE,
}


def get_system_prompt(mode: str) -> str:
    """
    여행 모드에 맞는 SystemPrompt 문자열을 반환한다.

    # WHY: 호출부(generate_course)가 프롬프트 내용을 몰라도 되도록 캡슐화
    # INPUT: travel_mode ("근교" | "타지" | "장거리")
    # OUTPUT: 해당 모드의 SystemPrompt 문자열 (수백 줄짜리 프롬프트)
    # ⚠️ Graceful Degradation: 알 수 없는 모드 → "타지" 프롬프트로 폴백
    #    시스템이 죽지 않고 가장 범용적인 타지 모드로 동작함
    """
    return _PROMPT_MAP.get(mode, SYSTEM_PROMPT_TOURIST)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [BLOCK 6] user 메시지 조립 함수 (2종)
#
# build_context_message(ctx: ContextData) — ★ 현재 메인 인터페이스 (v1.3)
#   이수연의 RAG 출력(ContextData)을 LLM user 메시지로 변환.
#   course_generator.py가 이 함수를 사용함.
#
# build_user_message(req: TravelRequest)  — 하위 호환성용 (v1.2)
#   직접 TravelRequest를 만들어 쓰는 레거시 코드용으로 유지.
#   새 코드는 build_context_message()를 사용할 것.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_context_message(ctx: ContextData) -> str:
    """
    ContextData 객체를 LLM에 보낼 user 메시지 문자열로 변환한다.

    # WHY 이 함수가 필요한가?
    #   이수연의 RAG 파이프라인은 ContextData를 출력한다.
    #   GPT-4o는 문자열(user message)을 입력으로 받는다.
    #   이 함수가 두 형식을 이어주는 브릿지 역할을 한다.
    #   이수연은 ContextData만 만들면 되고, GPT 프롬프트 형식은 몰라도 된다.
    #
    # INPUT:  ContextData — 이수연이 채운 컨텍스트 (RAG 결과 포함)
    # OUTPUT: str — GPT user 메시지 (candidate_spots 목록 포함)
    #
    # ⚠️ candidate_spots가 빈 리스트면 GPT가 장소를 자유롭게 생성하게 됨
    #   (Graceful Degradation: 데이터 없어도 동작, 단 품질은 낮아짐)
    """
    parts = [
        f"사용자 요청: {ctx.user_query}",
        f"현재 위치: 위도 {ctx.current_lat}, 경도 {ctx.current_lng}",
        f"현재 날씨: {ctx.weather}",
        f"가용 시간: {ctx.duration_hours}시간",
        f"이동 수단: {ctx.transport}",
        f"동행자: {ctx.companion}",
    ]

    # Optional 필드는 값이 있을 때만 추가 (Graceful Degradation)
    if ctx.mood:
        parts.append(f"원하는 분위기: {ctx.mood}")
    if ctx.route_summary:
        parts.append(f"경로 요약: {ctx.route_summary}")

    # 후보 장소 목록 주입
    # WHY candidate_spots를 user 메시지에 넣는가?
    #   GPT가 "이 목록에서 골라라"는 제약을 받아야 place_id가 SQLite PK와 1:1 매핑됨.
    #   목록 없이 자유 생성하면 존재하지 않는 place_id가 나와 DB 조회 실패.
    if ctx.candidate_spots:
        parts.append("\n[추천 후보 장소 — SQLite + Pinecone 필터링 결과]")
        parts.append("아래 목록에서 최적 코스를 선택하십시오. place_id는 반드시 아래 목록의 값만 사용.")
        for spot in ctx.candidate_spots:
            indoor_str = "실내" if spot.indoor else "실외"
            parts.append(
                f"  - place_id={spot.place_id} | {spot.place_name}"
                f" | {spot.category} | 이동 {spot.distance_min:.0f}분 | {indoor_str}"
            )

    return "\n".join(parts)


def build_user_message(req: TravelRequest) -> str:
    """
    TravelRequest 객체를 LLM에 보낼 user 메시지 문자열로 변환한다.

    # WHY travel_mode를 user 메시지에도 포함하는가?
    #   System Prompt에 이미 모드가 반영되어 있지만,
    #   user 메시지에도 명시하면 LLM이 컨텍스트를 더 확실히 인식한다.
    #   (특히 대화가 길어질 때 시스템 프롬프트 내용을 "잊는" 현상 방지)
    # WHY Optional 필드를 별도 처리하는가?
    #   Graceful Degradation: 날씨·예산·특이사항이 없어도 메시지가 구성됨
    #   None인 필드를 "없음"으로 채우지 않고 아예 생략 → LLM이 불필요한 정보를 받지 않음
    """
    parts = [
        f"여행지: {req.destination}",
        f"여행 유형: {req.travel_mode}",      # v1.2 추가 — Triple Mode 맥락 전달
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
