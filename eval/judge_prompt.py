# judge_prompt_id: course_quality_judge
# version: v1.0
# updated: 2026-08-03
# author: 이정훈
"""
LLM-as-Judge 5축 채점 프롬프트 — Sprint 5 Task 2.

[설계 근거] docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §3
judge 프롬프트를 별도 파일 + 버전 헤더로 분리하는 이유: judge 프롬프트가 바뀌면
점수가 통째로 이동한다. "프롬프트가 나빠짐"과 "심판이 엄해짐"을 구분하려면
judge 자체도 버전 관리 대상이어야 한다(eval versioning) — traveler/core/prompt.py의
버전 헤더 관례를 그대로 따른다.

halo effect 완화 장치 2개(JUDGE_SYSTEM_PROMPT에 반영):
  1. reason-then-score: 각 축에서 근거를 먼저, 점수를 나중에 출력하도록 강제.
     점수를 먼저 정하면 근거가 사후 정당화(post-hoc rationalization)되기 쉽다.
  2. anchor rubric: 1점/3점/5점이 각각 무엇을 의미하는지 구체적으로 명시.
     "좋다/나쁘다"만 주면 중심화 경향(3~4점 몰림)이 생겨 변별력이 사라진다.
"""

from traveler.rag.types import ContextData
from traveler.validation.schema import StreamingCourseModel

# WHY 상수로 노출하는가(Sprint 5 Task 4): 파일 상단 버전 헤더 주석("# version: v1.0")은
# import해서 report.json에 찍을 수 없다. traveler/core/prompt.py의 PROMPT_VERSION과
# 동일한 SSOT 패턴 — eval/run_judge.py가 이 상수를 import해 judge_prompt_version으로 기록한다.
JUDGE_PROMPT_VERSION = "v1.0"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JUDGE_SYSTEM_PROMPT — 5축 anchor rubric
#
# WHY 5축 1콜인가? (spec §3.2)
#   축마다 API를 나누면 비용·지연이 5배가 되고, 서술 일관성도 떨어진다.
#   한 번의 호출로 5축을 모두 판단하게 하되, 각 축 내부에서 reason-then-score
#   순서를 강제해 halo effect(하나 좋아 보이면 전체를 후하게 주는 경향)를 억제한다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JUDGE_SYSTEM_PROMPT = """\
당신은 여행 코스 추천 품질을 채점하는 엄격한 심사위원입니다.

[역할]
사용자의 요청·제약 조건과 GPT가 생성한 여행 코스를 비교해, 아래 5개 축을
각각 1~5점(정수)으로 채점하고 근거를 작성합니다.

[채점 절차 — 반드시 지킬 것]
각 축마다 먼저 "reason"(근거 문장)을 작성한 뒤에 "score"(점수)를 정하십시오.
점수를 먼저 정하고 근거를 나중에 짜맞추면(post-hoc rationalization) 채점의
일관성이 무너집니다. 근거가 점수를 결정해야지, 점수가 근거를 결정해서는
안 됩니다.

[5축 채점 기준 — anchor rubric]

1. condition_fit (조건반영도) — 날씨·시간·이동 수단 등 명시된 제약을 얼마나
   정확히 반영했는가.
   - 1점: 날씨/시간 제약을 완전히 무시함 (예: 비 오는 날 실외 위주 구성,
     가용 시간을 크게 초과하는 일정)
   - 3점: 제약을 일부만 반영함 (예: 날씨는 고려했지만 시간 배분이 어긋남)
   - 5점: 날씨·시간·이동 수단 등 모든 명시 제약을 정확히 반영함

2. mood_match (감성일치) — 사용자가 원한 분위기·감성을 코스와 문구가 얼마나
   살렸는가.
   - 1점: 요청한 분위기와 완전히 동떨어짐 (예: "조용하고 힐링되는" 요청에
     번화가·액티비티 위주로 구성)
   - 3점: 분위기를 부분적으로만 반영하거나 문구가 형식적임
   - 5점: 장소 선택과 bot_message 문구 모두 요청한 분위기를 일관되게 살림

3. route_sanity (동선합리성) — 방문 순서와 이동이 현실적이고 무리가 없는가.
   - 1점: 방문 순서가 비논리적이거나(예: 이동 거리를 고려하지 않은 왕복 동선),
     동행자 특성(유아/고령자 등)에 무리가 되는 일정
   - 3점: 동선은 성립하지만 비효율적이거나 일부 무리한 구간이 있음
   - 5점: 방문 순서가 자연스럽고 이동 부담이 동행자 특성에 맞게 적절함

4. diversity (다양성) — 장소 카테고리·경험이 얼마나 다채로운가(단, 억지로
   섞을 필요는 없음 — 사용자 요청과 맞는 선에서 평가).
   - 1점: 전부 같은 카테고리로 단조로움 (예: 카페만 3곳)
   - 3점: 카테고리가 2종 정도로 제한적
   - 5점: 관광·음식·체험·휴식 등 서로 다른 경험이 균형 있게 섞여 있음
     (단, 사용자가 특정 유형을 명시적으로 원했다면 그 요청을 우선함)

5. completeness (정보완결성) — bot_message와 장소 정보가 사용자가 코스를
   이해하고 실행하기에 충분한가.
   - 1점: bot_message가 지나치게 짧거나 장소 정보(순서·체류시간 등)가
     비어 있어 코스를 이해할 수 없음
   - 3점: 최소한의 정보는 있으나 구체성이 부족함
   - 5점: bot_message가 코스의 흐름을 명확히 설명하고, 각 장소의 방문
     순서·체류시간 등 실행에 필요한 정보가 충분함

[출력 형식]
반드시 아래 JSON 객체 하나로만 응답하십시오. 마크다운 코드 펜스, 설명,
사과문 등 JSON 외 텍스트는 금지합니다. 각 축의 필드 순서는 반드시
"reason"이 "score"보다 먼저 와야 합니다.

{
  "condition_fit": {"reason": "...", "score": 1},
  "mood_match": {"reason": "...", "score": 1},
  "route_sanity": {"reason": "...", "score": 1},
  "diversity": {"reason": "...", "score": 1},
  "completeness": {"reason": "...", "score": 1}
}
"""


def build_judge_user_message(
    ctx: ContextData, mode: str, course: StreamingCourseModel
) -> str:
    """
    채점 대상(사용자 요청/제약 + GPT가 생성한 코스)을 user 메시지 문자열로 조립한다.

    # WHY case: GoldenCase가 아니라 ctx+mode를 직접 받는가?
    #   eval/dataset.py(GoldenCase)는 Task 1이 아직 작업 중이라 이 시점에 존재를
    #   보장할 수 없다. eval/metrics.py의 measure()가 GoldenCase 대신 ctx+expectations를
    #   직접 받도록 이미 조정된 것과 같은 이유로, judge도 파일 레벨 의존성을 없애기
    #   위해 ContextData·mode·StreamingCourseModel만 직접 받는다.
    #
    # 스타일: traveler/core/prompt.py의 build_context_message()와 필드 구성을
    #   맞춘다 — user_query/weather/duration_hours/companion/mood 등 판단에 쓰이는
    #   요청 조건을 먼저 나열하고, 이어서 채점 대상인 GPT 출력을 붙인다.
    """
    parts = [
        "[사용자 요청 및 제약 조건]",
        f"여행 유형: {mode}",
        f"사용자 요청: {ctx.user_query}",
        f"날씨: {ctx.weather}",
        f"가용 시간: {ctx.duration_hours}시간",
        f"이동 수단: {ctx.transport}",
    ]

    # Optional 필드는 값이 있을 때만 추가 (course_generator.py의 build_context_message와 동일한
    # Graceful Degradation 패턴 — 없는 정보를 "없음"으로 채우면 채점 판단에 잡음만 더함)
    if ctx.companion:
        parts.append(f"동행자: {ctx.companion}")
    if ctx.mood:
        parts.append(f"원하는 분위기: {ctx.mood}")
    if ctx.route_summary:
        parts.append(f"경로 요약: {ctx.route_summary}")

    if ctx.candidate_spots:
        parts.append("\n[추천 후보 장소 목록]")
        for spot in ctx.candidate_spots:
            indoor_str = "실내" if spot.indoor else "실외"
            parts.append(
                f"  - place_id={spot.place_id} | {spot.place_name}"
                f" | {spot.category} | 이동 {spot.distance_min:.0f}분 | {indoor_str}"
            )

    parts.append("\n[채점 대상 — GPT가 생성한 코스]")
    parts.append(f"bot_message: {course.bot_message}")
    parts.append("recommended_places:")
    for place in course.recommended_places:
        name = place.place_name or f"place_id={place.place_id}"
        duration = place.duration_min if place.duration_min is not None else "미정"
        parts.append(
            f"  - visit_sequence={place.visit_sequence} | {name}"
            f" | 체류시간 {duration}분"
        )

    return "\n".join(parts)
