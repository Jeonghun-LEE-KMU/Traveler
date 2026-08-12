"""
LLM-as-Judge 5축 채점기 — Sprint 5 Task 2.

[설계 근거] docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §3
코스 1개 + 입력 컨텍스트를 gpt-4o(JUDGE_MODEL, D2)에게 보내 5축(조건반영도·
감성일치·동선합리성·다양성·정보완결성)을 1~5점 + 근거로 채점한다.

judge_course()는 eval/metrics.py의 measure()와 동일한 게이트 뒤에서 호출된다
(구조 메트릭 FAIL → judge 호출 안 함, §4). 채점 자체가 실패(JSON 파싱 실패,
Pydantic 검증 실패, API 예외)해도 예외를 밖으로 던지지 않고 None을 반환한다
— course_generator._build_fallback()과 같은 Graceful Degradation 철학이지만,
여기는 fallback 객체 대신 None을 반환해 호출부(러너)가 이 케이스를 skip 처리한다.
"""

import logging

from pydantic import BaseModel, Field, ValidationError

from eval.judge_prompt import JUDGE_SYSTEM_PROMPT, build_judge_user_message
from traveler.core.llm_client import JUDGE_MODEL, get_client
from traveler.rag.types import ContextData
from traveler.validation.schema import StreamingCourseModel

logger = logging.getLogger(__name__)

# WHY seed 고정: temperature=0과 함께 결정론적 채점을 보장하기 위함(spec §3.2·3.4
# 수용 기준 — "동일 입력 2회 실행 시 점수가 동일"). OpenAI는 seed만으로 완전한
# 결정론을 100% 보장하지는 않지만, temperature=0과 함께 쓰면 사실상 재현성이 높아진다.
JUDGE_SEED = 42


class AxisScore(BaseModel):
    """
    채점 축 하나의 결과.

    # WHY 필드 순서가 reason → score인가?
    #   Pydantic 필드 순서는 JSON 직렬화/역직렬화 순서와 무관하지만, 여기서는
    #   JUDGE_SYSTEM_PROMPT가 "reason을 먼저, score를 나중에 출력하라"고 GPT에게
    #   지시하는 것과 스키마 선언 순서를 일치시켜 halo effect 방지 의도를
    #   코드에서도 한눈에 드러낸다(reason-then-score).
    """
    reason: str
    score: int = Field(ge=1, le=5)


class JudgeResult(BaseModel):
    """5축 채점 결과 묶음."""
    condition_fit: AxisScore     # 조건반영도
    mood_match: AxisScore        # 감성일치
    route_sanity: AxisScore      # 동선합리성
    diversity: AxisScore         # 다양성
    completeness: AxisScore      # 정보완결성

    @property
    def mean(self) -> float:
        """
        5축 점수의 평균.

        # ⚠️ 주의(PM 리뷰 반영): run_judge.py의 report.json 집계는 이 프로퍼티를
        # 쓰지 않는다 — 여러 케이스를 모아 축별 평균을 내는 _axis_means()가 따로
        # 있다. 이건 JudgeResult 하나만 놓고 임시로 평균을 보고 싶을 때 쓰는
        # 편의 프로퍼티다.
        """
        scores = [
            self.condition_fit.score,
            self.mood_match.score,
            self.route_sanity.score,
            self.diversity.score,
            self.completeness.score,
        ]
        return sum(scores) / len(scores)


async def judge_course(
    ctx: ContextData,
    mode: str,
    course: StreamingCourseModel,
    *,
    model: str = JUDGE_MODEL,
) -> JudgeResult | None:
    """
    코스 1개를 5축으로 채점한다.

    # WHY case: GoldenCase가 아니라 ctx+mode를 직접 받는가?
    #   eval/dataset.py(GoldenCase)는 Task 1이 별도 진행 중이라 아직 존재를
    #   보장할 수 없다. eval/metrics.py의 measure()가 같은 이유로 GoldenCase 대신
    #   ctx+expectations를 직접 받도록 조정된 선례를 그대로 따른다.
    #
    # WHY 단발 호출(스트리밍 아님)인가?
    #   course_generator.generate_course()는 사용자에게 즉시 텍스트를 보여줘야
    #   해서 스트리밍(A 방식)이 필요하지만, judge는 배치로 도는 평가 러너라
    #   완성된 JSON 한 번에 받아 파싱하는 게 더 단순하고 충분하다.
    #
    # OUTPUT: JudgeResult | None — None은 채점 실패(Graceful Degradation).
    #   JSON 파싱 실패·Pydantic 검증 실패·API 예외를 전부 여기서 흡수해
    #   러너가 이 케이스를 skip 처리하고 계속 진행할 수 있게 한다.
    """
    user_message = build_judge_user_message(ctx, mode, course)

    try:
        response = await get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},  # JSON mode 강제
            temperature=0,     # WHY 0: 일관된 채점 판단(spec §3.2)
            seed=JUDGE_SEED,   # WHY: temperature=0과 함께 재현성 확보
        )
        content = response.choices[0].message.content
        return JudgeResult.model_validate_json(content)
    except (ValidationError, ValueError, TypeError) as e:
        # JSON 파싱 실패(ValueError) / 스키마 불일치(ValidationError) /
        # content가 None 등 예상 밖 응답 구조(TypeError)
        logger.warning("judge 응답 파싱 실패, skip 처리: %s", e)
        return None
    except Exception as e:
        # API 호출 자체의 실패(타임아웃·네트워크·rate limit 등)
        logger.warning("judge API 호출 실패, skip 처리: %s", e)
        return None
