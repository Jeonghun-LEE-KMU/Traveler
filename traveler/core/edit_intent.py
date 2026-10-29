"""
멀티턴 수정 인텐트 5종 — Sprint 7.

파이프라인 위치:
  StreamingCourseModel(1차 생성) → 사용자 후속 발화 → ★ revise_course() → StreamingCourseModel(수정본)

[설계 근거] docs/superpowers/specs/2026-08-30-sprint7-multiturn-edit-intent-design.md
(§0~§9, /plan-eng-review + Outside Voice 반영본 — D1~D13)

바꿔줘(replace) / 추가(add) / 빼줘(remove) / 순서변경(reorder) / 전체재생성(regenerate_all)
5종 중 regenerate_all만 LLM(generate_course)을 다시 부른다(D4). 나머지 4종은 LLM 없는
순수 리스트 조작이다(D3) — 사용자가 유지하길 원한 장소까지 통째로 다시 생성하지 않기
위함이고, 부수 효과로 API 없이 테스트 가능하다.
"""

import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

from traveler.core.course_generator import generate_course
from traveler.core.course_schedule import default_stay_minutes
from traveler.core.edit_intent_prompt import (
    EDIT_INTENT_SYSTEM_PROMPT,
    build_edit_intent_user_message,
)
from traveler.core.llm_client import INTENT_MODEL, get_client
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel

logger = logging.getLogger(__name__)


def _has_jongseong(name: str) -> bool:
    if not name:
        return False
    code = ord(name[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def _josa_eul_reul(name: str) -> str:
    return "을" if _has_jongseong(name) else "를"


def _josa_euro(name: str) -> str:
    """받침 유무에 따라 '으로'/'로' 반환. 받침이 'ㄹ'이면 '로'."""
    if not name:
        return "로"
    code = ord(name[-1])
    if 0xAC00 <= code <= 0xD7A3:
        jongseong = (code - 0xAC00) % 28
        if jongseong == 0:
            return "로"   # 받침 없음
        if jongseong == 8:
            return "로"   # ㄹ 받침
        return "으로"
    return "로"


class EditIntent(BaseModel):
    """
    분류기(classify_edit_intent)의 출력. action별로 어떤 필드가 채워지는지는
    설계 spec §3의 표를 참고. 대부분 필드가 Optional인 이유는 action마다 쓰는
    필드가 다르기 때문 — reason만 항상 필수다.
    """

    action: Literal["replace", "add", "remove", "reorder", "regenerate_all"]
    target_sequence: int | None = None   # replace·remove 대상의 visit_sequence
    new_place_id: str | None = None      # replace·add 시 넣을 후보 place_id
    insert_after: int | None = None      # add 전용(D13) — 이 visit_sequence 뒤에 삽입.
                                          #   None이면 끝에 추가(기존 동작과 동일)
    new_order: list[int] | None = None   # reorder 전용, 전체 순열 — 새 순서
    move_sequence: int | None = None     # reorder 전용(D13), 부분 이동 — 옮길 장소
    move_to_position: int | None = None  # reorder 전용(D13), 부분 이동 — 새 위치(1부터)
    reason: str                          # 사용자에게 보여줄 근거 한 문장


def _unused_candidates(
    course: StreamingCourseModel,
    ctx: ContextData,
    visited: set[str] | None = None,
) -> list[SpotCandidate]:
    """
    아직 코스에 없고, 아직 방문 처리되지 않은 후보만 뽑는다.

    # WHY 공유 헬퍼로 뺐는가(D10, /plan-eng-review 지적): classify_edit_intent()의
    # 프롬프트 주입과 apply_edit()의 replace/add 거절 판정이 둘 다 이 계산이
    # 필요하다. 두 곳에서 각자 재구현하면 판정 기준이 바뀔 때 한쪽만 고치는 게
    # 전형적인 버그 지점이라, 여기 하나만 SSOT로 둔다.
    #
    # WHY visited 파라미터가 추가됐는가(#31 PM 리뷰 11번): ctx.candidate_spots는
    # /api/chat 최초 호출 시점에 고정된다. 그 뒤 사용자가 POST /api/session/{id}/visited로
    # 코스를 확정해 방문 처리해도 ctx는 그대로라, replace/add가 이미 방문한 장소를
    # 다시 추천할 수 있었다. visited는 None이면 기존과 동일(회귀 0) — 호출부(main.py)가
    # revise 시점에 세션의 최신 visited 집합을 조회해서 넘겨줘야 실제로 걸러진다.
    """
    used = {p.place_id for p in course.recommended_places}
    excluded = used | (visited or set())
    return [s for s in ctx.candidate_spots if s.place_id not in excluded]


async def classify_edit_intent(
    user_message: str,
    course: StreamingCourseModel,
    ctx: ContextData,
    *,
    model: str = INTENT_MODEL,
    visited: set[str] | None = None,
) -> EditIntent | None:
    """
    후속 발화가 5종 중 무엇인지, 어느 장소를 대상으로 하는지 판정한다.

    # INPUT:  사용자 후속 발화 + 현재 코스 + ContextData(후보 풀) + visited(#31 PM
    #         리뷰 11번 — 이미 방문 처리된 장소는 그라운딩 목록에서도 제외)
    # OUTPUT: EditIntent | None  (None = 분류 실패 → 호출부가 재질문, D5)
    # COST:   ~700 tokens · gpt-4o-mini · 약 $0.15 per 1,000 calls
    #
    # WHY 이 구조(단발 호출 + JSON mode + Pydantic 검증 + 실패 시 None)인가?
    #   eval/judge.py judge_course()와 동일한 선례를 그대로 따른다 — 이미 검증된
    #   Graceful Degradation 패턴을 재사용할 뿐 새로 발명하지 않는다.
    """
    user_msg = build_edit_intent_user_message(
        user_message, course, _unused_candidates(course, ctx, visited)
    )

    try:
        response = await get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EDIT_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},  # JSON mode 강제
            temperature=0,  # WHY 0: 분류는 판단이므로 결정론적 출력이 필요
        )
        content = response.choices[0].message.content
        return EditIntent.model_validate_json(content)
    except (ValidationError, ValueError, TypeError) as e:
        # JSON 파싱 실패(ValueError) / 스키마 불일치(ValidationError) /
        # content가 None 등 예상 밖 응답 구조(TypeError)
        logger.warning("edit intent 분류 응답 파싱 실패, skip 처리: %s", e)
        return None
    except Exception as e:
        # API 호출 자체의 실패(타임아웃·네트워크·rate limit 등)
        logger.warning("edit intent 분류 API 호출 실패, skip 처리: %s", e)
        return None


def _resequence(places: list[RecommendedPlaceModel]) -> list[RecommendedPlaceModel]:
    """
    visit_sequence를 1..n으로 재부여한다(D7).

    # WHY 실행기 공통 마지막 단계인가?
    #   삭제·삽입 후 번호를 그대로 두면 gap이 생겨 StreamingCourseModel의
    #   check_sequence_sequential 검증(F4)에서 코스 전체가 거부된다. 모든 action
    #   핸들러가 각자 재시퀀싱을 잊지 않도록, 공통 함수로 못박아 마지막에 한 번만
    #   부른다.
    """
    return [
        p.model_copy(update={"visit_sequence": i})
        for i, p in enumerate(places, start=1)
    ]


def apply_edit(
    course: StreamingCourseModel,
    edit: EditIntent,
    ctx: ContextData,
    visited: set[str] | None = None,
) -> tuple[list[RecommendedPlaceModel], str]:
    """
    LLM 없이 코스를 실제로 수정한다. 4종(replace·add·remove·reorder) 모두 리스트
    조작이라 순수 함수로 충분하다(D3). regenerate_all은 여기서 처리하지 않는다 —
    호출부(revise_course)가 generate_course()를 직접 재호출한다(D4).

    # visited(#31 PM 리뷰 11번): 이미 방문 처리된 장소는 replace/add 후보에서도
    # 빠져야 한다. classify_edit_intent()의 그라운딩 목록과 동일한 기준을 쓰지
    # 않으면, LLM은 "쓸 수 없다"고 안내했는데 실행기는 넣어주는 모순이 생긴다.
    #
    # OUTPUT: (수정된 장소 목록, 사용자에게 보여줄 설명 한 문장)
    #         적용 불가 시 원본 목록 + 거절 사유를 돌려준다(D6) — 예외가 아니라
    #         정상적인 "거절"이다. 예외를 던지면 호출부가 매번 try/except를 써야
    #         하고, 조용히 무시하면 사용자가 왜 안 바뀌었는지 모른다.
    """
    places = list(course.recommended_places)
    original = list(course.recommended_places)
    unused_by_id = {s.place_id: s for s in _unused_candidates(course, ctx, visited)}

    if edit.action == "remove":
        if edit.target_sequence is None:
            return original, "어느 장소를 빼야 할지 알 수 없었어요."
        if len(places) <= 1:
            return original, "장소가 하나뿐이라 뺄 수 없어요."
        target = next(
            (p for p in places if p.visit_sequence == edit.target_sequence), None
        )
        if target is None:
            return original, "말씀하신 순번의 장소를 찾을 수 없었어요."
        places = [p for p in places if p is not target]
        name = target.place_name or "해당 장소"
        return _resequence(places), f"{name}{_josa_eul_reul(name)} 코스에서 뺐어요."

    if edit.action == "replace":
        if edit.target_sequence is None or edit.new_place_id is None:
            return original, "무엇을 무엇으로 바꿔야 할지 알 수 없었어요."
        target = next(
            (p for p in places if p.visit_sequence == edit.target_sequence), None
        )
        if target is None:
            return original, "말씀하신 순번의 장소를 찾을 수 없었어요."
        new_spot = unused_by_id.get(edit.new_place_id)
        if new_spot is None:
            return original, "말씀하신 장소는 추천 후보에 없거나 이미 코스에 있어요."
        new_place = RecommendedPlaceModel(
            place_id=new_spot.place_id,
            visit_sequence=target.visit_sequence,
            # ⚠️ 주의(#31 PM 리뷰 4번): Pinecone에 name이 없으면 retriever.py가
            # place_name=""을 만든다. RecommendedPlaceModel의 not_empty 검증기는
            # 빈 문자열이면 ValueError를 내지만 None은 통과시킨다(백엔드가 나중에
            # SQLite로 채우는 정상 경로) — 그래서 ""을 None으로 정규화해야 한다.
            place_name=new_spot.place_name or None,
            duration_min=default_stay_minutes(new_spot.category),
        )
        places = [new_place if p is target else p for p in places]
        old_name = target.place_name or "해당 장소"
        new_name = new_spot.place_name or "새 장소"
        return _resequence(places), f"{old_name} 대신 {new_name}{_josa_euro(new_name)} 바꿨어요."

    if edit.action == "add":
        if edit.new_place_id is None:
            return original, "어떤 장소를 추가해야 할지 알 수 없었어요."
        if len(places) >= 10:
            return original, "이미 장소가 10곳이라 더 추가할 수 없어요."
        new_spot = unused_by_id.get(edit.new_place_id)
        if new_spot is None:
            return original, "말씀하신 장소는 추천 후보에 없거나 이미 코스에 있어요."
        # WHY visit_sequence=1 placeholder: RecommendedPlaceModel은 Field(ge=1)이라
        # 0은 즉시 ValidationError. 어차피 아래서 _resequence()가 실제 값으로
        # 덮어쓰므로 여기선 제약을 만족하는 아무 값(1)이면 된다.
        new_place = RecommendedPlaceModel(
            place_id=new_spot.place_id,
            visit_sequence=1,
            place_name=new_spot.place_name or None,  # #31 PM 리뷰 4번 — replace와 동일 이유
            duration_min=default_stay_minutes(new_spot.category),
        )
        if edit.insert_after is None:
            places = places + [new_place]
        else:
            idx = next(
                (i for i, p in enumerate(places) if p.visit_sequence == edit.insert_after),
                None,
            )
            if idx is None:
                return original, "말씀하신 위치를 찾을 수 없었어요."
            places = places[: idx + 1] + [new_place] + places[idx + 1 :]
        added_name = new_spot.place_name or "새 장소"
        return _resequence(places), f"{added_name}{_josa_eul_reul(added_name)} 코스에 추가했어요."

    if edit.action == "reorder":
        has_full = edit.new_order is not None
        has_partial = edit.move_sequence is not None or edit.move_to_position is not None
        if has_full and has_partial:
            return original, "순서 변경 요청을 정확히 이해하지 못했어요."

        if has_full:
            current_seqs = {p.visit_sequence for p in places}
            if set(edit.new_order) != current_seqs or len(edit.new_order) != len(places):
                return original, "말씀하신 순서가 기존 장소들과 맞지 않았어요."
            by_seq = {p.visit_sequence: p for p in places}
            places = [by_seq[seq] for seq in edit.new_order]
            return _resequence(places), "말씀하신 순서로 바꿨어요."

        if has_partial:
            if edit.move_sequence is None or edit.move_to_position is None:
                return original, "어느 장소를 어디로 옮겨야 할지 알 수 없었어요."
            target = next(
                (p for p in places if p.visit_sequence == edit.move_sequence), None
            )
            if target is None:
                return original, "말씀하신 순번의 장소를 찾을 수 없었어요."
            if not (1 <= edit.move_to_position <= len(places)):
                return original, "옮길 위치가 코스 범위를 벗어났어요."
            remaining = [p for p in places if p is not target]
            new_index = edit.move_to_position - 1
            places = remaining[:new_index] + [target] + remaining[new_index:]
            name = target.place_name or "해당 장소"
            return _resequence(places), f"{name}{_josa_eul_reul(name)} {edit.move_to_position}번째로 옮겼어요."

        return original, "순서 변경 요청을 정확히 이해하지 못했어요."

    # regenerate_all은 revise_course()가 여기 도달하기 전에 분기한다(D4).
    return original, "요청을 처리하지 못했어요."


async def revise_course(
    user_message: str,
    course: StreamingCourseModel,
    ctx: ContextData,
    travel_mode: str = "타지",
    *,
    visited: set[str] | None = None,
) -> StreamingCourseModel:
    """
    시연·API가 호출할 단일 진입점. 분류 → 분기 → 코스 재조립까지 한 번에.

    # OUTPUT: StreamingCourseModel — 항상 유효한 코스. 실패해도 원본 + 설명 bot_message.
    # ⚠️ course·ctx의 출처는 이 함수의 관심사가 아니다(D11) — 호출부(main.py)가
    #   세션에서 조회해 넘겨주고, 반환값을 다시 저장한다. 이 함수는 세션이라는
    #   개념을 전혀 모른다.
    # visited(#31 PM 리뷰 11번): 마찬가지로 호출부가 세션의 최신 visited 집합을
    #   조회해서 넘겨준다 — None이면 기존과 동일하게 아무것도 걸러지지 않는다.
    """
    edit = await classify_edit_intent(user_message, course, ctx, visited=visited)
    if edit is None:
        return course.model_copy(
            update={"bot_message": "어떻게 바꿔드릴까요? 예를 들어 '카페 말고 다른 걸로 바꿔줘', '2번째 빼줘'처럼 말씀해 주시면 바로 반영할게요."}
        )

    if edit.action == "regenerate_all":
        # WHY refinement=user_message(D12): generate_course()는 원래 ctx.user_query만
        # 쓴다. 후속 발화("카페 중심으로 다시 짜줘")를 안 넘기면 원래 첫 질문으로
        # 그냥 한 번 더 뽑기만 하게 된다 — EditIntent에 별도 필드를 만들 필요 없이
        # 사용자 원문 자체가 곧 refinement다.
        return await generate_course(ctx, travel_mode, refinement=user_message)

    # WHY apply_edit()까지 try로 감싸는가(D9, /plan-eng-review 지적을 정확히
    # 반영): 설계 문서 초안은 StreamingCourseModel(...) 재조립만 감쌌지만, "어떤
    # 경로로도 예외가 밖으로 나가지 않는다"는 목표를 그대로 지키려면 apply_edit()
    # 내부의 RecommendedPlaceModel(...) 구성(예: place_name이 빈 문자열인 이상
    # SpotCandidate가 들어온 경우)도 같은 방어선 안에 있어야 한다.
    try:
        updated_places, explanation = apply_edit(course, edit, ctx, visited)
        return StreamingCourseModel(
            bot_message=explanation, recommended_places=updated_places
        )
    except (ValidationError, ValueError) as e:
        logger.warning("코스 수정 처리 실패, 원본 유지: %s", e)
        return course.model_copy(
            update={"bot_message": "수정을 반영하지 못했어요, 원래 코스를 유지할게요."}
        )
