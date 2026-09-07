"""
edit_intent.py 단위 테스트 — Sprint 7 멀티턴 수정 인텐트 5종.

WHY 이 테스트가 필요한가:
  classify_edit_intent()는 tests/test_eval_judge.py의 `@patch(".get_client")` +
  AsyncMock 패턴을 그대로 따른다(단발 호출, judge_course()와 동일 구조).
  apply_edit()·_resequence()·_unused_candidates()는 순수 함수라 목업이 필요 없다.
  revise_course()는 classify_edit_intent()와 generate_course() 둘 다 목업해서
  오케스트레이션(분기·재조립·D9 폴백)만 검증한다.

[설계 근거] docs/superpowers/specs/2026-08-30-sprint7-multiturn-edit-intent-design.md §7

검증값 출처: 스모크 테스트로 직접 확인한 실제 apply_edit() 입출력(재추론 없이 그대로 단언).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from traveler.core.edit_intent import (
    EditIntent,
    _resequence,
    _unused_candidates,
    apply_edit,
    classify_edit_intent,
    revise_course,
)
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

def minimal_ctx(**overrides) -> ContextData:
    """테스트용 최소 ContextData. 코스 [1:불국사,2:첨성대,3:교리김밥] + 미사용 후보 4:카페."""
    defaults = dict(
        user_query="경주 여행 추천해줘",
        current_lat=35.8562,
        current_lng=129.2247,
        weather="맑음",
        candidate_spots=[
            SpotCandidate("1", "불국사", "관광", 20.0, False),
            SpotCandidate("2", "첨성대", "관광", 10.0, False),
            SpotCandidate("3", "교리김밥", "음식", 15.0, True),
            SpotCandidate("4", "카페", "휴식", 12.0, True),
        ],
    )
    defaults.update(overrides)
    return ContextData(**defaults)


def sample_course() -> StreamingCourseModel:
    """테스트용 코스: [1:불국사, 2:첨성대, 3:교리김밥]."""
    return StreamingCourseModel(
        bot_message="경주 코스를 추천해요.",
        recommended_places=[
            RecommendedPlaceModel(place_id="1", visit_sequence=1, place_name="불국사", duration_min=90),
            RecommendedPlaceModel(place_id="2", visit_sequence=2, place_name="첨성대", duration_min=40),
            RecommendedPlaceModel(place_id="3", visit_sequence=3, place_name="교리김밥", duration_min=30),
        ],
    )


def make_edit(action: str, **kwargs) -> EditIntent:
    """EditIntent 생성 헬퍼 — action별로 필요한 필드만 kwargs로 채운다."""
    defaults = dict(
        target_sequence=None,
        new_place_id=None,
        insert_after=None,
        new_order=None,
        move_sequence=None,
        move_to_position=None,
        reason="test",
    )
    defaults.update(kwargs)
    return EditIntent(action=action, **defaults)


def _mock_response(content: str) -> SimpleNamespace:
    """OpenAI chat.completions.create() 단발 응답 모형 (test_eval_judge.py와 동일 패턴)."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _places(course_or_list) -> list[tuple[int, str]]:
    """검증 편의: (visit_sequence, place_id) 튜플 리스트로 변환."""
    places = (
        course_or_list.recommended_places
        if isinstance(course_or_list, StreamingCourseModel)
        else course_or_list
    )
    return [(p.visit_sequence, p.place_id) for p in places]


# ── 분류기 classify_edit_intent() ──────────────────────────────────────────────

REPLACE_JSON = """\
{"action": "replace", "target_sequence": 1, "new_place_id": "4",
 "insert_after": null, "new_order": null, "move_sequence": null,
 "move_to_position": null, "reason": "불국사를 카페로 바꿔달라는 요청"}
"""
ADD_JSON = """\
{"action": "add", "target_sequence": null, "new_place_id": "4",
 "insert_after": 2, "new_order": null, "move_sequence": null,
 "move_to_position": null, "reason": "카페를 세 번째에 넣어달라는 요청"}
"""
REMOVE_JSON = """\
{"action": "remove", "target_sequence": 2, "new_place_id": null,
 "insert_after": null, "new_order": null, "move_sequence": null,
 "move_to_position": null, "reason": "첨성대를 빼달라는 요청"}
"""
REORDER_FULL_JSON = """\
{"action": "reorder", "target_sequence": null, "new_place_id": null,
 "insert_after": null, "new_order": [3, 1, 2], "move_sequence": null,
 "move_to_position": null, "reason": "전체 순서를 다시 정해달라는 요청"}
"""
REORDER_PARTIAL_JSON = """\
{"action": "reorder", "target_sequence": null, "new_place_id": null,
 "insert_after": null, "new_order": null, "move_sequence": 3,
 "move_to_position": 1, "reason": "교리김밥만 먼저 가고 싶다는 요청"}
"""
REGENERATE_ALL_JSON = """\
{"action": "regenerate_all", "target_sequence": null, "new_place_id": null,
 "insert_after": null, "new_order": null, "move_sequence": null,
 "move_to_position": null, "reason": "전체가 마음에 안 든다는 요청"}
"""


@pytest.mark.parametrize(
    ("response_json", "expected_action"),
    [
        (REPLACE_JSON, "replace"),
        (ADD_JSON, "add"),
        (REMOVE_JSON, "remove"),
        (REORDER_FULL_JSON, "reorder"),
        (REORDER_PARTIAL_JSON, "reorder"),
        (REGENERATE_ALL_JSON, "regenerate_all"),
    ],
)
@patch("traveler.core.edit_intent.get_client")
async def test_classify_edit_intent_parses_each_action(
    mock_get_client, response_json, expected_action
):
    """5종(reorder는 전체·부분 2케이스) 각각 정상 JSON이 EditIntent로 파싱되는지 확인."""
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_mock_response(response_json))
            )
        )
    )
    mock_get_client.return_value = mock_client

    result = await classify_edit_intent("아무 발화", sample_course(), minimal_ctx())

    assert isinstance(result, EditIntent)
    assert result.action == expected_action


@patch("traveler.core.edit_intent.get_client")
async def test_classify_edit_intent_malformed_json_returns_none(mock_get_client):
    """JSON이 깨지면 예외 없이 None을 반환하는지 확인(D5)."""
    broken_json = '{"action": "replace", "target_sequence": 1'  # 닫는 괄호 없이 잘림
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_mock_response(broken_json))
            )
        )
    )
    mock_get_client.return_value = mock_client

    result = await classify_edit_intent("아무 발화", sample_course(), minimal_ctx())

    assert result is None


@patch("traveler.core.edit_intent.get_client")
async def test_classify_edit_intent_api_error_returns_none(mock_get_client):
    """API 호출 자체가 예외(타임아웃 등)를 던져도 None을 반환하는지 확인(D5)."""
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=Exception("simulated API timeout"))
            )
        )
    )
    mock_get_client.return_value = mock_client

    result = await classify_edit_intent("아무 발화", sample_course(), minimal_ctx())

    assert result is None


@patch("traveler.core.edit_intent.get_client")
async def test_classify_edit_intent_schema_violation_returns_none(mock_get_client):
    """action이 5종 밖의 값이면(Literal 위반) None을 반환하는지 확인."""
    invalid_json = """\
{"action": "teleport", "target_sequence": null, "new_place_id": null,
 "insert_after": null, "new_order": null, "move_sequence": null,
 "move_to_position": null, "reason": "지원하지 않는 액션"}
"""
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=_mock_response(invalid_json))
            )
        )
    )
    mock_get_client.return_value = mock_client

    result = await classify_edit_intent("아무 발화", sample_course(), minimal_ctx())

    assert result is None


# ── 실행기 apply_edit() ────────────────────────────────────────────────────────

def test_apply_edit_remove_valid_sequence():
    course = sample_course()
    edit = make_edit("remove", target_sequence=2)

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "3")]
    assert msg == "첨성대를 코스에서 뺐어요."


def test_apply_edit_replace_valid():
    course = sample_course()
    edit = make_edit("replace", target_sequence=1, new_place_id="4")

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "4"), (2, "2"), (3, "3")]
    assert places[0].duration_min == 60
    assert places[1].duration_min == 40
    assert places[2].duration_min == 30
    assert msg == "불국사 대신 카페로 바꿨어요."


def test_apply_edit_add_with_insert_after():
    course = sample_course()
    edit = make_edit("add", new_place_id="4", insert_after=1)

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "4"), (3, "2"), (4, "3")]
    assert msg == "카페를 코스에 추가했어요."


def test_apply_edit_add_without_insert_after_appends_to_end():
    course = sample_course()
    edit = make_edit("add", new_place_id="4")

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3"), (4, "4")]
    assert msg == "카페를 코스에 추가했어요."


def test_apply_edit_replace_empty_place_name_normalized_to_none():
    """#31 PM 리뷰 4번: Pinecone에 name이 없으면 SpotCandidate.place_name==""이
    될 수 있다. RecommendedPlaceModel의 not_empty 검증기는 빈 문자열이면
    ValueError를 내므로, apply_edit()이 ""을 None으로 정규화해야 크래시하지 않는다."""
    course = sample_course()
    ctx = minimal_ctx(
        candidate_spots=[
            SpotCandidate("1", "불국사", "관광", 20.0, False),
            SpotCandidate("2", "첨성대", "관광", 10.0, False),
            SpotCandidate("3", "교리김밥", "음식", 15.0, True),
            SpotCandidate("5", "", "관광", 8.0, False),  # 이름 없는 후보
        ]
    )
    edit = make_edit("replace", target_sequence=1, new_place_id="5")

    places, msg = apply_edit(course, edit, ctx)

    assert places[0].place_name is None
    assert msg == "불국사 대신 새 장소로 바꿨어요."


def test_apply_edit_add_empty_place_name_normalized_to_none():
    course = sample_course()
    ctx = minimal_ctx(
        candidate_spots=[
            SpotCandidate("1", "불국사", "관광", 20.0, False),
            SpotCandidate("2", "첨성대", "관광", 10.0, False),
            SpotCandidate("3", "교리김밥", "음식", 15.0, True),
            SpotCandidate("5", "", "관광", 8.0, False),
        ]
    )
    edit = make_edit("add", new_place_id="5")

    places, msg = apply_edit(course, edit, ctx)

    assert places[-1].place_name is None
    assert msg == "새 장소를 코스에 추가했어요."


def test_apply_edit_reorder_full_new_order():
    course = sample_course()
    edit = make_edit("reorder", new_order=[3, 1, 2])

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "3"), (2, "1"), (3, "2")]
    assert msg == "말씀하신 순서로 바꿨어요."


def test_apply_edit_reorder_partial_move():
    course = sample_course()
    edit = make_edit("reorder", move_sequence=3, move_to_position=1)

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "3"), (2, "1"), (3, "2")]
    assert msg == "교리김밥를 1번째로 옮겼어요."


def test_apply_edit_remove_last_place_rejected():
    course = StreamingCourseModel(
        bot_message="코스",
        recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1)],
    )
    edit = make_edit("remove", target_sequence=1)

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1")]
    assert msg == "장소가 하나뿐이라 뺄 수 없어요."


def test_apply_edit_add_already_in_course_rejected():
    course = sample_course()
    edit = make_edit("add", new_place_id="1")  # 이미 코스에 있음

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "말씀하신 장소는 추천 후보에 없거나 이미 코스에 있어요."


def test_apply_edit_add_visited_place_rejected():
    """#31 PM 리뷰 11번: 코스에는 없어도 이미 방문 처리된 후보는 추가할 수 없다."""
    course = sample_course()
    edit = make_edit("add", new_place_id="4")  # 미사용 후보지만 방문 처리됨

    places, msg = apply_edit(course, edit, minimal_ctx(), visited={"4"})

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "말씀하신 장소는 추천 후보에 없거나 이미 코스에 있어요."


def test_apply_edit_replace_visited_place_rejected():
    course = sample_course()
    edit = make_edit("replace", target_sequence=1, new_place_id="4")

    places, msg = apply_edit(course, edit, minimal_ctx(), visited={"4"})

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "말씀하신 장소는 추천 후보에 없거나 이미 코스에 있어요."


def test_apply_edit_add_non_visited_candidate_still_allowed():
    """visited가 있어도 그 안에 없는 후보는 정상적으로 추가돼야 한다(회귀 방지)."""
    course = sample_course()
    edit = make_edit("add", new_place_id="4")

    places, msg = apply_edit(course, edit, minimal_ctx(), visited={"999"})

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3"), (4, "4")]
    assert msg == "카페를 코스에 추가했어요."


def test_unused_candidates_excludes_visited():
    course = sample_course()
    result = _unused_candidates(course, minimal_ctx(), visited={"4"})
    assert result == []


def test_unused_candidates_visited_none_unchanged():
    """visited를 안 넘기면(None) 기존과 동일하게 동작한다(회귀 0)."""
    course = sample_course()
    result = _unused_candidates(course, minimal_ctx())
    assert [s.place_id for s in result] == ["4"]


def test_apply_edit_reorder_both_full_and_partial_filled_rejected():
    course = sample_course()
    edit = make_edit(
        "reorder", new_order=[3, 1, 2], move_sequence=1, move_to_position=2
    )

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "순서 변경 요청을 정확히 이해하지 못했어요."


def test_apply_edit_add_rejected_when_course_already_has_ten():
    ten_places = [
        RecommendedPlaceModel(place_id=str(i), visit_sequence=i) for i in range(1, 11)
    ]
    course = StreamingCourseModel(bot_message="코스", recommended_places=ten_places)
    ctx = minimal_ctx(
        candidate_spots=[SpotCandidate("11", "새장소", "관광", 5.0, False)]
    )
    edit = make_edit("add", new_place_id="11")

    places, msg = apply_edit(course, edit, ctx)

    assert len(places) == 10
    assert msg == "이미 장소가 10곳이라 더 추가할 수 없어요."


def test_apply_edit_replace_target_sequence_not_found_rejected():
    course = sample_course()
    edit = make_edit("replace", target_sequence=99, new_place_id="4")

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "말씀하신 순번의 장소를 찾을 수 없었어요."


def test_apply_edit_remove_target_sequence_not_found_rejected():
    course = sample_course()
    edit = make_edit("remove", target_sequence=99)

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "말씀하신 순번의 장소를 찾을 수 없었어요."


def test_apply_edit_add_insert_after_not_found_rejected():
    course = sample_course()
    edit = make_edit("add", new_place_id="4", insert_after=99)

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "말씀하신 위치를 찾을 수 없었어요."


def test_apply_edit_reorder_new_order_not_a_permutation_rejected():
    course = sample_course()
    edit = make_edit("reorder", new_order=[1, 2])  # 3이 빠짐 — 기존 순번 집합과 불일치

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "말씀하신 순서가 기존 장소들과 맞지 않았어요."


def test_apply_edit_reorder_move_to_position_out_of_range_rejected():
    course = sample_course()
    edit = make_edit("reorder", move_sequence=1, move_to_position=99)

    places, msg = apply_edit(course, edit, minimal_ctx())

    assert _places(places) == [(1, "1"), (2, "2"), (3, "3")]
    assert msg == "옮길 위치가 코스 범위를 벗어났어요."


# ── _resequence() ──────────────────────────────────────────────────────────────

def test_resequence_reassigns_gapped_sequence_to_one_through_n():
    places = [
        RecommendedPlaceModel(place_id="a", visit_sequence=5),
        RecommendedPlaceModel(place_id="b", visit_sequence=9),
        RecommendedPlaceModel(place_id="c", visit_sequence=20),
    ]

    result = _resequence(places)

    assert [p.visit_sequence for p in result] == [1, 2, 3]
    assert [p.place_id for p in result] == ["a", "b", "c"]


# ── _unused_candidates() ────────────────────────────────────────────────────────

def test_unused_candidates_empty_when_course_uses_all_candidates():
    ctx = minimal_ctx(
        candidate_spots=[
            SpotCandidate("1", "불국사", "관광", 20.0, False),
            SpotCandidate("2", "첨성대", "관광", 10.0, False),
            SpotCandidate("3", "교리김밥", "음식", 15.0, True),
        ]
    )
    course = sample_course()

    assert _unused_candidates(course, ctx) == []


def test_unused_candidates_returns_only_unused():
    ctx = minimal_ctx()  # 후보 1,2,3,4 — 코스는 1,2,3만 씀
    course = sample_course()

    result = _unused_candidates(course, ctx)

    assert [s.place_id for s in result] == ["4"]


# ── 오케스트레이터 revise_course() ────────────────────────────────────────────

@patch("traveler.core.edit_intent.classify_edit_intent", new_callable=AsyncMock)
async def test_revise_course_classification_failure_returns_original_with_reask_message(
    mock_classify,
):
    mock_classify.return_value = None
    course = sample_course()

    result = await revise_course("무슨 말인지 모를 발화", course, minimal_ctx())

    assert _places(result) == _places(course)
    assert "무슨 말씀인지 잘 모르겠어요" in result.bot_message


@patch("traveler.core.edit_intent.generate_course", new_callable=AsyncMock)
@patch("traveler.core.edit_intent.classify_edit_intent", new_callable=AsyncMock)
async def test_revise_course_regenerate_all_forwards_refinement_to_generate_course(
    mock_classify, mock_generate
):
    mock_classify.return_value = make_edit("regenerate_all")
    fallback_course = sample_course()
    mock_generate.return_value = fallback_course
    course = sample_course()
    ctx = minimal_ctx()

    result = await revise_course(
        "카페 중심으로 다시 짜줘", course, ctx, travel_mode="근교"
    )

    mock_generate.assert_called_once_with(
        ctx, "근교", refinement="카페 중심으로 다시 짜줘"
    )
    assert result is fallback_course


@patch("traveler.core.edit_intent.classify_edit_intent", new_callable=AsyncMock)
async def test_revise_course_normal_edit_reassembles_streaming_course_model(
    mock_classify,
):
    mock_classify.return_value = make_edit("remove", target_sequence=2)
    course = sample_course()

    result = await revise_course("첨성대는 빼줘", course, minimal_ctx())

    assert isinstance(result, StreamingCourseModel)
    assert _places(result) == [(1, "1"), (2, "3")]
    assert result.bot_message == "첨성대를 코스에서 뺐어요."


@patch("traveler.core.edit_intent.apply_edit")
@patch("traveler.core.edit_intent.classify_edit_intent", new_callable=AsyncMock)
async def test_revise_course_reassembly_failure_falls_back_to_original(
    mock_classify, mock_apply_edit
):
    """
    D9 — apply_edit()이 스키마 위반(visit_sequence 중복)을 만드는 결과를 반환해도
    revise_course()가 예외를 던지지 않고 원본 코스 + 안내 메시지로 폴백하는지 확인.
    이게 없으면 발표 시연 중 서버가 죽을 수 있는 지점.
    """
    mock_classify.return_value = make_edit("remove", target_sequence=2)
    course = sample_course()
    # visit_sequence가 둘 다 1로 중복 — StreamingCourseModel 재조립 시 ValidationError
    mock_apply_edit.return_value = (
        [
            RecommendedPlaceModel(place_id="1", visit_sequence=1),
            RecommendedPlaceModel(place_id="3", visit_sequence=1),
        ],
        "잘못 수정된 코스",
    )

    result = await revise_course("첨성대는 빼줘", course, minimal_ctx())

    assert _places(result) == _places(course)
    assert result.bot_message == "수정을 반영하지 못했어요, 원래 코스를 유지할게요."
