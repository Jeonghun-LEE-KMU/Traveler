"""PM의 장소 수/수정 일정 회귀를 LLM 없이 검증한다."""

from dataclasses import replace

import pytest

from traveler.core.course_schedule import _explicit_place_count, apply_schedule
from traveler.core.edit_intent import EditIntent, apply_edit
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


@pytest.mark.parametrize('query', [
    '카페 1곳 추가해줘', '불국사 말고 다른 곳 1개 넣어줘',
    '맛집 2개 포함해서 경주 코스', '5개월 만에 가는 여행',
    '3~4곳 추천해줘', '3-4곳 코스', '1곳 빼줘',
    '총 2곳만 빼줘', '총 2곳 추가해줘', '맛집 2곳 관광지 3곳 추천해줘',
])
def test_partial_counts_are_not_course_totals(query):
    assert _explicit_place_count(query) is None


@pytest.mark.parametrize('query, expected', [
    ('총 2곳으로 줄여줘', 2), ('경주 2곳만 추천해줘', 2),
    ('경주 총4곳 코스', 4), ('카페 1곳 추가해서 총 4곳으로', 4),
    ('경주 4곳 코스 추천', 4),
])
def test_clear_total_counts_still_work(query, expected):
    assert _explicit_place_count(query) == expected


def context(query='경주 코스', *, minutes=None, count=6):
    ctx = ContextData(
        user_query=query, current_lat=35.8, current_lng=129.2, weather='맑음',
        duration_hours=5, duration_hours_min=None,
        candidate_spots=[SpotCandidate(str(i), f'장소{i}', '관광', 0, False, 35.8, 129.2)
                         for i in range(1, count + 1)],
    )
    if minutes is not None:
        ctx.duration_minutes = minutes
    return ctx


def course(count):
    return StreamingCourseModel(bot_message='코스', recommended_places=[
        RecommendedPlaceModel(place_id=str(i), visit_sequence=i, duration_min=30)
        for i in range(1, count + 1)
    ])


def test_add_one_preserves_edited_course_instead_of_truncating_to_one(monkeypatch):
    monkeypatch.setattr('traveler.core.course_schedule._estimate_leg_minutes', lambda *a: 5)
    result = apply_schedule(course(4), context('카페 1곳 추가해줘'), fill_minimum=False)
    assert [p.place_id for p in result.places] == ['1', '2', '3', '4']


def test_explicit_edit_can_exceed_default_five_when_time_allows(monkeypatch):
    monkeypatch.setattr('traveler.core.course_schedule._estimate_leg_minutes', lambda *a: 5)
    result = apply_schedule(course(6), context('카페 1곳 추가해줘'), fill_minimum=False)
    assert len(result.places) == 6


def test_partial_total_prefix_does_not_retruncate_edited_course(monkeypatch):
    monkeypatch.setattr('traveler.core.course_schedule._estimate_leg_minutes', lambda *a: 5)
    result = apply_schedule(course(4), context('총 2곳만 빼줘'), fill_minimum=False)
    assert len(result.places) == 4  # 편집 계층이 이미 삭제한 목록을 다시 2곳으로 줄이지 않는다.


def test_minute_budget_overrides_legacy_hours_and_explains_trim(monkeypatch):
    monkeypatch.setattr('traveler.core.course_schedule._estimate_leg_minutes', lambda *a: 5)
    result = apply_schedule(course(3), context(minutes=40), fill_minimum=False)
    assert result.budget_min == 40
    assert len(result.places) == 1
    assert '시간' in result.note and '줄' in result.note
    assert '장소 수를 유지' not in result.note


def test_assumed_transport_is_disclosed(monkeypatch):
    monkeypatch.setattr('traveler.core.course_schedule._estimate_leg_minutes', lambda *a: 5)
    ctx = context()
    ctx.transport_assumed = True
    result = apply_schedule(course(3), ctx)
    assert '미지정' in result.note and '가정' in result.note


@pytest.mark.parametrize('action, category, expected', [
    ('add', '쇼핑', 40), ('replace', '축제', 90),
])
def test_edits_share_category_stay_defaults(action, category, expected):
    ctx = context(count=2)
    ctx.candidate_spots[1] = replace(ctx.candidate_spots[1], category=category)
    intent = EditIntent(action=action, target_sequence=1 if action == 'replace' else None,
                        new_place_id='2', reason='수정')
    places, _ = apply_edit(course(1), intent, ctx)
    assert next(p for p in places if p.place_id == '2').duration_min == expected
