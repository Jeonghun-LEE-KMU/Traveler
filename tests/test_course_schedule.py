"""Deterministic schedule-policy tests (no LLM or routing API required)."""

from traveler.core.course_schedule import apply_schedule
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


def _spot(place_id: str, *, category: str = "관광", lat: float = 35.0, lng: float = 129.0):
    return SpotCandidate(
        place_id=place_id,
        place_name=f"장소{place_id}",
        category=category,
        distance_min=0.0,
        indoor=False,
        lat=lat,
        lng=lng,
    )


def _ctx(*, spots, hours: int = 4, query: str = "경주 추천", transport: str = "자차",
         lat: float = 35.5, lng: float = 129.5):
    return ContextData(
        user_query=query,
        current_lat=lat,
        current_lng=lng,
        weather="맑음",
        candidate_spots=spots,
        duration_hours=hours,
        duration_hours_min=None,
        transport=transport,
    )


def _course(*places: RecommendedPlaceModel) -> StreamingCourseModel:
    return StreamingCourseModel(bot_message="코스", recommended_places=list(places))


def _fixed_40_min(monkeypatch):
    monkeypatch.setattr(
        "traveler.core.course_schedule._estimate_leg_minutes", lambda *_args: 40.0
    )


def test_budget_counts_origin_and_each_incoming_leg_then_rejects_fourth(monkeypatch):
    """240분에서 (이동40 + 체류30) 3곳=210, 4곳=280이므로 3곳까지만 채운다."""
    _fixed_40_min(monkeypatch)
    spots = [_spot(str(i), lat=35.0 + i / 100, lng=129.0 + i / 100) for i in range(1, 5)]
    course = _course(*[
        RecommendedPlaceModel(place_id=str(i), visit_sequence=i, duration_min=30)
        for i in range(1, 5)
    ])

    result = apply_schedule(course, _ctx(spots=spots, hours=4))

    assert result.status == "within_budget"
    assert [place.place_id for place in result.places] == ["1", "2", "3"]
    assert result.travel_min == 120.0
    assert result.stay_min == 90
    assert result.total_min == 210.0
    assert result.adjusted is True


def test_candidate_scarcity_allows_fewer_than_default_three(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot("1"), _spot("2", lat=35.1, lng=129.1)]
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=spots))

    assert result.status == "within_budget"
    assert [place.place_id for place in result.places] == ["1", "2"]
    assert "후보" in result.note


def test_explicit_two_place_request_is_not_filled_to_three(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot(str(i), lat=35.0 + i / 100, lng=129.0 + i / 100) for i in range(1, 5)]
    course = _course(
        RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30),
        RecommendedPlaceModel(place_id="2", visit_sequence=2, duration_min=30),
    )

    result = apply_schedule(course, _ctx(spots=spots, query="경주 2곳만 추천해줘"))

    assert result.status == "within_budget"
    assert [place.place_id for place in result.places] == ["1", "2"]


def test_ui_configuration_count_is_not_treated_as_an_explicit_user_request(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot(str(i), lat=35.0 + i / 100, lng=129.0 + i / 100) for i in range(1, 4)]
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(
        course,
        _ctx(spots=spots, hours=5, query="요청: 경주 둘러봐줘\n구성: 2곳으로 만들어줘"),
    )

    assert [place.place_id for place in result.places] == ["1", "2", "3"]


def test_short_budget_overrides_default_three(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot("1"), _spot("2", lat=35.1, lng=129.1), _spot("3", lat=35.2, lng=129.2)]
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=spots, hours=2))

    assert result.status == "within_budget"
    assert [place.place_id for place in result.places] == ["1"]
    assert "시간" in result.note


def test_no_feasible_place_returns_empty_without_fabricating(monkeypatch):
    _fixed_40_min(monkeypatch)
    # Raw LLM output remains non-empty by contract; this is the server-built
    # no-selection state supported by the integration response model.
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30)).model_copy(
        update={"recommended_places": []}
    )

    result = apply_schedule(course, _ctx(spots=[_spot("1")]))

    assert result.status == "no_feasible"
    assert result.places == []
    assert result.total_min is None


def test_empty_candidate_fallback_placeholder_is_not_returned_as_a_stop(monkeypatch):
    _fixed_40_min(monkeypatch)
    fallback_like_course = _course(
        RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=None)
    )

    result = apply_schedule(fallback_like_course, _ctx(spots=[]))

    assert result.status == "no_feasible"
    assert result.places == []
    assert result.budget_min == 240
    assert "가짜" in result.note


def test_legacy_integer_candidate_id_still_matches_string_schema_output(monkeypatch):
    _fixed_40_min(monkeypatch)
    spot = _spot("1")
    spot.place_id = 1  # retriever's legacy int-content-id path
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=[spot], hours=2), fill_minimum=False)

    assert result.status == "within_budget"
    assert result.places[0].place_id == "1"


def test_unknown_selected_id_is_removed_while_grounded_selection_is_retained(monkeypatch):
    _fixed_40_min(monkeypatch)
    course = _course(
        RecommendedPlaceModel(place_id="unknown", visit_sequence=1, duration_min=30),
        RecommendedPlaceModel(place_id="1", visit_sequence=2, duration_min=30),
    )

    result = apply_schedule(course, _ctx(spots=[_spot("1")], hours=2), fill_minimum=False)

    assert result.status == "within_budget"
    assert [place.place_id for place in result.places] == ["1"]
    assert result.adjusted is True


def test_initial_generation_rebuilds_from_real_candidates_when_all_selected_ids_unknown(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot(str(i), lat=35.0 + i / 100, lng=129.0 + i / 100) for i in range(1, 4)]
    course = _course(RecommendedPlaceModel(place_id="unknown", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=spots, hours=5))

    assert result.status == "within_budget"
    assert [place.place_id for place in result.places] == ["1", "2", "3"]


def test_revise_does_not_rebuild_when_all_selected_ids_unknown(monkeypatch):
    _fixed_40_min(monkeypatch)
    course = _course(RecommendedPlaceModel(place_id="unknown", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=[_spot("1")], hours=2), fill_minimum=False)

    assert result.status == "no_feasible"
    assert result.places == []


def test_later_feasible_candidate_replaces_an_over_budget_first_selection(monkeypatch):
    def estimate(_start_lat, _start_lng, end_lat, _end_lng, _speed):
        return 40.0 if end_lat < 35.2 else 10.0

    monkeypatch.setattr("traveler.core.course_schedule._estimate_leg_minutes", estimate)
    far = _spot("far", lat=35.1, lng=129.1)
    near = _spot("near", category="쇼핑", lat=35.3, lng=129.3)
    course = _course(RecommendedPlaceModel(place_id="far", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=[far, near], hours=1))

    assert result.status == "within_budget"
    assert [place.place_id for place in result.places] == ["near"]


def test_unverified_coordinates_still_trim_stay_only_budget_excess(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot(str(i), lat=35.0 + i / 100, lng=129.0 + i / 100) for i in range(1, 4)]
    course = _course(*[
        RecommendedPlaceModel(place_id=str(i), visit_sequence=i, duration_min=60)
        for i in range(1, 4)
    ])

    result = apply_schedule(course, _ctx(spots=spots, hours=2, lat=0.0))

    assert result.status == "unverified"
    assert [place.place_id for place in result.places] == ["1", "2"]
    assert result.stay_min == 120


def test_trimmed_selected_candidate_keeps_its_original_stay_if_reconsidered(monkeypatch):
    monkeypatch.setattr(
        "traveler.core.course_schedule._estimate_leg_minutes", lambda *_args: 0.0
    )
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=120))

    result = apply_schedule(course, _ctx(spots=[_spot("1")], hours=1))

    assert result.status == "no_feasible"
    assert result.places == []


def test_within_budget_note_declares_coordinate_estimate_assumptions(monkeypatch):
    _fixed_40_min(monkeypatch)
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=[_spot("1")], hours=2), fill_minimum=False)

    assert "좌표 직선거리" in result.note
    assert "자차 평균30km/h" in result.note
    assert "도로/교통/주차 미반영" in result.note


def test_invalid_origin_coordinates_are_unverified_and_do_not_fill(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot("1"), _spot("2", lat=35.1, lng=129.1), _spot("3", lat=35.2, lng=129.2)]
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=spots, lat=0.0))

    assert result.status == "unverified"
    assert [place.place_id for place in result.places] == ["1"]
    assert result.travel_min is None
    assert result.total_min is None


def test_schedule_origin_overrides_the_retrieval_search_center(monkeypatch):
    starts: list[tuple[float, float]] = []

    def estimate(start_lat, start_lng, *_args):
        starts.append((start_lat, start_lng))
        return 40.0

    monkeypatch.setattr("traveler.core.course_schedule._estimate_leg_minutes", estimate)
    ctx = _ctx(spots=[_spot("1")], hours=2, lat=35.5, lng=129.5)
    # main.py provides these when a region changes the retrieval search center.
    ctx.schedule_origin_lat = 37.5
    ctx.schedule_origin_lng = 127.0
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, ctx, fill_minimum=False)

    assert result.status == "within_budget"
    assert starts == [(37.5, 127.0)]


def test_duplicate_selected_places_are_deduplicated_before_filling(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot(str(i), lat=35.0 + i / 100, lng=129.0 + i / 100) for i in range(1, 4)]
    course = _course(
        RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30),
        RecommendedPlaceModel(place_id="1", visit_sequence=2, duration_min=30),
    )

    result = apply_schedule(course, _ctx(spots=spots, hours=5))

    assert [place.place_id for place in result.places] == ["1", "2", "3"]
    assert [place.visit_sequence for place in result.places] == [1, 2, 3]
    assert result.adjusted is True


def test_missing_duration_uses_category_default_without_shortening(monkeypatch):
    _fixed_40_min(monkeypatch)
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=None))

    result = apply_schedule(course, _ctx(spots=[_spot("1", category="쇼핑")], hours=2), fill_minimum=False)

    assert result.status == "within_budget"
    assert result.places[0].duration_min == 40
    assert result.stay_min == 40
    assert result.total_min == 80.0


def test_public_transport_is_unverified_and_does_not_add_candidates(monkeypatch):
    _fixed_40_min(monkeypatch)
    spots = [_spot("1"), _spot("2", lat=35.1, lng=129.1), _spot("3", lat=35.2, lng=129.2)]
    course = _course(RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=30))

    result = apply_schedule(course, _ctx(spots=spots, transport="대중교통"))

    assert result.status == "unverified"
    assert [place.place_id for place in result.places] == ["1"]
    assert "이동 수단" in result.note
