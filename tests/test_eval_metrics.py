import json

from eval.metrics import FALLBACK_BOT_MESSAGE, measure
from traveler.rag.types import ContextData, SpotCandidate


def _ctx(*, duration_hours: int = 4) -> ContextData:
    return ContextData(
        user_query="분위기 좋은 곳 추천해줘",
        current_lat=35.8,
        current_lng=129.2,
        weather="맑음",
        candidate_spots=[
            SpotCandidate(place_id="1", place_name="A", category="관광", distance_min=10.0, indoor=True),
            SpotCandidate(place_id="2", place_name="B", category="음식", distance_min=15.0, indoor=False),
        ],
        duration_hours=duration_hours,
    )


def _course_json(recommended_places: list[dict], bot_message: str = "좋은 코스를 준비했어요.") -> str:
    return json.dumps({"bot_message": bot_message, "recommended_places": recommended_places})


def test_normal_course_passes_all_checks():
    output = _course_json(
        [{"place_id": "1", "visit_sequence": 1, "place_name": "A", "duration_min": 60}]
    )
    metrics = measure(_ctx(), {"expect_places_range": [1, 3]}, output)

    assert metrics.json_valid is True
    assert metrics.is_fallback is False
    assert metrics.place_count_ok is True
    assert metrics.grounded is True
    assert metrics.hallucinated_ids == []
    assert metrics.time_budget_ok is True
    assert metrics.passed is True


def test_broken_json_fails_json_valid_only():
    metrics = measure(_ctx(), {"expect_places_range": [1, 3]}, "{not valid json")

    assert metrics.json_valid is False
    assert metrics.passed is False


def test_eleven_places_rejected_by_schema():
    # RecommendedPlaceModel의 max_length=10 제약으로 스키마 검증 단계에서 걸린다.
    places = [
        {"place_id": "1", "visit_sequence": i, "place_name": "A", "duration_min": 10}
        for i in range(1, 12)
    ]
    metrics = measure(_ctx(), {"expect_places_range": [1, 10]}, _course_json(places))

    assert metrics.json_valid is False
    assert metrics.passed is False


def test_hallucinated_place_id_fails_grounded_only():
    output = _course_json(
        [{"place_id": "99", "visit_sequence": 1, "place_name": "없는곳", "duration_min": 30}]
    )
    metrics = measure(_ctx(), {"expect_places_range": [1, 3]}, output)

    assert metrics.json_valid is True
    assert metrics.grounded is False
    assert metrics.hallucinated_ids == ["99"]
    assert metrics.passed is False


def test_time_budget_exceeded_fails_time_budget_only():
    # duration_hours=1 → 예산 72분(1*60*1.2). duration_min 400 + 이동 20분(10*2) 초과.
    output = _course_json(
        [{"place_id": "1", "visit_sequence": 1, "place_name": "A", "duration_min": 400}]
    )
    metrics = measure(_ctx(duration_hours=1), {"expect_places_range": [1, 3]}, output)

    assert metrics.json_valid is True
    assert metrics.grounded is True
    assert metrics.time_budget_ok is False
    assert metrics.passed is False


def test_fallback_message_detected():
    output = _course_json(
        [{"place_id": "1", "visit_sequence": 1}],
        bot_message=FALLBACK_BOT_MESSAGE,
    )
    metrics = measure(_ctx(), {"expect_places_range": [1, 3]}, output)

    assert metrics.json_valid is True
    assert metrics.is_fallback is True
    assert metrics.passed is False
