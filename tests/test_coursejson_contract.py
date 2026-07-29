"""CourseJSON 문서와 Pydantic 런타임 계약의 핵심 불변식 테스트."""

from pathlib import Path

from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


SPEC_PATH = Path(__file__).parents[1] / "docs" / "coursejson-spec.md"


def test_streaming_course_field_order_is_stable():
    assert list(StreamingCourseModel.model_fields) == [
        "bot_message",
        "recommended_places",
    ]


def test_coursejson_required_and_optional_place_fields_are_stable():
    fields = RecommendedPlaceModel.model_fields

    assert fields["place_id"].is_required()
    assert fields["visit_sequence"].is_required()
    assert not fields["place_name"].is_required()
    assert not fields["duration_min"].is_required()


def test_coursejson_spec_documents_runtime_contract():
    text = SPEC_PATH.read_text(encoding="utf-8")

    for required_text in (
        "bot_message",
        "recommended_places",
        "place_id",
        "visit_sequence",
        "place_name",
        "duration_min",
        "1~10",
        "10~480",
        "fallback",
    ):
        assert required_text in text
