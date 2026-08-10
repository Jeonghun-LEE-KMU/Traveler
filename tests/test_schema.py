"""
Pydantic Guardrails Stage 1 단위 테스트.

WHY 이 테스트가 필요한가:
  GPT-4o는 프롬프팅을 아무리 잘 해도 간혹 스키마를 어기는 JSON을 반환한다.
  이 테스트들이 있어야 "Guardrails가 실제로 막는다"는 것을 보장할 수 있다.

파이프라인 위치:
  GPT-4o 코스 생성 → ★ Pydantic 검증 → FastAPI → 지도 시각화

[T7 — 2026-05-27]
  경계값(Boundary Value) 테스트 2종 추가:
    - 최대 장소 수(10개 통과, 11개 실패)
    - place_id=0 차단 (SQLite PK는 1부터 시작)
"""

import pytest
from pydantic import ValidationError

from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

def valid_place(place_id: int = 1, sequence: int = 1) -> dict:
    """테스트용 정상 장소 dict. 반복 작성을 줄이기 위한 헬퍼 (DRY)."""
    return {
        "place_id": place_id,
        "place_name": "불국사",
        "visit_sequence": sequence,
        "duration_min": 60,
    }


def valid_course(places: list[dict] | None = None) -> dict:
    """테스트용 정상 코스 dict."""
    if places is None:
        places = [valid_place(1, 1)]
    return {
        "bot_message": "경주의 고즈넉한 코스를 추천드립니다.",
        "recommended_places": places,
    }


# ── Test 1: 정상 케이스 ────────────────────────────────────────────────────────

def test_valid_course_passes():
    """
    정상적인 StreamingCourseModel이 검증을 통과하는지 확인.

    WHY: Guardrails가 정상 응답까지 막아버리면 안 된다.
    경계 케이스가 아닌 '골든 패스'를 먼저 검증하는 것이 테스트 설계의 기본.
    """
    places = [
        valid_place(1, 1),
        valid_place(2, 2),
        valid_place(3, 3),
    ]
    course = StreamingCourseModel.model_validate(valid_course(places))

    assert course.bot_message == "경주의 고즈넉한 코스를 추천드립니다."
    assert len(course.recommended_places) == 3
    assert course.recommended_places[0].place_name == "불국사"


# ── Test 2: place_name 빈 문자열 ──────────────────────────────────────────────

def test_empty_place_name_raises():
    """
    GPT가 place_name을 빈 문자열로 반환할 때 ValidationError가 발생하는지 확인.

    WHY: GPT는 간혹 장소명을 ""로 반환한다.
    빈 문자열이 DB 조회 키로 들어가면 SQLite에서 오류가 나거나 잘못된 장소가 핀으로 찍힌다.
    """
    bad_place = valid_place()
    bad_place["place_name"] = "   "  # 공백만 있는 경우도 방어

    with pytest.raises(ValidationError) as exc_info:
        StreamingCourseModel.model_validate(valid_course([bad_place]))

    # 어느 필드에서 에러가 났는지 확인
    assert "place_name" in str(exc_info.value)


# ── Test 3: visit_sequence 불연속 (gap) ──────────────────────────────────────

def test_non_sequential_visit_sequence_raises():
    """
    visit_sequence 값 자체에 gap이 있으면 정렬 후에도 여전히 ValidationError.

    [동작 변경 — 2026-07-05] check_sequence_sequential이 "배열 순서 강제"에서
    "visit_sequence로 정렬 후 1..n 연속성 검증"으로 바뀌었다. 이 테스트는 gap
    케이스(2를 건너뜀)라 정렬해도 연속되지 않으므로 여전히 raise돼야 맞다 —
    즉 옛 동작과 결과는 같지만, 통과 이유가 "배열 순서"가 아니라 "정렬 후 값
    자체의 gap"으로 바뀌었다.

    WHY: 지도에 Polyline을 그릴 때 방문 순서가 곧 경로 순서다.
    GPT가 1, 3처럼 값 자체를 건너뛰면(진짜 malformation) 경로가 엉킨다.
    """
    places = [
        valid_place(1, 1),
        valid_place(2, 3),  # 2를 건너뜀 → gap (정렬해도 연속 아님)
    ]
    with pytest.raises(ValidationError) as exc_info:
        StreamingCourseModel.model_validate(valid_course(places))

    assert "연속" in str(exc_info.value)


# ── Test 3b: visit_sequence 배열 순서만 뒤섞임 (신규) ────────────────────────

def test_shuffled_but_continuous_visit_sequence_passes():
    """
    배열 순서는 뒤섞였지만 visit_sequence 값 자체는 연속(1, 2)이면 통과하고,
    정렬된 리스트([1, 2] 순서)가 반환되는지 확인.

    [동작 변경 — 2026-07-05] 옛 로직은 enumerate(places)로 "배열 인덱스"와
    visit_sequence를 비교했다. GPT가 순서 값은 맞게 냈는데 배열만 [2, 1]로
    섞어 내면(정상 출력의 사소한 이탈) 전체 코스가 폴백으로 강등됐다.
    Guardrails ③ "부분 재생성" 철학에 맞춰, 값 기준 정렬 검증으로 바꿨다.
    """
    places = [
        valid_place(2, 2),  # 배열상 1번째지만 visit_sequence=2
        valid_place(1, 1),  # 배열상 2번째지만 visit_sequence=1
    ]
    course = StreamingCourseModel.model_validate(valid_course(places))

    # 정렬된 리스트가 반환되어 다운스트림(Polyline)이 순서대로 받는지 확인
    assert [p.visit_sequence for p in course.recommended_places] == [1, 2]
    assert course.recommended_places[0].place_id == 1
    assert course.recommended_places[1].place_id == 2


# ── Test 3c: visit_sequence 중복 (신규) ──────────────────────────────────────

def test_duplicate_visit_sequence_raises():
    """
    visit_sequence 값이 중복(1, 2, 2)이면 정렬해도 연속이 아니므로 여전히 raise.

    WHY: 중복은 gap과 마찬가지로 진짜 malformation이다.
    두 장소가 같은 순번을 주장하면 Polyline이 어느 쪽을 먼저 그려야 할지
    알 수 없다 — clamp나 정렬로 흡수할 수 없는 케이스이므로 계속 거부해야 한다.
    """
    places = [
        valid_place(1, 1),
        valid_place(2, 2),
        valid_place(3, 2),  # 2가 중복
    ]
    with pytest.raises(ValidationError) as exc_info:
        StreamingCourseModel.model_validate(valid_course(places))

    assert "연속" in str(exc_info.value)


# ── Test 4: duration_min 범위 초과 → clamp ───────────────────────────────────

def test_duration_min_out_of_range_clamps():
    """
    duration_min이 허용 범위(10~480분) 밖이면 raise 대신 clamp되는지 확인.

    [동작 변경 — 2026-07-05] 옛 로직은 Field(ge=10, le=480)로 하드 거부했다.
    10곳 중 1곳만 duration_min이 이상해도(예: 600) course_generator가
    ValidationError를 잡아 코스 전체를 폴백("죄송해요…" 1줄)으로 강등시켰다.
    Guardrails ③ "부분 재생성" 철학에 맞춰, 범위 밖 값은 clamp로 흡수하고
    코스 전체는 그대로 통과시키도록 바꿨다.

    WHY: GPT가 "5분 방문" 같은 비현실적 수치를 내놓거나
    "600분(10시간)" 같은 일정 전체를 하나의 장소에 쏟아붓는 경우,
    이제는 그 장소 하나만 클램프되고 나머지 코스는 정상 진행된다.
    """
    # 너무 짧은 경우 (10분 미만) → 10으로 clamp
    short_place = valid_place()
    short_place["duration_min"] = 5
    course = StreamingCourseModel.model_validate(valid_course([short_place]))
    assert course.recommended_places[0].duration_min == 10

    # 너무 긴 경우 (8시간 초과) → 480으로 clamp
    long_place = valid_place()
    long_place["duration_min"] = 600
    course = StreamingCourseModel.model_validate(valid_course([long_place]))
    assert course.recommended_places[0].duration_min == 480

    # None은 그대로 유지 (백엔드가 나중에 채워주는 흐름과 호환)
    none_place = valid_place()
    none_place["duration_min"] = None
    course = StreamingCourseModel.model_validate(valid_course([none_place]))
    assert course.recommended_places[0].duration_min is None


# ── Test 5: recommended_places 비어있음 ──────────────────────────────────────

def test_empty_recommended_places_raises():
    """
    GPT가 recommended_places를 빈 리스트로 반환할 때 ValidationError가 발생하는지 확인.

    WHY: 빈 코스는 서비스 장애와 동일하다.
    지도에 핀이 하나도 없으면 사용자는 아무것도 할 수 없다.
    Field(min_length=1)로 선언했지만 실제로 막히는지 이 테스트로 보장한다.
    """
    with pytest.raises(ValidationError):
        StreamingCourseModel.model_validate(valid_course(places=[]))


# ── Test 6: 이수연 포맷 호환 (place_name, duration_min 없어도 통과) ───────────

def test_suyeon_format_passes():
    """
    이수연 /api/chat 출력처럼 place_name, duration_min이 없어도 통과하는지 확인.

    WHY: 이수연 출력 포맷: {"place_id": 1, "visit_sequence": 1}
    두 필드를 Optional로 바꿨으므로 이 포맷이 Pydantic 검증을 통과해야 한다.
    백엔드가 place_id로 SQLite 조회해 나중에 채우는 흐름 (lat/lng 제거와 동일 패턴).
    """
    suyeon_place = {
        "place_id": 1,
        "visit_sequence": 1,
        # place_name, duration_min 없음 — 이수연 포맷
    }
    course = StreamingCourseModel.model_validate(valid_course([suyeon_place]))
    assert course.recommended_places[0].place_name is None
    assert course.recommended_places[0].duration_min is None


# ── Test 7: recommended_places 최대 개수 경계값 ───────────────────────────────

def test_max_places_boundary():
    """
    10개 장소(최대) → 통과, 11개 장소 → ValidationError 확인.

    WHY: Field(max_length=10) 선언이 실제로 작동하는지 경계값으로 검증.
    GPT가 10개 이상의 장소를 반환하면 사용자 화면이 감당 못 할 정도로 핀이 많아진다.
    경계값 테스트는 "딱 그 숫자"를 직접 넣어보는 것이 핵심 — 9개나 12개로 테스트하면 안 됨.
    """
    # 10개 → 통과해야 함 (최대값 포함)
    places_10 = [valid_place(i, i) for i in range(1, 11)]
    course = StreamingCourseModel.model_validate(valid_course(places_10))
    assert len(course.recommended_places) == 10

    # 11개 → ValidationError (최대값 초과)
    places_11 = [valid_place(i, i) for i in range(1, 12)]
    with pytest.raises(ValidationError):
        StreamingCourseModel.model_validate(valid_course(places_11))


# ── Test 8: place_id 경계값 (0 차단) ─────────────────────────────────────────

def test_place_id_zero_raises():
    """
    place_id=0은 Field(ge=1)에 의해 ValidationError.

    WHY: SQLite PK는 1부터 시작하므로 0은 존재하지 않는 레코드.
    place_id=0이 통과하면 SQLite 조회 시 "장소 없음" 오류가 서비스 레이어에서 발생.
    Guardrails에서 미리 막는 게 올바른 설계.
    """
    bad_place = valid_place()
    bad_place["place_id"] = 0
    with pytest.raises(ValidationError):
        StreamingCourseModel.model_validate(valid_course([bad_place]))
