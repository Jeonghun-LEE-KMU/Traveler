"""요청 문장에서 여행 가능 시간을 뽑는 로직 단위 테스트 (LLM 호출 없음)."""

import pytest

from traveler.core.prompt import build_context_message
from traveler.rag.preference import (
    extract_duration_minutes,
    extract_duration_hours,
    extract_prefs_from_message,
    extract_user_request,
)
from traveler.rag.types import ContextData, SpotCandidate


@pytest.mark.parametrize("text, expected", [
    ("경주 5시간 코스 추천해줘", 5),
    ("경주 8 시간 코스", 8),
    ("반나절 경주 여행", 5),          # 4~6시간의 중앙값
    ("하루 종일 경주 돌아다닐래", 8),
    ("하루종일 코스 짜줘", 8),
    ("경주 하루 코스", 8),
    ("1박 2일로 경주 갈 건데 첫날 코스", 8),
])
def test_시간_표현을_인식한다(text, expected):
    assert extract_duration_hours(text) == expected


@pytest.mark.parametrize("text", [
    "경주 코스 추천해줘",
    "황리단길 카페 가고 싶어",
    "",
])
def test_시간_언급이_없으면_None(text):
    assert extract_duration_hours(text) is None


@pytest.mark.parametrize("text", [
    "여기서 3시간 거리에 있는 곳 추천해줘",
    "차로 2시간 걸리는 바다 보러 가고 싶어",
    "차로 3시간 소요되는 바다 보러 가고 싶어",
])
def test_이동시간_표현은_가용시간으로_읽지_않는다(text):
    assert extract_duration_hours(text) is None


def test_세자리_시간의_끝자리만_부분_매칭하지_않는다():
    """123시간은 끝자리만 자르지 않고 명시된 상한 위반으로 알린다."""
    with pytest.raises(ValueError):
        extract_duration_hours("경주 123시간 코스")


def test_소수_시간은_분_단위로_보존한다():
    assert extract_duration_minutes("경주 1.5시간 코스") == 90
    assert extract_duration_hours("경주 1.5시간 코스") == 1.5


@pytest.mark.parametrize("text", ["24시간 코스 짜줘", "0시간 코스"])
def test_비현실적인_명시시간은_오류로_알린다(text):
    with pytest.raises(ValueError):
        extract_duration_minutes(text)


def test_숫자가_서술형보다_우선한다():
    assert extract_duration_hours("하루 종일은 아니고 4시간만") == 4


def test_요청_구획_밖의_숫자는_읽지_않는다():
    """프론트가 앞뒤에 붙이는 UI 줄을 여행 길이로 오인하면 안 된다."""
    message = (
        "여행 유형: 타지\n"
        "동행자: 친구\n"
        "요청: 경주 코스 추천해줘\n"
        "구성: 이동 순서가 있는 장소 4~5곳으로 코스를 만들어줘."
    )
    assert "duration_hours" not in extract_prefs_from_message(message)


def test_요청_뒤의_UI_구성시간은_가용시간으로_읽지_않는다():
    message = "요청: 경주 코스 추천해줘\n구성: 4시간 코스를 만들어줘"
    assert "duration_hours" not in extract_prefs_from_message(message)


def test_사용자_요청은_멀티라인을_보존하고_UI_라벨에서_끝난다():
    message = (
        "동행자: 친구\n"
        "요청: 경주 코스 추천해줘\n"
        "바다도 보고 싶어\n"
        "구성: 이동 순서가 있는 장소 4~5곳으로 코스를 만들어줘."
    )
    assert extract_user_request(message) == "경주 코스 추천해줘\n바다도 보고 싶어"


def test_요청_라벨이_없어도_UI_구성줄은_사용자_제약에서_제외한다():
    message = "구성: 8시간 코스를 만들어줘\n경주 코스 추천해줘"
    assert extract_user_request(message) == "경주 코스 추천해줘"
    assert "duration_hours" not in extract_prefs_from_message(message)


def test_요청_안의_시간은_prefs에_담긴다():
    message = "동행자: 친구\n요청: 경주 8시간 코스 추천해줘"
    assert extract_prefs_from_message(message)["duration_hours"] == 8


def _ctx(**kwargs) -> ContextData:
    return ContextData(
        user_query="경주 코스 추천해줘",
        current_lat=35.8347,
        current_lng=129.2189,
        weather="맑음",
        candidate_spots=[SpotCandidate("1", "경주 첨성대", "관광", 2.0, False)],
        **kwargs,
    )


def test_미지정이면_프롬프트에_범위로_나간다():
    assert "가용 시간: 3~5시간" in build_context_message(_ctx())


def test_지정되면_프롬프트에_단일값으로_나간다():
    message = build_context_message(_ctx(duration_hours=8, duration_hours_min=None))
    assert "가용 시간: 8시간" in message
    assert "~" not in message.split("가용 시간: ")[1].split("\n")[0]


@pytest.mark.parametrize(("text", "expected"), [
    ("30분밖에 없어", 30),
    ("1.5시간만 가능해", 90),
    ("4시간 30분 코스", 270),
    ("1시간 반 코스", 90),
])
def test_분_단위_가용시간을_정확히_추출한다(text, expected):
    assert extract_duration_minutes(text) == expected


@pytest.mark.parametrize("text", [
    "1시간 이내 거리",
    "하루 전에 예약했어",
    "3시간 걸리는 바다",
])
def test_이동_과거_시간은_가용시간으로_읽지_않는다(text):
    assert extract_duration_minutes(text) is None


def test_이내_코스와_앞선_이동시간은_실제_가용시간을_숨기지_않는다():
    assert extract_duration_minutes("1시간 이내로 코스 짜줘") == 60
    assert extract_duration_minutes("30분 걸리는 곳에서 4시간 여행") == 240


def test_이동시간_라벨은_가용시간이_아니고_뒤_가용시간은_계속_읽는다():
    assert extract_duration_minutes("이동시간 40분") is None
    assert extract_duration_minutes("이동시간 40분 가용시간 90분") == 90
    assert extract_duration_minutes("차로 40분 걸리는 곳에서 4시간 여행") == 240


@pytest.mark.parametrize(("text", "expected"), [
    ("1,5시간 코스", 90),
    (".5시간 코스", 30),
])
def test_콤마와_선행_점_소수시간은_부분_매칭하지_않는다(text, expected):
    assert extract_duration_minutes(text) == expected


def test_음수_복합_분은_가용시간_오류다():
    with pytest.raises(ValueError, match="가용 시간"):
        extract_duration_minutes("1시간 -30분 코스")


@pytest.mark.parametrize("text", ["0분만 가능해", "3~5시간 코스", "0.5분 코스"])
def test_지원하지_않는_명시시간은_기본값으로_숨기지_않는다(text):
    with pytest.raises(ValueError, match="가용 시간"):
        extract_duration_minutes(text)


def test_분_단위_시간을_프롬프트에_보존한다():
    message = build_context_message(_ctx(
        duration_minutes=90, duration_hours=1.5, duration_hours_min=None,
    ))
    assert "가용 시간: 90분 (1.5시간)" in message
