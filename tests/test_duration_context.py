"""사용자 요청의 시간 정보가 RAG ContextData까지 전달되는 배선 테스트."""

from unittest.mock import patch

from traveler.rag.preference import extract_prefs_from_message, extract_user_request
from traveler.rag.retriever import get_context


def test_요청의_정확한_시간만_context에_전달한다():
    """UI 구성값이 아니라 요청의 4시간이 ContextData에 남아야 한다."""
    message = "요청: 경주 4시간 코스 추천해줘\n구성: 8시간 코스를 만들어줘"
    prefs = extract_prefs_from_message(message)

    # WHY: 검색 I/O와 무관하게 preference → ContextData 배선만 검증한다.
    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query=extract_user_request(message), prefs=prefs)

    assert ctx.duration_hours == 4
    assert ctx.duration_hours_min is None
    assert ctx.duration_minutes == 240


def test_대중교통_선호가_context에_보존되어_자차로_바뀌지_않는다():
    prefs = {"transport": "대중교통"}

    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주로 갈게", prefs=prefs)

    assert ctx.transport == "대중교통"
    assert ctx.transport_assumed is False


def test_교통수단이_없을때만_context의_기본_자차를_쓴다():
    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주로 갈게", prefs={})

    assert ctx.transport == "자차"
    assert ctx.transport_assumed is True


def test_요청의_교통수단_표현을_정규화한다():
    expected_by_message = {
        "요청: 경주를 걸어서 둘러볼래": "도보",
        "요청: 경주 도보 코스 추천해줘": "도보",
        "요청: 자동차로 경주 갈래": "자차",
        "요청: 차로 경주 바다 보러 갈래": "자차",
        "요청: 대중교통으로 경주 갈래": "대중교통",
        "요청: 버스로 경주 갈래": "대중교통",
        "요청: 기차로 경주 갈래": "대중교통",
        "요청: 열차로 경주 갈래": "대중교통",
        "요청: KTX 타고 경주 갈래": "대중교통",
        "요청: SRT로 경주 갈래": "대중교통",
        "요청: 지하철 타고 갈래": "대중교통",
    }
    for message, expected in expected_by_message.items():
        prefs = extract_prefs_from_message(message)
        assert prefs["transport"] == expected
        assert prefs["transport_assumed"] is False


def test_UI_구성문구의_교통수단은_선호도로_읽지_않는다():
    message = "교통수단: 버스로 이동\n경주 코스 추천해줘"
    assert "transport" not in extract_prefs_from_message(message)


def test_교통수단_오인과_혼합_선택은_검증불가로_보존한다():
    assert "transport" not in extract_prefs_from_message("버스킹 보러 가고 싶어")
    for message in ("차나 대중교통으로 갈게", "기차로 가서 도보 여행할래"):
        prefs = extract_prefs_from_message(message)
        assert prefs["transport"] == "혼합"
        assert prefs["transport_assumed"] is False

    prefs = extract_prefs_from_message("대중교통 말고 경주 갈래")
    assert prefs["transport"] == "불명"
    assert prefs["transport_assumed"] is False


def test_명시적으로_부정한_자차는_기본_자차가_되지_않는다():
    prefs = extract_prefs_from_message("차 없이 경주 갈래")
    assert prefs["transport"] == "불명"
    assert prefs["transport_assumed"] is False


def test_지원하지_않는_이동수단도_기본_자차로_가정하지_않는다():
    for message in ("자전거로 경주 갈래", "킥보드로 이동할래"):
        prefs = extract_prefs_from_message(message)
        assert prefs["transport"] == "불명"
        assert prefs["transport_assumed"] is False

    mixed = extract_prefs_from_message("자전거로 가거나 차로 갈래")
    assert mixed["transport"] == "혼합"
    assert mixed["transport_assumed"] is False


def test_교통수단_언급이_전혀_없으면_가정_표시를_만들지_않는다():
    prefs = extract_prefs_from_message("경주 코스 추천해줘")
    assert "transport" not in prefs
    assert "transport_assumed" not in prefs
