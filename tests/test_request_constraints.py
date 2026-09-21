"""PR #47 입력 제약: 시간·이동수단은 LLM 없이 보수적으로 해석한다."""

from unittest.mock import patch

from traveler.rag.preference import extract_prefs_from_message
from traveler.rag.retriever import get_context


def test_기차와_분_단위_시간이_동일_context_계약으로_전달된다():
    prefs = extract_prefs_from_message("기차로 경주 가서 1.5시간만 가능해")

    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주 추천", prefs=prefs)

    assert prefs["transport"] == "대중교통"
    assert prefs["duration_minutes"] == 90
    assert prefs["duration_hours"] == 1.5
    assert ctx.duration_minutes == 90
    assert ctx.duration_hours == 1.5
    assert ctx.transport_assumed is False


def test_분_예산이_레거시_시간값보다_우선한다():
    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(
            user_query="경주 추천",
            prefs={"duration_minutes": 90, "duration_hours": 4},
        )

    assert ctx.duration_minutes == 90
    assert ctx.duration_hours == 1.5


def test_분_예산이_없는_레거시_시간값은_보존한다():
    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주 추천", prefs={"duration_hours": 4})

    assert ctx.duration_minutes is None
    assert ctx.duration_hours == 4


def test_혼합_또는_부정_교통수단은_기본_자차로_가정하지_않는다():
    for message, expected in (("차나 대중교통으로 갈래", "혼합"), ("차 없이 갈래", "불명")):
        prefs = extract_prefs_from_message(message)
        with patch("traveler.rag.retriever.search_places", return_value=[]):
            ctx, _ = get_context(user_query="경주 추천", prefs=prefs)
        assert ctx.transport == expected
        assert ctx.transport_assumed is False


def test_지원하지_않는_이동수단도_검증불가_교통값으로_전달된다():
    prefs = extract_prefs_from_message("자전거로 경주 갈래")
    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주 추천", prefs=prefs)

    assert ctx.transport == "불명"
    assert ctx.transport_assumed is False
