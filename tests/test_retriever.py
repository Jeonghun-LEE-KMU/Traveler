"""
retriever.py 통합 테스트 — preference extraction 결과가 get_context()에 전달되는지.

WHY 이 테스트가 필요한가:
  preference extraction 단위 동작과 별개로, 프론트 형식 메시지에서 추출한
  prefs가 get_context()를 거쳐 ContextData.companion에 반영되는지 보장한다.
  get_context() 자체는 query를 다시 파싱하지 않고 호출자가 전달한 prefs를
  보존하는 경계이므로, 실제 호출 경로의 extraction과 경계 양쪽을 함께 검증한다.

  get_context()는 (ContextData, spots_context) 튜플을 반환한다. 검색 경계는
  mock으로 차단하고, query extraction 결과와 명시적 prefs 전달이 각각
  ContextData.companion에 보존되는지 검증한다(테스트 1개 = 검증 1개).
"""

from unittest.mock import patch

from traveler.rag.preference import extract_prefs_from_message, extract_user_request
from traveler.rag.retriever import DEFAULT_TOP_K, get_context


def test_get_context_uses_extracted_companion_from_request():
    """실제 요청 본문에서 추출한 커플 선호가 컨텍스트에 반영된다."""
    message = "요청: 여자친구랑 부산 데이트\n구성: 4시간 코스"
    prefs = extract_prefs_from_message(message)

    with patch("traveler.rag.retriever.search_places", return_value=[]) as mock_search:
        ctx, _ = get_context(
            user_query=extract_user_request(message),
            prefs=prefs,
        )
    assert prefs["companion"] == "커플"
    assert ctx.companion == "커플"
    assert mock_search.called


def test_get_context_uses_extracted_solo_companion():
    """실제 요청 본문에서 추출한 솔로 선호가 컨텍스트에 반영된다."""
    message = "요청: 혼자 훌쩍 떠나고 싶어요"
    prefs = extract_prefs_from_message(message)

    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(
            user_query=extract_user_request(message),
            prefs=prefs,
        )
    assert prefs["companion"] == "혼자"
    assert ctx.companion == "혼자"


def test_get_context_defaults_missing_companion_to_honja():
    """동행자를 말하지 않으면 컨텍스트 기본값은 혼자다."""
    message = "요청: 맛집 추천해줘"
    prefs = extract_prefs_from_message(message)

    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(
            user_query=extract_user_request(message),
            prefs=prefs,
        )
    assert "companion" not in prefs
    assert ctx.companion == "혼자"


def test_get_context_preserves_explicit_companion_preference():
    """세션에서 이미 정규화된 명시적 선호는 query와 무관하게 보존된다."""
    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(
            user_query="맛집 추천해줘",
            prefs={"companion": "친구"},
        )
    assert ctx.companion == "친구"


# ── #31 PM 리뷰 8번: get_context()의 top_k 패스스루 ──────────────────────────
# WHY mock이 필요한가: search_places()가 hybrid_search 실패 시 _fallback_spots()
# (고정 개수 더미)로 빠지는데, 그건 top_k를 아예 무시한다. "get_context()가
# top_k를 search_places()에 제대로 전달하는가"라는 배선(wiring) 자체를 검증하려면
# search_places를 직접 관찰해야 한다 — 실제 랭킹 결과 개수를 세는 통합 테스트가
# 아니다.

def test_get_context_forwards_top_k_to_search_places():
    with patch("traveler.rag.retriever.search_places") as mock_search:
        mock_search.return_value = []
        get_context(user_query="아무거나", top_k=20)

    assert mock_search.call_args.kwargs["top_k"] == 20


def test_get_context_default_top_k_unchanged():
    """top_k를 안 넘기면 기존과 동일하게 DEFAULT_TOP_K가 쓰인다(회귀 0)."""
    with patch("traveler.rag.retriever.search_places") as mock_search:
        mock_search.return_value = []
        get_context(user_query="아무거나")

    assert mock_search.call_args.kwargs["top_k"] == DEFAULT_TOP_K
