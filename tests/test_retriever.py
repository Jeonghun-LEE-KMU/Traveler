"""
retriever.py 통합 테스트 — get_context()가 동승자를 query에서 채우는지.

WHY 이 테스트가 필요한가:
  parse_companion 단위 테스트(test_intent.py)와 별개로, get_context()가 더 이상
  companion="친구"로 하드코딩하지 않고 parse_companion(user_query)의 반환값을
  그대로 ContextData.companion에 채우는지 — 즉 두 모듈이 실제로 연결됐는지 —
  를 보장하는 통합 지점이다(하드코딩 회귀를 막는다).

  get_context()는 동기(sync) def이고 더미라 실제 I/O가 없으므로, mock 없이
  평이한 sync def로 .companion 필드 하나만 검증한다(테스트 1개 = 검증 1개).
"""

from unittest.mock import patch

from traveler.rag.retriever import DEFAULT_TOP_K, get_context


def test_get_context_fills_companion_from_query():
    """동승자 키워드가 있는 query → 해당 라벨이 companion에 채워지는지 확인."""
    ctx = get_context(user_query="여자친구랑 부산 데이트")
    assert ctx.companion == "연인"


def test_get_context_solo_query_fills_honja():
    """"혼자" 명시 query → companion이 "혼자"로 채워지는지 확인 (미언급과 구별)."""
    ctx = get_context(user_query="혼자 훌쩍 떠나고 싶어요")
    assert ctx.companion == "혼자"


def test_get_context_no_companion_fills_empty():
    """동승자 미언급 query → companion이 빈 문자열 ""로 채워지는지 확인."""
    ctx = get_context(user_query="맛집 추천해줘")
    assert ctx.companion == ""


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
