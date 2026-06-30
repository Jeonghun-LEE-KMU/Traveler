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

from traveler.rag.retriever import get_context


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
