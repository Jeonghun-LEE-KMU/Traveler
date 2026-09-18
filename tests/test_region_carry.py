"""
후속 발화(지역명 없는 재요청)에 이전 지역을 이어받는 fallback_region 테스트 (2026-09-15).

WHY 이 테스트가 필요한가:
  "경주 코스 추천해줘" → 코스 생성 → "다른 코스 추천해줘"(지역명 없음)로 이어지면
  extract_region()이 None을 반환해 hybrid_search가 지역 필터 없이 전국 풀에서
  검색한다 — 경주 축제/쇼핑 필터, ContextData.region → 경주 프롬프트 블록이 전부
  날아간다. search_places()/get_context()에 fallback_region을 추가해, query 자체의
  지역이 없을 때만 이전 지역을 대신 쓰게 한다. query 자체에 지역이 있으면 항상 그게 이긴다.

파이프라인 위치:
  main.py chat_endpoint (이전 세션 지역 조회) → get_context(fallback_region=...)
    → search_places(fallback_region=...) → region = extract_region(query) or fallback_region
"""

from unittest.mock import MagicMock, patch

from traveler.rag.hybrid_search import HybridSearchResult
from traveler.rag.retriever import get_context, search_places


def make_result(content_id: str, content_type_id: str, name: str = "장소") -> HybridSearchResult:
    """테스트용 HybridSearchResult. GPS 반경(100km) 안에 들어오도록 경주 좌표 근처로 고정."""
    return HybridSearchResult(
        content_id=content_id,
        name=name,
        content_type_id=content_type_id,
        category="기타",
        address="경상북도 경주시",
        lat=35.8562,
        lng=129.2247,
        indoor=False,
        semantic_score=0.5,
        keyword_score=0.5,
        preference_score=0.5,
        iconic_score=0.5,
        final_score=0.5,
    )


# ── 1. query에 지역이 없으면 fallback_region을 쓴다 ───────────────────────────

def test_fallback_region_used_when_query_has_no_region():
    """지역명 없는 query + fallback_region="경주" → hybrid_search에 region="경주" 전달,
    경주 쇼핑 필터도 적용된다(쇼핑 키워드 없으므로 쇼핑몰 제외)."""
    raw_results = [make_result("1", "12", "불국사"), make_result("2", "38", "쇼핑몰")]
    mock_search = MagicMock(return_value=raw_results)
    with patch("traveler.rag.hybrid_search.search_places", mock_search):
        spots = search_places(
            user_query="근처 맛집 추천해줘",
            lat=35.8562, lon=129.2247, visited=set(), prefs={},
            fallback_region="경주",
        )
    assert mock_search.call_args.kwargs["region"] == "경주"
    assert {s.place_id for s in spots} == {"1"}  # 쇼핑몰(38)은 키워드 없어 제외


def test_get_context_uses_fallback_region_for_context_data_region():
    """get_context()도 fallback_region을 받아 ContextData.region에 반영한다."""
    raw_results = [make_result("1", "12", "불국사")]
    with patch("traveler.rag.hybrid_search.search_places", return_value=raw_results):
        ctx, _ = get_context(user_query="맛집 추천해줘", fallback_region="경주")
    assert ctx.region == "경주"


# ── 2. query 자체의 지역이 fallback_region보다 항상 우선한다 ─────────────────

def test_query_region_overrides_fallback_region():
    """query에 지역("안동")이 있으면 fallback_region("경주")을 무시하고 query 쪽이 이긴다."""
    raw_results = [make_result("1", "12", "장소")]
    mock_search = MagicMock(return_value=raw_results)
    with patch("traveler.rag.hybrid_search.search_places", mock_search):
        search_places(
            user_query="안동 코스 추천해줘",
            lat=35.8562, lon=129.2247, visited=set(), prefs={},
            fallback_region="경주",
        )
    assert mock_search.call_args.kwargs["region"] == "안동"


# ── 3. fallback_region이 None이면 기존 동작과 동일하다 ───────────────────────

def test_fallback_none_keeps_existing_behavior():
    """fallback_region 생략(None) → region 미해결, 경주 필터도 미적용(기존 동작 그대로)."""
    raw_results = [make_result("1", "12", "장소"), make_result("2", "38", "쇼핑몰")]
    mock_search = MagicMock(return_value=raw_results)
    with patch("traveler.rag.hybrid_search.search_places", mock_search):
        spots = search_places(
            user_query="맛집 추천해줘",
            lat=35.8562, lon=129.2247, visited=set(), prefs={},
        )
    assert mock_search.call_args.kwargs["region"] is None
    assert {s.place_id for s in spots} == {"1", "2"}  # 필터 미적용 — 쇼핑몰도 그대로
