"""
RAG Retriever - Sprint 4
hybrid_search.py(Pinecone + SQLite) + GPS 반경 필터 + RRF 순위 결합 + SpotCandidate 변환
"""

import logging
import math

from traveler.core.rrf import reciprocal_rank_fusion
from traveler.rag.types import ContextData, SpotCandidate

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_KM = 100.0
DEFAULT_TOP_K = 10


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _to_place_id(content_id: str) -> str:
    """
    content_id → place_id 변환.
    PR #17(place_id 문자열 전환) 이후: content_id 문자열을 그대로 사용.
    hash() 기반 임시 변환 제거 — 서버 재시작에도 안전하고 SQLite PK와 1:1 매핑.
    """
    return str(content_id) if content_id else ""


def _apply_diversity(spots: list[SpotCandidate], max_per_category: int = 3) -> list[SpotCandidate]:
    """카테고리 다양성 보장. 같은 카테고리가 max_per_category 초과하면 뒤로 밀고 나머지로 채움."""
    category_count: dict[str, int] = {}
    primary: list[SpotCandidate] = []
    overflow: list[SpotCandidate] = []
    for spot in spots:
        count = category_count.get(spot.category, 0)
        if count < max_per_category:
            primary.append(spot)
            category_count[spot.category] = count + 1
        else:
            overflow.append(spot)
    return primary + overflow


def search_places(
    user_query: str,
    lat: float,
    lon: float,
    visited: set,
    prefs: dict,
    top_k: int = DEFAULT_TOP_K,
    radius_km: float = DEFAULT_RADIUS_KM,
    dest_lat: float | None = None,
    dest_lon: float | None = None,
) -> list[SpotCandidate]:
    """
    장소 검색 메인 함수.

    Args:
        dest_lat, dest_lon: 목적지 좌표 (장거리 모드에서만 사용).
            설정 시 Tmap searchPoiAlong으로 경로 상 POI를 추가 탐색한다.
    """
    from traveler.rag.hybrid_search import search_places as hybrid_search_places

    try:
        raw_results = hybrid_search_places(
            user_query=user_query,
            preferences=prefs or None,
            visited_ids={str(v) for v in visited} if visited else None,
            top_k=top_k * 5,
        )
        logger.info("[Retriever] hybrid_search 결과: %d건", len(raw_results))

        # GPS 반경 필터 + (SpotCandidate, raw_result) 튜플 생성
        filtered: list[tuple[SpotCandidate, object]] = []
        for r in raw_results:
            if r.lat == 0.0 and r.lng == 0.0:
                continue
            dist_km = _haversine(lat, lon, r.lat, r.lng)
            if dist_km > radius_km:
                continue
            distance_min = round(dist_km, 1)  # 60km/h 기준 1km = 1분, dist_km == distance_min
            spot = SpotCandidate(
                place_id=_to_place_id(r.content_id),
                place_name=r.name,
                category=r.category,
                distance_min=distance_min,
                indoor=r.indoor,
                lat=r.lat,
                lng=r.lng,
            )
            filtered.append((spot, r))

        # 장거리 모드 Tmap 병합: /api/chat에 dest_lat/dest_lon 파라미터 추가 후 활성화 예정
        # if dest_lat is not None and dest_lon is not None:
        #     filtered = _merge_tmap_results(
        #         filtered, lat, lon, dest_lat, dest_lon, visited
        #     )

        if not filtered:
            logger.warning("[Retriever] GPS 필터 후 결과 없음, fallback 사용")
            return _fallback_spots()

        spots_by_id = {spot.place_id: spot for spot, _ in filtered}

        # RRF 세 신호 준비
        # semantic: Pinecone 의미 유사도 순위
        semantic_ranking = [
            _to_place_id(r.content_id)
            for _, r in sorted(filtered, key=lambda x: x[1].semantic_score, reverse=True)
        ]
        # iconic: 대표 명소 점수 순위
        iconic_ranking = [
            _to_place_id(r.content_id)
            for _, r in sorted(filtered, key=lambda x: x[1].iconic_score, reverse=True)
        ]
        # diversity: 카테고리 다양성 기반 순위
        diversity_ranking = [
            spot.place_id for spot in _apply_diversity([s for s, _ in filtered])
        ]

        rrf_result = reciprocal_rank_fusion({
            "semantic": semantic_ranking,
            "iconic": iconic_ranking,
            "diversity": diversity_ranking,
        })

        final_spots = [spots_by_id[pid] for pid in rrf_result if pid in spots_by_id][:top_k]
        logger.info("[Retriever] 최종 결과: %d건 (반경 %.0fkm)", len(final_spots), radius_km)
        return final_spots

    except Exception as e:
        logger.error("[Retriever] hybrid_search 실패, fallback 사용: %s", e)
        return _fallback_spots()


def _merge_tmap_results(
    filtered: list[tuple[SpotCandidate, object]],
    lat: float,
    lon: float,
    dest_lat: float,
    dest_lon: float,
    visited: set,
) -> list[tuple[SpotCandidate, object]]:
    """
    Tmap searchPoiAlong 결과를 hybrid_search 결과에 추가.
    이미 있는 place_id는 중복 추가하지 않는다.
    """
    import asyncio
    from traveler.api.tmap import search_poi_along_route

    try:
        loop = asyncio.get_event_loop()
        tmap_results = loop.run_until_complete(
            search_poi_along_route(lat, lon, dest_lat, dest_lon)
        )
    except Exception as e:
        logger.warning("[Retriever] Tmap 호출 실패: %s", e)
        return filtered

    existing_ids = {spot.place_id for spot, _ in filtered}
    visited_str = {str(v) for v in visited}

    for t in tmap_results:
        place_id = f"tmap_{t.lat}_{t.lon}"
        if place_id in existing_ids:
            continue
        if str(place_id) in visited_str:
            continue
        dist_km = _haversine(lat, lon, t.lat, t.lon)
        distance_min = round((dist_km / 60.0) * 60, 1)
        spot = SpotCandidate(
            place_id=place_id,
            place_name=t.name,
            category=t.category,
            distance_min=distance_min,
            indoor=False,
        )
        # Tmap 결과는 semantic_score/iconic_score가 없으므로
        # dummy 객체로 감싸서 필터 리스트에 추가
        filtered.append((spot, _TmapDummy(t.lat, t.lon)))
        existing_ids.add(place_id)

    logger.info("[Retriever] Tmap 병합 후 총 %d건", len(filtered))
    return filtered


class _TmapDummy:
    """Tmap POI용 dummy raw_result (RRF 신호 최솟값 부여)."""
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lng = lon
        self.semantic_score = 0.0
        self.iconic_score = 0.0


def _fallback_spots() -> list[SpotCandidate]:
    return [
        SpotCandidate(place_id="125266",  place_name="불국사",   category="관광", distance_min=5.0,  indoor=False, lat=35.7898, lng=129.3320),
        SpotCandidate(place_id="126508",  place_name="첨성대",   category="관광", distance_min=12.0, indoor=False, lat=35.8348, lng=129.2189),
        SpotCandidate(place_id="2733144", place_name="교리김밥", category="음식", distance_min=15.0, indoor=True, lat=35.8347, lng=129.2119),
    ]


def build_spots_context(spots: list[SpotCandidate]) -> str:
    if not spots:
        return ""
    lines = ["[실시간 검색된 주변 장소 후보 - 이 목록에서만 추천할 것]"]
    for i, s in enumerate(spots, 1):
        indoor_str = "실내" if s.indoor else "실외"
        lines.append(
            f"{i}. {s.place_name} | 카테고리: {s.category} | "
            f"이동 {s.distance_min:.0f}분 | {indoor_str} | place_id: {s.place_id}"
        )
    return "\n".join(lines)


def get_context(
    user_query: str = "경주에서 반나절 자차로 여행하고 싶어요",
    lat: float = 35.8562,
    lon: float = 129.2247,
    weather: str = "맑음",
    visited: set = None,
    prefs: dict = None,
    radius_km: float = DEFAULT_RADIUS_KM,
    top_k: int = DEFAULT_TOP_K,
    dest_lat: float | None = None,
    dest_lon: float | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple:
    if visited is None:
        visited = set()
    if prefs is None:
        prefs = {}

    spots = search_places(
        user_query=user_query,
        lat=lat,
        lon=lon,
        visited=visited,
        prefs=prefs,
        radius_km=radius_km,
        top_k=top_k,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        top_k=top_k,
    )

    context_data = ContextData(
        user_query=user_query,
        current_lat=lat,
        current_lng=lon,
        weather=weather,
        candidate_spots=spots,
        companion=prefs.get("companion", "혼자"),
        mood=prefs.get("mood", ""),
    )

    return context_data, build_spots_context(spots)
