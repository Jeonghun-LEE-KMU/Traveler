"""
Tmap searchPoiAlong 래퍼 - Sprint 5

장거리 모드에서 출발지~목적지 경로 상 주변 POI를 검색한다.

[API 스펙]
POST https://apis.openapi.sk.com/tmap/pois/along-route
Header: appKey={TMAP_APP_KEY}
Body: startX, startY, endX, endY, passList, count, radius, categories

[사용 흐름]
retriever.py에서 mode="longdist"일 때 호출:
  results = await search_poi_along_route(start_lat, start_lon, end_lat, end_lon)

[주의]
- lat/lon 순서: Tmap은 X=경도(lon), Y=위도(lat) 순서
- 장애 시 빈 리스트 반환 (retriever.py fallback으로 처리)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

TMAP_APP_KEY = os.getenv("TMAP_APP_KEY", "")
TMAP_ALONG_ROUTE_URL = "https://apis.openapi.sk.com/tmap/pois/along-route"

# Tmap 카테고리 코드 -> 우리 카테고리 매핑
_CATEGORY_MAP: dict[str, str] = {
    "관광": "관광",
    "음식점": "음식",
    "카페": "음식",
    "숙박": "휴식",
    "레저": "체험",
    "문화": "관광",
    "쇼핑": "체험",
}


@dataclass
class TmapPoiResult:
    """Tmap POI 검색 결과 하나."""
    name: str
    lat: float
    lon: float
    category: str
    address: str


def _parse_category(upper_biz_name: str) -> str:
    """Tmap upperBizName -> 우리 카테고리 변환."""
    for keyword, category in _CATEGORY_MAP.items():
        if keyword in upper_biz_name:
            return category
    return "관광"


def _build_pass_list(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> str:
    """
    중간 경유 좌표 목록을 Tmap passList 형식으로 생성.
    출발-목적지 중간점을 경유지로 추가해 경로 상 탐색 범위를 확장한다.
    형식: "경도,위도_경도,위도"
    """
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2
    return f"{mid_lon},{mid_lat}"


async def search_poi_along_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    count: int = 20,
    radius: int = 10000,  # 경로 반경 10km
    categories: list[str] | None = None,
) -> list[TmapPoiResult]:
    """
    출발지~목적지 경로 상 주변 POI 검색.

    Args:
        start_lat: 출발지 위도
        start_lon: 출발지 경도
        end_lat: 목적지 위도
        end_lon: 목적지 경도
        count: 반환할 POI 수 (기본 20)
        radius: 경로 기준 탐색 반경 m (기본 10km)
        categories: 카테고리 필터 (None이면 전체)

    Returns:
        TmapPoiResult 리스트. 장애 시 빈 리스트.
    """
    if not TMAP_APP_KEY:
        logger.warning("[Tmap] TMAP_APP_KEY 미설정, 빈 결과 반환")
        return []

    payload = {
        "startX": str(start_lon),
        "startY": str(start_lat),
        "endX": str(end_lon),
        "endY": str(end_lat),
        "passList": _build_pass_list(start_lat, start_lon, end_lat, end_lon),
        "count": str(count),
        "radius": str(radius),
        "resCoordType": "WGS84GEO",
        "reqCoordType": "WGS84GEO",
    }
    if categories:
        payload["categories"] = ",".join(categories)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                TMAP_ALONG_ROUTE_URL,
                data=payload,
                headers={"appKey": TMAP_APP_KEY},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("[Tmap] searchPoiAlong 요청 실패: %s", e)
        return []

    results: list[TmapPoiResult] = []
    try:
        poi_list = (
            data.get("searchPoiInfo", {})
                .get("poiList", {})
                .get("poi", [])
        )
        for poi in poi_list:
            lat = float(poi.get("frontLat") or poi.get("noorLat") or 0)
            lon = float(poi.get("frontLon") or poi.get("noorLon") or 0)
            if lat == 0.0 and lon == 0.0:
                continue
            results.append(TmapPoiResult(
                name=poi.get("name", ""),
                lat=lat,
                lon=lon,
                category=_parse_category(poi.get("upperBizName", "")),
                address=poi.get("secondaryAddressName", ""),
            ))
    except Exception as e:
        logger.warning("[Tmap] 응답 파싱 실패: %s", e)

    logger.info("[Tmap] searchPoiAlong 결과: %d건", len(results))
    return results
