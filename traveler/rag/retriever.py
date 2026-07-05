"""
RAG Retriever - Sprint 3
Pinecone 벡터 검색 + GPS 반경 필터 + SpotCandidate 변환
"""

import logging
import math
import os
from typing import Optional

from openai import OpenAI
from pinecone import Pinecone

from traveler.rag.types import ContextData, SpotCandidate

logger = logging.getLogger(__name__)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "traveler-places"
EMBED_MODEL      = "text-embedding-3-small"
DEFAULT_RADIUS_KM = 100.0
DEFAULT_TOP_K     = 10

_oai_client: Optional[OpenAI] = None
_pinecone_index = None


def _get_oai() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _oai_client


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _embed_query(text: str) -> list:
    resp = _get_oai().embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def _is_indoor(content_type_id: str) -> bool:
    # 14=문화시설, 38=쇼핑, 39=음식점 -> 실내
    return content_type_id in ("14", "38", "39")


def _pinecone_filter(prefs: dict) -> Optional[dict]:
    mood = prefs.get("mood", "")
    food_kws    = ["맛집", "카페", "브런치", "디저트", "음식"]
    history_kws = ["역사", "전통", "문화", "예술"]

    if any(kw in mood for kw in food_kws):
        return {"content_type_id": {"$eq": "39"}}
    elif "쇼핑" in mood:
        return {"content_type_id": {"$eq": "38"}}
    elif any(kw in mood for kw in history_kws):
        return {"content_type_id": {"$in": ["12", "14"]}}
    return None


def search_places(
    user_query: str,
    lat: float,
    lon: float,
    visited: set,
    prefs: dict,
    top_k: int = DEFAULT_TOP_K,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> list:
    try:
        query_vec = _embed_query(user_query)
        pc_filter = _pinecone_filter(prefs)
        index     = _get_index()

        results = index.query(
            vector=query_vec,
            top_k=top_k * 5,
            include_metadata=True,
            filter=pc_filter,
        )
        logger.info("[Retriever] Pinecone 결과: %d건", len(results.matches))

        spots = []
        for match in results.matches:
            content_id = match.id
            if content_id in visited:
                continue

            meta = match.metadata
            mlat = float(meta.get("lat") or 0.0)
            mlng = float(meta.get("lng") or 0.0)

            if mlat == 0.0 and mlng == 0.0:
                continue

            dist_km = _haversine(lat, lon, mlat, mlng)
            if dist_km > radius_km:
                continue

            distance_min = round((dist_km / 60.0) * 60, 1)

            try:
                place_id = int(content_id)
            except (ValueError, TypeError):
                place_id = abs(hash(content_id)) % 1_000_000

            ctype = str(meta.get("content_type_id") or "")
            spots.append(SpotCandidate(
                place_id=place_id,
                place_name=str(meta.get("name") or ""),
                category=str(meta.get("category") or ctype),
                distance_min=distance_min,
                indoor=_is_indoor(ctype),
            ))

            if len(spots) >= top_k:
                break

        logger.info("[Retriever] 최종 결과: %d건 (반경 %.0fkm)", len(spots), radius_km)
        return spots

    except Exception as e:
        logger.error("[Retriever] 검색 실패, fallback 사용: %s", e)
        return _fallback_spots()


def _fallback_spots() -> list:
    return [
        SpotCandidate(place_id=1, place_name="불국사",   category="관광", distance_min=5.0,  indoor=False),
        SpotCandidate(place_id=2, place_name="첨성대",   category="관광", distance_min=12.0, indoor=False),
        SpotCandidate(place_id=3, place_name="교리김밥", category="음식", distance_min=15.0, indoor=True),
    ]


def build_spots_context(spots: list) -> str:
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
