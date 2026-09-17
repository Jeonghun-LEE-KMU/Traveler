from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Query

from traveler.rag.retriever import search_places

router = APIRouter(prefix="/api/dev", tags=["dev"])


@router.get("/search-cache")
async def search_cache_endpoint(
    user_message: str,
    lat: float = Query(35.8562, description="검색 기준 위도"),
    lon: float = Query(129.2247, description="검색 기준 경도"),
    mode: Literal["nearby", "longdist", "remote"] = Query(
        "nearby",
        description="여행 모드",
    ),
):
    """
    검색 캐시/검색 레이어 부하테스트용 개발 엔드포인트.

    /api/chat 전체 경로와 달리 LLM 응답 생성을 호출하지 않는다.
    Locust 테스트에서 Redis/memory cache hit 성능만 분리 측정하기 위해 사용한다.
    """
    radius_by_mode = {
        "nearby": 30,
        "longdist": 120,
        "remote": 300,
    }

    prefs = {
        "companion": "",
        "mood": "",
        "travel_mode": "",
    }

    radius_km = radius_by_mode[mode]

    results = await asyncio.to_thread(
        search_places,
        user_query=user_message,
        lat=lat,
        lon=lon,
        visited=set(),
        prefs=prefs,
        top_k=10,
        radius_km=radius_km,
    )

    return {
        "count": len(results),
        "places": [
            {
                "place_id": result.place_id,
                "name": result.place_name,
            }
            for result in results
        ],
    }