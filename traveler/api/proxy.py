"""
TourAPI 장소 상세 · 네이버 Directions 프록시 — 배포 환경용.

파이프라인 위치:
  프론트(마커·경로 렌더링) → ★ [이 라우터] → TourAPI / 네이버 Directions

WHY 이 파일이 필요한가:
  두 경로는 지금까지 traveler-front 의 `vite.config.ts` 안에서 개발 서버
  미들웨어로만 존재했다. `npm run build` 산출물에는 포함되지 않으므로,
  배포하면 프론트의 enrichRecommendedPlaces() 와 getRouteNaver() 가 404 를 받고
  장소명·마커·경로선이 통째로 사라진다. (traveler-front #3)

  프론트에서 직접 호출하지 않는 이유:
    - TourAPI serviceKey 와 네이버 시크릿이 번들에 노출된다
    - 두 API 모두 브라우저 CORS 를 허용하지 않는다

[Graceful Degradation 설계]
  weather.py 와 같은 원칙. 외부 API 가 실패해도 500 을 내지 않고
  프론트가 자체 폴백을 쓸 수 있도록 4xx/5xx 를 그대로 전달하거나
  빈 결과를 돌려준다. 코스 생성 자체는 이 라우터와 무관하게 계속된다.

[적용]
  main.py 에서:
      from traveler.api.proxy import router as proxy_router
      app.include_router(proxy_router)
  config.py 에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 추가.
"""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from config import KORWITHSERVICE_API_KEY, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

logger = logging.getLogger(__name__)

router = APIRouter(tags=["proxy"])

_TOUR_DETAIL_URL = "https://apis.data.go.kr/B551011/KorService2/detailCommon2"
_NAVER_DIRECTIONS_URL = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
_TIMEOUT = 10.0

# contentId 는 불변 정보라 프로세스 수명 동안 캐싱해도 안전하다.
# vite 미들웨어도 같은 방식이었다. 규모가 커지면 Redis 로 옮긴다.
_tour_cache: dict[str, dict] = {}


def _to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # NaN 방어


@router.get("/tour-place")
async def get_tour_place(contentId: str = Query(..., pattern=r"^\d+$")):
    """
    TourAPI detailCommon2 로 장소 한 건의 이름·좌표·설명을 조회한다.

    프론트 계약(traveler-front src/types/api.ts RecommendedPlacePayload):
        {place_id: str, place_name, category, lat, lng, description}

    place_id 는 **문자열**이다. 백엔드 schema.py 의 `place_id: str` 과 맞춰야
    preference.build_visited_context 의 isdigit() 경로에서 터지지 않는다.
    """
    if not KORWITHSERVICE_API_KEY:
        raise HTTPException(status_code=503, detail="TourAPI 키가 설정되지 않았습니다.")

    cached = _tour_cache.get(contentId)
    if cached is not None:
        return cached

    params = {
        "serviceKey": KORWITHSERVICE_API_KEY,
        "MobileOS": "ETC",
        "MobileApp": "Traveler",
        "_type": "json",
        "contentId": contentId,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(_TOUR_DETAIL_URL, params=params)
    except httpx.HTTPError as error:
        logger.warning("[tour-place] 요청 실패 contentId=%s: %s", contentId, error)
        raise HTTPException(status_code=502, detail="TourAPI 장소 조회에 실패했습니다.")

    try:
        data = response.json()
    except ValueError:
        # 키가 잘못되면 공공데이터포털이 XML 에러를 돌려준다.
        logger.warning("[tour-place] JSON 이 아님 contentId=%s: %s", contentId, response.text[:200])
        raise HTTPException(status_code=502, detail="TourAPI 응답을 해석하지 못했습니다.")

    header = (data.get("response") or {}).get("header") or {}
    body = (data.get("response") or {}).get("body") or {}
    items = body.get("items")
    raw_item = items.get("item") if isinstance(items, dict) else None
    item = raw_item[0] if isinstance(raw_item, list) and raw_item else raw_item

    if not response.is_success or header.get("resultCode") != "0000" or not isinstance(item, dict):
        detail = header.get("resultMsg") or "장소 정보를 찾지 못했습니다."
        raise HTTPException(status_code=404 if response.is_success else response.status_code, detail=detail)

    payload = {
        "place_id": str(item.get("contentid") or contentId),
        "place_name": item.get("title") or None,
        "category": item.get("lclsSystm1") or None,
        "lat": _to_float(item.get("mapy")),
        "lng": _to_float(item.get("mapx")),
        "description": item.get("overview") or None,
    }
    _tour_cache[contentId] = payload
    return payload


@router.get("/naver-directions/driving")
async def get_naver_directions(request: Request):
    """
    네이버 클라우드 Directions 5 를 중계한다.

    프론트는 start / goal / option / waypoints 를 그대로 붙여 호출하고,
    응답의 route.<option>[0].path 를 읽어 Polyline 을 그린다.
    실패하면 프론트가 OSRM 폴백으로 넘어가므로 여기서 예외를 삼키지 않고
    상태 코드를 그대로 전달한다.
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise HTTPException(status_code=503, detail="네이버 Directions 키가 설정되지 않았습니다.")

    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(
                _NAVER_DIRECTIONS_URL,
                params=dict(request.query_params),
                headers=headers,
            )
    except httpx.HTTPError as error:
        logger.warning("[naver-directions] 요청 실패: %s", error)
        raise HTTPException(status_code=502, detail="경로 조회에 실패했습니다.")

    if not response.is_success:
        raise HTTPException(status_code=response.status_code, detail="경로 조회에 실패했습니다.")

    try:
        return response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="경로 응답을 해석하지 못했습니다.")
