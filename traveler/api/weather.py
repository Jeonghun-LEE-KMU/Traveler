"""
기상청 단기예보 API 래퍼 — 현재 위치의 날씨를 한글 문자열로 반환.

파이프라인 위치:
  사용자 입력 → [RAG 장소 목록] → ★ [weather.py → "맑음"] → ContextData → GPT-4o

WHY 이 파일이 필요한가:
  GPT가 코스를 짤 때 날씨를 모르면 비 오는 날에도 야외 명소를 추천할 수 있다.
  실제 날씨를 ContextData.weather에 주입해야 GPT가 날씨 반영 코스를 생성한다.
  예: weather="비" → GPT가 실내 카페, 박물관 위주로 코스를 짬

[기상청 단기예보 API 기본 지식]
  - 엔드포인트: https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst
  - 발표 주기: 하루 8번 (0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300)
  - 좌표계: 위경도(WGS84) → 격자 좌표(nx, ny) 변환 필요 (Lambert Conformal Conic)
  - 주요 카테고리:
      PTY: 강수형태 (0=없음, 1=비, 2=비/눈, 3=눈, 4=소나기)
      SKY: 하늘상태 (1=맑음, 3=구름많음, 4=흐림)

[Graceful Degradation 설계]
  기상청 API 실패(키 없음, 네트워크 오류, 파싱 실패) → "정보없음" 반환
  WHY: 날씨는 선택적 컨텍스트. API 실패해도 코스 생성은 계속되어야 함.
  OpenAI API 키와 다르게 Fail Fast 하지 않는 이유가 이것.

[Sprint 1 — 2026-06-01]
  기본 날씨 조회 구현. 캐싱은 Sprint 2에서 Redis로 추가 예정.
"""

import math
import os
from datetime import datetime, timedelta

import httpx


# ── 상수 ─────────────────────────────────────────────────────────────────────

# 기상청 발표 시간 8종 (HH:MM 형식)
# WHY 10분을 더하나: API 데이터는 발표 후 약 10분 뒤에 실제로 제공됨
# 예: 08:00 발표 → 08:10부터 조회 가능. 08:05에 요청하면 아직 없으므로 이전 회(05:00) 사용
_BASE_TIMES = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]
_API_DELAY_MINUTES = 10  # 발표 후 실제 제공까지 걸리는 시간

# 날씨 코드 → 한글 매핑
# WHY PTY를 SKY보다 먼저 확인하나:
#   PTY가 0이 아니면(비/눈) 하늘 상태보다 강수 여부가 더 중요한 정보이기 때문.
#   "구름많음+비" 보다 "비"가 GPT에게 더 명확한 컨텍스트.
_PTY_MAP = {0: None, 1: "비", 2: "비/눈", 3: "눈", 4: "소나기"}
_SKY_MAP = {1: "맑음", 3: "구름많음", 4: "흐림"}

_FALLBACK = "정보없음"  # API 실패 시 반환값 (Graceful Degradation)


# ── 내부 헬퍼 함수들 ─────────────────────────────────────────────────────────

def _latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """
    위경도(WGS84) → 기상청 격자 좌표(nx, ny) 변환.

    WHY 이 변환이 필요한가:
      기상청은 한반도를 5km 격자로 나눠 예보를 계산한다.
      위경도를 그대로 보내면 API가 이해하지 못함.
      기상청 공식 Lambert Conformal Conic 투영 수식을 Python으로 구현한 것.

    참고: 기상청 기술문서 "격자 변환 프로그램" (공식 배포)
    """
    # 기상청 표준 상수 — 임의로 변경하면 안 됨
    RE = 6371.00877   # 지구 반경 (km)
    GRID = 5.0        # 격자 간격 (km)
    SLAT1 = 30.0      # 표준 위도 1 (Lambert 투영 기준)
    SLAT2 = 60.0      # 표준 위도 2 (Lambert 투영 기준)
    OLON = 126.0      # 기준 경도 (한반도 중심)
    OLAT = 38.0       # 기준 위도 (한반도 중심)
    XO = 43.0         # 기준점 격자 X
    YO = 136.0        # 기준점 격자 Y

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon

    # 경도 차이가 ±180도를 넘지 않도록 보정
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    nx = int(ra * math.sin(theta) + XO + 0.5)
    ny = int(ro - ra * math.cos(theta) + YO + 0.5)
    return nx, ny


def _get_base_datetime() -> tuple[str, str]:
    """
    현재 시각 기준 가장 최근 기상청 발표 시간(base_date, base_time) 반환.

    WHY 이 계산이 필요한가:
      기상청 API는 발표 시간 기준으로 데이터를 제공한다.
      현재 시각이 09:15이면 0800 발표 데이터를 요청해야 한다.
      발표 후 10분 내에는 아직 데이터가 없으므로 이전 회 데이터를 사용.

    반환: ("20260601", "0800") 형태의 (날짜, 시간) 튜플
    """
    now = datetime.now()

    # 발표 시간 리스트를 역순으로 순회해 현재 시각보다 이전인 것을 찾음
    for base_time in reversed(_BASE_TIMES):
        hour = int(base_time[:2])
        minute = int(base_time[2:])

        # ★ 핵심: API 데이터 제공까지 10분 딜레이를 고려
        available_at = now.replace(hour=hour, minute=minute, second=0) + timedelta(
            minutes=_API_DELAY_MINUTES
        )

        if now >= available_at:
            return now.strftime("%Y%m%d"), base_time

    # 자정 이후 0210 이전이면 → 어제 2300 데이터 사용
    yesterday = now - timedelta(days=1)
    return yesterday.strftime("%Y%m%d"), "2300"


def _parse_weather_code(items: list[dict]) -> str:
    """
    기상청 API 응답 items에서 PTY/SKY 코드를 추출해 한글 문자열로 변환.

    WHY PTY → SKY 순서로 확인하나:
      PTY(강수형태)가 0이 아니면(=비/눈이 내리면) 그게 더 중요한 날씨 정보.
      강수 없을 때만 SKY(하늘상태)로 판단.

    INPUT: 기상청 API response.json()["response"]["body"]["items"]["item"] 리스트
    OUTPUT: "맑음", "비", "눈" 등 → ContextData.weather 필드에 바로 사용 가능
    """
    # 현재 시각과 가장 가까운 예보 시간(fcstTime) 추출
    # WHY: items에는 여러 시간대 예보가 섞여 있음, 지금 시각에 맞는 것만 필요
    now_hhmm = datetime.now().strftime("%H%M")
    # 현재 시각 이후 첫 번째 예보 시간 찾기 (없으면 마지막 시간대)
    forecast_times = sorted({item["fcstTime"] for item in items})
    target_time = next((t for t in forecast_times if t >= now_hhmm), forecast_times[-1])

    pty_val: int | None = None
    sky_val: int | None = None

    for item in items:
        if item["fcstTime"] != target_time:
            continue
        if item["category"] == "PTY":
            pty_val = int(item["fcstValue"])
        elif item["category"] == "SKY":
            sky_val = int(item["fcstValue"])

    # PTY 우선 확인
    if pty_val is not None and pty_val != 0:
        return _PTY_MAP.get(pty_val, _FALLBACK)

    # PTY=0 (강수 없음)이면 SKY로 판단
    if sky_val is not None:
        return _SKY_MAP.get(sky_val, _FALLBACK)

    return _FALLBACK


# ── 공개 함수 ─────────────────────────────────────────────────────────────────

async def get_weather(lat: float, lng: float) -> str:
    """
    현재 위치(위도, 경도)의 날씨를 기상청 단기예보 API로 조회해 반환.

    파이프라인 위치:
      ContextData 조립 시 호출 → weather 필드에 주입 → GPT-4o 프롬프트로 전달

    INPUT: 위도(lat), 경도(lng) — ContextData.current_lat, current_lng 와 동일 타입
    OUTPUT: "맑음", "비", "눈", "흐림", "소나기", "구름많음", "정보없음" 중 하나
    # COST: 기상청 API 무료 (data.go.kr 공공 API, 일 1,000회 제한)

    대안: OpenWeatherMap API → 유료, 영어 응답이라 번역 필요. 기상청이 무료·국내 정확도↑.
    """
    api_key = os.getenv("WEATHER_API_KEY")

    # ⚠️ 주의: OpenAI 키와 달리 RuntimeError 발생시키지 않음 (Graceful Degradation)
    #   날씨는 선택적 컨텍스트 — 없어도 GPT는 코스를 생성할 수 있음
    if not api_key:
        return _FALLBACK

    nx, ny = _latlon_to_grid(lat, lng)
    base_date, base_time = _get_base_datetime()

    url = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": api_key,
        "pageNo": 1,
        "numOfRows": 100,   # 여러 시간대 예보를 한 번에 받기 위해 충분히 크게 설정
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    try:
        # ★ 핵심: AsyncClient 사용 — FastAPI async 환경에서 동기 requests 사용 금지
        #   대안: requests.get() → 응답 대기 중 서버 전체가 블로킹됨
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()  # 4xx, 5xx → httpx.HTTPStatusError 발생

        data = response.json()

        # API 응답 구조: response → body → items → item (리스트)
        items = data["response"]["body"]["items"]["item"]
        return _parse_weather_code(items)

    except httpx.TimeoutException:
        # ⚠️ 기상청 API는 간혹 응답이 느림 — timeout=5s 초과 시 폴백
        return _FALLBACK
    except (httpx.HTTPStatusError, KeyError, ValueError):
        # KeyError: 응답 구조가 예상과 다를 때 (API 스펙 변경 등)
        # ValueError: int() 변환 실패 (비정상 fcstValue)
        return _FALLBACK
