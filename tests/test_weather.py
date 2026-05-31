"""
기상청 날씨 API 래퍼 단위 테스트.

파이프라인 위치:
  사용자 입력 → ★ [weather.py → "맑음"] → ContextData → GPT-4o

WHY 테스트가 필요한가:
  실제 기상청 API를 호출하면 세 가지 문제가 생긴다:
  1. 인터넷이 없으면 테스트 실패 (CI 환경, 지하철 안 등)
  2. API 호출 제한(일 1,000회)을 개발 중에 소진할 수 있음
  3. 날씨는 매일 달라지므로 테스트 결과도 달라짐 → "재현 불가능한 테스트"

  Mock = 외부 의존성을 제거하고 "내 코드의 로직만" 테스트.

[테스트 대상 함수 분류]
  _latlon_to_grid()    — 순수 수학 함수 → Mock 불필요, 직접 입출력 검증
  _parse_weather_code()— 순수 파싱 함수 → "2359" 트릭으로 시간 의존성 우회
  get_weather()        — httpx 호출 함수 → AsyncMock으로 HTTP 요청 대체

[비동기 테스트 실행 방법]
  pytest.ini에 asyncio_mode = auto 설정됨
  → async def test_*() 함수는 자동으로 비동기 테스트로 실행됨
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from traveler.api.weather import (
    _FALLBACK,
    _latlon_to_grid,
    _parse_weather_code,
    get_weather,
)


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def make_items(pty: str, sky: str, time: str = "2359") -> list[dict]:
    """
    테스트용 기상청 API 응답 item 리스트 생성 헬퍼 (DRY).

    WHY time="2359"가 기본값인가:
      _parse_weather_code()는 datetime.now()로 현재 시각을 구해 target_time을 찾는다.
      "2359"는 하루 중 가장 늦은 시각이므로 어떤 시각에 테스트해도 항상 선택된다.
      → 시간을 Mock하지 않아도 테스트가 항상 동일하게 동작한다 (결정론적 테스트).
    """
    return [
        {"fcstTime": time, "category": "PTY", "fcstValue": pty},
        {"fcstTime": time, "category": "SKY", "fcstValue": sky},
    ]


def make_api_response(items: list[dict]) -> dict:
    """기상청 전체 API 응답 구조 생성 헬퍼."""
    return {"response": {"body": {"items": {"item": items}}}}


def make_mock_http_client(json_data: dict) -> AsyncMock:
    """
    httpx.AsyncClient() 호출을 대체하는 Mock 객체 생성 헬퍼.

    WHY 이 복잡한 Mock 구조가 필요한가:
      weather.py 코드:
        async with httpx.AsyncClient(...) as client:   ← (1) async context manager
            response = await client.get(...)           ← (2) async GET 호출
            response.json()                            ← (3) json 파싱

      각 단계를 순서대로 Mock해야 한다:
        (1) AsyncClient() → mock_cm (async with를 지원하는 Mock)
        (2) mock_cm.__aenter__ → mock_client (실제 client처럼 동작)
        (3) mock_client.get() → mock_response (.json() 호출 가능한 Mock)
    """
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()  # 에러 없이 통과
    mock_response.json.return_value = json_data

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    return mock_cm


# ── Test 1~2: 격자 좌표 변환 ─────────────────────────────────────────────────

def test_latlon_to_grid_gyeongju():
    """
    경주역 좌표 → 기상청 격자 (100, 91) 변환 검증.

    WHY 이 테스트가 필요한가:
      Lambert 투영 수식이 한 줄이라도 틀리면 전혀 다른 지역 날씨를 가져온다.
      경주역 격자 (100, 91)은 기상청 공식 변환 도구로 사전 검증된 값.
      이 테스트가 실패하면 "수식 자체가 깨진 것"을 바로 알 수 있다.
    """
    nx, ny = _latlon_to_grid(35.8562, 129.2247)
    assert nx == 100
    assert ny == 91


def test_latlon_to_grid_seoul():
    """
    서울 시청 좌표 → 격자 (60, 127) 검증.

    WHY 두 번째 도시를 테스트하나:
      "경주 좌표만 특수 처리한 버그"를 잡기 위해.
      두 도시가 서로 다른 격자를 반환해야 수식이 올바르게 동작하는 것.
    """
    nx, ny = _latlon_to_grid(37.5665, 126.9780)
    assert nx == 60
    assert ny == 127


# ── Test 3~7: 날씨 코드 파싱 ─────────────────────────────────────────────────

def test_parse_rain_overrides_sky():
    """
    PTY=1(비)이면 SKY=1(맑음)이어도 "비" 반환.

    WHY PTY가 SKY보다 우선인가:
      강수형태(PTY)는 하늘상태(SKY)보다 직접적인 날씨 정보다.
      비가 오는데 하늘이 맑아 보여도(일부 소나기 상황), "비"가 더 정확한 컨텍스트.
      이 우선순위가 바뀌면 비 오는 날 야외 관광지를 추천하는 버그로 이어진다.
    """
    result = _parse_weather_code(make_items(pty="1", sky="1"))
    assert result == "비"


def test_parse_snow():
    """PTY=3(눈) → "눈"."""
    result = _parse_weather_code(make_items(pty="3", sky="4"))
    assert result == "눈"


def test_parse_sunny():
    """
    PTY=0(강수없음), SKY=1(맑음) → "맑음".

    WHY: 강수 없을 때는 SKY(하늘상태)로 날씨를 판단한다.
    이 경로도 테스트해야 PTY 우선처리가 SKY 판단을 실수로 막지 않는지 확인 가능.
    """
    result = _parse_weather_code(make_items(pty="0", sky="1"))
    assert result == "맑음"


def test_parse_cloudy():
    """PTY=0(강수없음), SKY=3(구름많음) → "구름많음"."""
    result = _parse_weather_code(make_items(pty="0", sky="3"))
    assert result == "구름많음"


def test_parse_overcast():
    """PTY=0(강수없음), SKY=4(흐림) → "흐림"."""
    result = _parse_weather_code(make_items(pty="0", sky="4"))
    assert result == "흐림"


# ── Test 8~11: get_weather (비동기 + httpx Mock) ─────────────────────────────

async def test_get_weather_no_api_key(monkeypatch):
    """
    WEATHER_API_KEY 없으면 즉시 "정보없음" 반환 (Graceful Degradation).

    WHY: 날씨는 선택적 컨텍스트 — API 키 없어도 코스 생성은 가능해야 한다.
    OPENAI_API_KEY처럼 RuntimeError를 발생시키지 않는 설계가 올바른지 확인.

    비교:
      OPENAI_API_KEY 없음 → RuntimeError (코스 생성 자체 불가 → Fail Fast)
      WEATHER_API_KEY 없음 → "정보없음" (날씨 없이 코스 생성 계속 → Graceful)
    """
    monkeypatch.delenv("WEATHER_API_KEY", raising=False)
    result = await get_weather(35.8562, 129.2247)
    assert result == _FALLBACK


async def test_get_weather_success(monkeypatch):
    """
    정상 API 응답(맑음) → "맑음" 반환.

    WHY 통합 테스트가 필요한가:
      격자 변환, 발표 시간 계산, HTTP 호출, 파싱을 각각 테스트했어도
      연결 지점에서 타입 불일치 같은 버그가 생길 수 있다.
      전체 흐름을 한 번에 통과하는지 여기서 확인한다.
    """
    monkeypatch.setenv("WEATHER_API_KEY", "test_key")
    mock_cm = make_mock_http_client(
        make_api_response(make_items(pty="0", sky="1"))  # 맑음
    )

    with patch("traveler.api.weather.httpx.AsyncClient", return_value=mock_cm):
        result = await get_weather(35.8562, 129.2247)

    assert result == "맑음"


async def test_get_weather_timeout(monkeypatch):
    """
    기상청 API가 5초 내 응답 없으면 "정보없음" 반환.

    WHY: 기상청 API는 간혹 응답이 느리다.
    timeout 시 전체 파이프라인이 멈추면 안 된다 — 폴백이 작동하는지 확인.
    """
    monkeypatch.setenv("WEATHER_API_KEY", "test_key")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("traveler.api.weather.httpx.AsyncClient", return_value=mock_cm):
        result = await get_weather(35.8562, 129.2247)

    assert result == _FALLBACK


async def test_get_weather_http_error(monkeypatch):
    """
    기상청 API 5xx 서버 에러 → "정보없음" 반환.

    WHY: 기상청 서버 점검, API 키 만료 등 상황에서도 서비스가 중단되면 안 된다.
    raise_for_status()가 HTTPStatusError를 던질 때 except가 잡는지 확인.
    """
    monkeypatch.setenv("WEATHER_API_KEY", "test_key")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error", request=MagicMock(), response=MagicMock()
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("traveler.api.weather.httpx.AsyncClient", return_value=mock_cm):
        result = await get_weather(35.8562, 129.2247)

    assert result == _FALLBACK
