"""PR #47 리베이스 회귀: develop 계약과 분 단위 시간 계약이 함께 유지되는지 검증한다."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import main
from tests.test_main_chat_persist import _patch_llm_layer
from tests.test_schedule_integration import _context, _raw
from traveler.rag.preference import (
    extract_duration,
    extract_duration_hours,
    extract_duration_minutes,
    extract_region,
)
from traveler.rag.retriever import get_context


@pytest.mark.parametrize(
    "text, expected_minutes",
    [
        ("당일치기 경주 코스 추천해줘", 480),
        ("당일치기지만 1.5시간만 가능해", 90),
        ("당일치기지만 30분만 가능해", 30),
        ("오전 경주 코스", 180),
        ("오후 경주 코스", 180),
        ("저녁 경주 코스", 120),
        ("반나절 경주 여행", 300),
        ("하루 경주 코스", 480),
    ],
)
def test_duration_minutes_preserves_develop_keywords_and_explicit_overrides(text, expected_minutes):
    assert extract_duration_minutes(text) == expected_minutes


@pytest.mark.parametrize(
    "text, expected_hours",
    [
        ("반나절 경주 여행", 5),
        ("하루 경주 코스", 8),
        ("당일치기지만 1.5시간만 가능해", 1.5),
    ],
)
def test_duration_hours_preserves_legacy_hour_contract(text, expected_hours):
    assert extract_duration_hours(text) == expected_hours


def test_extract_duration_compatibility_uses_minute_parser_errors():
    assert extract_duration("90분만 가능해") == 1.5
    with pytest.raises(ValueError):
        extract_duration("0분만 가능해")


@pytest.mark.parametrize(
    "query, expected_region",
    [
        ("가평 드라이브 코스", "가평"),
        ("양평 당일치기", "양평"),
        ("남양주 카페 추천", "남양주"),
        ("파주 가족 여행", "파주"),
    ],
)
def test_develop_region_keywords_are_retained(query, expected_region):
    assert extract_region(query) == expected_region


def test_get_context_duration_hours_argument_is_compatible_and_does_not_mutate_prefs():
    prefs = {"companion": "친구"}

    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주 추천", prefs=prefs, duration_hours=8)

    assert ctx.duration_hours == 8
    assert ctx.duration_minutes is None
    assert prefs == {"companion": "친구"}


def test_get_context_duration_hours_argument_does_not_override_prefs_minutes_or_hours():
    prefs = {"duration_minutes": 90, "duration_hours": 1.5}

    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주 추천", prefs=prefs, duration_hours=8)

    assert ctx.duration_minutes == 90
    assert ctx.duration_hours == 1.5
    assert prefs == {"duration_minutes": 90, "duration_hours": 1.5}


def test_get_context_duration_hours_argument_does_not_override_prefs_hours_only():
    prefs = {"duration_hours": 4}

    with patch("traveler.rag.retriever.search_places", return_value=[]):
        ctx, _ = get_context(user_query="경주 추천", prefs=prefs, duration_hours=8)

    assert ctx.duration_minutes is None
    assert ctx.duration_hours == 4
    assert prefs == {"duration_hours": 4}


@pytest.mark.asyncio
async def test_stream_adds_only_final_time_budget_filtered_ids_to_visited(add_visited_mock, monkeypatch):
    _patch_llm_layer(monkeypatch, [_raw(count=4, stay=30)])
    sid = "rebase-final-visited-only"
    queue: asyncio.Queue = asyncio.Queue()

    try:
        await main._stream_and_persist(queue, sid, "타지", "문맥", _context(hours=4, count=4))
        chunks = [queue.get_nowait() for _ in range(queue.qsize())]
        assert chunks[-1] is None
        final = json.loads(chunks[-2].removeprefix("data: "))

        final_ids = [place["place_id"] for place in final["recommended_places"]]
        assert final_ids == ["1", "2", "3"]
        assert final["time_budget"]["status"] == "within_budget"
        assert final["time_budget"]["total_min"] <= final["time_budget"]["budget_min"] == 240
        add_visited_mock.assert_awaited_once_with(sid, final_ids)
    finally:
        main._session_courses.pop(sid, None)


@pytest.mark.asyncio
async def test_stream_no_feasible_does_not_add_visited_and_clears_saved_course(add_visited_mock, monkeypatch):
    _patch_llm_layer(monkeypatch, [_raw(count=1, stay=120)])
    sid = "rebase-no-feasible-clears"
    queue: asyncio.Queue = asyncio.Queue()
    main._session_courses[sid] = (
        main.StreamingCourseModel.model_validate_json(_raw(count=1)),
        _context(hours=4, count=1),
        "타지",
    )

    await main._stream_and_persist(queue, sid, "타지", "문맥", _context(hours=1, count=1))
    chunks = [queue.get_nowait() for _ in range(queue.qsize())]
    assert chunks[-1] is None
    final = json.loads(chunks[-2].removeprefix("data: "))

    assert final["recommended_places"] == []
    assert final["time_budget"]["status"] == "no_feasible"
    add_visited_mock.assert_not_awaited()
    assert sid not in main._session_courses


def _main_has_dev_endpoint(enabled: bool) -> bool:
    repo_root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "OPENAI_API_KEY": "",
        "PINECONE_API_KEY": "",
        "DATABASE_URL": "",
        "REDIS_URL": "",
        "KORSERVICE_API_KEY": "",
        "KMA_API_KEY": "",
        "KORWITHSERVICE_API_KEY": "",
        "TARRITETARSERVICE_API_KEY": "",
        "MDCLTURSMSERVICE_API_KEY": "",
    }
    env["ENABLE_DEV_ENDPOINTS"] = "true" if enabled else ""

    # app.routes의 내부 구조가 아닌 실제 HTTP 접근과 검색 호출 여부를 검사한다.
    # FastAPI의 지연 등록 라우터는 app.routes에 path=None으로 보일 수 있다.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json\n"
                "from unittest.mock import patch\n"
                "from fastapi.testclient import TestClient\n"
                "import main\n"
                "with patch('traveler.api.dev_search_cache.search_places', return_value=[]) as search:\n"
                "    with TestClient(main.app) as client:\n"
                "        response = client.get('/api/dev/search-cache', params={'user_message': '경주'})\n"
                "    print(json.dumps({'status': response.status_code, 'body': response.json(), "
                "'search_calls': search.call_count}))\n"
            ),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == (200 if enabled else 404)
    assert payload["search_calls"] == (1 if enabled else 0)
    if enabled:
        assert payload["body"] == {"count": 0, "places": []}
    return payload["status"] == 200


def test_dev_search_cache_endpoint_is_gated_by_env_in_isolated_import():
    assert _main_has_dev_endpoint(enabled=False) is False
    assert _main_has_dev_endpoint(enabled=True) is True
