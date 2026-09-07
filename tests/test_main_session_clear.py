"""
main.py clear_session_data() 회귀 테스트 — Sprint 7, #31 PM 리뷰 3번 대응.

[문제] "새 여행 시작" 시 호출되는 clear_session_data()가 Redis(visited·prefs)와
memory_store(대화 이력)는 지우지만 _session_courses(방금 만든 코스를 담아두는
딕셔너리)는 건드리지 않았다. session_id가 새로 발급되지 않고 재사용되는 구조라,
초기화 후에도 /api/chat/revise가 이전 여행의 코스를 계속 수정 대상으로 삼는 버그.

Redis 연결 없이 검증하기 위해 main.clear_session(Redis 접근 함수)만 no-op으로
monkeypatch한다 — 이 테스트가 확인하려는 건 "_session_courses 딕셔너리 정리"이지
Redis 동작이 아니다.
"""

import pytest

import main


@pytest.mark.asyncio
async def test_clear_session_data_pops_session_courses(monkeypatch):
    async def fake_clear_session(session_id: str) -> None:
        return None

    monkeypatch.setattr(main, "clear_session", fake_clear_session)

    session_id = "test-session-clear-courses"
    main._session_courses[session_id] = ("dummy_course", "dummy_ctx", "근교")

    result = await main.clear_session_data(session_id)

    assert session_id not in main._session_courses
    assert "course" in result["cleared"]


@pytest.mark.asyncio
async def test_clear_session_data_no_course_is_noop(monkeypatch):
    """애초에 코스가 없던 세션을 초기화해도 KeyError 없이 정상 동작해야 한다."""

    async def fake_clear_session(session_id: str) -> None:
        return None

    monkeypatch.setattr(main, "clear_session", fake_clear_session)

    session_id = "test-session-without-course"
    main._session_courses.pop(session_id, None)  # 혹시 남아있으면 정리

    result = await main.clear_session_data(session_id)

    assert session_id not in main._session_courses
    assert result["ok"] is True
