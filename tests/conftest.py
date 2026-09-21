from unittest.mock import AsyncMock

import pytest

import main


@pytest.fixture
def add_visited_mock() -> AsyncMock:
    """기본 테스트는 실제 visited 저장소(DB)에 쓰지 않는다."""
    return AsyncMock()


@pytest.fixture(autouse=True)
def _patch_main_add_visited(monkeypatch, add_visited_mock: AsyncMock) -> None:
    """main.add_visited의 새 DB 부작용을 테스트 기본값에서 차단한다.

    개별 테스트가 monkeypatch.setattr(main, "add_visited", ...)를 다시 호출하면
    그 테스트의 명시적 override가 그대로 우선한다.
    """
    monkeypatch.setattr(main, "add_visited", add_visited_mock)
