"""
traveler/core/prompt.py 단위 테스트.

검증 대상:
  - MODE_WEIGHTS: 모드 존재 여부, 가중치 합 1.0
  - get_system_prompt(): 모드별 올바른 프롬프트 반환, unknown 모드 fallback
"""

from traveler.core import prompt as prompt_module
from traveler.core.prompt import (
    MODE_WEIGHTS,
    SYSTEM_PROMPT_LONGDRIVE,
    SYSTEM_PROMPT_NEARBY,
    SYSTEM_PROMPT_TOURIST,
    get_system_prompt,
)

_TRAVEL_MODES = {"근교", "타지", "장거리"}
_WEIGHT_LABELS = {
    "hidden": "숨은 명소",
    "iconic": "대표 명소",
    "creative": "창의적 추천",
}


def _assert_prompt_uses_mode_weight(prompt: str, mode: str, key: str) -> None:
    pct = round(MODE_WEIGHTS[mode][key] * 100)
    label = _WEIGHT_LABELS[key]
    assert any(
        line.strip().startswith(f"- {label}") and line.strip().endswith(f": {pct}%")
        for line in prompt.splitlines()
    ), f"{mode} 프롬프트에 {key}={pct}% 라인이 없음"


def test_mode_weights_include_all_travel_modes():
    assert set(MODE_WEIGHTS.keys()) == _TRAVEL_MODES


def test_mode_weights_sum_to_one():
    for mode, weights in MODE_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"{mode} 가중치 합이 1.0이 아님: {total}"


def test_get_system_prompt_returns_nearby_prompt():
    assert get_system_prompt("근교") == SYSTEM_PROMPT_NEARBY


def test_get_system_prompt_returns_longdrive_prompt():
    prompt = get_system_prompt("장거리")
    assert prompt == SYSTEM_PROMPT_LONGDRIVE
    assert "장거리 경유지 안내" in prompt


def test_get_system_prompt_unknown_mode_falls_back_to_tourist():
    assert get_system_prompt("이상한값") == SYSTEM_PROMPT_TOURIST


def test_nearby_prompt_reflects_mode_weights():
    prompt = get_system_prompt("근교")
    for key in ("hidden", "iconic", "creative"):
        _assert_prompt_uses_mode_weight(prompt, "근교", key)


def test_tourist_prompt_reflects_mode_weights():
    prompt = get_system_prompt("타지")
    for key in ("iconic", "hidden", "creative"):
        _assert_prompt_uses_mode_weight(prompt, "타지", key)


def test_longdrive_prompt_reflects_its_own_mode_weights():
    prompt = get_system_prompt("장거리")
    for key in ("iconic", "hidden", "creative"):
        _assert_prompt_uses_mode_weight(prompt, "장거리", key)


def test_destination_persona_renderer_uses_requested_mode(monkeypatch):
    monkeypatch.setitem(
        MODE_WEIGHTS,
        "장거리",
        {"iconic": 0.50, "hidden": 0.30, "creative": 0.20},
    )

    prompt = prompt_module._build_destination_persona("장거리")

    for key in ("iconic", "hidden", "creative"):
        _assert_prompt_uses_mode_weight(prompt, "장거리", key)
