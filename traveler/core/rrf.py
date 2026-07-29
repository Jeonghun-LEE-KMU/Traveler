"""세 개 검색 신호를 결합하는 순수 Reciprocal Rank Fusion(RRF).

이 모듈의 가중치는 검색 결과를 합치는 용도이며, 여행 모드의 장소 구성 비율인
``prompt.MODE_WEIGHTS``와는 별개의 계약이다.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar


ItemId = TypeVar("ItemId", bound=Hashable)

RRF_SIGNALS: tuple[str, ...] = ("semantic", "iconic", "diversity")
DEFAULT_RRF_WEIGHTS: dict[str, float] = {
    "semantic": 1.0,
    "iconic": 1.0,
    "diversity": 1.0,
}


def _validate_signal_names(values: Mapping[str, object], *, label: str) -> None:
    """입력 키가 지원 신호 세 개와 정확히 일치하는지 확인한다."""
    actual = set(values)
    expected = set(RRF_SIGNALS)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} signal 구성이 올바르지 않습니다 "
            f"(missing={missing}, unknown={unknown})"
        )


def _item_sort_key(item_id: Hashable) -> tuple[str, object]:
    """동점일 때 item_id 오름차순이 되도록 안정적인 정렬 키를 만든다."""
    if isinstance(item_id, (int, float)) and not isinstance(item_id, bool):
        return ("number", item_id)
    if isinstance(item_id, str):
        return ("string", item_id)
    return (type(item_id).__name__, repr(item_id))


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ItemId]],
    *,
    weights: Mapping[str, float] | None = None,
    k: float = 60,
) -> list[ItemId]:
    """semantic/iconic/diversity 순위를 정통 RRF 점수로 결합한다.

    각 항목의 최종 점수는 모든 신호에 대한
    ``sum(weight / (k + rank))``이며, ``rank``는 1부터 시작한다.
    점수가 같으면 item_id 오름차순으로 정렬해 결과를 항상 동일하게 만든다.

    Args:
        rankings: 정확히 ``semantic``, ``iconic``, ``diversity`` 키를 가진 순위 목록.
        weights: 신호별 가중치. 생략하면 모든 신호에 1.0을 사용한다.
        k: 상위 순위의 영향도를 완화하는 RRF 상수. 0보다 커야 한다.

    Returns:
        RRF 점수 내림차순으로 정렬된 item_id 목록. 모든 목록이 비면 빈 목록.

    Raises:
        ValueError: 신호 키, k, 가중치가 잘못됐거나 한 신호 안에 중복 ID가 있을 때.
    """
    if isinstance(k, bool) or not isinstance(k, (int, float)):
        raise ValueError("k는 0보다 큰 숫자여야 합니다.")
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k는 0보다 큰 유한한 숫자여야 합니다.")

    _validate_signal_names(rankings, label="rankings")

    resolved_weights = DEFAULT_RRF_WEIGHTS if weights is None else weights
    _validate_signal_names(resolved_weights, label="weights")
    for signal, weight in resolved_weights.items():
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(f"{signal} weight는 0 이상의 유한한 숫자여야 합니다.")

    scores: dict[ItemId, float] = {}
    for signal in RRF_SIGNALS:
        seen: set[ItemId] = set()
        for rank, item_id in enumerate(rankings[signal], start=1):
            try:
                if item_id in seen:
                    raise ValueError(
                        f"{signal} signal에 중복 item_id={item_id!r}가 있습니다."
                    )
                seen.add(item_id)
            except TypeError as exc:
                raise ValueError("item_id는 hash 가능한 값이어야 합니다.") from exc

            scores[item_id] = scores.get(item_id, 0.0) + (
                float(resolved_weights[signal]) / (float(k) + rank)
            )

    return sorted(
        scores,
        key=lambda item_id: (-scores[item_id], _item_sort_key(item_id)),
    )
