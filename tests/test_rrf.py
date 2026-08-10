"""Sprint 4 RRF 재정렬 순수 함수 테스트."""

import pytest

from traveler.core.rrf import reciprocal_rank_fusion


def test_rrf_combines_semantic_iconic_and_diversity_rankings():
    rankings = {
        "semantic": [1, 2, 3],
        "iconic": [2, 1, 3],
        "diversity": [2, 3, 1],
    }

    assert reciprocal_rank_fusion(rankings) == [2, 1, 3]


def test_rrf_is_deterministic_when_scores_tie():
    rankings = {
        "semantic": [20, 10],
        "iconic": [10, 20],
        "diversity": [],
    }

    assert reciprocal_rank_fusion(rankings) == [10, 20]
    assert reciprocal_rank_fusion(rankings) == [10, 20]


def test_rrf_accepts_explicit_signal_weights():
    rankings = {
        "semantic": [1, 2],
        "iconic": [2, 1],
        "diversity": [2, 1],
    }
    weights = {
        "semantic": 3.0,
        "iconic": 1.0,
        "diversity": 1.0,
    }

    assert reciprocal_rank_fusion(rankings, weights=weights) == [1, 2]


def test_rrf_empty_signal_lists_degrade_to_empty_result():
    assert reciprocal_rank_fusion(
        {"semantic": [], "iconic": [], "diversity": []}
    ) == []


def test_rrf_rejects_non_positive_k():
    rankings = {"semantic": [1], "iconic": [], "diversity": []}

    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion(rankings, k=0)


def test_rrf_rejects_duplicate_item_within_one_signal():
    rankings = {"semantic": [1, 1], "iconic": [], "diversity": []}

    with pytest.raises(ValueError, match="중복"):
        reciprocal_rank_fusion(rankings)


@pytest.mark.parametrize(
    "rankings",
    [
        {"semantic": [1], "iconic": [1]},
        {
            "semantic": [1],
            "iconic": [1],
            "diversity": [1],
            "popularity": [1],
        },
    ],
)
def test_rrf_requires_exact_supported_signal_names(rankings):
    with pytest.raises(ValueError, match="signal"):
        reciprocal_rank_fusion(rankings)
