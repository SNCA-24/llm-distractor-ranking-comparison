import pytest

from llm_distractor_ranking.metrics import (
    aggregate_metrics,
    calculate_metrics,
    clean_question,
    parse_ranking_string,
)


def test_parse_ranking_string_returns_triplet_for_standard_case():
    assert parse_ranking_string("2,3,0") == [2, 3, 0]


def test_parse_ranking_string_skips_leading_index_for_longer_legacy_format():
    assert parse_ranking_string("9,2,3,0") == [2, 3, 0]


@pytest.mark.parametrize("value", ["", None, "abc", "1,2"])
def test_parse_ranking_string_returns_empty_list_for_invalid_values(value):
    assert parse_ranking_string(value) == []


def test_clean_question_normalizes_case_and_whitespace():
    assert clean_question('  Which   DATA type is "Mutable"?  ') == 'which data type is "mutable"?'


def test_calculate_metrics_returns_exact_match_for_identical_rankings():
    metrics = calculate_metrics([2, 3, 0], [2, 3, 0])

    assert metrics["exact_match"] == 1
    assert metrics["top1_accuracy"] is True
    assert metrics["top2_accuracy"] is True
    assert metrics["pairwise_concordance"] == 1.0


def test_aggregate_metrics_returns_percentages():
    metrics = aggregate_metrics(
        [
            calculate_metrics([2, 3, 0], [2, 3, 0]),
            calculate_metrics([2, 3, 0], [3, 0, 2]),
        ],
        n_items=2,
    )

    assert metrics["exact_match_pct"] == 50.0
    assert metrics["top1_accuracy"] == 50.0
    assert "avg_spearman_rho" in metrics
