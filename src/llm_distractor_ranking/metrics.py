from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr


def clean_question(value: Any) -> str:
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, str) and value.startswith("(") and value.endswith("',)"):
        try:
            value = ast.literal_eval(value)[0]
        except Exception:
            pass
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = value.replace("—", "-").replace("–", "-")
    return value


def parse_ranking_string(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []

    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    integers: list[int] = []
    for token in tokens:
        try:
            integers.append(int(token))
        except ValueError:
            continue

    if len(integers) == 3:
        return integers
    if len(integers) >= 4:
        return integers[1:4]
    return []


def calculate_pairwise_concordance(ranking_one: list[int], ranking_two: list[int]) -> float:
    if len(ranking_one) != len(ranking_two) or not ranking_one:
        return 0.0

    concordant = 0
    total_pairs = 0
    for index in range(len(ranking_one)):
        for second_index in range(index + 1, len(ranking_one)):
            if (ranking_one[index] - ranking_one[second_index]) * (
                ranking_two[index] - ranking_two[second_index]
            ) > 0:
                concordant += 1
            total_pairs += 1
    return concordant / total_pairs if total_pairs else 0.0


def calculate_top_k_accuracy(ranking_one: list[int], ranking_two: list[int], k: int = 1) -> bool:
    if len(ranking_one) < k or len(ranking_two) < k:
        return False
    top_k_one = sorted(range(len(ranking_one)), key=lambda idx: ranking_one[idx])[:k]
    top_k_two = sorted(range(len(ranking_two)), key=lambda idx: ranking_two[idx])[:k]
    return set(top_k_one) == set(top_k_two)


def calculate_weighted_kappa(ranking_one: list[int], ranking_two: list[int]) -> float:
    try:
        labels = sorted(set(ranking_one + ranking_two))
        size = len(labels)
        if size <= 1:
            return 1.0

        ranking_one_categories = [labels.index(item) for item in ranking_one]
        ranking_two_categories = [labels.index(item) for item in ranking_two]

        observed = np.zeros((size, size), dtype=float)
        for left, right in zip(ranking_one_categories, ranking_two_categories, strict=False):
            observed[left, right] += 1
        observed /= observed.sum()

        row_marginals = observed.sum(axis=1)
        col_marginals = observed.sum(axis=0)
        expected = np.outer(row_marginals, col_marginals)

        weights = np.zeros((size, size), dtype=float)
        for row in range(size):
            for col in range(size):
                weights[row, col] = ((row - col) ** 2) / ((size - 1) ** 2)

        observed_score = float((weights * observed).sum())
        expected_score = float((weights * expected).sum())
        if expected_score == 0:
            return 1.0
        return 1.0 - (observed_score / expected_score)
    except Exception:
        return 0.0


def calculate_metrics(ranking_one: list[int], ranking_two: list[int]) -> dict[str, float | int | bool]:
    if not ranking_one or not ranking_two or len(ranking_one) != len(ranking_two):
        return {
            "exact_match": 0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "kendall_tau": 0.0,
            "kendall_p": 1.0,
            "pairwise_concordance": 0.0,
            "weighted_kappa": 0.0,
            "top1_accuracy": False,
            "top2_accuracy": False,
            "tie_cases": 0,
        }

    spearman = spearmanr(ranking_one, ranking_two)
    kendall = kendalltau(ranking_one, ranking_two)
    return {
        "exact_match": int(ranking_one == ranking_two),
        "spearman_rho": float(0.0 if np.isnan(spearman.statistic) else spearman.statistic),
        "spearman_p": float(1.0 if np.isnan(spearman.pvalue) else spearman.pvalue),
        "kendall_tau": float(0.0 if np.isnan(kendall.statistic) else kendall.statistic),
        "kendall_p": float(1.0 if np.isnan(kendall.pvalue) else kendall.pvalue),
        "pairwise_concordance": calculate_pairwise_concordance(ranking_one, ranking_two),
        "weighted_kappa": calculate_weighted_kappa(ranking_one, ranking_two),
        "top1_accuracy": calculate_top_k_accuracy(ranking_one, ranking_two, 1),
        "top2_accuracy": calculate_top_k_accuracy(ranking_one, ranking_two, 2),
        "tie_cases": int(len(set(ranking_one)) < len(ranking_one) or len(set(ranking_two)) < len(ranking_two)),
    }


def aggregate_metrics(metrics_list: list[dict[str, float | int | bool]], n_items: int) -> dict[str, float | int]:
    if not metrics_list or n_items <= 0:
        return {
            "exact_match_pct": 0.0,
            "avg_spearman_rho": 0.0,
            "avg_spearman_p": 1.0,
            "avg_kendall_tau": 0.0,
            "avg_kendall_p": 1.0,
            "avg_pairwise_concordance": 0.0,
            "avg_weighted_kappa": 0.0,
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "tie_cases": 0,
            "tie_case_pct": 0.0,
        }

    return {
        "exact_match_pct": (sum(int(metric["exact_match"]) for metric in metrics_list) / n_items) * 100,
        "avg_spearman_rho": float(np.nanmean([float(metric["spearman_rho"]) for metric in metrics_list])),
        "avg_spearman_p": float(np.nanmean([float(metric["spearman_p"]) for metric in metrics_list])),
        "avg_kendall_tau": float(np.nanmean([float(metric["kendall_tau"]) for metric in metrics_list])),
        "avg_kendall_p": float(np.nanmean([float(metric["kendall_p"]) for metric in metrics_list])),
        "avg_pairwise_concordance": float(np.nanmean([float(metric["pairwise_concordance"]) for metric in metrics_list])) * 100,
        "avg_weighted_kappa": float(np.nanmean([float(metric["weighted_kappa"]) for metric in metrics_list])),
        "top1_accuracy": (sum(bool(metric["top1_accuracy"]) for metric in metrics_list) / n_items) * 100,
        "top2_accuracy": (sum(bool(metric["top2_accuracy"]) for metric in metrics_list) / n_items) * 100,
        "tie_cases": sum(int(metric["tie_cases"]) for metric in metrics_list),
        "tie_case_pct": (sum(int(metric["tie_cases"]) for metric in metrics_list) / n_items) * 100,
    }
