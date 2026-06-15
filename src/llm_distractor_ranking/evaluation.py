from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import aggregate_metrics, calculate_metrics, clean_question, parse_ranking_string

PREDICTION_COLUMNS = ["question", "model_name", "variant", "predicted_choice", "correct_choice"]

DOMAIN_MAP = {
    "college_biology": "Medical science",
    "high_school_biology": "Medical science",
    "college_chemistry": "Chemistry",
    "high_school_chemistry": "Chemistry",
    "college_physics": "Physics",
    "high_school_physics": "Physics",
    "college_computer_science": "Computer Science",
    "high_school_computer_science": "Computer Science",
    "computer_security": "Computer Science",
    "machine_learning": "Computer Science",
    "college_mathematics": "Mathematics",
    "high_school_mathematics": "Mathematics",
    "elementary_mathematics": "Mathematics",
    "high_school_statistics": "Mathematics",
    "formal_logic": "Mathematics",
    "high_school_microeconomics": "Social Sciences",
    "world_history": "General Knowledge",
    "global_facts": "General Knowledge",
}


def load_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in PREDICTION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")
    return df


def get_triplet(frequencies: pd.Series) -> list[int]:
    sorted_votes = frequencies.sort_index().sort_values(ascending=False)
    return [int(value) for value in sorted_votes.index.tolist()[:3]]


def create_prediction_subsets(
    combined_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correct_df = combined_df[combined_df["predicted_choice"] == combined_df["correct_choice"]].copy()
    incorrect_df = combined_df[combined_df["predicted_choice"] != combined_df["correct_choice"]].copy()

    tie_questions: list[str] = []
    valid_questions: list[str] = []
    for question, group in incorrect_df.groupby("question"):
        frequency = group["predicted_choice"].value_counts()
        if len(frequency) < 3 or frequency.nunique() < len(frequency):
            tie_questions.append(question)
        else:
            valid_questions.append(question)

    tie_df = incorrect_df[incorrect_df["question"].isin(tie_questions)].copy()
    valid_df = incorrect_df[incorrect_df["question"].isin(valid_questions)].copy()

    triplet_records = []
    for question in valid_questions:
        group = incorrect_df[incorrect_df["question"] == question]
        frequency = group["predicted_choice"].value_counts()
        triplet_records.append({"question": question, "model_triplet": get_triplet(frequency)})

    return correct_df, tie_df, valid_df, pd.DataFrame(triplet_records)


def calculate_iaa(human_ranked_df: pd.DataFrame) -> pd.DataFrame:
    overlap_df = human_ranked_df.dropna(
        subset=[
            "distractor_ranking_best_to_worst_Annotator_1",
            "distractor_ranking_best_to_worst_Annotator_2",
        ]
    ).copy()

    metrics_list = []
    for _, row in overlap_df.iterrows():
        ranking_one = parse_ranking_string(row["distractor_ranking_best_to_worst_Annotator_1"])
        ranking_two = parse_ranking_string(row["distractor_ranking_best_to_worst_Annotator_2"])
        if ranking_one and ranking_two:
            metrics_list.append(calculate_metrics(ranking_one, ranking_two))

    results = aggregate_metrics(metrics_list, len(overlap_df))
    results["n_overlapping"] = len(overlap_df)
    return pd.DataFrame([results])


def evaluate_method1(model_triplets_df: pd.DataFrame, human_ranked_df: pd.DataFrame) -> pd.DataFrame:
    overlap = human_ranked_df.dropna(
        subset=[
            "distractor_ranking_best_to_worst_Annotator_1",
            "distractor_ranking_best_to_worst_Annotator_2",
        ]
    ).copy()
    overlap["h1"] = overlap["distractor_ranking_best_to_worst_Annotator_1"].apply(parse_ranking_string)
    overlap["h2"] = overlap["distractor_ranking_best_to_worst_Annotator_2"].apply(parse_ranking_string)
    overlap["q_norm"] = overlap["question"].apply(clean_question)

    triplets = model_triplets_df.copy()
    triplets["q_norm"] = triplets["question"].apply(clean_question)

    merged = pd.merge(overlap[["q_norm", "h1", "h2"]], triplets[["q_norm", "model_triplet"]], on="q_norm", how="left")
    tie_cases = int(merged["model_triplet"].isna().sum())

    merged = merged[
        merged["h1"].apply(lambda value: isinstance(value, list) and len(value) == 3)
        & merged["h2"].apply(lambda value: isinstance(value, list) and len(value) == 3)
        & merged["model_triplet"].apply(lambda value: isinstance(value, list) and len(value) == 3)
    ].copy()

    human_human_metrics = [calculate_metrics(row["h1"], row["h2"]) for _, row in merged.iterrows()]
    model_human_metrics = [calculate_metrics(row["model_triplet"], row["h1"]) for _, row in merged.iterrows()]
    hh_aggregate = aggregate_metrics(human_human_metrics, len(merged))
    mh_aggregate = aggregate_metrics(model_human_metrics, len(merged))

    results = {
        "human_n_full": len(overlap),
        "eval_n": len(merged),
        **{f"hh_{key}": value for key, value in hh_aggregate.items()},
        "method1_exact_match_pct": mh_aggregate["exact_match_pct"],
        "method1_pairwise_concordance": mh_aggregate["avg_pairwise_concordance"],
        "method1_top1_accuracy": mh_aggregate["top1_accuracy"],
        "method1_top2_accuracy": mh_aggregate["top2_accuracy"],
        "method1_spearman_rho": mh_aggregate["avg_spearman_rho"],
        "method1_kendall_tau": mh_aggregate["avg_kendall_tau"],
        "method1_spearman_p": mh_aggregate["avg_spearman_p"],
        "method1_kendall_p": mh_aggregate["avg_kendall_p"],
        "method1_weighted_kappa": mh_aggregate["avg_weighted_kappa"],
        "method1_tie_case": tie_cases,
        "method1_tie_case_pct": (tie_cases / len(overlap) * 100) if len(overlap) else 0.0,
    }
    return pd.DataFrame([results])


def evaluate_method2(model_triplets_df: pd.DataFrame, human_ranked_df: pd.DataFrame) -> pd.DataFrame:
    overlap_df = human_ranked_df.dropna(
        subset=[
            "distractor_ranking_best_to_worst_Annotator_1",
            "distractor_ranking_best_to_worst_Annotator_2",
        ]
    ).copy()
    overlap_df["q_norm"] = overlap_df["question"].apply(clean_question)
    overlap_questions = set(overlap_df["q_norm"])

    single = human_ranked_df.copy()
    single["q_norm"] = single["question"].apply(clean_question)
    single = single[~single["q_norm"].isin(overlap_questions)].copy()
    single["human_triplet"] = single["distractor_ranking_best_to_worst_Annotator_1"].apply(parse_ranking_string)

    triplets = model_triplets_df.copy()
    triplets["q_norm"] = triplets["question"].apply(clean_question)

    merged = pd.merge(triplets[["q_norm", "model_triplet"]], single[["q_norm", "human_triplet"]], on="q_norm", how="inner")
    valid = merged[
        merged["model_triplet"].apply(lambda value: isinstance(value, list) and len(value) == 3)
        & merged["human_triplet"].apply(lambda value: isinstance(value, list) and len(value) == 3)
    ].copy()

    metrics_list = [calculate_metrics(row["model_triplet"], row["human_triplet"]) for _, row in valid.iterrows()]
    aggregated = aggregate_metrics(metrics_list, len(valid))
    tie_cases = int(single["q_norm"].nunique() - valid["q_norm"].nunique())

    results = {
        "method2_exact_match_pct": aggregated["exact_match_pct"],
        "method2_pairwise_concordance": aggregated["avg_pairwise_concordance"],
        "method2_top1_accuracy": aggregated["top1_accuracy"],
        "method2_top2_accuracy": aggregated["top2_accuracy"],
        "method2_spearman_rho": aggregated["avg_spearman_rho"],
        "method2_kendall_tau": aggregated["avg_kendall_tau"],
        "method2_spearman_p": aggregated["avg_spearman_p"],
        "method2_kendall_p": aggregated["avg_kendall_p"],
        "method2_weighted_kappa": aggregated["avg_weighted_kappa"],
        "method2_tie_case": tie_cases,
        "method2_tie_case_pct": (tie_cases / single["q_norm"].nunique() * 100) if single["q_norm"].nunique() else 0.0,
        "total_single_questions": single["q_norm"].nunique(),
        "eval_n": len(valid),
    }
    return pd.DataFrame([results])


def evaluate_domain_level(model_triplets_df: pd.DataFrame, human_ranked_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_df = human_ranked_df.dropna(
        subset=[
            "distractor_ranking_best_to_worst_Annotator_1",
            "distractor_ranking_best_to_worst_Annotator_2",
        ]
    ).copy()
    overlap_df["q_norm"] = overlap_df["question"].apply(clean_question)
    overlap_df["domain"] = overlap_df["subject"].map(DOMAIN_MAP)
    overlap_df["h1_triplet"] = overlap_df["distractor_ranking_best_to_worst_Annotator_1"].apply(parse_ranking_string)
    overlap_df["h2_triplet"] = overlap_df["distractor_ranking_best_to_worst_Annotator_2"].apply(parse_ranking_string)

    triplets = model_triplets_df.copy()
    triplets["q_norm"] = triplets["question"].apply(clean_question)

    method1_merged = pd.merge(
        overlap_df[["q_norm", "domain", "h1_triplet", "h2_triplet"]],
        triplets[["q_norm", "model_triplet"]],
        on="q_norm",
        how="left",
    )
    method1_merged["is_tie"] = method1_merged["model_triplet"].apply(
        lambda value: not (isinstance(value, list) and len(value) == 3)
    )

    method1_rows = []
    for domain in method1_merged["domain"].dropna().unique():
        domain_data = method1_merged[method1_merged["domain"] == domain]
        valid = domain_data[~domain_data["is_tie"]]
        if len(valid):
            h1_metrics = aggregate_metrics(
                [calculate_metrics(row["model_triplet"], row["h1_triplet"]) for _, row in valid.iterrows()],
                len(valid),
            )
            h2_metrics = aggregate_metrics(
                [calculate_metrics(row["model_triplet"], row["h2_triplet"]) for _, row in valid.iterrows()],
                len(valid),
            )
            metrics = {key: (h1_metrics[key] + h2_metrics[key]) / 2 for key in h1_metrics}
        else:
            metrics = aggregate_metrics([], 0)
        metrics.update(
            {
                "domain": domain,
                "n_total_questions": len(domain_data),
                "n_valid_questions": len(valid),
                "n_tie_cases": int(domain_data["is_tie"].sum()),
                "tie_case_pct": (domain_data["is_tie"].sum() / len(domain_data) * 100) if len(domain_data) else 0.0,
                "method": "Method 1 (Overlapping)",
            }
        )
        method1_rows.append(metrics)

    single = human_ranked_df.copy()
    single["q_norm"] = single["question"].apply(clean_question)
    single["domain"] = single["subject"].map(DOMAIN_MAP)
    single = single[~single["q_norm"].isin(set(overlap_df["q_norm"]))].copy()
    single["human_triplet"] = single["distractor_ranking_best_to_worst_Annotator_1"].apply(parse_ranking_string)

    method2_merged = pd.merge(
        single[["q_norm", "human_triplet", "domain"]],
        triplets[["q_norm", "model_triplet"]],
        on="q_norm",
        how="left",
    )
    method2_merged["is_tie"] = method2_merged["model_triplet"].apply(
        lambda value: not (isinstance(value, list) and len(value) == 3)
    )

    method2_rows = []
    for domain in method2_merged["domain"].dropna().unique():
        domain_data = method2_merged[method2_merged["domain"] == domain]
        valid = domain_data[~domain_data["is_tie"]]
        if len(valid):
            metrics = aggregate_metrics(
                [calculate_metrics(row["model_triplet"], row["human_triplet"]) for _, row in valid.iterrows()],
                len(valid),
            )
        else:
            metrics = aggregate_metrics([], 0)
        metrics.update(
            {
                "domain": domain,
                "n_total_questions": len(domain_data),
                "n_valid_questions": len(valid),
                "n_tie_cases": int(domain_data["is_tie"].sum()),
                "tie_case_pct": (domain_data["is_tie"].sum() / len(domain_data) * 100) if len(domain_data) else 0.0,
                "method": "Method 2 (Single-Annotated)",
            }
        )
        method2_rows.append(metrics)

    return pd.DataFrame(method1_rows), pd.DataFrame(method2_rows)


def create_overall_metrics(
    iaa_metrics: pd.DataFrame,
    method1_metrics: pd.DataFrame,
    method2_metrics: pd.DataFrame,
) -> pd.DataFrame:
    iaa_row = iaa_metrics.iloc[0].to_dict()
    method1_row = method1_metrics.iloc[0].to_dict()
    method2_row = method2_metrics.iloc[0].to_dict()

    return pd.DataFrame(
        [
            {
                "iaa_exact_match_pct": iaa_row.get("exact_match_pct", 0.0),
                "iaa_pairwise_concordance": iaa_row.get("avg_pairwise_concordance", 0.0),
                "iaa_top1_accuracy": iaa_row.get("top1_accuracy", 0.0),
                "iaa_top2_accuracy": iaa_row.get("top2_accuracy", 0.0),
                "iaa_tie_case_pct": iaa_row.get("tie_case_pct", 0.0),
                "iaa_spearman_rho": iaa_row.get("avg_spearman_rho", 0.0),
                "iaa_spearman_p": iaa_row.get("avg_spearman_p", 1.0),
                "iaa_kendall_tau": iaa_row.get("avg_kendall_tau", 0.0),
                "iaa_kendall_p": iaa_row.get("avg_kendall_p", 1.0),
                "iaa_weighted_kappa": iaa_row.get("avg_weighted_kappa", 0.0),
                "method1_exact_match_pct": method1_row.get("method1_exact_match_pct", 0.0),
                "method1_pairwise_concordance": method1_row.get("method1_pairwise_concordance", 0.0),
                "method1_top1_accuracy": method1_row.get("method1_top1_accuracy", 0.0),
                "method1_top2_accuracy": method1_row.get("method1_top2_accuracy", 0.0),
                "method1_tie_case_pct": method1_row.get("method1_tie_case_pct", 0.0),
                "method1_spearman_rho": method1_row.get("method1_spearman_rho", 0.0),
                "method1_spearman_p": method1_row.get("method1_spearman_p", 1.0),
                "method1_kendall_tau": method1_row.get("method1_kendall_tau", 0.0),
                "method1_kendall_p": method1_row.get("method1_kendall_p", 1.0),
                "method1_weighted_kappa": method1_row.get("method1_weighted_kappa", 0.0),
                "method2_exact_match_pct": method2_row.get("method2_exact_match_pct", 0.0),
                "method2_pairwise_concordance": method2_row.get("method2_pairwise_concordance", 0.0),
                "method2_top1_accuracy": method2_row.get("method2_top1_accuracy", 0.0),
                "method2_top2_accuracy": method2_row.get("method2_top2_accuracy", 0.0),
                "method2_tie_case_pct": method2_row.get("method2_tie_case_pct", 0.0),
                "method2_spearman_rho": method2_row.get("method2_spearman_rho", 0.0),
                "method2_spearman_p": method2_row.get("method2_spearman_p", 1.0),
                "method2_kendall_tau": method2_row.get("method2_kendall_tau", 0.0),
                "method2_kendall_p": method2_row.get("method2_kendall_p", 1.0),
                "method2_weighted_kappa": method2_row.get("method2_weighted_kappa", 0.0),
            }
        ]
    )
