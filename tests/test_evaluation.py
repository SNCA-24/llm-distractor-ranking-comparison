import pandas as pd

from llm_distractor_ranking.evaluation import (
    calculate_iaa,
    create_prediction_subsets,
    evaluate_method1,
    evaluate_method2,
    get_triplet,
)


def _predictions_df():
    return pd.DataFrame(
        [
            {"question": "Q1", "model_name": "m1", "variant": "low", "predicted_choice": 1, "correct_choice": 0},
            {"question": "Q1", "model_name": "m2", "variant": "low", "predicted_choice": 2, "correct_choice": 0},
            {"question": "Q1", "model_name": "m3", "variant": "low", "predicted_choice": 1, "correct_choice": 0},
            {"question": "Q1", "model_name": "m4", "variant": "low", "predicted_choice": 3, "correct_choice": 0},
            {"question": "Q1", "model_name": "m5", "variant": "low", "predicted_choice": 1, "correct_choice": 0},
            {"question": "Q1", "model_name": "m6", "variant": "low", "predicted_choice": 2, "correct_choice": 0},
            {"question": "Q2", "model_name": "m1", "variant": "low", "predicted_choice": 1, "correct_choice": 0},
            {"question": "Q2", "model_name": "m2", "variant": "low", "predicted_choice": 1, "correct_choice": 0},
            {"question": "Q2", "model_name": "m3", "variant": "low", "predicted_choice": 2, "correct_choice": 0},
            {"question": "Q3", "model_name": "m1", "variant": "low", "predicted_choice": 0, "correct_choice": 0},
        ]
    )


def _human_ranked_df():
    return pd.DataFrame(
        [
            {
                "subject": "college_biology",
                "question_id": 1,
                "question": "Q1",
                "correct_answer": 0,
                "option_0": "A",
                "option_1": "B",
                "option_2": "C",
                "option_3": "D",
                "distractor_ranking_best_to_worst_Annotator_1": "1,2,3",
                "distractor_ranking_best_to_worst_Annotator_2": "1,2,3",
            },
            {
                "subject": "college_biology",
                "question_id": 2,
                "question": "Q2",
                "correct_answer": 0,
                "option_0": "A",
                "option_1": "B",
                "option_2": "C",
                "option_3": "D",
                "distractor_ranking_best_to_worst_Annotator_1": "1,2,3",
                "distractor_ranking_best_to_worst_Annotator_2": None,
            },
        ]
    )


def test_get_triplet_orders_by_descending_vote_count():
    triplet = get_triplet(pd.Series({2: 1, 1: 3, 3: 2}))

    assert triplet == [1, 3, 2]


def test_create_prediction_subsets_separates_correct_tie_and_valid_rows():
    correct_df, tie_df, valid_df, triplets_df = create_prediction_subsets(_predictions_df())

    assert correct_df["question"].tolist() == ["Q3"]
    assert sorted(tie_df["question"].unique().tolist()) == ["Q2"]
    assert sorted(valid_df["question"].unique().tolist()) == ["Q1"]
    assert triplets_df.to_dict("records") == [{"question": "Q1", "model_triplet": [1, 2, 3]}]


def test_calculate_iaa_returns_single_row_summary():
    result = calculate_iaa(_human_ranked_df())

    assert result.loc[0, "n_overlapping"] == 1
    assert result.loc[0, "exact_match_pct"] == 100.0


def test_evaluate_method1_uses_overlap_questions_and_counts_tie_cases():
    _, _, _, triplets_df = create_prediction_subsets(_predictions_df())

    result = evaluate_method1(triplets_df, _human_ranked_df())

    assert result.loc[0, "human_n_full"] == 1
    assert result.loc[0, "eval_n"] == 1
    assert result.loc[0, "method1_tie_case"] == 0
    assert result.loc[0, "method1_exact_match_pct"] == 100.0


def test_evaluate_method2_counts_missing_triplets_as_tie_cases():
    _, _, _, triplets_df = create_prediction_subsets(_predictions_df())

    result = evaluate_method2(triplets_df, _human_ranked_df())

    assert result.loc[0, "total_single_questions"] == 1
    assert result.loc[0, "eval_n"] == 0
    assert result.loc[0, "method2_tie_case"] == 1
    assert result.loc[0, "method2_tie_case_pct"] == 100.0
