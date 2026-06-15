import pandas as pd
import pytest

from llm_distractor_ranking.data import (
    HUMAN_RANKED_COLUMNS,
    TRAINING_COLUMNS,
    build_input_text,
    preprocess_human_ranked_data,
    preprocess_training_data,
    validate_columns,
)


def test_validate_columns_accepts_expected_training_schema():
    df = pd.DataFrame(columns=TRAINING_COLUMNS)

    validate_columns(df, TRAINING_COLUMNS, "training_data")


def test_validate_columns_rejects_missing_required_columns():
    df = pd.DataFrame(columns=["question", "subject"])

    with pytest.raises(ValueError, match="training_data"):
        validate_columns(df, TRAINING_COLUMNS, "training_data")


def test_preprocess_training_data_drops_rows_with_missing_options():
    df = pd.DataFrame(
        [
            {
                "question": "Q1",
                "subject": "math",
                "choices": "['1' '2' '3' '4']",
                "correct_answer": 0,
                "option_0": "1",
                "option_1": "2",
                "option_2": "3",
                "option_3": "4",
            },
            {
                "question": "Q2",
                "subject": "math",
                "choices": "['1' '2' '3' '4']",
                "correct_answer": 1,
                "option_0": None,
                "option_1": "2",
                "option_2": "3",
                "option_3": "4",
            },
        ]
    )

    cleaned, dropped = preprocess_training_data(df)

    assert cleaned["question"].tolist() == ["Q1"]
    assert dropped["question"].tolist() == ["Q2"]


def test_preprocess_human_ranked_preserves_known_invalid_ranking_rows():
    df = pd.DataFrame(
        [
            {
                "subject": "biology",
                "question_id": 1,
                "question": "Q1",
                "correct_answer": 2,
                "option_0": "A",
                "option_1": "B",
                "option_2": "C",
                "option_3": "D",
                "distractor_ranking_best_to_worst_Annotator_1": "2,1,3",
                "distractor_ranking_best_to_worst_Annotator_2": None,
            }
        ]
    )

    cleaned, dropped = preprocess_human_ranked_data(df)

    assert cleaned["question_id"].tolist() == [1]
    assert dropped.empty


def test_build_input_text_formats_question_and_options():
    row = pd.Series(
        {
            "question": "What is 2 + 2?",
            "option_0": "1",
            "option_1": "4",
            "option_2": "8",
            "option_3": "16",
        }
    )

    text = build_input_text(row)

    assert text == "Question: What is 2 + 2? Options: A: 1 B: 4 C: 8 D: 16"


def test_human_ranked_schema_constant_matches_current_csv_contract():
    assert HUMAN_RANKED_COLUMNS == [
        "subject",
        "question_id",
        "question",
        "correct_answer",
        "option_0",
        "option_1",
        "option_2",
        "option_3",
        "distractor_ranking_best_to_worst_Annotator_1",
        "distractor_ranking_best_to_worst_Annotator_2",
    ]
