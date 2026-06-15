from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

TRAINING_COLUMNS = [
    "question",
    "subject",
    "choices",
    "correct_answer",
    "option_0",
    "option_1",
    "option_2",
    "option_3",
]

HUMAN_RANKED_COLUMNS = [
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

_TRAINING_REQUIRED = [
    "question",
    "subject",
    "correct_answer",
    "option_0",
    "option_1",
    "option_2",
    "option_3",
]

_HUMAN_REQUIRED = [
    "subject",
    "question_id",
    "question",
    "correct_answer",
    "option_0",
    "option_1",
    "option_2",
    "option_3",
    "distractor_ranking_best_to_worst_Annotator_1",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, expected_columns: Iterable[str], dataset_name: str) -> None:
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def _drop_missing_rows(df: pd.DataFrame, required_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_mask = df[required_columns].notna().all(axis=1)
    cleaned = df.loc[keep_mask].copy()
    dropped = df.loc[~keep_mask].copy()
    return cleaned, dropped


def preprocess_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(df, TRAINING_COLUMNS, "training_data")
    cleaned, dropped = _drop_missing_rows(df, _TRAINING_REQUIRED)
    cleaned["correct_answer"] = pd.to_numeric(cleaned["correct_answer"], errors="coerce")
    invalid_answer_mask = ~cleaned["correct_answer"].isin([0, 1, 2, 3])
    if invalid_answer_mask.any():
        dropped = pd.concat([dropped, cleaned.loc[invalid_answer_mask]], ignore_index=True)
        cleaned = cleaned.loc[~invalid_answer_mask].copy()
    cleaned["correct_answer"] = cleaned["correct_answer"].astype(int)
    return cleaned.reset_index(drop=True), dropped.reset_index(drop=True)


def preprocess_human_ranked_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(df, HUMAN_RANKED_COLUMNS, "human_ranked")
    cleaned, dropped = _drop_missing_rows(df, _HUMAN_REQUIRED)
    cleaned["correct_answer"] = pd.to_numeric(cleaned["correct_answer"], errors="coerce")
    invalid_answer_mask = ~cleaned["correct_answer"].isin([0, 1, 2, 3])
    if invalid_answer_mask.any():
        dropped = pd.concat([dropped, cleaned.loc[invalid_answer_mask]], ignore_index=True)
        cleaned = cleaned.loc[~invalid_answer_mask].copy()
    cleaned["correct_answer"] = cleaned["correct_answer"].astype(int)
    return cleaned.reset_index(drop=True), dropped.reset_index(drop=True)


def build_input_text(row: pd.Series) -> str:
    return (
        f"Question: {row['question']} "
        f"Options: A: {row['option_0']} B: {row['option_1']} "
        f"C: {row['option_2']} D: {row['option_3']}"
    )


def add_input_text(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["input_text"] = enriched.apply(build_input_text, axis=1)
    return enriched
