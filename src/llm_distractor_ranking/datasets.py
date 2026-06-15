from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ProjectConfig
from .data import add_input_text


def map_answer_to_choice(row: pd.Series, answer_column: str = "correct_answer") -> str | None:
    try:
        answer_index = int(row[answer_column])
    except (TypeError, ValueError, KeyError):
        return None
    if answer_index not in [0, 1, 2, 3]:
        return None
    return chr(65 + answer_index)


def apply_runtime_limits(
    training_df: pd.DataFrame,
    human_ranked_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.runtime.mode != "smoke":
        return training_df, human_ranked_df
    return (
        training_df.head(config.smoke.training_row_limit).copy(),
        human_ranked_df.head(config.smoke.evaluation_row_limit).copy(),
    )


def prepare_model_frames(
    training_df: pd.DataFrame,
    human_ranked_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_df, human_ranked_df = apply_runtime_limits(training_df, human_ranked_df, config)

    training_frame = add_input_text(training_df.copy())
    human_frame = add_input_text(human_ranked_df.copy())

    training_frame["correct_choice"] = training_frame.apply(map_answer_to_choice, axis=1)
    human_frame["correct_choice"] = human_frame.apply(map_answer_to_choice, axis=1)
    training_frame["correct_index"] = training_frame["correct_answer"].astype(int)
    human_frame["correct_index"] = human_frame["correct_answer"].astype(int)

    training_frame = training_frame.dropna(subset=["correct_choice"]).reset_index(drop=True)
    human_frame = human_frame.dropna(subset=["correct_choice"]).reset_index(drop=True)
    return training_frame, human_frame


def prepare_hf_datasets(
    training_df: pd.DataFrame,
    human_ranked_df: pd.DataFrame,
    model_names: list[str],
) -> tuple[dict[str, dict[str, Any]], Any, Any]:
    from datasets import Dataset
    from transformers import AutoTokenizer

    train_dataset = Dataset.from_pandas(
        training_df[
            [
                "input_text",
                "correct_choice",
                "question",
                "option_0",
                "option_1",
                "option_2",
                "option_3",
                "correct_index",
            ]
        ],
        preserve_index=False,
    )
    eval_dataset = Dataset.from_pandas(
        human_ranked_df[
            [
                "input_text",
                "correct_choice",
                "question",
                "option_0",
                "option_1",
                "option_2",
                "option_3",
                "correct_index",
            ]
        ],
        preserve_index=False,
    )

    tokenized_datasets: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples: dict[str, list[Any]]) -> dict[str, Any]:
            model_inputs = tokenizer(
                examples["input_text"],
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            labels = tokenizer(examples["correct_choice"], add_special_tokens=False).input_ids
            model_inputs["labels"] = [
                label[0] if len(label) == 1 else tokenizer.unk_token_id for label in labels
            ]
            return model_inputs

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        tokenized_datasets[model_name] = {"train": tokenized_train, "eval": tokenized_eval}

    return tokenized_datasets, train_dataset, eval_dataset
