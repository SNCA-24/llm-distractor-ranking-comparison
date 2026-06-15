from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_predictions(
    eval_dataset,
    model_name: str,
    hyperparameter_name: str,
    hyperparameters: dict,
    checkpoint_dir: Path,
    device: str = "auto",
) -> pd.DataFrame:
    from .modeling import DistractorAnalysis

    analysis = DistractorAnalysis(
        model_name=model_name,
        hyperparameters=hyperparameters,
        output_dir=checkpoint_dir,
        log_dir=checkpoint_dir.parent / f"{checkpoint_dir.name}_logs",
    )
    analysis.model = analysis.model.from_pretrained(str(checkpoint_dir))
    predictions, log_probabilities = analysis.predict_logprob(eval_dataset, device=device)

    rows = []
    for index, example in enumerate(eval_dataset):
        rows.append(
            {
                "question": example["question"],
                "model_name": model_name,
                "variant": hyperparameter_name,
                "predicted_choice": int(predictions[index]),
                "correct_choice": int(example["correct_index"]),
                "logprob_A": float(log_probabilities[index][0]),
                "logprob_B": float(log_probabilities[index][1]),
                "logprob_C": float(log_probabilities[index][2]),
                "logprob_D": float(log_probabilities[index][3]),
                "option_0": example["option_0"],
                "option_1": example["option_1"],
                "option_2": example["option_2"],
                "option_3": example["option_3"],
            }
        )
    return pd.DataFrame(rows)
