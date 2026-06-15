from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class TrainingVariant:
    model_name: str
    hyperparameter_name: str
    checkpoint_dir: Path
    log_dir: Path


class DistractorAnalysis:
    def __init__(self, model_name: str, hyperparameters: dict[str, Any], output_dir: Path, log_dir: Path) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments

        self.model_name = model_name
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(hyperparameters["batch_size"]),
            per_device_eval_batch_size=int(hyperparameters["batch_size"]),
            num_train_epochs=float(hyperparameters["epochs"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            evaluation_strategy="steps",
            eval_steps=1000,
            logging_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            logging_dir=str(log_dir),
        )

    def train(self, train_dataset: Any, eval_dataset: Any) -> Path:
        from transformers import Trainer

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        return self.output_dir

    def predict_logprob(self, dataset: Any, device: str = "auto") -> tuple[list[int], list[list[float]]]:
        resolved_device = _resolve_device(device)
        self.model.to(resolved_device).eval()
        predictions: list[int] = []
        log_probabilities: list[list[float]] = []

        with torch.no_grad():
            for example in dataset:
                prompt = example["input_text"]
                encoded = self.tokenizer(prompt, return_tensors="pt").to(resolved_device)
                letter_scores: list[float] = []
                for label in ["A", "B", "C", "D"]:
                    labels = self.tokenizer(label, add_special_tokens=False, return_tensors="pt").input_ids.to(resolved_device)
                    output = self.model(
                        input_ids=encoded.input_ids,
                        attention_mask=encoded.attention_mask,
                        labels=labels,
                    )
                    letter_scores.append(float(-output.loss.item() * labels.size(1)))
                predictions.append(int(max(range(len(letter_scores)), key=lambda idx: letter_scores[idx])))
                log_probabilities.append(letter_scores)

        return predictions, log_probabilities


def _resolve_device(device: str) -> torch.device:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
