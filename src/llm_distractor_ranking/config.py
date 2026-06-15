from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    training_data: Path
    human_ranked: Path
    parquet_source: Path
    predictions: Path
    metrics_dir: Path
    figures_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    device: str
    random_seed: int


@dataclass(frozen=True)
class SmokeConfig:
    model_limit: int
    hyperparameter_limit: int
    training_row_limit: int
    evaluation_row_limit: int


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    paths: PathsConfig
    runtime: RuntimeConfig
    models: list[str]
    hyperparameters: dict[str, dict[str, Any]]
    smoke: SmokeConfig

    @property
    def active_models(self) -> list[str]:
        if self.runtime.mode == "smoke":
            return self.models[: self.smoke.model_limit]
        return self.models

    @property
    def active_hyperparameters(self) -> dict[str, dict[str, Any]]:
        if self.runtime.mode == "smoke":
            keys = list(self.hyperparameters)[: self.smoke.hyperparameter_limit]
            return {key: self.hyperparameters[key] for key in keys}
        return self.hyperparameters


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    root = config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text())

    paths = PathsConfig(
        training_data=root / raw["paths"]["training_data"],
        human_ranked=root / raw["paths"]["human_ranked"],
        parquet_source=root / raw["paths"]["parquet_source"],
        predictions=root / raw["paths"]["predictions"],
        metrics_dir=root / raw["paths"]["metrics_dir"],
        figures_dir=root / raw["paths"]["figures_dir"],
        checkpoints_dir=root / raw["paths"]["checkpoints_dir"],
        logs_dir=root / raw["paths"]["logs_dir"],
    )
    runtime = RuntimeConfig(
        mode=raw["runtime"]["mode"],
        device=raw["runtime"]["device"],
        random_seed=int(raw["runtime"]["random_seed"]),
    )
    smoke = SmokeConfig(
        model_limit=int(raw["smoke"]["model_limit"]),
        hyperparameter_limit=int(raw["smoke"]["hyperparameter_limit"]),
        training_row_limit=int(raw["smoke"]["row_limit"]["training"]),
        evaluation_row_limit=int(raw["smoke"]["row_limit"]["evaluation"]),
    )
    return ProjectConfig(
        root=root,
        paths=paths,
        runtime=runtime,
        models=list(raw["models"]),
        hyperparameters=dict(raw["hyperparameters"]),
        smoke=smoke,
    )


def ensure_runtime_directories(config: ProjectConfig) -> None:
    for path in [
        config.paths.metrics_dir,
        config.paths.figures_dir,
        config.paths.checkpoints_dir,
        config.paths.logs_dir,
        config.paths.predictions.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)
