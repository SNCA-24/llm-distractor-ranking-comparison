from pathlib import Path
from types import SimpleNamespace

from llm_distractor_ranking.cli import cmd_validate_data
from llm_distractor_ranking.config import load_config


def test_load_config_resolves_paths_relative_to_repo_root():
    config = load_config("configs/default.yaml")

    assert config.paths.training_data.name == "training_data.csv"
    assert config.paths.human_ranked.name == "human_ranked.csv"
    assert config.root == Path("configs/default.yaml").resolve().parent.parent


def test_validate_data_cli_smoke(tmp_path, capsys):
    training_csv = tmp_path / "training.csv"
    human_csv = tmp_path / "human.csv"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    training_csv.write_text(
        "question,subject,choices,correct_answer,option_0,option_1,option_2,option_3\n"
        "Q1,math,\"['1' '2' '3' '4']\",1,1,2,3,4\n",
        encoding="utf-8",
    )
    human_csv.write_text(
        "subject,question_id,question,correct_answer,option_0,option_1,option_2,option_3,distractor_ranking_best_to_worst_Annotator_1,distractor_ranking_best_to_worst_Annotator_2\n"
        "math,1,Q1,1,1,2,3,4,\"0,2,3\",\n",
        encoding="utf-8",
    )
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "paths:",
                "  training_data: training.csv",
                "  human_ranked: human.csv",
                "  parquet_source: source.parquet",
                "  predictions: outputs/predictions/predictions.csv",
                "  metrics_dir: reports/metrics",
                "  figures_dir: reports/figures",
                "  checkpoints_dir: outputs/checkpoints",
                "  logs_dir: outputs/logs",
                "runtime:",
                "  mode: smoke",
                "  device: auto",
                "  random_seed: 42",
                "models:",
                "  - google-t5/t5-small",
                "hyperparameters:",
                "  low:",
                "    batch_size: 4",
                "    epochs: 1",
                "    learning_rate: 1.0e-5",
                "smoke:",
                "  model_limit: 1",
                "  hyperparameter_limit: 1",
                "  row_limit:",
                "    training: 8",
                "    evaluation: 8",
            ]
        ),
        encoding="utf-8",
    )

    cmd_validate_data(SimpleNamespace(config=str(config_dir / "default.yaml")))
    captured = capsys.readouterr()

    assert "'training_rows': 1" in captured.out
    assert "'human_rows': 1" in captured.out
