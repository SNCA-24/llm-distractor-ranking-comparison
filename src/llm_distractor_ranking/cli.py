from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ensure_runtime_directories, load_config
from .data import load_csv, preprocess_human_ranked_data, preprocess_training_data
from .evaluation import (
    calculate_iaa,
    create_overall_metrics,
    create_prediction_subsets,
    evaluate_domain_level,
    evaluate_method1,
    evaluate_method2,
    load_predictions,
)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-distractor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in [
        ("validate-data", cmd_validate_data),
        ("train", cmd_train),
        ("predict", cmd_predict),
        ("evaluate", cmd_evaluate),
        ("plot", cmd_plot),
        ("run-all", cmd_run_all),
    ]:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="configs/default.yaml")
        subparser.set_defaults(func=handler)
    return parser


def cmd_validate_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    training_df, training_dropped, human_df, human_dropped = _load_and_preprocess(config)
    print(
        {
            "training_rows": len(training_df),
            "training_dropped": len(training_dropped),
            "human_rows": len(human_df),
            "human_dropped": len(human_dropped),
        }
    )


def cmd_train(args: argparse.Namespace) -> None:
    from .datasets import prepare_hf_datasets, prepare_model_frames
    from .modeling import DistractorAnalysis

    config = load_config(args.config)
    ensure_runtime_directories(config)
    training_df, _, human_df, _ = _load_and_preprocess(config)
    train_frame, eval_frame = prepare_model_frames(training_df, human_df, config)
    tokenized, _, _ = prepare_hf_datasets(train_frame, eval_frame, config.active_models)

    for model_name in config.active_models:
        for hyper_name, hyperparameters in config.active_hyperparameters.items():
            checkpoint_dir = config.paths.checkpoints_dir / f"{model_name.replace('/', '_')}_{hyper_name}"
            log_dir = config.paths.logs_dir / f"{model_name.replace('/', '_')}_{hyper_name}"
            analysis = DistractorAnalysis(model_name, hyperparameters, checkpoint_dir, log_dir)
            analysis.train(tokenized[model_name]["train"], tokenized[model_name]["eval"])


def cmd_predict(args: argparse.Namespace) -> None:
    from .datasets import prepare_hf_datasets, prepare_model_frames
    from .prediction import generate_predictions

    config = load_config(args.config)
    ensure_runtime_directories(config)
    training_df, _, human_df, _ = _load_and_preprocess(config)
    train_frame, eval_frame = prepare_model_frames(training_df, human_df, config)
    _, _, eval_dataset = prepare_hf_datasets(train_frame, eval_frame, config.active_models)

    prediction_frames: list[pd.DataFrame] = []
    for model_name in config.active_models:
        for hyper_name, hyperparameters in config.active_hyperparameters.items():
            checkpoint_dir = config.paths.checkpoints_dir / f"{model_name.replace('/', '_')}_{hyper_name}"
            prediction_frames.append(
                generate_predictions(
                    eval_dataset,
                    model_name=model_name,
                    hyperparameter_name=hyper_name,
                    hyperparameters=hyperparameters,
                    checkpoint_dir=checkpoint_dir,
                    device=config.runtime.device,
                )
            )
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions_df.to_csv(config.paths.predictions, index=False)


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_runtime_directories(config)
    _, _, human_df, _ = _load_and_preprocess(config)
    predictions_df = load_predictions(config.paths.predictions)
    correct_df, tie_df, valid_df, triplets_df = create_prediction_subsets(predictions_df)
    iaa_df = calculate_iaa(human_df)
    method1_df = evaluate_method1(triplets_df, human_df)
    method2_df = evaluate_method2(triplets_df, human_df)
    domain1_df, domain2_df = evaluate_domain_level(triplets_df, human_df)
    overall_df = create_overall_metrics(iaa_df, method1_df, method2_df)

    _write_csv(config.paths.metrics_dir / "iaa_results.csv", iaa_df)
    _write_csv(config.paths.metrics_dir / "method1_results.csv", method1_df)
    _write_csv(config.paths.metrics_dir / "method2_results.csv", method2_df)
    _write_csv(config.paths.metrics_dir / "domain_level_method1.csv", domain1_df)
    _write_csv(config.paths.metrics_dir / "domain_level_method2.csv", domain2_df)
    _write_csv(config.paths.metrics_dir / "overall_metrics.csv", overall_df)
    _write_csv(config.paths.metrics_dir / "correct_predictions.csv", correct_df)
    _write_csv(config.paths.metrics_dir / "tie_predictions.csv", tie_df)
    _write_csv(config.paths.metrics_dir / "valid_predictions.csv", valid_df)
    _write_csv(config.paths.metrics_dir / "model_triplets.csv", triplets_df)


def cmd_plot(args: argparse.Namespace) -> None:
    from .plots import configure_style, plot_data_funnel, plot_method_comparison, save_figure

    config = load_config(args.config)
    ensure_runtime_directories(config)
    configure_style()
    predictions_df = load_predictions(config.paths.predictions)
    _, _, human_df, _ = _load_and_preprocess(config)
    correct_df, tie_df, valid_df, triplets_df = create_prediction_subsets(predictions_df)
    iaa_df = pd.read_csv(config.paths.metrics_dir / "iaa_results.csv")
    method1_df = pd.read_csv(config.paths.metrics_dir / "method1_results.csv")
    method2_df = pd.read_csv(config.paths.metrics_dir / "method2_results.csv")
    overall_df = create_overall_metrics(iaa_df, method1_df, method2_df)

    save_figure(plot_data_funnel(predictions_df, correct_df, tie_df, valid_df, triplets_df), config.paths.figures_dir / "data_funnel.png")
    save_figure(plot_method_comparison(overall_df), config.paths.figures_dir / "method_comparison.png")
    print({"figures_dir": str(config.paths.figures_dir), "human_rows": len(human_df)})


def cmd_run_all(args: argparse.Namespace) -> None:
    cmd_validate_data(args)
    cmd_train(args)
    cmd_predict(args)
    cmd_evaluate(args)
    cmd_plot(args)


def _load_and_preprocess(config):
    training_df = load_csv(config.paths.training_data)
    human_df = load_csv(config.paths.human_ranked)
    cleaned_training_df, dropped_training_df = preprocess_training_data(training_df)
    cleaned_human_df, dropped_human_df = preprocess_human_ranked_data(human_df)
    return cleaned_training_df, dropped_training_df, cleaned_human_df, dropped_human_df


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
