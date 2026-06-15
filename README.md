# LLM Distractor Ranking Comparison

Refactored repository for training seq2seq models on multiple-choice question data and evaluating model-generated distractor rankings against human annotations.

## What this repo does

- Loads curated MMLU-style training data and human-ranked evaluation data from `data/raw/`
- Trains encoder-decoder models to predict answer choices
- Scores distractor preferences from model outputs
- Computes inter-annotator agreement plus Method 1, Method 2, and domain-level evaluation summaries
- Saves machine-readable metrics and report figures under `reports/`

## Repository layout

- `src/llm_distractor_ranking/`: package code for data loading, modeling, evaluation, plotting, and CLI entrypoints
- `data/raw/`: committed project datasets
- `notebooks/archive/`: original notebook artifacts kept for provenance
- `configs/default.yaml`: default runtime and path configuration
- `tests/`: unit tests and smoke coverage for core extraction logic

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

Validate the real CSVs:

```bash
llm-distractor validate-data --config configs/default.yaml
```

Run the full pipeline:

```bash
llm-distractor run-all --config configs/default.yaml
```

Or run stages independently:

```bash
llm-distractor train --config configs/default.yaml
llm-distractor predict --config configs/default.yaml
llm-distractor evaluate --config configs/default.yaml
llm-distractor plot --config configs/default.yaml
```

## Data notes

- The refactor preserves the current annotation data as-is, including known ranking rows where a human distractor ranking includes the correct answer index.
- Missing option rows in `training_data.csv` are filtered during preprocessing and reported by `validate-data`.
- `mmlu_dataset.parquet` is retained as a raw source artifact but the default pipeline uses `training_data.csv`.

## Outputs

- `outputs/predictions/predictions.csv`
- `reports/metrics/*.csv`
- `reports/figures/*.png`

## Notebook lineage

The original notebook/export sources were moved to:

- [notebooks/archive/LLM Distractor Ranking - Part 1](</Users/chaitu/Downloads/llm-distractor-ranking-comparison-main/notebooks/archive/LLM Distractor Ranking - Part 1>)
- [notebooks/archive/LLM_Distractor_Ranking_Part_2.ipynb](/Users/chaitu/Downloads/llm-distractor-ranking-comparison-main/notebooks/archive/LLM_Distractor_Ranking_Part_2.ipynb)
