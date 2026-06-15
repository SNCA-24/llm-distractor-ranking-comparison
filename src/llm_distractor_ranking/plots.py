from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = {
    "blue": "#2E86AB",
    "yellow": "#F6C85F",
    "red": "#FF6F59",
    "green": "#6BAB90",
    "purple": "#9B5094",
}


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-talk")
    sns.set_palette("husl")


def plot_data_funnel(combined_df: pd.DataFrame, correct_df: pd.DataFrame, tie_df: pd.DataFrame, valid_df: pd.DataFrame, model_triplets_df: pd.DataFrame):
    counts = {
        "Combined rows": len(combined_df),
        "Correct predictions": len(correct_df),
        "Incorrect predictions": len(combined_df) - len(correct_df),
        "Tie-case rows": len(tie_df),
        "Valid-case rows": len(valid_df),
        "Triplets generated": len(model_triplets_df),
    }
    stages, values = zip(*counts.items())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(stages, values, color=[COLORS[key] for key in ["blue", "green", "red", "yellow", "purple", "blue"]])
    max_value = max(values) if values else 0
    for idx, value in enumerate(values):
        ax.text(value + max_value * 0.01, idx, f"{value:,}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Data Processing Funnel")
    plt.tight_layout()
    return fig


def plot_method_comparison(overall_metrics_df: pd.DataFrame):
    metrics = ["Exact-match %", "Top-1 %", "Top-2 %", "Tie-case %"]
    method1 = [
        float(overall_metrics_df["method1_exact_match_pct"].iloc[0]),
        float(overall_metrics_df["method1_top1_accuracy"].iloc[0]),
        float(overall_metrics_df["method1_top2_accuracy"].iloc[0]),
        float(overall_metrics_df["method1_tie_case_pct"].iloc[0]),
    ]
    method2 = [
        float(overall_metrics_df["method2_exact_match_pct"].iloc[0]),
        float(overall_metrics_df["method2_top1_accuracy"].iloc[0]),
        float(overall_metrics_df["method2_top2_accuracy"].iloc[0]),
        float(overall_metrics_df["method2_tie_case_pct"].iloc[0]),
    ]

    x_axis = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x_axis - width / 2, method1, width, color=COLORS["blue"], label="Method 1")
    bars2 = ax.bar(x_axis + width / 2, method2, width, color=COLORS["yellow"], label="Method 2")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(metrics, rotation=45)
    ax.set_ylabel("Percentage")
    ax.set_title("Method 1 vs Method 2 Trade-offs")
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%", ha="center", va="bottom")
    ax.legend()
    plt.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
