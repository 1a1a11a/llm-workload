#!/usr/bin/env python3
"""Token usage distribution analysis script

Plot token distributions for metrics traces using data_loader.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from readers.data_loader import load_metrics_dataframe
from utils.plot import setup_plot_style


def _load_records(csv_path: str) -> pd.DataFrame:
    """Load all metrics records from the given CSV path using data_loader."""
    return load_metrics_dataframe(csv_path, apply_transforms=True)


def extract_token_data(
    records: pd.DataFrame, model_name: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract arrays of input and output tokens for the requested model."""
    input_tokens: List[int] = []
    output_tokens: List[int] = []
    for record in records.itertuples():
        model = getattr(record, "model_name", None)
        chute = getattr(record, "chute_id", None)
        if model == model_name or chute == model_name:
            input_tokens.append(record.input_tokens)
            output_tokens.append(record.output_tokens)
    if not input_tokens:
        return None, None
    return np.array(input_tokens), np.array(output_tokens)


def plot_cdf(
    data_dict: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    output_file: str,
    logx: bool = True,
) -> None:
    """Plot the CDF for each dataset in data_dict and save the figure."""
    plt.figure(figsize=(12, 8))
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    colors = tuple(plt.cm.tab10.colors + plt.cm.Set1.colors)

    for index, (label, data) in enumerate(data_dict.items()):
        if data is None or len(data) == 0:
            continue
        sorted_data = np.sort(data)
        y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        line_style = line_styles[index % len(line_styles)]
        color = colors[index % len(colors)]
        plt.plot(
            sorted_data,
            y_values,
            label=f"{label} (n={len(data):,})",
            linewidth=3,
            alpha=0.8,
            linestyle=line_style,
            color=color,
        )

    plt.title(title, pad=20)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative Probability Distribution (CDF)")
    if logx:
        plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_cdf_individual(
    data_dict: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    output_dir: str,
    logx: bool = True,
) -> None:
    """Plot individual CDFs for each model in separate figures."""
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    colors = tuple(plt.cm.tab10.colors + plt.cm.Set1.colors)
    
    for model_name, data in data_dict.items():
        if data is None or len(data) == 0:
            continue
            
        plt.figure(figsize=(12, 8))
        sorted_data = np.sort(data)
        y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        line_style = line_styles[0]  # Use same style for all individual plots
        color = colors[0]  # Use same color for all individual plots
        plt.plot(
            sorted_data,
            y_values,
            label=f"{model_name} (n={len(data):,})",
            linewidth=3,
            alpha=0.8,
            linestyle=line_style,
            color=color,
        )

        plt.title(f"{title} - {model_name}", pad=20)
        plt.xlabel(xlabel)
        plt.ylabel("Cumulative Probability Distribution (CDF)")
        if logx:
            plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f"{model_name.replace('/', '_')}_cdf.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved individual plot: {output_file}")


def _select_models(
    records: pd.DataFrame, requested: Optional[List[str]], limit: int
) -> List[str]:
    """Return the models to plot, defaulting to the top-N by frequency."""
    if requested:
        seen = []
        for model in requested:
            if model not in seen:
                seen.append(model)
        return seen
    counts = Counter(
        getattr(record, "model_name", getattr(record, "chute_id", "unknown"))
        for record in records.itertuples()
    )
    return [model for model, _ in counts.most_common(limit)]


def main(
    csv_path: str, models: Optional[List[str]], output_dir: str, top_k: int
) -> None:
    records = _load_records(csv_path)
    if records.empty:
        raise RuntimeError(f"No records found in {csv_path}")

    target_models = _select_models(records, models, top_k)
    if not target_models:
        raise RuntimeError("No models selected for plotting")

    print(f"Loaded {len(records):,} records from {csv_path}")
    print("Models to plot: " + ", ".join(target_models))

    input_tokens_data: Dict[str, np.ndarray] = {}
    output_tokens_data: Dict[str, np.ndarray] = {}
    ratio_data: Dict[str, np.ndarray] = {}

    for model in target_models:
        input_tokens, output_tokens = extract_token_data(records, model)
        if input_tokens is None or output_tokens is None:
            print(f"  -> No samples for {model}")
            continue

        input_tokens_data[model] = input_tokens
        output_tokens_data[model] = output_tokens
        positive_mask = input_tokens > 0
        ratio_data[model] = output_tokens[positive_mask] / input_tokens[positive_mask]
        print(f"  -> {model}: {len(input_tokens)} samples")

    if not input_tokens_data:
        raise RuntimeError("No token data available after filtering")

    trace_name = os.path.basename(csv_path).split(".")[0]
    output_path = Path(f"{output_dir}/figures/token_distributions/{trace_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    print(output_path)

    # Create combined plots for all models
    print("\nCreating combined input tokens CDF plot...")
    plot_cdf(
        input_tokens_data,
        "Input Tokens Distribution (All Models)",
        "Input Tokens",
        str(output_path / "input_tokens_cdf_combined.png"),
    )

    print("Creating combined output tokens CDF plot...")
    plot_cdf(
        output_tokens_data,
        "Output Tokens Distribution (All Models)",
        "Output Tokens",
        str(output_path / "output_tokens_cdf_combined.png"),
    )

    print("Creating combined output/input ratio CDF plot...")
    plot_cdf(
        ratio_data,
        "Output/Input Token Distribution (All Models)",
        "Output/Input Ratio",
        str(output_path / "input_output_ratio_cdf_combined.png"),
        # logx=False,
    )

    # Create individual plots for each model
    print("\nCreating individual plots for each model...")
    individual_output_path = output_path / "individual"
    individual_output_path.mkdir(exist_ok=True)
    
    plot_cdf_individual(
        input_tokens_data,
        "Input Tokens Distribution",
        "Input Tokens",
        str(individual_output_path),
    )

    plot_cdf_individual(
        output_tokens_data,
        "Output Tokens Distribution",
        "Output Tokens",
        str(individual_output_path),
    )

    plot_cdf_individual(
        ratio_data,
        "Output/Input Token Distribution",
        "Output/Input Ratio",
        str(individual_output_path),
        # logx=False,
    )

    print("\nPlotting complete! Outputs saved to:")
    for filename in (
        "input_tokens_cdf_combined.png",
        "output_tokens_cdf_combined.png",
        "input_output_ratio_cdf_combined.png",
    ):
        print(f"- {output_path / filename}")
    
    print("Individual plots saved to:")
    print(f"- {individual_output_path}")


if __name__ == "__main__":
    setup_plot_style()
    parser = argparse.ArgumentParser(
        description="Plot token distributions for metrics traces"
    )
    parser.add_argument("csv_path", help="Path to the metrics CSV file")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to plot (defaults to top-K by frequency)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for saving the generated plots",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of most common models to plot when --models is not provided",
    )
    arguments = parser.parse_args()
    main(arguments.csv_path, arguments.models, arguments.output_dir, arguments.top_k)
