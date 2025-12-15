#!/usr/bin/env python3
"""Token usage distribution analysis script

Plot token distributions for metrics traces using data_loader.
"""

import argparse
import os
import sys
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


def main(
    csv_files: List[str], model_names: Optional[List[str]], output_dir: str
) -> None:
    if len(csv_files) == 0:
        raise RuntimeError("No CSV files provided")
    
    # Prepare model names - use provided names or derive from filenames
    if model_names is None:
        model_names = []
        for csv_file in csv_files:
            base_name = os.path.basename(csv_file)
            model_name = os.path.splitext(base_name)[0]
            model_names.append(model_name)
    elif len(model_names) != len(csv_files):
        raise RuntimeError(f"Number of model names ({len(model_names)}) must match number of CSV files ({len(csv_files)})")

    input_tokens_data: Dict[str, np.ndarray] = {}
    output_tokens_data: Dict[str, np.ndarray] = {}
    ratio_data: Dict[str, np.ndarray] = {}

    total_records = 0
    for csv_file, model_name in zip(csv_files, model_names):
        records = _load_records(csv_file)
        if records.empty:
            print(f"  -> No records found in {csv_file}")
            continue

        total_records += len(records)
        
        # Extract all token data from this file (treating the entire file as one model)
        input_tokens: List[int] = []
        output_tokens: List[int] = []
        
        for record in records.itertuples():
            input_tokens.append(record.input_tokens)
            output_tokens.append(record.output_tokens)

        if not input_tokens:
            print(f"  -> No token data found in {csv_file}")
            continue

        input_tokens_data[model_name] = np.array(input_tokens)
        output_tokens_data[model_name] = np.array(output_tokens)
        
        # Calculate output/input ratio for positive input tokens only
        positive_mask = np.array(input_tokens) > 0
        if np.any(positive_mask):
            ratio_data[model_name] = np.array(output_tokens)[positive_mask] / np.array(input_tokens)[positive_mask]
        
        print(f"  -> {model_name}: {len(input_tokens)} samples from {os.path.basename(csv_file)}")

    if not input_tokens_data:
        raise RuntimeError("No token data available from any file")

    print(f"Loaded {total_records:,} total records from {len(csv_files)} file(s)")
    print("Models to plot: " + ", ".join(input_tokens_data.keys()))

    # Create output directory based on the first file name or a combined name
    if len(csv_files) == 1:
        trace_name = os.path.basename(csv_files[0]).split(".")[0]
    else:
        trace_name = "combined_models"
    
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
        description="Plot token distributions for metrics traces (each file represents one model)"
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Path to one or more metrics CSV files (each file represents one model)"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for saving the generated plots",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Model names corresponding to each CSV file (defaults to filename without extension)",
    )
    arguments = parser.parse_args()
    main(arguments.csv_files, arguments.model_names, arguments.output_dir)
