#!/usr/bin/env python3
"""Token usage distribution analysis script

Plot token distributions for metrics traces.
Reads CSV files line-by-line in parallel to minimize memory usage.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import setup_plot_style


def _read_tokens_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read input_tokens and output_tokens from CSV line by line.
    
    Returns:
        Tuple of (input_tokens, output_tokens) as numpy arrays
    """
    print(f"    Counting valid records in {csv_path}...", flush=True)
    
    # First pass: count valid records
    valid_count = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('input_tokens') and row.get('output_tokens'):
                valid_count += 1
    
    if valid_count == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    print(f"    Reading {valid_count:,} records from {csv_path}...", flush=True)
    
    # Second pass: pre-allocate arrays and fill
    input_tokens = np.empty(valid_count, dtype=np.int32)
    output_tokens = np.empty(valid_count, dtype=np.int32)
    
    idx = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('input_tokens') or not row.get('output_tokens'):
                continue
            input_tokens[idx] = int(row['input_tokens'])
            output_tokens[idx] = int(row['output_tokens'])
            idx += 1
    
    print(f"    Done reading {csv_path}", flush=True)
    return input_tokens, output_tokens


def _process_single_csv(args: Tuple[str, str]) -> Tuple[str, np.ndarray, np.ndarray]:
    """Process a single CSV file and return model_name, input_tokens, output_tokens.
    
    This function is used for parallel processing.
    """
    csv_file, model_name = args
    input_tokens, output_tokens = _read_tokens_from_csv(csv_file)
    return model_name, input_tokens, output_tokens


def _plot_single_model(args: Tuple[str, np.ndarray, str, str, str, bool]) -> str:
    """Plot a single model's CDF. Used for parallel processing.
    
    Returns:
        Path to the saved plot
    """
    model_name, data, title, xlabel, output_dir, logx = args
    
    if data is None or len(data) == 0:
        return ""
    
    plt.figure(figsize=(12, 8))
    sorted_data = np.sort(data)
    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    plt.plot(
        sorted_data,
        y_values,
        label=f"{model_name} (n={len(data):,})",
        linewidth=3,
        alpha=0.8,
        linestyle="-",
        color=plt.cm.tab10.colors[0],
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
    return output_file


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
    """Plot individual CDFs for each model in separate figures (parallelized)."""
    # Prepare arguments for parallel processing
    plot_args = [
        (model_name, data, title, xlabel, output_dir, logx)
        for model_name, data in data_dict.items()
        if data is not None and len(data) > 0
    ]
    
    if not plot_args:
        return
    
    # Plot in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_plot_single_model, args) for args in plot_args]
        for future in as_completed(futures):
            output_file = future.result()
            if output_file:
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
    
    # Read files in parallel
    print(f"Reading {len(csv_files)} file(s) in parallel...")
    with ProcessPoolExecutor() as executor:
        # Submit all CSV reading jobs
        future_to_csv = {
            executor.submit(_process_single_csv, (csv_file, model_name)): csv_file
            for csv_file, model_name in zip(csv_files, model_names)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_csv):
            csv_file = future_to_csv[future]
            model_name, input_tokens, output_tokens = future.result()
            
            if len(input_tokens) == 0:
                print(f"  -> No token data found in {csv_file}")
                continue

            total_records += len(input_tokens)
            input_tokens_data[model_name] = input_tokens
            output_tokens_data[model_name] = output_tokens
            
            # Calculate output/input ratio for positive input tokens only
            positive_mask = input_tokens > 0
            if np.any(positive_mask):
                ratio_data[model_name] = output_tokens[positive_mask] / input_tokens[positive_mask]
            
            print(f"  -> {model_name}: {len(input_tokens):,} samples from {os.path.basename(csv_file)}")

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
