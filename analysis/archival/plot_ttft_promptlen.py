#!/usr/bin/env python3
"""
TTFT vs Input Tokens Analysis Script (plot_ttft_promptlen.py)

This script analyzes the relationship between Time to First Token (TTFT) and input tokens
by creating scatter plots and basic statistical analysis with log-log scale visualization.

Usage:
    python plot_ttft_promptlen.py /path/to/metrics.csv
    python plot_ttft_promptlen.py ../data/metrics_30day/DeepSeek-R1.csv --hexbin
    python plot_ttft_promptlen.py ../data/metrics_30day/DeepSeek-R1.csv --max-input-tokens 800000
    python plot_ttft_promptlen.py ../data/metrics_30day/DeepSeek-R1.csv --max-ttft 20.0

The script expects CSV files with columns:
- invocation_id, chute_id, function_name, user_id, started_at, completed_at
- input_tokens, output_tokens, ttft

Rows with missing ttft values, input_tokens > max_input_tokens, or ttft > max_ttft are automatically filtered out.
Both axes use log scale for better visualization of wide ranges.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style

setup_plot_style()


def load_and_clean_data(filepath, max_input_tokens=1000000, max_ttft=10.0):
    """
    Load CSV data and clean it by removing rows with missing ttft values
    and filtering out rows with input_tokens > max_input_tokens or ttft > max_ttft.

    Args:
        filepath (str): Path to the CSV file
        max_input_tokens (int): Maximum input tokens to include (default: 1,000,000)
        max_ttft (float): Maximum TTFT to include (default: 10.0 seconds)

    Returns:
        pd.DataFrame: Cleaned dataframe with valid ttft values, input_tokens <= max_input_tokens, and ttft <= max_ttft
    """
    # Read CSV
    df = pd.read_csv(filepath)

    # Convert ttft to numeric, handling empty strings
    df["ttft"] = pd.to_numeric(df["ttft"], errors="coerce")

    # Filter out rows with missing ttft
    df_clean = df.dropna(subset=["ttft"])

    # Filter out rows with input_tokens > max_input_tokens
    initial_clean_count = len(df_clean)
    df_clean = df_clean[df_clean["input_tokens"] <= max_input_tokens]
    input_tokens_filtered = initial_clean_count - len(df_clean)

    # Filter out rows with ttft > max_ttft
    df_clean = df_clean[df_clean["ttft"] <= max_ttft]
    ttft_filtered = initial_clean_count - input_tokens_filtered - len(df_clean)

    print(f"Loaded {len(df)} rows, {len(df_clean)} rows with valid ttft values")
    print(f"Removed {len(df) - initial_clean_count} rows with missing ttft")
    if input_tokens_filtered > 0:
        print(f"Removed {input_tokens_filtered} rows with input_tokens > {max_input_tokens:,}")
    if ttft_filtered > 0:
        print(f"Removed {ttft_filtered} rows with ttft > {max_ttft:.1f}s")

    return df_clean


def create_scatter_plot(df, title="TTFT vs Input Tokens", save_path=None):
    """
    Create a scatter plot of TTFT vs input tokens.

    Args:
        df (pd.DataFrame): DataFrame with 'input_tokens' and 'ttft' columns
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    scatter = plt.scatter(
        df["input_tokens"],
        df["ttft"],
        alpha=0.6,
        s=50,
        c="blue",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add labels and title
    plt.xlabel("Input Tokens (log scale)", fontsize=12)
    plt.ylabel("Time to First Token (TTFT, log scale)", fontsize=12)
    plt.title(title, fontsize=14, pad=20)

    # Use log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Add grid
    plt.grid(True, alpha=0.3)

    # Format axes (skip axis formatting for log scale)

    # Add statistics text
    stats_text = f"""
    n = {len(df):,}
    Mean TTFT: {df["ttft"].mean():.3f}s
    Median TTFT: {df["ttft"].median():.3f}s
    Mean Input Tokens: {df["input_tokens"].mean():.0f}
    Median Input Tokens: {df["input_tokens"].median():.0f}
    """
    plt.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def analyze_correlation(df):
    """
    Calculate and print correlation between TTFT and input tokens.

    Args:
        df (pd.DataFrame): DataFrame with 'input_tokens' and 'ttft' columns
    """
    if len(df) < 2:
        print("Insufficient data points for correlation analysis (need at least 2)")
        return

    correlation = df["input_tokens"].corr(df["ttft"])
    print(f"Pearson correlation between input_tokens and ttft: {correlation:.3f}")

    # Also calculate correlation for log-transformed data
    df_log = df.copy()
    df_log["input_tokens_log"] = np.log1p(df_log["input_tokens"])
    df_log["ttft_log"] = np.log1p(df_log["ttft"])
    log_correlation = df_log["input_tokens_log"].corr(df_log["ttft_log"])
    print(f"Pearson correlation (log-transformed): {log_correlation:.3f}")


def create_hexbin_plot(df, title="TTFT vs Input Tokens (Hexbin)", save_path=None):
    """
    Create a hexbin plot for denser data visualization.

    Args:
        df (pd.DataFrame): DataFrame with 'input_tokens' and 'ttft' columns
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Create hexbin plot
    hb = plt.hexbin(df["input_tokens"], df["ttft"], gridsize=50, cmap="Blues", mincnt=1)

    # Add colorbar
    cb = plt.colorbar(hb)
    cb.set_label("Count")

    # Add labels and title
    plt.xlabel("Input Tokens (log scale)", fontsize=12)
    plt.ylabel("Time to First Token (TTFT, log scale)", fontsize=12)
    plt.title(title, fontsize=14, pad=20)

    # Use log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Add grid
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Hexbin plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze TTFT vs Input Tokens")
    parser.add_argument(
        "file", type=str, help="Path to CSV file"
    )
    parser.add_argument("--output", "-o", type=str, help="Output directory for plots")
    parser.add_argument(
        "--hexbin", action="store_true", help="Also create hexbin plot for dense data"
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1000000,
        help="Maximum input tokens to include (default: 1,000,000)",
    )
    parser.add_argument(
        "--max-ttft",
        type=float,
        default=120.0,
        help="Maximum TTFT to include in seconds (default: 120.0)",
    )

    args = parser.parse_args()

    # Load data
    filepath = args.file
    df_clean = load_and_clean_data(filepath, args.max_input_tokens, args.max_ttft)

    if len(df_clean) == 0:
        print("No valid data found with ttft values. Exiting.")
        return

    # Basic statistics
    print("\nBasic Statistics:")
    print(df_clean[["input_tokens", "ttft"]].describe())

    # Correlation analysis
    print("\nCorrelation Analysis:")
    analyze_correlation(df_clean)

    # Create plots
    output_dir = args.output or "."
    Path(output_dir).mkdir(exist_ok=True)

    # Scatter plot
    title = f"TTFT vs Input Tokens (n={len(df_clean)})"
    if filepath:
        title += f"\n{Path(filepath).name}"

    scatter_path = Path(output_dir) / "ttft_vs_input_scatter.png"
    create_scatter_plot(df_clean, title=title, save_path=str(scatter_path))

    # Hexbin plot if requested or if data is dense
    if args.hexbin or len(df_clean) > 1000:
        hexbin_path = Path(output_dir) / "ttft_vs_input_hexbin.png"
        create_hexbin_plot(df_clean, title=title, save_path=str(hexbin_path))

if __name__ == "__main__":
    main()
