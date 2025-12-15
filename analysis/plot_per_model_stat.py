#!/usr/bin/env python3
"""Per-model statistics analysis and visualization

Analyzes statistics for each model from per-model CSV traces and creates
three bar plots showing different metrics across models.

Input:
    Directory containing per-model CSV files. Each CSV represents one model's requests
    with columns: invocation_id, user_id, input_tokens, output_tokens, etc.

Output:
    - figures/per_model_stats/requests_per_model.png
        Bar plot showing total number of requests for each model
    - figures/per_model_stats/tokens_per_model.png
        Stacked bar plot showing input and output tokens for each model
    - figures/per_model_stats/users_per_model.png
        Bar plot showing number of unique users for each model

Usage:
    python3 plot_per_model_stat.py <per_model_csv_dir> [OPTIONS]

Options:
    --output-dir DIR        Directory for saving plots (default: current directory)
    --top-n N              Show only top N models by request count (default: all)
    --sort-by {requests,tokens,users}
                           Sort models by this metric (default: requests)
    --force-recompute      Recompute statistics ignoring cache

Features:
    - Caches computed statistics for faster subsequent runs
    - Cache file: model_stats_cache.csv in output directory

Examples:
    # Analyze all models in directory (uses cache if available)
    python3 plot_per_model_stat.py data/metrics_30day/per_model/100k

    # Force recompute statistics (ignore cache)
    python3 plot_per_model_stat.py data/metrics_30day/per_model/100k --force-recompute

    # Show top 20 models sorted by request count
    python3 plot_per_model_stat.py data/metrics_30day/per_model/100k --top-n 20

    # Sort by number of users
    python3 plot_per_model_stat.py data/metrics_30day/per_model/100k --sort-by users

    # Custom output directory
    python3 plot_per_model_stat.py data/metrics_30day/per_model/100k --output-dir results/
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot import setup_plot_style


def analyze_model_csv(csv_path: str) -> Tuple[int, int, int, int]:
    """Analyze a single model CSV file and extract statistics.

    Args:
        csv_path: Path to model CSV file

    Returns:
        Tuple of (num_requests, input_tokens, output_tokens, num_users)
    """
    num_requests = 0
    total_input_tokens = 0
    total_output_tokens = 0
    unique_users = set()

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_requests += 1

            # Accumulate tokens
            if row.get("input_tokens"):
                total_input_tokens += int(row["input_tokens"])
            if row.get("output_tokens"):
                total_output_tokens += int(row["output_tokens"])

            # Track unique users
            if row.get("user_id"):
                unique_users.add(row["user_id"])

    return num_requests, total_input_tokens, total_output_tokens, len(unique_users)


def analyze_model_csv_wrapper(csv_file_path: Path) -> Tuple[str, Tuple[int, int, int, int]]:
    """Wrapper function for parallel processing of model CSV files.

    Args:
        csv_file_path: Path object to model CSV file

    Returns:
        Tuple of (model_name, (num_requests, input_tokens, output_tokens, num_users))
    """
    model_name = csv_file_path.stem
    stats = analyze_model_csv(str(csv_file_path))
    return model_name, stats


def save_model_stats_cache(
    model_stats: Dict[str, Tuple[int, int, int, int]], cache_file: str
) -> None:
    """Save model statistics to cache file.

    Args:
        model_stats: Dict mapping model_name to (requests, input_tokens, output_tokens, users)
        cache_file: Path to cache file
    """
    df = pd.DataFrame(
        [
            {
                "model_name": model,
                "requests": stats[0],
                "input_tokens": stats[1],
                "output_tokens": stats[2],
                "users": stats[3],
            }
            for model, stats in model_stats.items()
        ]
    )
    df.to_csv(cache_file, index=False)
    print(f"Statistics cached to {cache_file}")


def load_model_stats_cache(cache_file: str) -> Optional[Dict[str, Tuple[int, int, int, int]]]:
    """Load model statistics from cache file.

    Args:
        cache_file: Path to cache file

    Returns:
        Dict mapping model_name to (requests, input_tokens, output_tokens, users)
        or None if cache doesn't exist or is invalid
    """
    if not os.path.exists(cache_file):
        return None

    try:
        df = pd.read_csv(cache_file)
        model_stats = {
            row["model_name"]: (
                int(row["requests"]),
                int(row["input_tokens"]),
                int(row["output_tokens"]),
                int(row["users"]),
            )
            for _, row in df.iterrows()
        }
        print(f"Loaded cached statistics from {cache_file}")
        return model_stats
    except Exception as e:
        print(f"Warning: Could not load cache ({e}), will recompute")
        return None


def plot_requests_per_model(
    model_stats: Dict[str, Tuple[int, int, int, int]], output_path: Path
) -> None:
    """Plot bar chart of request counts per model.

    Args:
        model_stats: Dict mapping model_name to (requests, input_tokens, output_tokens, users)
        output_path: Path where plot will be saved
    """
    models = list(model_stats.keys())
    requests = [stats[0] for stats in model_stats.values()]

    plt.figure(figsize=(max(12, len(models) * 0.5), 16))
    plt.bar(range(len(models)), requests, color="steelblue", alpha=0.8)

    plt.xlabel("Model")
    plt.ylabel("Number of Requests")
    plt.title("Requests per Model")
    plt.xticks(range(len(models)), models, rotation=90, ha="center")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_tokens_per_model(
    model_stats: Dict[str, Tuple[int, int, int, int]], output_path: Path
) -> None:
    """Plot stacked bar chart of input/output tokens per model.

    Args:
        model_stats: Dict mapping model_name to (requests, input_tokens, output_tokens, users)
        output_path: Path where plot will be saved
    """
    models = list(model_stats.keys())
    input_tokens = [stats[1] for stats in model_stats.values()]
    output_tokens = [stats[2] for stats in model_stats.values()]

    x = np.arange(len(models))
    width = 0.8

    plt.figure(figsize=(max(12, len(models) * 0.5), 16))

    # Create stacked bars
    plt.bar(x, input_tokens, width, label="Input Tokens", color="steelblue", alpha=0.8)
    plt.bar(
        x,
        output_tokens,
        width,
        bottom=input_tokens,
        label="Output Tokens",
        color="coral",
        alpha=0.8,
    )

    plt.xlabel("Model")
    plt.ylabel("Number of Tokens")
    plt.title("Tokens per Model (Stacked: Input + Output)")
    plt.xticks(x, models, rotation=90, ha="center")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_users_per_model(
    model_stats: Dict[str, Tuple[int, int, int, int]], output_path: Path
) -> None:
    """Plot bar chart of unique user counts per model.

    Args:
        model_stats: Dict mapping model_name to (requests, input_tokens, output_tokens, users)
        output_path: Path where plot will be saved
    """
    models = list(model_stats.keys())
    users = [stats[3] for stats in model_stats.values()]

    plt.figure(figsize=(max(12, len(models) * 0.5), 16))
    plt.bar(range(len(models)), users, color="forestgreen", alpha=0.8)

    plt.xlabel("Model")
    plt.ylabel("Number of Unique Users")
    plt.title("Users per Model")
    plt.xticks(range(len(models)), models, rotation=90, ha="center")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main(csv_dir: str, output_dir: str, top_n: int, sort_by: str, force_recompute: bool) -> None:
    """Main entry point for per-model statistics analysis.

    Args:
        csv_dir: Directory containing per-model CSV files
        output_dir: Base directory for saving output plots
        top_n: Number of top models to show (None for all)
        sort_by: Metric to sort models by ('requests', 'tokens', 'users')
        force_recompute: Whether to recompute statistics ignoring cache
    """
    csv_path = Path(csv_dir)

    if not csv_path.exists():
        raise RuntimeError(f"Directory not found: {csv_dir}")

    # Create output directory and cache file path
    output_path = Path(output_dir) / "figures" / "per_model_stats"
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = str(output_path / "model_stats_cache.csv")

    # Try to load from cache
    model_stats = None
    if not force_recompute:
        model_stats = load_model_stats_cache(cache_file)

    # Compute statistics if not loaded from cache
    if model_stats is None:
        # Find all CSV files
        csv_files = list(csv_path.glob("*.csv"))

        if len(csv_files) == 0:
            raise RuntimeError(f"No CSV files found in {csv_dir}")

        print(f"Processing {len(csv_files):,} model files in parallel...")

        # Analyze each model in parallel
        model_stats = {}
        with ProcessPoolExecutor() as executor:
            # Submit all jobs
            futures = [
                executor.submit(analyze_model_csv_wrapper, csv_file) 
                for csv_file in csv_files
            ]
            
            # Process results as they complete
            for future in as_completed(futures):
                model_name, stats = future.result()
                num_requests, input_tokens, output_tokens, num_users = stats
                
                if num_requests > 0:  # Only include models with data
                    model_stats[model_name] = stats
                    print(
                        f"  {model_name}: {num_requests:,} requests, "
                        f"{input_tokens + output_tokens:,} tokens, {num_users:,} users"
                    )

        if not model_stats:
            raise RuntimeError("No valid model data found")

        # Save to cache
        save_model_stats_cache(model_stats, cache_file)

    print(f"\nAnalyzed {len(model_stats)} models with data")

    # Sort models by specified metric
    if sort_by == "requests":
        sort_idx = 0
        sort_name = "request count"
    elif sort_by == "tokens":
        # Sort by total tokens (input + output)
        model_stats = {
            k: v for k, v in sorted(
                model_stats.items(),
                key=lambda x: x[1][1] + x[1][2],
                reverse=True
            )
        }
        print("Sorted by total tokens (descending)")
    elif sort_by == "users":
        sort_idx = 3
        sort_name = "user count"
    else:
        raise ValueError(f"Invalid sort_by value: {sort_by}")

    if sort_by != "tokens":
        model_stats = {
            k: v
            for k, v in sorted(
                model_stats.items(), key=lambda x: x[1][sort_idx], reverse=True
            )
        }
        print(f"Sorted by {sort_name} (descending)")

    # Select top N if specified
    if top_n and top_n < len(model_stats):
        model_stats = dict(list(model_stats.items())[:top_n])
        print(f"Showing top {top_n} models")

    # Generate plots
    print("\nGenerating plots...")
    plot_requests_per_model(model_stats, output_path / "requests_per_model.png")
    plot_tokens_per_model(model_stats, output_path / "tokens_per_model.png")
    plot_users_per_model(model_stats, output_path / "users_per_model.png")

    print("\nAll plots saved to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    setup_plot_style()
    parser = argparse.ArgumentParser(
        description="Plot per-model statistics from per-model CSV traces"
    )
    parser.add_argument("csv_dir", help="Directory containing per-model CSV files")
    parser.add_argument("--output-dir", default=".", help="Directory for saving plots")
    parser.add_argument(
        "--top-n", type=int, default=20, help="Show only top N models (default: 20)"
    )
    parser.add_argument(
        "--sort-by",
        choices=["requests", "tokens", "users"],
        default="tokens",
        help="Sort models by this metric (default: tokens)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute statistics ignoring cache",
    )
    args = parser.parse_args()
    main(args.csv_dir, args.output_dir, args.top_n, args.sort_by, args.force_recompute)

