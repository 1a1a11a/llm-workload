#!/usr/bin/env python3
"""Per-user model diversity analysis

Analyzes how diverse users' model choices are within fixed-size request windows.
Creates two visualizations:
1. Histogram + CDF showing overall distribution of model diversity across all users
2. Box plot showing per-user diversity distributions (top N users by request count)

Input:
    Directory containing per-user CSV files. Each CSV represents one user's requests
    with columns: invocation_id, chute_id, function_name, user_id, started_at, etc.

Output:
    - figures/per_user/model_diversity_per_<window_size>_requests.png
        Histogram and CDF showing how many unique models users access per window
    - figures/per_user/model_diversity_boxplot_per_<window_size>_requests.png
        Box plot for top N users sorted by request count

Usage:
    python3 plot_per_user.py <per_user_csv_dir> [OPTIONS]

Options:
    --output-dir DIR        Directory for saving plots (default: current directory)
    --window-size N         Number of requests per window (default: 1000)
    --top-n N              Number of top users to show in box plot (default: 50)

Examples:
    # Analyze users with <1M requests, using default 1000-request windows
    python3 plot_per_user.py data/metrics_30day/per_user/1000k

    # Analyze all users with 5000-request windows, show top 100
    python3 plot_per_user.py data/metrics_30day/per_user/ --window-size 5000 --top-n 100

    # Custom output directory
    python3 plot_per_user.py data/metrics_30day/per_user/1000k --output-dir results/
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import setup_plot_style


def analyze_user_model_diversity(csv_file: str, window_size: int = 1000) -> List[int]:
    """Analyze model diversity for a user in fixed-size request windows.

    Divides user's requests into consecutive windows of size `window_size` and
    counts the number of unique models (chute_id) used in each window.

    Args:
        csv_file: Path to per-user CSV file containing request history
        window_size: Number of consecutive requests per window (default: 1000)

    Returns:
        List of unique model counts, one entry per window.
        Example: [3, 5, 2] means 3 unique models in window 1, 5 in window 2, etc.
    """
    model_counts = []
    current_window = set()
    request_count = 0

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # chute_id represents the model
            model_id = row.get("chute_id")
            if not model_id:
                continue

            current_window.add(model_id)
            request_count += 1

            if request_count >= window_size:
                model_counts.append(len(current_window))
                current_window = set()
                request_count = 0

    # Include partial window if it has requests
    if request_count > 0:
        model_counts.append(len(current_window))

    return model_counts


def plot_distribution(
    data: List[int],
    output_path: Path,
    window_size: int,
    total_users: int,
    total_windows: int,
) -> None:
    """Plot histogram and CDF of model diversity across all users and windows.

    Creates a figure with two subplots:
    1. Histogram showing frequency distribution with mean/median/p95 lines
    2. CDF showing cumulative probability distribution

    Args:
        data: Flat list of unique model counts from all users and windows
        output_path: Path where the plot will be saved
        window_size: Size of request windows used in analysis
        total_users: Total number of users analyzed
        total_windows: Total number of windows across all users
    """

    if len(data) == 0:
        print("No data to plot")
        return

    data_array = np.array(data)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.hist(data_array, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel(f"Number of Unique Models per {window_size:,} Requests")
    ax1.set_ylabel("Frequency")
    ax1.set_title(
        f"Distribution of Model Diversity\n({total_users:,} users, {total_windows:,} windows)"
    )
    ax1.grid(True, alpha=0.3)

    # Add statistics
    mean_val = np.mean(data_array)
    median_val = np.median(data_array)
    p95_val = np.percentile(data_array, 95)
    ax1.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.1f}",
    )
    ax1.axvline(
        median_val,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.1f}",
    )
    ax1.axvline(
        p95_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"P95: {p95_val:.1f}",
    )
    ax1.legend()

    # CDF
    sorted_data = np.sort(data_array)
    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, y_values, linewidth=2)
    ax2.set_xlabel(f"Number of Unique Models per {window_size:,} Requests")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("CDF of Model Diversity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print("\nStatistics:")
    print(f"  Mean:   {mean_val:.2f} models per {window_size:,} requests")
    print(f"  Median: {median_val:.2f} models per {window_size:,} requests")
    print(f"  P95:    {p95_val:.2f} models per {window_size:,} requests")
    print(f"  Min:    {np.min(data_array)}")
    print(f"  Max:    {np.max(data_array)}")


def plot_boxplot_per_user(
    per_user_data: Dict[str, List[int]],
    output_path: Path,
    window_size: int,
    top_n: int = 50,
) -> None:
    """Plot box plot showing model diversity distribution for top N users.

    Each box represents one user and shows the distribution of unique models
    they used across all their windows. Users are ranked by total number of
    requests (windows), and only the top N are shown.

    Args:
        per_user_data: Dict mapping user_id to list of model counts per window
        output_path: Path where the plot will be saved
        window_size: Size of request windows used in analysis
        top_n: Number of top users to include in the plot (default: 50)

    Note:
        Only includes users with at least 2 windows for meaningful box plots.
    """

    if len(per_user_data) == 0:
        print("No per-user data to plot")
        return

    # Filter users with at least 2 windows for meaningful box plots
    filtered_data = {
        user_id: counts
        for user_id, counts in per_user_data.items()
        if len(counts) >= 2
    }

    if len(filtered_data) == 0:
        print("No users with multiple windows for box plot")
        return

    # Sort users by number of requests (descending)
    user_stats = [
        (user_id, np.median(counts), len(counts))
        for user_id, counts in filtered_data.items()
    ]
    user_stats.sort(key=lambda x: x[2], reverse=True)

    # Select top N users
    top_users = user_stats[:top_n]
    user_ids = [u[0] for u in top_users]

    # Prepare data for box plot
    plot_data = [filtered_data[user_id] for user_id in user_ids]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))

    # Create box plot
    bp = ax.boxplot(plot_data, patch_artist=True, showfliers=True)

    # Customize box plot
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_xlabel("User Rank (Sorted by Number of Requests)")
    ax.set_ylabel(f"Unique Models per {window_size:,} Requests")
    ax.set_title(
        f"Model Diversity Distribution per User\n(Top {len(top_users)} users by number of requests)",
        fontsize=14,
        pad=20,
    )
    ax.grid(True, alpha=0.3, axis="y")

    # Remove x-axis tick labels
    ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Box plot shows top {len(top_users)} users with multiple windows")


def main(csv_dir: str, output_dir: str, window_size: int, top_n_users: int) -> None:
    """Main entry point for per-user model diversity analysis.

    Args:
        csv_dir: Directory containing per-user CSV files (one CSV per user)
        output_dir: Base directory for saving output plots
        window_size: Number of consecutive requests per analysis window
        top_n_users: Number of top users to show in box plot
    """
    csv_path = Path(csv_dir)

    if not csv_path.exists():
        raise RuntimeError(f"Directory not found: {csv_dir}")

    # Find all CSV files
    csv_files = list(csv_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found in {csv_dir}")

    print(f"Processing {len(csv_files):,} user files...")
    print(f"Window size: {window_size:,} requests\n")

    all_model_counts = []
    per_user_data = {}
    users_processed = 0

    for csv_file in csv_files:
        user_id = csv_file.stem
        model_counts = analyze_user_model_diversity(str(csv_file), window_size)

        if len(model_counts) > 0:
            all_model_counts.extend(model_counts)
            per_user_data[user_id] = model_counts
            users_processed += 1

            if users_processed % 100 == 0:
                print(
                    f"  Processed {users_processed:,}/{len(csv_files):,} users...",
                    flush=True,
                )

    print(
        f"\nProcessed {users_processed:,} users with {len(all_model_counts):,} total windows"
    )

    # Create output directory
    output_path = Path(output_dir) / "figures" / "per_user_model_diversity"
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot the distribution
    print("\nCreating distribution plots...")
    plot_file = output_path / f"model_diversity_per_{window_size}_requests.png"
    plot_distribution(
        all_model_counts, plot_file, window_size, users_processed, len(all_model_counts)
    )

    # Plot box plot per user
    print("\nCreating per-user box plot...")
    boxplot_file = (
        output_path / f"model_diversity_boxplot_per_{window_size}_requests.png"
    )
    plot_boxplot_per_user(per_user_data, boxplot_file, window_size, top_n_users)


if __name__ == "__main__":
    setup_plot_style()
    parser = argparse.ArgumentParser(
        description="Plot distribution of model diversity per user"
    )
    parser.add_argument("csv_dir", help="Directory containing per-user CSV files")
    parser.add_argument("--output-dir", default=".", help="Directory for saving plots")
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Number of requests per window (default: 1000)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top users to show in box plot (default: 50)",
    )
    args = parser.parse_args()
    main(args.csv_dir, args.output_dir, args.window_size, args.top_n)

