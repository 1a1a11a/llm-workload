#!/usr/bin/env python3
"""
Inter-Arrival Time Analysis Script
This requires the per-model CSV files. The input can be one or more files.

This script analyzes the inter-arrival times of requests across all users
by plotting the time differences between consecutive requests as a CDF, boxplot by hour,
daily boxplot, and correlation between consecutive inter-arrival times.
When multiple traces are provided, all-user and per-user inter-arrival times for
each trace are processed in parallel and plotted side-by-side in a single grouped figure.

Usage:
    python plot_per_model_arrival_time.py /path/to/metrics.csv
    python plot_per_model_arrival_time.py ../data/metrics_30day/DeepSeek-R1.csv --per-user
    python plot_per_model_arrival_time.py model_a.csv model_b.csv --per-user
    python plot_per_model_arrival_time.py model_a.csv model_b.csv --workers 4

The --per-user flag calculates inter-arrival times only within each user (not across users).
Use --workers to control the parallelism when loading multiple traces.

Creates and saves CDF, boxplot, daily boxplot, correlation plots, and probability heatmap automatically (no display).
Only the inter-arrival duration frequency maps are cached (for CDF reuse); full dataframes are reloaded each run to avoid stale results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style

setup_plot_style()


def infer_output_dir(sample_file, per_user, override_dir=None):
    """
    Infer an output directory based on the sample file path and mode.

    Args:
        sample_file (str): A representative file path
        per_user (bool): Whether we are plotting per-user inter-arrival times
        override_dir (str|None): User-specified output directory

    Returns:
        Path: Directory to write figures to
    """
    if override_dir:
        return Path(override_dir)

    base_dir = (
        "figures/per_model_arrival_time_per_user/"
        if per_user
        else "figures/per_model_arrival_time/"
    )
    if "/1k/" in sample_file:
        return Path(base_dir) / "1k"
    if "/10k/" in sample_file:
        return Path(base_dir) / "10k"
    if "/100k/" in sample_file:
        return Path(base_dir) / "100k"
    if "/1000k/" in sample_file:
        return Path(base_dir) / "1000k"
    if "/large/" in sample_file:
        return Path(base_dir) / "large"
    return Path(base_dir)


def get_duration_map_cache_path(output_dir, trace_name, per_user=False):
    """Return cache path for duration frequency map."""
    suffix = "per_user" if per_user else "all_users"
    return Path(output_dir) / "pkl" / f"{trace_name}_duration_freq_map_{suffix}.pkl"


def build_duration_frequency_map(
    df: pd.DataFrame, decimals: int = 6
) -> Dict[float, int]:
    """Create frequency map of inter-arrival durations rounded to given decimals."""
    if "inter_arrival_time" not in df.columns or len(df) == 0:
        return {}

    rounded = df["inter_arrival_time"].round(decimals)
    freq_series = rounded.value_counts().sort_index()
    freq_map = {float(duration): int(count) for duration, count in freq_series.items()}
    print(
        f"Built duration frequency map with {len(freq_map):,} unique durations "
        f"(rounded to {decimals} decimals)."
    )
    return freq_map


def load_or_process_trace(
    file_path, per_user, output_dir
) -> Tuple[pd.DataFrame, Dict[float, int]]:
    """
    Load CSV data, compute inter-arrival statistics, and cache only the duration map.

    Args:
        file_path (str): Original CSV file path
        per_user (bool): Whether to calculate inter-arrival times per user
        output_dir (Path): Directory to store cached data

    Returns:
        Tuple[pd.DataFrame, Dict[float, int]]: Processed dataframe with inter-arrival
        information and the cached duration frequency map for plotting.
    """
    trace_name = Path(file_path).stem
    duration_cache_path = get_duration_map_cache_path(output_dir, trace_name, per_user)

    # Always reload and process CSV to avoid stale cached dataframes
    df = load_and_process_data(file_path, per_user=per_user)

    cache_is_fresh = False
    if duration_cache_path.exists():
        try:
            cache_is_fresh = (
                duration_cache_path.stat().st_mtime
                >= Path(file_path).stat().st_mtime
            )
        except FileNotFoundError:
            cache_is_fresh = False

    if cache_is_fresh:
        print(f"Loading cached duration frequency map from {duration_cache_path}")
        duration_freq_map = pd.read_pickle(duration_cache_path)
    else:
        duration_freq_map = build_duration_frequency_map(df)
        duration_cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(duration_freq_map, duration_cache_path)
        print(f"Saved duration frequency map to {duration_cache_path}")

    if duration_freq_map is None:
        duration_freq_map = {}

    return df, duration_freq_map


def load_and_process_data(filepath, per_user=False):
    """
    Load CSV data and process timestamps for inter-arrival time analysis.

    Args:
        filepath (str): Path to the CSV file
        per_user (bool): If True, calculate inter-arrival times per user

    Returns:
        pd.DataFrame: Processed dataframe with inter-arrival times
    """
    # Read CSV
    df = pd.read_csv(filepath)

    # Determine which timestamp column is available
    timestamp_col = None
    if "started_at" in df.columns:
        timestamp_col = "started_at"
    elif "timestamp" in df.columns:
        timestamp_col = "timestamp"
    else:
        raise ValueError(
            "No timestamp column found. Expected 'started_at' or 'timestamp' column."
        )

    # Convert timestamp to datetime
    df["started_at"] = pd.to_datetime(df[timestamp_col], format="mixed")

    # Sort by started_at to ensure chronological order
    df = df.sort_values("started_at").reset_index(drop=True)

    # Calculate inter-arrival times (in seconds)
    if per_user:
        # Calculate inter-arrival times per user
        if "user_id" not in df.columns:
            raise ValueError(
                "No user_id column found. Cannot calculate per-user inter-arrival times."
            )

        # Sort by user_id and started_at
        df = df.sort_values(["user_id", "started_at"]).reset_index(drop=True)

        # Calculate inter-arrival time within each user group
        df["inter_arrival_time"] = (
            df.groupby("user_id")["started_at"].diff().dt.total_seconds()
        )

        print(f"Loaded {len(df)} rows from {df['user_id'].nunique()} users")
    else:
        # Calculate inter-arrival times across all users
        df["inter_arrival_time"] = df["started_at"].diff().dt.total_seconds()
        print(f"Loaded {len(df)} rows")

    # Extract hour and day for grouping
    df["hour"] = df["started_at"].dt.hour
    df["date"] = df["started_at"].dt.date

    # Remove rows with NaN inter_arrival_time
    rows_before = len(df)
    df = df.dropna(subset=["inter_arrival_time"])
    rows_removed = rows_before - len(df)

    # Enforce a minimum inter-arrival time to avoid log-scale issues downstream
    df["inter_arrival_time"] = df["inter_arrival_time"].clip(lower=1e-3)

    print(
        f"Processed {len(df)} inter-arrival times (removed {rows_removed} NaN values)"
    )
    print(f"Time range: {df['started_at'].min()} to {df['started_at'].max()}")

    return df


def plot_inter_arrival_times(freq_map, title="Inter-Arrival Times", save_path=None):
    """
    Plot the inter-arrival times as a Cumulative Distribution Function (CDF)
    using a precomputed duration frequency map {duration: count}.

    Args:
        freq_map (Dict[float, int]): Mapping of duration to observation count
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    if not freq_map:
        print("No inter-arrival data available for CDF plot.")
        return

    plt.figure(figsize=(12, 10))

    sorted_items = sorted(freq_map.items())
    values = np.array([float(duration) for duration, _ in sorted_items])
    counts = np.array([int(count) for _, count in sorted_items])
    cumulative_counts = np.cumsum(counts)
    total = cumulative_counts[-1]
    if total == 0:
        print("Frequency map contained zero counts; skipping CDF plot.")
        return

    yvals = cumulative_counts / float(total)

    # Create CDF plot
    plt.plot(values, yvals, "b-", linewidth=2, alpha=0.8)

    # Calculate summary statistics using frequency map
    mean_val = float(np.sum(values * counts) / total)
    median_idx = np.searchsorted(cumulative_counts, total / 2)
    median_val = float(values[min(median_idx, len(values) - 1)])
    variance = float(np.sum(((values - mean_val) ** 2) * counts) / total)
    std_val = float(np.sqrt(variance))
    min_val = float(values[0])
    max_val = float(values[-1])

    # Add labels and title
    plt.xlabel("Inter-Arrival Time (seconds)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"{title} - CDF\n(n={total:,})", pad=20)

    # Use log scale for x-axis since inter-arrival times span multiple orders of magnitude
    plt.xscale("log")

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""
    Mean: {mean_val:,.3f}s
    Median: {median_val:,.3f}s
    Std: {std_val:,.3f}s
    Min: {min_val:,.6f}s
    Max: {max_val:,.3f}s
    """
    plt.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def plot_hourly_boxplot(df, title="Inter-Arrival Times by Hour", save_path=None):
    """
    Plot inter-arrival times as a boxplot grouped by hour of day.

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' and 'hour' columns
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(14, 10))

    # Group data by hour and prepare for boxplot
    hourly_groups = df.groupby("hour")["inter_arrival_time"]

    # Only include hours that have data
    hourly_data = []
    hour_labels = []
    positions = []

    for hour in range(24):
        if hour in hourly_groups.groups:
            hourly_data.append(hourly_groups.get_group(hour).values)
            hour_labels.append(f"{hour:02d}:00")
            positions.append(hour)

    # Create boxplot with custom positions and show means
    bp = plt.boxplot(
        hourly_data,
        patch_artist=True,
        positions=positions,
        showmeans=True,
    )

    # Set custom tick labels
    plt.xticks(positions, hour_labels, rotation=90)

    # Customize boxplot colors
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    for whisker in bp["whiskers"]:
        whisker.set(color="blue", linewidth=1.5)

    for cap in bp["caps"]:
        cap.set(color="blue", linewidth=1.5)

    for median in bp["medians"]:
        median.set(color="red", linewidth=2)

    for flier in bp["fliers"]:
        flier.set(marker="o", color="red", alpha=0.5)

    # Customize mean markers as triangles so plots consistently show the mean shape
    for mean in bp["means"]:
        mean.set(
            color="green",
            linewidth=0,
            linestyle="None",
            marker="^",
            markersize=10,
            markerfacecolor="green",
            markeredgecolor="black",
        )
        mean.set_alpha(0.9)

    # Add means label to legend matching triangle styling
    plt.plot([], [], color="green", marker="^", linestyle="None", markersize=10, label="Mean")
    plt.legend(loc="upper right")

    # Add labels and title
    plt.xlabel("Hour of Day")
    plt.ylabel("Inter-Arrival Time (seconds)")
    plt.title(f"{title}\n(n={len(df):,})", pad=20)

    # Use log scale for y-axis
    plt.yscale("log")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Add grid
    plt.grid(True, alpha=0.3, axis="y")

    # Add statistics text
    hours_with_data = df["hour"].nunique()
    stats_text = f"""
    Hours with data: {hours_with_data}
    Total requests: {len(df) + 1}
    """
    plt.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Boxplot saved to {save_path}")


def plot_daily_boxplot(df, title="Inter-Arrival Times by Day", save_path=None):
    """
    Plot inter-arrival times as a boxplot grouped by day.

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' and 'date' columns
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(16, 10))

    # Group data by date and prepare for boxplot
    daily_groups = df.groupby("date")["inter_arrival_time"]

    # Sort dates chronologically
    sorted_dates = sorted(daily_groups.groups.keys())

    # Only include dates that have data
    daily_data = []
    date_labels = []
    positions = []

    for i, date in enumerate(sorted_dates):
        daily_data.append(daily_groups.get_group(date).values)
        date_labels.append(str(date))
        positions.append(i)

    # Create boxplot with custom positions and show means
    bp = plt.boxplot(
        daily_data,
        patch_artist=True,
        positions=positions,
        showmeans=True,
    )

    # Set custom tick labels - show every nth date if too many
    n_dates = len(date_labels)
    step = max(1, n_dates // 10)  # Show approximately 10 x-axis labels
    tick_positions = positions[::step]
    tick_labels = date_labels[::step]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    # Customize boxplot colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(daily_data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for whisker in bp["whiskers"]:
        whisker.set(color="gray", linewidth=1.5)

    for cap in bp["caps"]:
        cap.set(color="gray", linewidth=1.5)

    for median in bp["medians"]:
        median.set(color="red", linewidth=2)

    for flier in bp["fliers"]:
        flier.set(marker="o", color="red", alpha=0.5)

    # Customize mean markers as triangles consistently across plots
    for mean in bp["means"]:
        mean.set(
            color="green",
            linewidth=0,
            linestyle="None",
            marker="^",
            markersize=10,
            markerfacecolor="green",
            markeredgecolor="black",
        )
        mean.set_alpha(0.9)

    plt.plot([], [], color="green", marker="^", linestyle="None", markersize=10, label="Mean")
    plt.legend(loc="upper right")

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Inter-Arrival Time (seconds)")
    plt.title(f"{title}\n(n={len(df):,})", pad=20)

    # Use log scale for y-axis
    plt.yscale("log")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Add grid
    plt.grid(True, alpha=0.3, axis="y")

    # Add statistics text
    dates_with_data = df["date"].nunique()
    date_range = f"{df['date'].min()} to {df['date'].max()}"
    stats_text = f"""
    Date range: {date_range}
    Days with data: {dates_with_data}
    Total requests: {len(df) + 1}
    """
    plt.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Daily boxplot saved to {save_path}")


def plot_multi_trace_boxplot_grouped(
    all_user_dfs,
    per_user_dfs,
    trace_labels,
    title="Inter-Arrival Times Comparison",
    save_path=None,
):
    """
    Plot grouped boxplots with all-user and per-user inter-arrival times per trace.

    Each trace contributes two side-by-side boxes (all users vs per user) so the
    comparison happens in a single figure.
    """
    if not all_user_dfs or not per_user_dfs or not trace_labels:
        print("No data provided for grouped multi-trace boxplot.")
        return

    plt.figure(figsize=(14, 10))

    all_data = [df["inter_arrival_time"].values for df in all_user_dfs]
    per_data = [df["inter_arrival_time"].values for df in per_user_dfs]

    num_traces = len(trace_labels)
    group_positions = np.arange(num_traces) * 2.0 + 1.0
    offset = 0.35
    positions_all = group_positions - offset
    positions_per = group_positions + offset

    all_color = "#1f77b4"
    per_color = "#ff7f0e"

    bp_all = plt.boxplot(
        all_data,
        patch_artist=True,
        positions=positions_all,
        showmeans=True,
        meanline=False,
        widths=0.5,
    )
    bp_per = plt.boxplot(
        per_data,
        patch_artist=True,
        positions=positions_per,
        showmeans=True,
        meanline=False,
        widths=0.5,
    )

    for patch in bp_all["boxes"]:
        patch.set_facecolor(all_color)
        patch.set_alpha(0.4)
    for patch in bp_per["boxes"]:
        patch.set_facecolor(per_color)
        patch.set_alpha(0.4)

    for whisker in bp_all["whiskers"] + bp_per["whiskers"]:
        whisker.set(color="gray", linewidth=1.5)
    for cap in bp_all["caps"] + bp_per["caps"]:
        cap.set(color="gray", linewidth=1.5)
    for median in bp_all["medians"]:
        median.set(color=all_color, linewidth=2)
    for median in bp_per["medians"]:
        median.set(color=per_color, linewidth=2)
    for flier in bp_all["fliers"]:
        flier.set(marker="o", color=all_color, alpha=0.4)
    for flier in bp_per["fliers"]:
        flier.set(marker="o", color=per_color, alpha=0.4)

    for mean in bp_all["means"]:
        mean.set(
            color=all_color,
            linewidth=0,
            linestyle="None",
            marker="^",
            markersize=10,
            markerfacecolor=all_color,
            markeredgecolor="black",
        )
        mean.set_alpha(0.9)
    for mean in bp_per["means"]:
        mean.set(
            color=per_color,
            linewidth=0,
            linestyle="None",
            marker="^",
            markersize=10,
            markerfacecolor=per_color,
            markeredgecolor="black",
        )
        mean.set_alpha(0.9)

    plt.plot(
        [],
        [],
        color=all_color,
        marker="^",
        linestyle="None",
        markersize=10,
        label="All users (mean)",
    )
    plt.plot(
        [],
        [],
        color=per_color,
        marker="^",
        linestyle="None",
        markersize=10,
        label="Per user (mean)",
    )
    plt.legend(loc="upper right")

    truncated_labels = [label[:32] for label in trace_labels]
    plt.xticks(group_positions, truncated_labels, rotation=90, ha="center", fontsize=11)
    plt.xlabel("Trace")
    plt.ylabel("Inter-Arrival Time (seconds)")
    plt.title(f"{title}\n({num_traces} traces)", pad=20)
    plt.yscale("log")
    plt.grid(True, alpha=0.3, axis="y")

    total_all = sum(len(values) for values in all_data)
    total_per = sum(len(values) for values in per_data)
    stats_text = f"Total traces: {num_traces}\nSamples (all users): {total_all:,}\nSamples (per user): {total_per:,}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Grouped multi-trace boxplot saved to {save_path}")


def plot_inter_arrival_correlation(
    df, title="Inter-Arrival Correlation", save_path=None
):
    """
    Plot correlation between consecutive inter-arrival times.

    Calculates and displays both Pearson and Spearman correlation coefficients
    to provide comprehensive correlation analysis.

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' column
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Get consecutive pairs of inter-arrival times
    inter_arrivals = df["inter_arrival_time"].values

    if len(inter_arrivals) < 2:
        print("Not enough data points for correlation analysis")
        return

    # Create pairs: (current, next)
    current = inter_arrivals[:-1]  # All except last
    next_arrival = inter_arrivals[1:]  # All except first

    # Calculate both Pearson and Spearman correlations
    pearson_corr = np.corrcoef(current, next_arrival)[0, 1]
    spearman_corr, spearman_p_value = spearmanr(current, next_arrival)

    # Create scatter plot
    plt.scatter(
        current,
        next_arrival,
        alpha=0.6,
        s=30,
        c="blue",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add diagonal line (y=x) to show perfect correlation
    max_val = max(current.max(), next_arrival.max())
    min_val = min(current.min(), next_arrival.min())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        alpha=0.7,
        linewidth=2,
        label="Perfect correlation (y=x)",
    )

    # Add labels and title
    plt.xlabel("Current Inter-Arrival Time (seconds)")
    plt.ylabel("Next Inter-Arrival Time (seconds)")
    plt.title(f"{title} - Consecutive Pairs\n(n={len(current):,})", pad=20)

    # Add large, prominent correlation coefficients at the top
    correlation_text = f"Pearson: {pearson_corr:.4f} | Spearman: {spearman_corr:.4f}"
    plt.text(
        0.5,
        0.95,
        correlation_text,
        transform=plt.gca().transAxes,
        horizontalalignment="center",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="yellow",
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        ),
    )

    # Use log scale for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add detailed statistics (smaller since we have the large correlation above)
    stats_text = f"""
    Total pairs: {len(current):,}
    Mean current: {current.mean():,.3f}s
    Mean next: {next_arrival.mean():,.3f}s
    Std current: {current.std():,.3f}s
    Std next: {next_arrival.std():,.3f}s
    Spearman p-value: {spearman_p_value:.2e}
    """

    # Position stats box in top-left
    plt.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add legend
    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Correlation plot saved to {save_path}")


def plot_inter_arrival_heatmap(
    df, title="Inter-Arrival Time Probability Heatmap", save_path=None
):
    """
    Create a heatmap showing the probability distribution of current vs next inter-arrival times.

    Uses 2D histogram to show joint probability density with log-scale in milliseconds.

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' column
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Get consecutive pairs of inter-arrival times
    inter_arrivals = df["inter_arrival_time"].values

    if len(inter_arrivals) < 2:
        print("Not enough data points for heatmap analysis")
        return

    # set minimum inter-arrival time to avoid log(0)
    inter_arrivals = np.maximum(inter_arrivals, 1e-3)  # 1 ms minimum
    inter_arrivals = np.minimum(inter_arrivals, 1e4)  # 10000 seconds maximum

    # Create pairs: (current, next)
    current = inter_arrivals[:-1]  # All except last
    next_arrival = inter_arrivals[1:]  # All except first

    # Convert to milliseconds for better readability
    current_ms = current * 1000
    next_arrival_ms = next_arrival * 1000

    print(f"Processing {len(current_ms)} points for heatmap.")

    # Create 2D histogram (probability density)
    # Use log scale for better visualization
    current_log = np.log10(current_ms + 1e-6)  # Add small value to avoid log(0)
    next_log = np.log10(next_arrival_ms + 1e-6)

    # Create bins for the heatmap
    bins = 50
    hist, xedges, yedges = np.histogram2d(current_log, next_log, bins=bins)

    # Convert to probability density (normalize)
    hist = hist / hist.sum()

    # Create heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = plt.imshow(
        hist.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="auto",
        interpolation="bilinear",
    )

    # Set custom tick labels (convert back from log scale)
    def format_time_label(value):
        if value == int(value):
            return f"{int(10**value)}"
        else:
            return f"{10**value:.1f}"

    # Set ticks at log scale intervals
    log_ticks = np.arange(int(current_log.min()), int(current_log.max()) + 1)

    plt.xticks(log_ticks, [format_time_label(t) for t in log_ticks])
    plt.yticks(log_ticks, [format_time_label(t) for t in log_ticks])

    # Add labels and title
    plt.xlabel("Current Inter-Arrival Time (ms)")
    plt.ylabel("Next Inter-Arrival Time (ms)")
    plt.title(f"{title}\n(n={len(current_ms):,} pairs)", pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("Probability Density", rotation=270, labelpad=20)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, color="white", linewidth=0.5)

    # Add diagonal line to show perfect correlation
    min_val = min(current_log.min(), next_log.min())
    max_val = max(current_log.max(), next_log.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        alpha=0.7,
        linewidth=2,
        label="Perfect correlation (y=x)",
    )

    # Add statistics text
    pearson_corr = np.corrcoef(current_ms / 1000, next_arrival_ms / 1000)[0, 1]
    spearman_corr, spearman_p = spearmanr(current_ms / 1000, next_arrival_ms / 1000)

    stats_text = f"""
    Pearson: {pearson_corr:.3f}
    Spearman: {spearman_corr:.3f}
    P-value: {spearman_p:.2e}
    Total pairs: {len(current_ms):,}
    """

    plt.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {save_path}")


def load_traces_for_modes(
    file_paths,
    output_dir,
    worker_count,
    per_user_modes=(False, True),
) -> Dict[str, Dict[bool, Tuple[pd.DataFrame, Dict[float, int]]]]:
    """Process each file for the requested per_user modes in parallel."""
    trace_results: Dict[str, Dict[bool, Tuple[pd.DataFrame, Dict[float, int]]]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_meta = {
            executor.submit(
                load_or_process_trace, file_path, per_user_mode, output_dir
            ): (file_path, per_user_mode)
            for file_path in file_paths
            for per_user_mode in per_user_modes
        }
        for future in as_completed(future_to_meta):
            file_path, per_user_mode = future_to_meta[future]
            try:
                df, duration_map = future.result()
                df["inter_arrival_time"] = df["inter_arrival_time"].clip(lower=1e-3)
                trace_results.setdefault(file_path, {})[per_user_mode] = (
                    df,
                    duration_map,
                )
            except Exception as exc:
                mode_desc = "per-user" if per_user_mode else "all-user"
                print(f"Failed to process {file_path} ({mode_desc}): {exc}")
    return trace_results


def main():
    parser = argparse.ArgumentParser(description="Analyze Inter-Arrival Times")
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Path(s) to per-model CSV file(s). Provide multiple to plot trace-level boxplot.",
    )
    parser.add_argument("--output", "-o", type=str, help="Output directory for plots")
    parser.add_argument(
        "--per-user",
        action="store_true",
        help="Calculate inter-arrival times per user (only for same user)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers when loading multiple traces (default: min(num_traces, cpu_count)).",
    )

    args = parser.parse_args()

    output_dir = infer_output_dir(args.files[0], args.per_user, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    multi_trace_mode = len(args.files) > 1

    if multi_trace_mode:
        worker_count = args.workers or min(len(args.files), os.cpu_count() or 4)
        print(
            f"Processing {len(args.files)} traces using {worker_count} parallel worker(s)..."
        )
        print("Generating multi-trace grouped boxplot for all-user & per-user data.")

        trace_results = load_traces_for_modes(
            args.files, output_dir, worker_count, per_user_modes=(False, True)
        )

        all_user_dfs = []
        per_user_dfs = []
        trace_labels = []
        for file_path in args.files:
            per_mode_map = trace_results.get(file_path, {})
            all_result = per_mode_map.get(False)
            per_result = per_mode_map.get(True)
            if all_result is None or per_result is None:
                print(
                    f"Skipping {file_path} because both all-user and per-user data are required."
                )
                continue
            all_df, _ = all_result
            per_df, _ = per_result
            if len(all_df) == 0 or len(per_df) == 0:
                print(f"No valid inter-arrival data found in {file_path}; skipping.")
                continue
            all_user_dfs.append(all_df)
            per_user_dfs.append(per_df)
            trace_labels.append(Path(file_path).stem)

        if not all_user_dfs:
            print(
                "No valid inter-arrival time data found across provided traces. Exiting."
            )
            return

        base_title = "Inter-Arrival Times (All Users vs Per User)"
        title = f"{base_title} - Multiple Traces"
        suffix = args.files[0].split("/")[-2]
        boxplot_path = (
            output_dir.parent / f"multi_trace_inter_arrival_boxplot_{suffix}.png"
        )
        plot_multi_trace_boxplot_grouped(
            all_user_dfs,
            per_user_dfs,
            trace_labels,
            title=title,
            save_path=str(boxplot_path),
        )
        return

    # Single trace path
    file_path = args.files[0]

    # Load and process data
    df, duration_freq_map = load_or_process_trace(
        file_path, per_user=args.per_user, output_dir=output_dir
    )

    if len(df) == 0:
        print("No valid inter-arrival time data found. Exiting.")
        return

    # Basic statistics
    print("\nInter-Arrival Time Statistics:")
    print(df["inter_arrival_time"].describe())

    trace_name = os.path.basename(file_path).split(".")[0]

    base_title = (
        "Inter-Arrival Times (Per User)" if args.per_user else "Inter-Arrival Times"
    )
    if file_path:
        base_title += f" - {Path(file_path).name}"

    # CDF
    cdf_path = Path(output_dir) / f"{trace_name}_inter_arrival_cdf.png"
    plot_inter_arrival_times(
        duration_freq_map, title=base_title, save_path=str(cdf_path)
    )

    # Boxplot by hour (always created)
    boxplot_path = Path(output_dir) / f"{trace_name}_inter_arrival_boxplot.png"
    plot_hourly_boxplot(df, title=base_title, save_path=str(boxplot_path))

    # Boxplot by day (always created)
    daily_boxplot_path = (
        Path(output_dir) / f"{trace_name}_inter_arrival_daily_boxplot.png"
    )
    plot_daily_boxplot(df, title=base_title, save_path=str(daily_boxplot_path))

    # Correlation between consecutive inter-arrivals
    correlation_path = Path(output_dir) / f"{trace_name}_inter_arrival_correlation.png"
    plot_inter_arrival_correlation(
        df, title=base_title, save_path=str(correlation_path)
    )

    # Heatmap showing probability distribution of current vs next inter-arrival times
    heatmap_path = Path(output_dir) / f"{trace_name}_inter_arrival_heatmap.png"
    plot_inter_arrival_heatmap(df, title=base_title, save_path=str(heatmap_path))


if __name__ == "__main__":
    main()
