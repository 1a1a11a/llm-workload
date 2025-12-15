#!/usr/bin/env python3
"""
Inter-Arrival Time Analysis Script
This requires the per-model CSV files.

This script analyzes the inter-arrival times of requests across all users
by plotting the time differences between consecutive requests as a CDF, boxplot by hour,
daily boxplot, and correlation between consecutive inter-arrival times.

Usage:
    python plot_arrival_time_model.py /path/to/metrics.csv
    python plot_arrival_time_model.py ../data/metrics_30day/DeepSeek-R1.csv --output plots

Creates and saves CDF, boxplot, daily boxplot, correlation plots, and probability heatmap automatically (no display).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.stats import spearmanr

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style

setup_plot_style()


def load_and_process_data(filepath):
    """
    Load CSV data and process timestamps for inter-arrival time analysis.

    Args:
        filepath (str): Path to the CSV file

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
    df["inter_arrival_time"] = df["started_at"].diff().dt.total_seconds()

    # Extract hour and day for grouping
    df["hour"] = df["started_at"].dt.hour
    df["date"] = df["started_at"].dt.date

    # Remove the first row (which will have NaN for inter_arrival_time)
    df = df.dropna(subset=["inter_arrival_time"])

    print(f"Loaded {len(df) + 1} rows, processed {len(df)} inter-arrival times")
    print(f"Time range: {df['started_at'].min()} to {df['started_at'].max()}")

    return df


def plot_inter_arrival_times(df, title="Inter-Arrival Times", save_path=None):
    """
    Plot the inter-arrival times as a Cumulative Distribution Function (CDF).

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' column
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Sort inter-arrival times for CDF
    sorted_times = np.sort(df["inter_arrival_time"])
    yvals = np.arange(len(sorted_times)) / float(len(sorted_times))

    # Create CDF plot
    plt.plot(sorted_times, yvals, "b-", linewidth=2, alpha=0.8)

    # Add labels and title
    plt.xlabel("Inter-Arrival Time (seconds)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"{title} - CDF\n(n={len(df):,})", pad=20)

    # Use log scale for x-axis since inter-arrival times span multiple orders of magnitude
    plt.xscale("log")

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""
    Mean: {df["inter_arrival_time"].mean():,.3f}s
    Median: {df["inter_arrival_time"].median():,.3f}s
    Std: {df["inter_arrival_time"].std():,.3f}s
    Min: {df["inter_arrival_time"].min():,.6f}s
    Max: {df["inter_arrival_time"].max():,.3f}s
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
    plt.figure(figsize=(14, 8))

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
        meanline=True,
    )

    # Set custom tick labels
    plt.xticks(positions, hour_labels, rotation=45)

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

    # Customize mean lines to be visually distinct
    for mean in bp["means"]:
        mean.set(color="green", linewidth=3)
        mean.set_alpha(0.8)

    # Add means label to legend
    plt.plot([], [], color="green", linewidth=3, label="Mean")
    plt.legend(loc="upper right")

    # Add labels and title
    plt.xlabel("Hour of Day")
    plt.ylabel("Inter-Arrival Time (seconds)")
    plt.title(f"{title}\n(n={len(df):,})", pad=20)

    # Use log scale for y-axis
    plt.yscale("log")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

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
    plt.figure(figsize=(16, 8))

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
        meanline=True,
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

    # Customize mean lines to be visually distinct
    for mean in bp["means"]:
        mean.set(color="green", linewidth=3)
        mean.set_alpha(0.8)

    # Add means label to legend
    plt.plot([], [], color="green", linewidth=3, label="Mean")
    plt.legend(loc="upper right")

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Inter-Arrival Time (seconds)")
    plt.title(f"{title}\n(n={len(df):,})", pad=20)

    # Use log scale for y-axis
    plt.yscale("log")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

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
    plt.figure(figsize=(12, 8))

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
        fontsize=16,
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
        fontsize=9,
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
    inter_arrivals = np.maximum(inter_arrivals, 1e-2)  # 10 ms minimum
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
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Inter-Arrival Times")
    parser.add_argument("file", type=str, help="Path to CSV file")
    parser.add_argument("--output", "-o", type=str, help="Output directory for plots")

    args = parser.parse_args()

    # Load and process data
    df = load_and_process_data(args.file)

    if len(df) == 0:
        print("No valid inter-arrival time data found. Exiting.")
        return

    # Basic statistics
    print("\nInter-Arrival Time Statistics:")
    print(df["inter_arrival_time"].describe())

    trace_name = os.path.basename(args.file).split(".")[0]

    # Create plots
    if "/1k/" in args.file:
        output_dir = args.output or "figures/arrival_time/1k/"
    elif "/10k/" in args.file:
        output_dir = args.output or "figures/arrival_time/10k/"
    elif "/100k/" in args.file:
        output_dir = args.output or "figures/arrival_time/100k/"
    else:
        output_dir = args.output or "figures/arrival_time/"
    Path(output_dir).mkdir(exist_ok=True)

    base_title = "Inter-Arrival Times"
    if args.file:
        base_title += f" - {Path(args.file).name}"

    # CDF
    cdf_path = Path(output_dir) / f"{trace_name}_inter_arrival_cdf.png"
    plot_inter_arrival_times(df, title=base_title, save_path=str(cdf_path))

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
