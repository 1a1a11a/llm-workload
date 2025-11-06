#!/usr/bin/env python3
"""
Inter-Arrival Time Analysis Script

This script analyzes the inter-arrival times of requests across all users
by plotting the time differences between consecutive requests as a CDF, boxplot by hour,
and correlation between consecutive inter-arrival times.

Usage:
    python plot_arrival_time_model.py /path/to/metrics.csv
    python plot_arrival_time_model.py ../data/metrics_30day/DeepSeek-R1.csv --output plots

Creates and saves CDF, boxplot, and correlation plots automatically (no display).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

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

    # Convert started_at to datetime
    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed")

    # Sort by started_at to ensure chronological order
    df = df.sort_values("started_at").reset_index(drop=True)

    # Calculate inter-arrival times (in seconds)
    df["inter_arrival_time"] = df["started_at"].diff().dt.total_seconds()

    # Extract hour for grouping
    df["hour"] = df["started_at"].dt.hour

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

    # Create boxplot with custom positions
    bp = plt.boxplot(hourly_data, patch_artist=True, positions=positions)

    # Set custom tick labels
    plt.xticks(positions, hour_labels, rotation=45)

    # Customize boxplot colors
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    for whisker in bp['whiskers']:
        whisker.set(color='blue', linewidth=1.5)

    for cap in bp['caps']:
        cap.set(color='blue', linewidth=1.5)

    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    for flier in bp['fliers']:
        flier.set(marker='o', color='red', alpha=0.5)

    # Add labels and title
    plt.xlabel("Hour of Day")
    plt.ylabel("Inter-Arrival Time (seconds)")
    plt.title(f"{title}\n(n={len(df):,})", pad=20)

    # Use log scale for y-axis
    plt.yscale('log')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add grid
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    hours_with_data = df["hour"].nunique()
    stats_text = f"""
    Hours with data: {hours_with_data}
    Total requests: {len(df) + 1}
    """
    plt.text(0.02, 0.98, stats_text.strip(),
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Boxplot saved to {save_path}")


def plot_inter_arrival_correlation(df, title="Inter-Arrival Correlation", save_path=None):
    """
    Plot correlation between consecutive inter-arrival times.

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' column
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Get consecutive pairs of inter-arrival times
    inter_arrivals = df['inter_arrival_time'].values

    if len(inter_arrivals) < 2:
        print("Not enough data points for correlation analysis")
        return

    # Create pairs: (current, next)
    current = inter_arrivals[:-1]  # All except last
    next_arrival = inter_arrivals[1:]  # All except first

    # Calculate correlation
    correlation = np.corrcoef(current, next_arrival)[0, 1]

    # Create scatter plot
    plt.scatter(current, next_arrival, alpha=0.6, s=30, c='blue', edgecolors='black', linewidth=0.5)

    # Add diagonal line (y=x) to show perfect correlation
    max_val = max(current.max(), next_arrival.max())
    min_val = min(current.min(), next_arrival.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, label='Perfect correlation (y=x)')

    # Add labels and title
    plt.xlabel("Current Inter-Arrival Time (seconds)")
    plt.ylabel("Next Inter-Arrival Time (seconds)")
    plt.title(f"{title} - Consecutive Pairs\n(n={len(current):,})", pad=20)

    # Use log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient and statistics
    stats_text = f"""
    Correlation: {correlation:.3f}
    Total pairs: {len(current):,}
    Mean current: {current.mean():,.3f}s
    Mean next: {next_arrival.mean():,.3f}s
    """

    # Position stats box in top-left
    plt.text(0.02, 0.98, stats_text.strip(),
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add legend
    plt.legend(loc='lower right')

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Correlation plot saved to {save_path}")


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
    output_dir = args.output or "figures/arrival_time/small/"
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

    # Correlation between consecutive inter-arrivals
    correlation_path = Path(output_dir) / f"{trace_name}_inter_arrival_correlation.png"
    plot_inter_arrival_correlation(df, title=base_title, save_path=str(correlation_path))


if __name__ == "__main__":
    main()
