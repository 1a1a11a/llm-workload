#!/usr/bin/env python3
"""
Script to analyze metrics data and generate statistical plots.
Creates CDF, distribution plots, and pie charts for each column.

This script analyzes LLM workload metrics data and generates comprehensive
statistical visualizations including:
- CDF plots of unique values per column
- Categorical distributions with pie charts and bar charts
- Numerical distributions with CDF plots for token metrics
- Temporal analysis with hourly/daily distribution plots

Usage:
    python3 overview.py [input_file]

The script expects data to be located at:
    /scratch/juncheng/data/prefix_cache/metrics_30day.csv
    /scratch/juncheng/data/prefix_cache/metrics_1day.csv
    /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/

Input data should contain columns like:
    - user_id, chute_id, model_name, function_name
    - input_tokens, output_tokens, ttft, duration
    - started_at, completed_at, invocation_id

Output:
    - CDF plots of unique values
    - Categorical distribution plots
    - Numerical distribution plots
    - Temporal analysis plots
    - All saved to figures/ directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readers.data_loader import load_metrics_dataframe
from utils.plot import setup_plot_style


# Columns to exclude from analysis (too unique or not useful)
# invocation_id is excluded because it's a unique identifier for each request
# timestamp columns are excluded because they represent specific points in time (very high uniqueness)
EXCLUDED_COLUMNS = {"invocation_id", "started_at", "completed_at"}


def analyze_column_uniques(df, column_name):
    """Analyze unique values in a column"""
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found")
        return None

    unique_vals = df[column_name].nunique()
    total_vals = len(df[column_name])
    null_count = df[column_name].isnull().sum()

    print(f"\n{column_name}:")
    print(
        f"  Total values:  {total_vals}, unique values: {unique_vals} ({unique_vals / total_vals:.4f})"
    )

    return {
        "total": total_vals,
        "unique": unique_vals,
        "null_count": null_count,
        "uniqueness_ratio": unique_vals / total_vals if total_vals > 0 else 0,
    }


def plot_unique_values_cdf(df, output_dir="figures"):
    """Plot CDF of unique values across all columns"""
    os.makedirs(output_dir, exist_ok=True)

    # Define which fields to analyze
    fields_to_analyze = [
        "chute_id",
        "function_name",
        "user_id",
        "model_name",
    ]

    # Calculate statistics for each column
    column_stats = {}
    for field_name in fields_to_analyze:
        if field_name in EXCLUDED_COLUMNS:
            print(f"Skipping excluded column: {field_name}")
            continue

        if field_name not in df.columns:
            continue

        unique_count = df[field_name].nunique()
        total_count = len(df[field_name])
        null_count = df[field_name].isnull().sum()

        column_stats[field_name] = {
            "total": total_count,
            "unique": unique_count,
            "null_count": null_count,
            "uniqueness_ratio": unique_count / total_count if total_count > 0 else 0,
        }

    if not column_stats:
        return

    # Create CDF plot
    fig, ax = plt.subplots(figsize=(12, 8))

    columns = list(column_stats.keys())
    unique_counts = [column_stats[col]["unique"] for col in columns]

    # Sort by unique count
    sorted_indices = np.argsort(unique_counts)
    sorted_columns = [columns[i] for i in sorted_indices]
    sorted_uniques = [unique_counts[i] for i in sorted_indices]

    # Plot unique counts
    ax.bar(range(len(sorted_columns)), sorted_uniques, alpha=0.7, label="Unique Values")
    ax.set_xticks(range(len(sorted_columns)))
    ax.set_xticklabels(sorted_columns, rotation=45, ha="right")
    ax.set_ylabel("Number of Unique Values")
    ax.set_title("Unique Values per Column (Sorted)")
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(sorted_uniques):
        ax.text(
            i,
            v + max(sorted_uniques) * 0.01,
            str(v),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/unique_values_per_column.png", dpi=300, bbox_inches="tight"
    )
    plt.clf()
    plt.close(fig)


def plot_categorical_distributions(df, output_dir="figures"):
    """Plot distributions for categorical columns"""
    os.makedirs(output_dir, exist_ok=True)

    categorical_cols = ["function_name", "model_name", "user_id"]

    # Initialize counters for each categorical column
    value_counters = {col: Counter() for col in categorical_cols}

    # Count values for each categorical column
    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts()
            for value, count in value_counts.items():
                if pd.notna(value):  # Skip NaN values
                    value_counters[col][value] += count

    for col in categorical_cols:
        value_counts = value_counters[col]
        if not value_counts:
            continue

        # Skip plotting model_name if there's only one model
        if col == "model_name" and len(value_counts) == 1:
            print(f"Skipping {col} distribution plot - only one model present")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Convert to sorted list for top_n selection
        sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        top_n = min(10, len(sorted_counts))

        # Pie chart (top categories)
        top_items = sorted_counts[:top_n]
        if len(sorted_counts) > top_n:
            other_count = sum(count for _, count in sorted_counts[top_n:])
            top_items.append(("Other", other_count))

        labels, values = zip(*top_items)
        ax1.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax1.set_title(f"{col} Distribution (Top {top_n})")
        ax1.axis("equal")

        # Bar chart
        top_labels, top_values = zip(*sorted_counts[:top_n])
        ax2.bar(range(len(top_labels)), top_values)
        ax2.set_title(f"{col} Frequency (Top {top_n})")
        ax2.set_ylabel("Count")
        ax2.set_xticks(range(len(top_labels)))
        ax2.set_xticklabels(top_labels, rotation=45, ha="right")

        # Add count labels on bars
        max_val = max(top_values)
        for i, v in enumerate(top_values):
            ax2.text(i, v + max_val * 0.01, str(v), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{col}_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


def plot_numerical_distributions(df, output_dir="figures"):
    """Plot distributions for numerical columns"""
    os.makedirs(output_dir, exist_ok=True)

    numerical_cols = ["input_tokens", "output_tokens", "ttft", "duration"]

    # Collect numerical data from DataFrame
    data_collectors = {}
    for col in numerical_cols:
        if col in df.columns:
            # Filter out NaN values and collect data
            data = df[col].dropna().tolist()
            data_collectors[col] = data
        else:
            data_collectors[col] = []

    for col in numerical_cols:
        data = data_collectors[col]
        if len(data) == 0:
            print(f"No valid data for {col}")
            continue

        # For token columns, only show CDF plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # CDF
        sorted_data = np.sort(data)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, yvals, "b-", linewidth=3, markersize=8)

        # Set larger fonts specifically for CDF plots
        ax.set_title(f"{col} CDF", pad=20)
        ax.set_xlabel(col, labelpad=10)
        ax.set_ylabel("Fraction of Requests (CDF)", labelpad=10)

        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")  # Set x-axis to log scale

        # Add minor grid lines for better readability
        ax.grid(True, which="minor", alpha=0.2, linestyle="--")

        # Set y-axis limits to ensure full range is visible
        ax.set_ylim(0, 1.05)

        # Add a subtle background color for better contrast
        ax.set_facecolor("#f8f9fa")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_cdf.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Convert to numpy array for statistics
        data_array = np.array(data)

        # Print statistics
        print(f"\n{col} Statistics:")
        print(f"  Count: {len(data_array)}")
        print(f"  Mean: {data_array.mean():.2f}")
        print(f"  Median: {np.median(data_array):.2f}")
        print(f"  Std: {data_array.std():.2f}")
        print(f"  Min: {data_array.min():.2f}")
        print(f"  Max: {data_array.max():.2f}")
        print(f"  25th percentile: {np.percentile(data_array, 25):.2f}")
        print(f"  75th percentile: {np.percentile(data_array, 75):.2f}")


def plot_timestamp_analysis(df, output_dir="figures"):
    """Analyze timestamp columns"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp_cols = [
        "started_at",
    ]

    # Collect timestamp data from DataFrame
    timestamp_collectors = {}
    for col in timestamp_cols:
        if col in df.columns:
            # Convert timestamps and filter out NaN values
            timestamps = pd.to_datetime(df[col], errors="coerce").dropna().tolist()
            timestamp_collectors[col] = timestamps
        else:
            timestamp_collectors[col] = []

    for col in timestamp_cols:
        timestamps = timestamp_collectors[col]
        if len(timestamps) == 0:
            continue

        try:
            timestamps_series = pd.Series(timestamps)
            timestamps_series = timestamps_series.dropna()

            if len(timestamps_series) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Hourly distribution
            hourly_counts = timestamps_series.dt.hour.value_counts().sort_index()
            ax1.bar(hourly_counts.index, hourly_counts.values, alpha=0.7)
            ax1.set_title(f"{col} - Hourly Distribution")
            ax1.set_xlabel("Hour of Day")
            ax1.set_ylabel("Count")
            ax1.grid(True, alpha=0.3)

            # Daily distribution (if multiple days)
            daily_counts = timestamps_series.dt.date.value_counts().sort_index()
            if len(daily_counts) > 1:
                ax2.bar(range(len(daily_counts)), daily_counts.values, alpha=0.7)
                ax2.set_xticks(range(len(daily_counts)))
                ax2.set_xticklabels(
                    [str(d) for d in daily_counts.index], rotation=45, ha="right"
                )
                ax2.set_title(f"{col} - Daily Distribution")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Count")
                ax2.grid(True, alpha=0.3)
            else:
                # If only one day, show minute-by-minute
                minute_counts = timestamps_series.dt.minute.value_counts().sort_index()
                ax2.bar(minute_counts.index, minute_counts.values, alpha=0.7)
                ax2.set_title(f"{col} - Per Minute Distribution")
                ax2.set_xlabel("Minute")
                ax2.set_ylabel("Count")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/{col}_temporal_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            print(f"\n{col} Temporal Statistics:")
            print(
                f"  Date range: {timestamps_series.min()} to {timestamps_series.max()}"
            )
            print(f"  Duration: {timestamps_series.max() - timestamps_series.min()}")

        except Exception as e:
            print(f"Error processing {col}: {e}")


def analyze_one_trace(file_path):
    """Analyze one trace"""

    print("Loading data...")
    path_obj = Path(file_path)
    trace_name = path_obj.stem

    # Extract folder name from file path for proper output organization
    # Expected path structure: .../per_model/{folder_name}/{model_name}.csv
    folder_name = "unknown"
    if "per_model" in path_obj.parts:
        idx = path_obj.parts.index("per_model")
        if len(path_obj.parts) > idx + 1:
            folder_name = path_obj.parts[idx + 1]
    elif path_obj.parent.name:
        folder_name = path_obj.parent.name

    # Load data using the utility function with transformations
    df = load_metrics_dataframe(file_path, apply_transforms=True)

    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Note: model_name mapping and transformations are handled in load_metrics_dataframe

    print("\n" + "=" * 50)
    print(f"ANALYZING METRICS DATA: {trace_name}")
    print("=" * 50)

    # Create output directory
    output_dir = f"figures/metrics_overview/{folder_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Analyze unique values (model_name mapping already applied in data loading)
    plot_unique_values_cdf(df, output_dir)

    # Categorical distributions (model_name mapping already applied in data loading)
    plot_categorical_distributions(df, output_dir)

    # Numerical distributions (model_name mapping already applied in data loading)
    plot_numerical_distributions(df, output_dir)

    # Timestamp analysis (model_name mapping already applied in data loading)
    plot_timestamp_analysis(df, output_dir)


    print("\nAnalysis complete! Plots saved to:", output_dir)
    print("Generated files:")
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            print(f"  - {file}")

    plt.close("all")


def main():
    """Main function"""
    # Set up plot style
    setup_plot_style()
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/scratch/juncheng/data/prefix_cache/metrics_1day_head.csv"

    analyze_one_trace(file_path)


if __name__ == "__main__":
    main()
