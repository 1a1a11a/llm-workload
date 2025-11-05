#!/usr/bin/env python3
"""
Script to analyze metrics_30_days_head.csv and generate statistical plots.
Creates CDF, distribution plots, and pie charts for each column.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Columns to exclude from analysis (too unique or not useful)
# invocation_id is excluded because it's a unique identifier for each request
# timestamp columns are excluded because they represent specific points in time (very high uniqueness)
EXCLUDED_COLUMNS = {'invocation_id', 'started_at', 'completed_at'}

def load_data(file_path):
    """Load CSV data, skipping empty first line"""
    try:
        df = pd.read_csv(file_path, skiprows=0)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_column_uniques(df, column_name):
    """Analyze unique values in a column"""
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found")
        return None

    unique_vals = df[column_name].nunique()
    total_vals = len(df[column_name])
    null_count = df[column_name].isnull().sum()

    print(f"\n{column_name}:")
    print(f"  Total values:  {total_vals}, unique values: {unique_vals} ({unique_vals / total_vals:.4f})")

    return {
        'total': total_vals,
        'unique': unique_vals,
        'null_count': null_count,
        'uniqueness_ratio': unique_vals / total_vals if total_vals > 0 else 0
    }

def plot_unique_values_cdf(df, output_dir="figures"):
    """Plot CDF of unique values across all columns"""
    os.makedirs(output_dir, exist_ok=True)

    column_stats = {}
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            print(f"Skipping excluded column: {col}")
            continue
        stats = analyze_column_uniques(df, col)
        if stats:
            column_stats[col] = stats

    if not column_stats:
        return

    # Create CDF plot
    fig, ax = plt.subplots(figsize=(12, 8))

    columns = list(column_stats.keys())
    unique_counts = [column_stats[col]['unique'] for col in columns]
    uniqueness_ratios = [column_stats[col]['uniqueness_ratio'] for col in columns]

    # Sort by unique count
    sorted_indices = np.argsort(unique_counts)
    sorted_columns = [columns[i] for i in sorted_indices]
    sorted_uniques = [unique_counts[i] for i in sorted_indices]
    sorted_ratios = [uniqueness_ratios[i] for i in sorted_indices]

    # Plot unique counts
    ax.bar(range(len(sorted_columns)), sorted_uniques, alpha=0.7, label='Unique Values')
    ax.set_xticks(range(len(sorted_columns)))
    ax.set_xticklabels(sorted_columns, rotation=45, ha='right')
    ax.set_ylabel('Number of Unique Values')
    ax.set_title('Unique Values per Column (Sorted)')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(sorted_uniques):
        ax.text(i, v + max(sorted_uniques) * 0.01, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/unique_values_per_column.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot uniqueness ratios
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(sorted_columns)), sorted_ratios, alpha=0.7, color='orange', label='Uniqueness Ratio')
    ax.set_xticks(range(len(sorted_columns)))
    ax.set_xticklabels(sorted_columns, rotation=45, ha='right')
    ax.set_ylabel('Uniqueness Ratio')
    ax.set_title('Column Uniqueness Ratio (Unique/Total)')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(sorted_ratios):
        ax.text(i, v + 0.01, '.3f', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/uniqueness_ratios.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_categorical_distributions(df, output_dir="figures"):
    """Plot distributions for categorical columns"""
    os.makedirs(output_dir, exist_ok=True)

    categorical_cols = ['function_name', 'chute_id', 'user_id']

    for col in categorical_cols:
        if col not in df.columns:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Get value counts
        value_counts = df[col].value_counts()
        top_n = min(10, len(value_counts))  # Show top 10

        # Pie chart (top categories)
        top_values = value_counts.head(top_n)
        if len(value_counts) > top_n:
            other_count = value_counts[top_n:].sum()
            top_values = pd.concat([top_values, pd.Series({'Other': other_count})])

        ax1.pie(top_values.values, labels=top_values.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'{col} Distribution (Top {top_n})')
        ax1.axis('equal')

        # Bar chart
        value_counts.head(top_n).plot(kind='bar', ax=ax2)
        ax2.set_title(f'{col} Frequency (Top {top_n})')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)

        # Add count labels on bars
        for i, v in enumerate(value_counts.head(top_n)):
            ax2.text(i, v + max(value_counts.head(top_n)) * 0.01, str(v), ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

def plot_numerical_distributions(df, output_dir="figures"):
    """Plot distributions for numerical columns"""
    os.makedirs(output_dir, exist_ok=True)

    numerical_cols = ['input_tokens', 'output_tokens', 'ttft']

    for col in numerical_cols:
        if col not in df.columns:
            continue

        # Remove null values for plotting
        data = df[col].dropna()

        if len(data) == 0:
            print(f"No valid data for {col}")
            continue

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram
        ax1.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title(f'{col} Histogram')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(data)
        ax2.set_title(f'{col} Box Plot')
        ax2.set_ylabel(col)
        ax2.grid(True, alpha=0.3)

        # CDF
        sorted_data = np.sort(data)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax3.plot(sorted_data, yvals, 'b-', linewidth=2)
        ax3.set_title(f'{col} CDF')
        ax3.set_xlabel(col)
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)

        # Q-Q plot (normal distribution)
        from scipy import stats
        stats.probplot(data, dist="norm", plot=ax4)
        ax4.set_title(f'{col} Q-Q Plot (vs Normal)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistics
        print(f"\n{col} Statistics:")
        print(f"  Count: {len(data)}")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Median: {data.median():.2f}")
        print(f"  Std: {data.std():.2f}")
        print(f"  Min: {data.min():.2f}")
        print(f"  Max: {data.max():.2f}")
        print(f"  25th percentile: {data.quantile(0.25):.2f}")
        print(f"  75th percentile: {data.quantile(0.75):.2f}")

def plot_timestamp_analysis(df, output_dir="figures"):
    """Analyze timestamp columns"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp_cols = ['started_at', 'completed_at']

    for col in timestamp_cols:
        if col not in df.columns:
            continue

        # Convert to datetime
        try:
            timestamps = pd.to_datetime(df[col], format='mixed')
            timestamps = timestamps.dropna()

            if len(timestamps) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Hourly distribution
            hourly_counts = timestamps.dt.hour.value_counts().sort_index()
            ax1.bar(hourly_counts.index, hourly_counts.values, alpha=0.7)
            ax1.set_title(f'{col} - Hourly Distribution')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)

            # Daily distribution (if multiple days)
            daily_counts = timestamps.dt.date.value_counts().sort_index()
            if len(daily_counts) > 1:
                ax2.bar(range(len(daily_counts)), daily_counts.values, alpha=0.7)
                ax2.set_xticks(range(len(daily_counts)))
                ax2.set_xticklabels([str(d) for d in daily_counts.index], rotation=45, ha='right')
                ax2.set_title(f'{col} - Daily Distribution')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Count')
                ax2.grid(True, alpha=0.3)
            else:
                # If only one day, show minute-by-minute
                minute_counts = timestamps.dt.minute.value_counts().sort_index()
                ax2.bar(minute_counts.index, minute_counts.values, alpha=0.7)
                ax2.set_title(f'{col} - Per Minute Distribution')
                ax2.set_xlabel('Minute')
                ax2.set_ylabel('Count')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_temporal_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()

            print(f"\n{col} Temporal Statistics:")
            print(f"  Date range: {timestamps.min()} to {timestamps.max()}")
            print(f"  Duration: {timestamps.max() - timestamps.min()}")

        except Exception as e:
            print(f"Error processing {col}: {e}")

def analyze_one_trace(file_path="/home/juncheng/workspace/prefix_cache/requests/gemma-3-27b-it.csv"):
    """Analyze one trace"""
    
    print("Loading data...")
    trace_name = os.path.basename(file_path).split('.')[0]
    df = load_data(file_path)

    if df is None or df.empty:
        print("No data to analyze")
        return

    print("\n" + "="*50)
    print(f"ANALYZING METRICS DATA: {trace_name}")
    print("="*50)

    # Create output directory
    output_dir = f"figures/metrics_analysis/{trace_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Analyze unique values
    plot_unique_values_cdf(df, output_dir)

    # Categorical distributions
    plot_categorical_distributions(df, output_dir)

    # Numerical distributions
    plot_numerical_distributions(df, output_dir)

    # Timestamp analysis
    plot_timestamp_analysis(df, output_dir)

    print("\nAnalysis complete! Plots saved to:", output_dir)
    print("Generated files:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")

def main():
    """Main function"""
    file_path = "/home/juncheng/workspace/prefix_cache/data/metrics_30day/"
    file_path = "/home/juncheng/workspace/prefix_cache/metrics_30day.csv"
    if os.path.isdir(file_path):
        for file in os.listdir(file_path):
            if file.endswith('.csv'):
                analyze_one_trace(os.path.join(file_path, file))
    else:
        analyze_one_trace(file_path)

if __name__ == "__main__":
    main()
