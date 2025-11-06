#!/usr/bin/env python3
"""
Inter-Arrival Time Analysis Script - Per User

This script analyzes the inter-arrival times of requests for each user separately,
showing how different users' request patterns vary. It creates a boxplot showing
inter-arrival time distributions across users and optionally summary statistics.

Usage:
    python plot_arrival_time_user.py /path/to/metrics.csv
    python plot_arrival_time_user.py ../data/metrics_30day/DeepSeek-R1.csv --max-users 15
    python plot_arrival_time_user.py ../data/metrics_30day/DeepSeek-R1.csv --summary --min-requests 10

Creates and saves boxplot by default, use --summary for additional statistical plots.
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


def load_and_process_data_per_user(filepath, min_requests_per_user=5):
    """
    Load CSV data and process timestamps for inter-arrival time analysis per user.

    Args:
        filepath (str): Path to the CSV file
        min_requests_per_user (int): Minimum number of requests per user to include

    Returns:
        pd.DataFrame: Processed dataframe with inter-arrival times per user
    """
    # Read CSV
    df = pd.read_csv(filepath)

    # Convert started_at to datetime
    df['started_at'] = pd.to_datetime(df['started_at'], format="mixed")

    # Group by user and process each user's data
    user_dataframes = []

    for user_id, user_df in df.groupby('user_id'):
        if len(user_df) < min_requests_per_user:
            continue  # Skip users with too few requests

        # Sort by started_at to ensure chronological order
        user_df = user_df.sort_values('started_at').reset_index(drop=True)

        # Calculate inter-arrival times (in seconds)
        user_df['inter_arrival_time'] = user_df['started_at'].diff().dt.total_seconds()

        # Remove the first row (which will have NaN for inter_arrival_time)
        user_df = user_df.dropna(subset=['inter_arrival_time'])

        user_dataframes.append(user_df)

    if not user_dataframes:
        print("No users found with sufficient data")
        return pd.DataFrame()

    # Combine all user data
    combined_df = pd.concat(user_dataframes, ignore_index=True)

    print(f"Processed {len(user_dataframes)} users")
    print(f"Total requests analyzed: {len(combined_df)}")
    print(f"Time range: {combined_df['started_at'].min()} to {combined_df['started_at'].max()}")

    return combined_df


def plot_inter_arrival_per_user_boxplot(df, title="Inter-Arrival Times by User", save_path=None, max_users=20):
    """
    Plot inter-arrival times as a boxplot grouped by user.

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' and 'user_id' columns
        title (str): Plot title
        save_path (str): Path to save the plot
        max_users (int): Maximum number of users to show (will select top users by request count)
    """
    plt.figure(figsize=(16, 10))

    # Get top users by number of requests
    user_counts = df['user_id'].value_counts().head(max_users)
    top_users = user_counts.index.tolist()

    # Filter data to only include top users
    df_filtered = df[df['user_id'].isin(top_users)]

    # Prepare data for boxplot
    user_data = []
    user_labels = []
    user_stats = []

    for user_id in top_users:
        user_times = df_filtered[df_filtered['user_id'] == user_id]['inter_arrival_time'].values
        if len(user_times) > 0:
            user_data.append(user_times)
            # Truncate user ID for display
            user_label = str(user_id)[:8] + "..." if len(str(user_id)) > 8 else str(user_id)
            user_labels.append(f"{user_label}\n({len(user_times)})")
            user_stats.append(len(user_times))

    # Create boxplot
    bp = plt.boxplot(user_data, patch_artist=True, tick_labels=user_labels)

    # Customize boxplot colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(user_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5)

    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.5)

    for median in bp['medians']:
        median.set(color='red', linewidth=2)

    for flier in bp['fliers']:
        flier.set(marker='o', color='red', alpha=0.5)

    # Add labels and title
    plt.xlabel("User ID (request count)", fontsize=12)
    plt.ylabel("Inter-Arrival Time (seconds)", fontsize=12)
    plt.title(f"{title}\n(Top {len(user_data)} users by request volume)", pad=20)

    # Use log scale for y-axis
    plt.yscale('log')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add grid
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    total_users = df['user_id'].nunique()
    total_requests = len(df)
    stats_text = f"""
    Total users: {total_users}
    Users shown: {len(user_data)}
    Total requests: {total_requests}
    Avg requests per user: {total_requests/total_users:.1f}
    """
    plt.text(0.02, 0.98, stats_text.strip(),
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Boxplot saved to {save_path}")


def plot_inter_arrival_per_user_summary(df, title="Inter-Arrival Time Summary by User", save_path=None):
    """
    Plot a summary of inter-arrival time statistics across all users.

    Args:
        df (pd.DataFrame): DataFrame with 'inter_arrival_time' and 'user_id' columns
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(14, 10))

    # Calculate statistics per user
    user_stats = df.groupby('user_id')['inter_arrival_time'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()

    # Filter out users with very few requests
    user_stats = user_stats[user_stats['count'] >= 3]

    if len(user_stats) == 0:
        print("No users with sufficient data for summary plot")
        return

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean inter-arrival time vs request count
    scatter1 = ax1.scatter(user_stats['count'], user_stats['mean'],
                          alpha=0.6, c=user_stats['std'], cmap='viridis')
    ax1.set_xlabel('Request Count')
    ax1.set_ylabel('Mean Inter-Arrival Time (s)')
    ax1.set_title('Mean vs Request Count')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Std Dev (s)')

    # Plot 2: Median inter-arrival time distribution
    ax2.hist(user_stats['median'], bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Median Inter-Arrival Time (s)')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('Median Inter-Arrival Distribution')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Coefficient of variation (std/mean)
    cv = user_stats['std'] / user_stats['mean']
    ax3.hist(cv.dropna(), bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax3.set_xlabel('Coefficient of Variation')
    ax3.set_ylabel('Number of Users')
    ax3.set_title('Inter-Arrival Variability')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Request frequency (requests per hour)
    # Calculate approximate hours of activity per user
    user_time_ranges = df.groupby('user_id')['started_at'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)
    request_frequency = user_stats.set_index('user_id')['count'] / user_time_ranges
    request_frequency = request_frequency.dropna()

    if len(request_frequency) > 0:
        ax4.hist(request_frequency, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax4.set_xlabel('Requests per Hour')
        ax4.set_ylabel('Number of Users')
        ax4.set_title('Request Frequency Distribution')
        ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Summary plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Inter-Arrival Times per User")
    parser.add_argument("file", type=str, help="Path to CSV file")
    parser.add_argument("--output", "-o", type=str, help="Output directory for plots")
    parser.add_argument(
        "--max-users", type=int, default=20,
        help="Maximum number of users to show in boxplot (default: 20)"
    )
    parser.add_argument(
        "--min-requests", type=int, default=5,
        help="Minimum requests per user to include (default: 5)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Create summary plots showing user statistics"
    )

    args = parser.parse_args()

    # Load and process data
    df = load_and_process_data_per_user(args.file, args.min_requests)

    if len(df) == 0:
        print("No valid data found. Exiting.")
        return

    # Basic statistics
    print("\nOverall Inter-Arrival Time Statistics:")
    print(df['inter_arrival_time'].describe())

    print("\nUser Statistics:")
    user_counts = df['user_id'].value_counts()
    print(f"Total users analyzed: {len(user_counts)}")
    print(f"Average requests per user: {user_counts.mean():.1f}")
    print(f"Median requests per user: {user_counts.median():.1f}")
    print(f"Max requests by single user: {user_counts.max()}")

    # Create plots
    output_dir = args.output or "."
    Path(output_dir).mkdir(exist_ok=True)

    base_title = "Inter-Arrival Times by User"
    if args.file:
        base_title += f" - {Path(args.file).name}"

    # Boxplot
    boxplot_path = Path(output_dir) / "inter_arrival_by_user_boxplot.png"
    plot_inter_arrival_per_user_boxplot(
        df, title=base_title, save_path=str(boxplot_path), max_users=args.max_users
    )

    # Summary plots if requested
    if args.summary:
        summary_path = Path(output_dir) / "inter_arrival_by_user_summary.png"
        plot_inter_arrival_per_user_summary(df, title=base_title, save_path=str(summary_path))


if __name__ == "__main__":
    main()
