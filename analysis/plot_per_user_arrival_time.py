#!/usr/bin/env python3
"""
Per-User Inter-Arrival Time Analysis Script

This script processes individual user CSV files and creates CDF plots and statistical
distribution analysis for each user's inter-arrival times. Each CSV file should contain
data for a single user. 
The aggregated statistical distribution plots show the distribution of each user's 
P10, P25, Median, Mean, P75, and P90 inter-arrival times. 

Features:
- Parallel processing for improved performance
- Automatic selection of top users by request count
- Individual CDF plots for each user (log-scale)
- Statistical distribution analysis with CDF format in single figure
- Statistical comparison boxplot with proper ordering (Mean, P10, P25, Median, P75, P90)
- Folder-based naming for output files

Usage:
    python plot_per_user_arrival_time.py <csv_directory> --output-dir <output_dir>
    python plot_per_user_arrival_time.py data/metrics_30day/per_user/100k --output-dir figures/per_user_arrival
    python plot_per_user_arrival_time.py data/metrics_30day/per_user/100k --max-users 50 --workers 8

Examples:
    # Process all users with default CPU count workers
    python plot_per_user_arrival_time.py data/metrics_30day/per_user/100k
    
    # Process top 20 users with 4 parallel workers
    python plot_per_user_arrival_time.py data/metrics_30day/per_user/100k --max-users 20 --workers 4
    
    # Custom output directory
    python plot_per_user_arrival_time.py data/metrics_30day/per_user/100k --output-dir ./my_analysis
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style


def calculate_inter_arrival_times(csv_file: str) -> List[float]:
    """
    Calculate inter-arrival times for a single user's CSV file.

    Args:
        csv_file: Path to the CSV file for a single user

    Returns:
        List of inter-arrival times in seconds
    """
    timestamps = []
    
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try to get timestamp - handle both 'timestamp' and 'started_at' columns
            timestamp_str = row.get('started_at') or row.get('timestamp')
            if timestamp_str:
                try:
                    # Parse timestamp - handle different formats
                    if '.' in timestamp_str:
                        timestamp = pd.to_datetime(timestamp_str, format="mixed")
                    else:
                        timestamp = pd.to_datetime(timestamp_str)
                    timestamps.append(timestamp)
                except Exception as e:
                    print(f"Warning: Could not parse timestamp '{timestamp_str}' in {csv_file}: {e}")
                    continue
    
    if len(timestamps) < 2:
        return []
    
    # Sort timestamps to ensure chronological order
    timestamps.sort()
    
    # Calculate inter-arrival times
    inter_arrival_times = []
    for i in range(1, len(timestamps)):
        time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
        inter_arrival_times.append(time_diff)
    
    return inter_arrival_times


def plot_user_cdf(inter_arrival_times: List[float], user_id: str, output_path: Path) -> None:
    """
    Create a CDF plot for a single user's inter-arrival times.

    Args:
        inter_arrival_times: List of inter-arrival times for the user
        user_id: User identifier for the plot title
        output_path: Path to save the CDF plot
    """
    if len(inter_arrival_times) == 0:
        print(f"No inter-arrival times found for user {user_id}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Sort the data for CDF
    sorted_times = np.sort(inter_arrival_times)
    y_values = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    
    # Create CDF plot
    plt.plot(sorted_times, y_values, linewidth=2, label='Empirical CDF')
    
    # Add mean line
    mean_time = np.mean(inter_arrival_times)
    plt.axvline(float(mean_time), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_time:.2f}s')
    
    # Add median line
    median_time = np.median(inter_arrival_times)
    plt.axvline(float(median_time), color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_time:.2f}s')
    plt.xlabel("Inter-Arrival Time (seconds)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of Inter-Arrival Times for User {user_id}\n({len(inter_arrival_times)} requests)")
    
    # Use log scale for x-axis
    plt.xscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CDF plot saved: {output_path}")


def process_single_user(args) -> Dict:
    """
    Process a single user file for parallel processing.
    
    Args:
        args: Tuple of (csv_file_path, user_id, cdf_dir)
    
    Returns:
        Dictionary with user statistics or error info
    """
    csv_file_path, user_id, cdf_dir = args
    
    try:
        # Calculate inter-arrival times for this user
        inter_arrival_times = calculate_inter_arrival_times(csv_file_path)
        
        if len(inter_arrival_times) < 2:
            return {
                'user_id': user_id,
                'success': False,
                'error': f'Insufficient data ({len(inter_arrival_times)} timestamps)'
            }
        
        # Create CDF plot
        cdf_path = Path(cdf_dir) / f"{user_id}_inter_arrival_cdf.png"
        plot_user_cdf(inter_arrival_times, user_id, cdf_path)
        
        # Collect statistics for distribution plots
        stats = {
            'user_id': user_id,
            'mean': np.mean(inter_arrival_times),
            'median': np.median(inter_arrival_times),
            'p10': np.percentile(inter_arrival_times, 10),
            'p25': np.percentile(inter_arrival_times, 25),
            'p75': np.percentile(inter_arrival_times, 75),
            'p90': np.percentile(inter_arrival_times, 90),
            'count': len(inter_arrival_times),
            'success': True
        }
        return stats
        
    except Exception as e:
        return {
            'user_id': user_id,
            'success': False,
            'error': str(e)
        }


def plot_statistical_distributions(user_stats: List[Dict], output_dir: Path, csv_dir: str = "") -> None:
    """
    Create distribution plots for statistical measures across all users.

    Args:
        user_stats: List of dictionaries containing user statistics
        output_dir: Directory to save the distribution plots
        csv_dir: Input directory path for naming
    """
    if len(user_stats) == 0:
        print("No user statistics available for distribution plots")
        return
    
    # Extract statistical measures
    measures_data = {
        'Mean': [stats['mean'] for stats in user_stats],
        'P10': [stats['p10'] for stats in user_stats],
        'P25': [stats['p25'] for stats in user_stats],
        'Median': [stats['median'] for stats in user_stats],
        'P75': [stats['p75'] for stats in user_stats],
        'P90': [stats['p90'] for stats in user_stats]
    }
    
    # Create single figure with CDF plots for all measures
    plt.figure(figsize=(12, 8))
    
    # Define colors for each measure
    colors = ['blue', 'orange', 'purple', 'green', 'red', 'brown']
    
    # Plot CDF for each measure
    for i, (measure_name, values) in enumerate(measures_data.items()):
        sorted_values = np.sort(values)
        y_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        plt.plot(sorted_values, y_values, linewidth=2, label=measure_name, color=colors[i])
    
    plt.xlabel('Inter-Arrival Time (seconds)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF of Statistical Measures Across {len(user_stats)} Users')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Extract folder name from input directory for naming
    input_folder_name = Path(csv_dir).name
    distribution_path = output_dir / f"{input_folder_name}_statistical_distributions.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical distributions plot saved: {distribution_path}")
    
    # Create a combined comparison plot
    create_statistical_comparison_plot(user_stats, output_dir, csv_dir)


def create_statistical_comparison_plot(user_stats: List[Dict], output_dir: Path, csv_dir: str = "") -> None:
    """
    Create a boxplot comparison of all statistical measures.

    Args:
        user_stats: List of dictionaries containing user statistics
        output_dir: Directory to save the comparison plot
        csv_dir: Input directory path for naming
    """
    if len(user_stats) == 0:
        return
    
    # Extract statistical measures
    data_for_boxplot = []
    labels = []
    measures = ['mean', 'p10', 'p25', 'median', 'p75', 'p90']
    measure_names = ['Mean', 'P10', 'P25', 'Median', 'P75', 'P90']
    
    for measure in measures:
        values = [stats[measure] for stats in user_stats]
        data_for_boxplot.append(values)
        labels.append(measure_names[measures.index(measure)])
    
    # Create boxplot
    plt.figure(figsize=(14, 8))
    bp = plt.boxplot(data_for_boxplot, patch_artist=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    
    # Customize boxplot
    colors = ['lightblue', 'lightgreen', 'orange', 'purple', 'lightcoral', 'brown']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5)
    
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.5)
    
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    
    plt.xlabel("Statistical Measures")
    plt.ylabel("Inter-Arrival Time (seconds)")
    plt.title(f"Distribution Comparison of Inter-Arrival Time Statistics\n({len(user_stats)} users)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Extract folder name from input directory for naming
    if csv_dir:
        input_folder_name = Path(csv_dir).name
        comparison_path = output_dir / f"{input_folder_name}_statistical_comparison_boxplot.png"
    else:
        comparison_path = output_dir / "statistical_comparison_boxplot.png"
    
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Statistical comparison boxplot saved: {comparison_path}")


def process_user_files(csv_dir: str, output_dir: str, max_users: Optional[int] = None, max_workers: Optional[int] = None) -> None:
    """
    Process all user CSV files and create plots for each user.

    Args:
        csv_dir: Directory containing individual user CSV files
        output_dir: Directory to save plots
        max_users: Maximum number of users to process (None for all)
        max_workers: Number of parallel workers (None for CPU count)
    """
    csv_path = Path(csv_dir)
    
    if not csv_path.exists():
        raise RuntimeError(f"Directory not found: {csv_dir}")
    
    # Find all CSV files
    csv_files = list(csv_path.glob("*.csv"))
    
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found in {csv_dir}")
    print(f"Found {len(csv_files)} user files")
    
    # If max_users is specified, select users with most requests
    if max_users:
        print(f"Selecting top {max_users} users by request count...")
        user_request_counts = []
        
        for csv_file in csv_files:
            try:
                # Count lines in CSV file (excluding header)
                with open(csv_file, "r") as f:
                    line_count = sum(1 for line in f) - 1  # Subtract header
                user_request_counts.append((csv_file, line_count))
            except Exception as e:
                print(f"Warning: Could not count requests for {csv_file}: {e}")
                user_request_counts.append((csv_file, 0))
        
        # Sort by request count (descending) and take top users
        user_request_counts.sort(key=lambda x: x[1], reverse=True)
        csv_files = [item[0] for item in user_request_counts[:max_users]]
        
        print(f"Selected users with request counts: {[item[1] for item in user_request_counts[:max_users]]}")
        print(f"Processing top {max_users} users...")
    else:
        print(f"Processing all {len(csv_files)} users...")
    
    # Create output directories
    cdf_dir = Path(output_dir) / "cdfs"
    cdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Set number of workers
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    print(f"Using {max_workers} parallel workers...")
    
    # Prepare arguments for parallel processing
    process_args = [(str(csv_file), csv_file.stem, str(cdf_dir)) for csv_file in csv_files]
    
    user_stats = []
    users_processed = 0
    users_with_data = 0
    
    # Process users in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_user, args): args for args in process_args}
        
        # Process completed tasks
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            user_id = args[1]
            users_processed += 1
            
            if users_processed % 50 == 0:
                print(f"  Processed {users_processed}/{len(csv_files)} users...", flush=True)
            
            try:
                result = future.result()
                if result['success']:
                    users_with_data += 1
                    user_stats.append(result)
                else:
                    print(f"  Skipping user {user_id}: {result['error']}")
            except Exception as e:
                print(f"  Error processing user {user_id}: {e}")
    
    # Create distribution plots
    if user_stats:
        plot_statistical_distributions(user_stats, Path(output_dir), csv_dir)
    
    print(f"\nCompleted processing:")
    print(f"  Total users found: {len(csv_files)}")
    print(f"  Users processed: {users_processed}")
    print(f"  Users with valid data: {users_with_data}")
    print(f"  CDF plots saved to: {cdf_dir}")
    print(f"  Distribution plots saved to: {Path(output_dir)}")


def main():
    setup_plot_style()
    
    parser = argparse.ArgumentParser(
        description="Plot inter-arrival times for individual users"
    )
    parser.add_argument("csv_dir", help="Directory containing individual user CSV files")
    parser.add_argument("--output-dir", default="figures/per_user_arrival_time", 
                       help="Directory for saving plots (default: figures/per_user_arrival_time)")
    parser.add_argument("--max-users", type=int, default=None,
                       help="Maximum number of users to process (default: all)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                       help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    
    # Process all user files
    process_user_files(args.csv_dir, args.output_dir, args.max_users, args.workers)


if __name__ == "__main__":
    main()