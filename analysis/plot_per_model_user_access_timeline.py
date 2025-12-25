#!/usr/bin/env python3
"""
User Access Timeline Visualization
==================================

This script generates a scatter plot showing how different users access a model over time.
It takes a single per-model trace as input, samples up to 240 users, and plots their
request timestamps on a timeline.

Features:
---------
1. **User Sampling**: Randomly samples up to 240 users if the trace contains more.
2. **Temporal Sorting**: Users are ordered on the y-axis by the time of their first request.
3. **Timeline Visualization**: X-axis shows the timeline of requests, Y-axis shows individual users.

Usage:
------
    python analysis/plot_per_model_user_access_timeline.py /path/to/trace.csv

Output:
-------
- Figure: figures/per_model_user_access_timeline/{scale}/{trace_name}.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os
import numpy as np

# Add root to sys.path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style

def infer_output_paths(input_file: str):
    """Infer output directory and filename based on input file path."""
    input_path = Path(input_file)
    trace_name = input_path.stem
    
    # Identify scale (1k, 10k, 100k, 1000k, large)
    scale = "unknown"
    for s in ["1k", "10k", "100k", "1000k", "large"]:
        if f"/{s}/" in str(input_path):
            scale = s
            break
            
    base_output_dir = Path("figures/per_model_user_access_timeline") / scale
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = base_output_dir / f"{trace_name}.png"
    
    return fig_path

def preprocess_data(df: pd.DataFrame, max_users: int = 240):
    """Sample users and prepare data for plotting."""
    # Ensure started_at is datetime
    df['started_at'] = pd.to_datetime(df['started_at'], format='ISO8601')
    
    # Get unique users
    unique_users = df['user_id'].unique()
    
    if len(unique_users) > max_users:
        # Randomly sample users
        np.random.seed(42)  # For reproducibility
        sampled_users = np.random.choice(unique_users, size=max_users, replace=False)
        df = df[df['user_id'].isin(sampled_users)].copy()
    else:
        sampled_users = unique_users

    # Determine first request time for each sampled user to sort them on Y-axis
    user_first_request = df.groupby('user_id')['started_at'].min().sort_values()
    user_to_idx = {user_id: i for i, user_id in enumerate(user_first_request.index)}
    
    df['user_idx'] = df['user_id'].map(user_to_idx)
    
    return df

def plot_timeline(df: pd.DataFrame, fig_path: Path, trace_name: str, total_requests: int):
    """Generate and save the scatter plot."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Use a colormap for different users
    # tab20, tab20b, or tab20c provide good distinct colors
    colors = plt.cm.tab20(df['user_idx'] % 20)
    
    # Scatter plot: x=started_at, y=user_idx
    ax.scatter(df['started_at'], df['user_idx'], s=10, alpha=0.6, c=colors)
    
    ax.set_xlabel("Timeline")
    ax.set_ylabel("User Index (sorted by arrival)")
    ax.set_title(f"Trace: {trace_name} (Total Requests: {total_requests})")
    
    # Rotate xticklabels
    plt.xticks(rotation=90)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved figure to {fig_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot user access timeline for a model trace.")
    parser.add_argument("trace_file", type=str, help="Path to the per-model trace CSV file.")
    parser.add_argument("--max-users", type=int, default=240, help="Maximum number of users to sample (default: 240).")
    args = parser.parse_args()
    
    trace_file = args.trace_file
    if not os.path.exists(trace_file):
        print(f"Error: File {trace_file} not found.")
        sys.exit(1)
        
    print(f"Loading data from {trace_file}...")
    # Read only necessary columns
    df = pd.read_csv(trace_file, usecols=['user_id', 'started_at'])
    total_requests = len(df)
    
    # Check if there are fewer than 20 users
    num_unique_users = df['user_id'].nunique()
    if num_unique_users < 20:
        print(f"Skipping {trace_file}: only {num_unique_users} users found (minimum 20 required).")
        return

    print(f"Preprocessing data (sampling up to {args.max_users} users)...")
    df_plot = preprocess_data(df, max_users=args.max_users)
    
    fig_path = infer_output_paths(trace_file)
    trace_name = Path(trace_file).stem
    
    plot_timeline(df_plot, fig_path, trace_name, total_requests)
    

if __name__ == "__main__":
    main()

