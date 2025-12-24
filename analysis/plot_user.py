#!/usr/bin/env python3
"""
User-level metrics analysis and visualization for LLM workload data.

This script analyzes and plots aggregate user behavior metrics from a metrics CSV file.
It generates four key visualizations showing user request patterns and model access patterns:

1. CDF of requests per user - shows distribution of request counts across users
2. Rank plot of requests per user - shows power-law distribution (log-log scale)
3. CDF of models accessed per user - shows how many unique models users access
4. Rank plot of models per user - shows model diversity distribution (log-log scale)

Key Features:
- Caching support for expensive computations (CDF calculations)
- Force recompute option to refresh cached data
- Works with large CSV files by computing statistics incrementally
- Handles both user_id and chute_id columns (with automatic column detection)
- Generates high-resolution plots (300 DPI)

Input Requirements:
    CSV file with columns: user_id, chute_id (model identifier)
    Expected data location: /scratch/juncheng/data/prefix_cache/metrics_30day.csv
    Default: metrics_30day.csv

Output:
    - figures/user_analysis/requests_per_user_cdf.png
    - figures/user_analysis/requests_per_user_rank.png
    - figures/user_analysis/models_per_user_cdf.png
    - figures/user_analysis/models_per_user_rank.png
    - Cache files: requests_per_user_cdf.csv, models_per_user_cdf.csv

Usage:
    python3 plot_user.py [INPUT_FILE] [OPTIONS]

Options:
    --output-dir DIR        Output directory for plots and cache (default: figures/user_analysis/)
    --force-recompute      Recompute CDF data ignoring cache

Examples:
    # Use default file and cached data if available
    python3 plot_user.py

    # Process specific file
    python3 plot_user.py metrics_7day.csv

    # Force recomputation (ignore cache)
    python3 plot_user.py metrics_30day.csv --force-recompute

    # Custom output directory
    python3 plot_user.py --output-dir results/user_stats/
"""



import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style

try:
    from readers.data_loader import load_metrics_dataframe
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from readers.data_loader import load_metrics_dataframe


def validate_data_columns(df, required_columns):
    """Validate that the dataframe has all required columns.
    
    Args:
        df: Pandas DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If any required columns are missing
        
    Note:
        Provides helpful suggestions if alternative column names are found.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for alternative column names and provide helpful suggestions
    alternative_mappings = {
        "user_id": ["user", "userid", "user_id"],
        "chute_id": ["chute_id", "model_id", "chute", "model"],
    }

    for required, alternatives in alternative_mappings.items():
        if required not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    print(
                        f"Info: Found '{alt}' column, did you mean to use '{required}' instead?"
                    )
                    break


def _request_distribution_cache(
    df: pd.DataFrame, cache_file: str, use_cached: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute or load cached request distribution data.
    
    Calculates the CDF of requests per user. If cache exists and use_cached
    is True, loads from cache instead of recomputing.
    
    Args:
        df: DataFrame with user_id column
        cache_file: Path to cache file (CSV format)
        use_cached: Whether to use cached data if available
        
    Returns:
        Tuple of (sorted request counts, CDF values)
    """
    if use_cached and cache_file and os.path.exists(cache_file):
        print(f"Using cached data from {cache_file}")
        cached = pd.read_csv(cache_file)
        return (
            np.array(cached["requests"], dtype=float),
            np.array(cached["cdf"], dtype=float),
        )

    user_requests = df.groupby("user_id").size()
    requests = np.sort(user_requests.to_numpy())
    cdf = np.arange(len(requests), dtype=float) / float(len(requests) - 1)

    if cache_file:
        pd.DataFrame({"requests": requests, "cdf": cdf}).to_csv(cache_file, index=False)
        print(f"Data computed and saved to {cache_file}")

    return requests, cdf


def plot_requests_per_user_cdf(df, output_dir, use_cached=True):
    """Plot CDF of request counts per user.
    
    Shows what fraction of users have made up to N requests. Useful for
    understanding user activity distribution.
    
    Args:
        df: DataFrame with user_id column
        output_dir: Directory to save plot and cache
        use_cached: Whether to use cached CDF data if available
        
    Output:
        requests_per_user_cdf.png - CDF plot with log scale on x-axis
    """
    cache_file = os.path.join(output_dir, "requests_per_user_cdf.csv")
    requests, cdf = _request_distribution_cache(df, cache_file, use_cached)

    plt.figure(figsize=(10, 6))
    plt.plot(requests, cdf)
    plt.xlabel("Number of Requests")
    plt.ylabel("Fraction of Users")
    plt.title("CDF of Requests per User")
    plt.grid(True)
    plt.xscale("log")

    output_path = os.path.join(output_dir, "requests_per_user_cdf.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_requests_rank(df, output_dir, use_cached=True):
    """Plot rank vs request count on log-log scale.
    
    Shows power-law distribution of user activity. Top-ranked users have
    the most requests. Useful for identifying power users and overall
    activity skew.
    
    Args:
        df: DataFrame with user_id column
        output_dir: Directory to save plot
        use_cached: Whether to use cached data if available
        
    Output:
        requests_per_user_rank.png - Log-log rank plot
    """
    cache_file = os.path.join(output_dir, "requests_per_user_cdf.csv")
    requests, _ = _request_distribution_cache(df, cache_file, use_cached)
    requests = np.sort(requests)[::-1]
    ranks = np.arange(1, len(requests) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ranks, requests)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Number of Requests")
    ax.set_title("Requests per User Rank (Log-Log)")
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    output_path = os.path.join(output_dir, "requests_per_user_rank.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Rank plot saved to {output_path}")


def plot_models_per_user_cdf(df, output_dir, use_cached=True):
    """Plot CDF of unique models accessed per user.
    
    Shows the distribution of model diversity across users. Helps understand
    whether users stick to few models or explore many different ones.
    
    Args:
        df: DataFrame with user_id and chute_id columns
        output_dir: Directory to save plot and cache
        use_cached: Whether to use cached data if available
        
    Output:
        models_per_user_cdf.png - CDF plot with log scale on x-axis
    """
    cache_file = os.path.join(output_dir, "models_per_user_cdf.csv")
    if use_cached and os.path.exists(cache_file):
        print(f"Using cached data from {cache_file}")
        cached_data = pd.read_csv(cache_file)
        models = np.array(cached_data["models"], dtype=float)
        cdf = np.array(cached_data["cdf"], dtype=float)
    else:
        user_chutes = df.groupby("user_id")["chute_id"].apply(set)
        models = np.sort(user_chutes.apply(len).to_numpy())
        cdf = np.arange(len(models), dtype=float) / float(len(models) - 1)
        pd.DataFrame({"models": models, "cdf": cdf}).to_csv(cache_file, index=False)
        print(f"Data computed and saved to {cache_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(models, cdf)
    plt.xlabel("Number of Models")
    plt.ylabel("Fraction of Users")
    plt.title("CDF of Models Accessed per User")
    plt.grid(True)
    plt.xscale("log")

    output_path = os.path.join(output_dir, "models_per_user_cdf.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_models_rank(df, output_dir, use_cached=True):
    """Plot rank vs model count on log-log scale.
    
    Shows distribution of model diversity across users. Top-ranked users
    access the most unique models.
    
    Args:
        df: DataFrame with user_id and chute_id columns
        output_dir: Directory to save plot
        use_cached: Whether to use cached data if available
        
    Output:
        models_per_user_rank.png - Log-log rank plot
    """
    cache_file = os.path.join(output_dir, "models_per_user_cdf.csv")
    if use_cached and os.path.exists(cache_file):
        cached = pd.read_csv(cache_file)
        models = np.sort(np.array(cached["models"], dtype=float))[::-1]
    else:
        user_chutes = df.groupby("user_id")["chute_id"].apply(set)
        models = np.sort(user_chutes.apply(len).to_numpy())[::-1]
        pd.DataFrame({"models": models}).to_csv(cache_file, index=False)
        print(f"Data computed and saved to {cache_file}")

    ranks = np.arange(1, len(models) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ranks, models)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Number of Models")
    ax.set_title("Models per User Rank (Log-Log)")
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")

    output_path = os.path.join(output_dir, "models_per_user_rank.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Rank plot saved to {output_path}")


def _load_dataframe(input_file: str) -> pd.DataFrame:
    """Load metrics dataframe from CSV file.
    
    Attempts to use custom data loader if available, falls back to pandas read_csv.
    
    Args:
        input_file: Path to CSV file
        
    Returns:
        DataFrame with loaded metrics data
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        pd.errors.EmptyDataError: If input file is empty
    """
    try:
        return load_metrics_dataframe(input_file)
    except ImportError:
        print("Info: Standard data loader not available, using direct CSV loading")
        return pd.read_csv(input_file)


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for command-line interface.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Plot per-user usage statistics from a metrics CSV file"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="metrics_30day.csv",
        help="CSV file containing metrics data",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/user_analysis/",
        help="Directory where plots and cache files are written",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute and overwrite cached CDF data before plotting",
    )
    return parser


def main():
    """Main entry point for user-level metrics analysis.
    
    Orchestrates the entire analysis pipeline:
    1. Parse command-line arguments
    2. Load data (or skip if using cache)
    3. Validate required columns
    4. Generate all four plots
    
    The function uses caching to speed up repeated runs. Cache files store
    pre-computed CDF data. Use --force-recompute to refresh cache.
    """
    setup_plot_style()
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests_cache = output_dir / "requests_per_user_cdf.csv"
    models_cache = output_dir / "models_per_user_cdf.csv"
    have_cache = requests_cache.exists() and models_cache.exists()
    use_cached = (not args.force_recompute) and have_cache

    if use_cached:
        print("Using cached data only; skipping dataframe load")
        df = pd.DataFrame()  # unused because caches are read directly
    else:
        try:
            df = _load_dataframe(args.input_file)
        except FileNotFoundError:
            parser.error(f"Input file '{args.input_file}' not found")
        except pd.errors.EmptyDataError:
            parser.error(f"Input file '{args.input_file}' is empty")
        except Exception as exc:
            parser.error(f"Error loading data: {exc}")

    required_columns = ["user_id", "chute_id"]
    if not use_cached:
        validate_data_columns(df, required_columns)
        print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"Unique users: {df['user_id'].nunique()}")
        print(f"Unique models/chutes: {df['chute_id'].nunique()}")
    print(f"Using cached data: {use_cached}")

    plot_requests_per_user_cdf(df, str(output_dir), use_cached)
    plot_requests_rank(df, str(output_dir), use_cached)
    plot_models_per_user_cdf(df, str(output_dir), use_cached)
    plot_models_rank(df, str(output_dir), use_cached)


if __name__ == "__main__":
    main()
