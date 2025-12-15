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
    """Validate that the dataframe has all required columns."""
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
    """CDF of number of models each user accesses (cached when available)."""
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
    try:
        return load_metrics_dataframe(input_file)
    except ImportError:
        print("Info: Standard data loader not available, using direct CSV loading")
        return pd.read_csv(input_file)


def _build_parser() -> argparse.ArgumentParser:
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
