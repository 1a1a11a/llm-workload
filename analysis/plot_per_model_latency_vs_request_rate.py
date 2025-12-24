"""
Analysis Script: TPOT/TTFT vs Request Rate & Token Rate (Moving Window)

This script calculates TPOT (Time Per Output Token), TTFT (Time To First Token), and both request/token rates
from a metrics CSV file and plots their relationships. Rates are calculated using a moving window.

TPOT = (Duration - TTFT) / (Output Tokens - 1)
TTFT = Time To First Token (from CSV)
Request Rate = window_size / (started_at[i] - started_at[i-window_size])
Token Rate = sum(tokens in window) / (started_at[i] - started_at[i-window_size])

Output:
- figures/per_model_latency_vs_request_rate/tpot/$subfolder/$chute_id.png
- figures/per_model_latency_vs_token_rate/tpot/$subfolder/$chute_id.png
- figures/per_model_latency_vs_request_rate/ttft/$subfolder/$chute_id.png
- figures/per_model_latency_vs_request_rate/ttft/$subfolder/$chute_id.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style

setup_plot_style()


def plot_relationship(
    df, x_col, y_col, x_label, y_label, title, output_path, stats_text
):
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        df[x_col],
        df[y_col],
        alpha=0.6,
        s=30,
        c=df[y_col],
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(y_label, rotation=270, labelpad=20)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.xscale("log")
    plt.yscale("log")

    plt.grid(True, linestyle="--", alpha=0.7)

    plt.text(
        0.02,
        0.98,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()


def process_single_file(file_path):
    print(f"Processing file: {file_path}")
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Basic validation
    required_cols = {
        "started_at",
        "completed_at",
        "input_tokens",
        "output_tokens",
        "ttft",
    }
    if not required_cols.issubset(df.columns):
        print(f"Missing columns in {file_path}. Found: {df.columns}")
        return

    # Calculate Duration and TPOT
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["completed_at"] = pd.to_datetime(df["completed_at"], errors="coerce")
    df["duration"] = (df["completed_at"] - df["started_at"]).dt.total_seconds()

    # Filter for streaming requests with valid ttft and output_tokens > 1
    mask = df["ttft"].notna() & (df["ttft"] > 0) & (df["output_tokens"] > 1)
    df_filtered = df[mask].copy()

    df_filtered["tpot"] = (df_filtered["duration"] - df_filtered["ttft"]) / (
        df_filtered["output_tokens"] - 1
    )
    df_filtered = df_filtered[df_filtered["tpot"] > 0]

    if len(df_filtered) < 1000:
        print(
            f"Skipping {file_path}: only {len(df_filtered)} valid requests (less than 1000)"
        )
        return

    # Sort by started_at to ensure chronological order for rate calculation
    df_filtered = df_filtered.sort_values("started_at").reset_index(drop=True)

    # Calculate window size
    window_size = min(100, len(df_filtered) // 2)
    if window_size < 1:
        window_size = 1

    print(f"Using moving window size of {window_size} requests for rate calculation")

    # Calculate duration between current request and window_size requests ago
    df_filtered["window_duration"] = (
        df_filtered["started_at"].diff(periods=window_size).dt.total_seconds()
    )

    # Calculate moving sum of tokens (input + output)
    df_filtered["total_tokens"] = (
        df_filtered["input_tokens"] + df_filtered["output_tokens"]
    )
    df_filtered["window_tokens"] = (
        df_filtered["total_tokens"].rolling(window=window_size).sum()
    )

    # Drop rows where we don't have a full window yet
    df_filtered = df_filtered.dropna(subset=["window_duration", "window_tokens"])

    # Enforce a minimum window duration to avoid extremely high rates
    df_filtered["window_duration"] = df_filtered["window_duration"].clip(lower=1e-3)

    # Calculate rates
    df_filtered["request_rate"] = float(window_size) / df_filtered["window_duration"]
    df_filtered["token_rate"] = (
        df_filtered["window_tokens"] / df_filtered["window_duration"]
    )

    if df_filtered.empty:
        print(f"No valid rates calculated for {file_path}")
        return

    # Setup output directories
    trace_name = os.path.basename(file_path).replace(".csv", "")
    chute_id = (
        str(df["chute_id"].iloc[0])
        if "chute_id" in df.columns and not df["chute_id"].empty
        else trace_name
    )

    path_parts = file_path.split(os.sep)
    subfolder = path_parts[-2] if len(path_parts) > 1 else "unknown"

    # 1. Plot TPOT vs Request Rate
    req_output_base = os.path.join(
        "figures", "per_model_latency_vs_request_rate/tpot", subfolder
    )
    os.makedirs(req_output_base, exist_ok=True)
    req_stats = f"""
    Total requests: {len(df_filtered):,}
    TPOT range: {df_filtered["tpot"].min():.4f} - {df_filtered["tpot"].max():.4f}s
    Request rate range: {df_filtered["request_rate"].min():.4f} - {df_filtered["request_rate"].max():.4f} req/s
    Median TPOT: {df_filtered["tpot"].median():.4f}s
    Median request rate: {df_filtered["request_rate"].median():.4f} req/s
    """
    plot_relationship(
        df_filtered,
        "request_rate",
        "tpot",
        f"Request Rate (requests/second, window={window_size})",
        "TPOT (seconds/token)",
        f"TPOT vs Request Rate (Moving Window={window_size}) - {trace_name}",
        os.path.join(req_output_base, f"{trace_name}.png"),
        req_stats,
    )

    # 2. Plot TPOT vs Token Rate
    tok_output_base = os.path.join(
        "figures", "per_model_latency_vs_token_rate", subfolder
    )
    os.makedirs(tok_output_base, exist_ok=True)
    tok_stats = f"""
    Total requests: {len(df_filtered):,}
    TPOT range: {df_filtered["tpot"].min():.4f} - {df_filtered["tpot"].max():.4f}s
    Token rate range: {df_filtered["token_rate"].min():.4f} - {df_filtered["token_rate"].max():.4f} tokens/s
    Median TPOT: {df_filtered["tpot"].median():.4f}s
    Median token rate: {df_filtered["token_rate"].median():.4f} tokens/s
    """
    plot_relationship(
        df_filtered,
        "token_rate",
        "tpot",
        f"Token Rate (total tokens/second, window={window_size})",
        "TPOT (seconds/token)",
        f"TPOT vs Token Rate (Moving Window={window_size}) - {trace_name}",
        os.path.join(tok_output_base, f"{trace_name}.png"),
        tok_stats,
    )

    # 3. Plot TTFT vs Request Rate
    ttft_req_output_base = os.path.join(
        "figures", "per_model_latency_vs_request_rate/ttft", subfolder
    )
    os.makedirs(ttft_req_output_base, exist_ok=True)
    ttft_req_stats = f"""
    Total requests: {len(df_filtered):,}
    TTFT range: {df_filtered["ttft"].min():.4f} - {df_filtered["ttft"].max():.4f}s
    Request rate range: {df_filtered["request_rate"].min():.4f} - {df_filtered["request_rate"].max():.4f} req/s
    Median TTFT: {df_filtered["ttft"].median():.4f}s
    Median request rate: {df_filtered["request_rate"].median():.4f} req/s
    """
    plot_relationship(
        df_filtered,
        "request_rate",
        "ttft",
        f"Request Rate (requests/second, window={window_size})",
        "TTFT (seconds)",
        f"TTFT vs Request Rate (Moving Window={window_size}) - {trace_name}",
        os.path.join(ttft_req_output_base, f"{trace_name}.png"),
        ttft_req_stats,
    )

    # 4. Plot TTFT vs Token Rate
    ttft_tok_output_base = os.path.join(
        "figures", "per_model_latency_vs_request_rate/ttft", subfolder
    )
    os.makedirs(ttft_tok_output_base, exist_ok=True)
    ttft_tok_stats = f"""
    Total requests: {len(df_filtered):,}
    TTFT range: {df_filtered["ttft"].min():.4f} - {df_filtered["ttft"].max():.4f}s
    Token rate range: {df_filtered["token_rate"].min():.4f} - {df_filtered["token_rate"].max():.4f} tokens/s
    Median TTFT: {df_filtered["ttft"].median():.4f}s
    Median token rate: {df_filtered["token_rate"].median():.4f} tokens/s
    """
    plot_relationship(
        df_filtered,
        "token_rate",
        "ttft",
        f"Token Rate (total tokens/second, window={window_size})",
        "TTFT (seconds)",
        f"TTFT vs Token Rate (Moving Window={window_size}) - {trace_name}",
        os.path.join(ttft_tok_output_base, f"{trace_name}.png"),
        ttft_tok_stats,
    )

    return chute_id


def main():
    if len(sys.argv) < 2:
        # Default test file from rules
        test_file = "/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/100k/23346232-a0be-5448-91be-596f7ab832c2.csv"
        process_single_file(test_file)
    else:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            process_single_file(file_path)
        else:
            print(f"Error: {file_path} is not a valid file.")


if __name__ == "__main__":
    main()
