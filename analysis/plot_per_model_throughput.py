#!/usr/bin/env python3
"""
Throughput Analysis Script
Calculates effective prefill and decode throughput per second.

Prefill throughput: input_tokens / ttft
Decode throughput: output_tokens / (duration - ttft)

README:
This script takes one or more per-model CSV files and calculates the throughput
for each second of the trace. It produces:
1. Time-series plots of throughput (tokens/sec) aggregated by hour.
2. Time-series plots of throughput (tokens/sec) aggregated by minute.
The data used for the plots is saved in a 'data' subfolder within the output directory.

All calculations are done per second as requested.

Usage:
    python plot_per_model_throughput.py /path/to/trace.csv
    python plot_per_model_throughput.py trace1.csv trace2.csv --workers 4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys

# Add parent directory to path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plot import setup_plot_style

setup_plot_style()

def infer_output_dir(sample_file, analysis_type="per_model_throughput"):
    """
    Infer an output directory based on the sample file path.
    Example: /.../1000k/DeepSeek-R1.csv -> figures/per_model_throughput/1000k/
    """
    sample_path = Path(sample_file)
    # The parent directory name (e.g., 1k, 100k, 1000k)
    category = sample_path.parent.name
    
    base_dir = Path("figures") / analysis_type
    
    if category in ["1k", "10k", "100k", "1000k", "large"]:
        return base_dir / category
    return base_dir

def calculate_binned_throughput(events, trace_start_time, trace_end_time):
    """
    events: list of (timestamp, delta_rate)
    Returns: (throughput_array, bin_start_times)
    """
    if not events:
        return np.array([]), np.array([])
    
    events.sort()
    times, deltas = zip(*events)
    times = np.array(times)
    deltas = np.array(deltas).astype(float)
    
    # Instantaneous rates (piecewise constant)
    rates = np.cumsum(deltas)
    
    # Cumulative tokens F(t) = integral of rate from 0 to t
    # F(t) is piecewise linear
    durations = np.diff(times)
    token_increments = rates[:-1] * durations
    cum_tokens = np.concatenate(([0.0], np.cumsum(token_increments)))
    
    # Sample F(k) for each second k
    t_start = np.floor(times[0])
    t_end = np.ceil(times[-1])
    
    # We use a 1-second interval
    query_times = np.arange(t_start, t_end + 1)
    
    # F(t) is piecewise linear. interp works perfectly.
    F_sampled = np.interp(query_times, times, cum_tokens)
    
    # Throughput in bin k = F(k+1) - F(k)
    throughput = np.diff(F_sampled)
    
    return throughput, query_times[:-1]

def process_single_trace(file_path):
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)

    if df.empty:
        print(f"Empty dataframe for {file_path}")
        return None

    # Convert timestamps
    df['started_at'] = pd.to_datetime(df['started_at'], format='mixed')
    df['completed_at'] = pd.to_datetime(df['completed_at'], format='mixed')
    df['duration'] = (df['completed_at'] - df['started_at']).dt.total_seconds()
    
    # Handle ttft
    df['ttft'] = pd.to_numeric(df['ttft'], errors='coerce')
    
    prefill_events = []
    decode_events = []
    
    # Use float timestamps for calculation
    df['s_ts'] = df['started_at'].apply(lambda x: x.timestamp())
    df['e_ts'] = df['completed_at'].apply(lambda x: x.timestamp())
    
    # Prefill: throughput = input_tokens / ttft
    valid_prefill = df[df['ttft'] > 0].copy()
    if not valid_prefill.empty:
        valid_prefill['p_rate'] = valid_prefill['input_tokens'] / valid_prefill['ttft']
        for _, row in valid_prefill.iterrows():
            prefill_events.append((row['s_ts'], row['p_rate']))
            prefill_events.append((row['s_ts'] + row['ttft'], -row['p_rate']))
            
    # Decode: throughput = output_tokens / (duration - ttft)
    valid_decode = df[(df['duration'] > df['ttft']) & (df['ttft'] >= 0)].copy()
    if not valid_decode.empty:
        valid_decode['d_duration'] = valid_decode['duration'] - valid_decode['ttft']
        valid_decode['d_rate'] = valid_decode['output_tokens'] / valid_decode['d_duration']
        for _, row in valid_decode.iterrows():
            decode_events.append((row['s_ts'] + row['ttft'], row['d_rate']))
            decode_events.append((row['e_ts'], -row['d_rate']))

    if not prefill_events and not decode_events:
        print(f"No valid events found in {file_path}")
        return None

    trace_start = df['s_ts'].min()
    trace_end = df['e_ts'].max()
    
    p_throughput, p_times = calculate_binned_throughput(prefill_events, trace_start, trace_end)
    d_throughput, d_times = calculate_binned_throughput(decode_events, trace_start, trace_end)
    
    # Align prefill and decode onto a common second-by-second timeline
    all_times = np.unique(np.concatenate([p_times, d_times]))
    res_df = pd.DataFrame({'ts': all_times})
    
    if len(p_times) > 0:
        p_df = pd.DataFrame({'ts': p_times, 'prefill': p_throughput})
        res_df = res_df.merge(p_df, on='ts', how='left').fillna(0)
    else:
        res_df['prefill'] = 0.0
        
    if len(d_times) > 0:
        d_df = pd.DataFrame({'ts': d_times, 'decode': d_throughput})
        res_df = res_df.merge(d_df, on='ts', how='left').fillna(0)
    else:
        res_df['decode'] = 0.0
        
    res_df['timestamp'] = pd.to_datetime(res_df['ts'], unit='s')
    res_df['hour'] = res_df['timestamp'].dt.hour
    res_df['date'] = res_df['timestamp'].dt.date
    
    trace_name = Path(file_path).stem
    return {
        'trace_name': trace_name,
        'data': res_df,
        'file_path': file_path
    }

def plot_throughput_timeseries(df, resample_rate, title, ylabel, save_path, data_dir, filename_prefix):
    """
    Plots a time series of throughput after resampling.
    """
    resampled = df.set_index('timestamp').resample(resample_rate).mean(numeric_only=True).reset_index()
    
    # Save the resampled data
    resampled.to_csv(data_dir / f"{filename_prefix}_{resample_rate}_mean.csv", index=False)
    
    plt.figure(figsize=(12, 6))
    
    # Filter out zeros to avoid cluttering if the trace has long idle periods? 
    # Actually, keep them to show gaps.
    
    plt.plot(resampled['timestamp'], resampled['prefill'], label='Prefill', alpha=0.8)
    plt.plot(resampled['timestamp'], resampled['decode'], label='Decode', alpha=0.8)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Time")
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze throughput per second.")
    parser.add_argument("files", nargs="+", help="Per-model CSV files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = parser.parse_args()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_trace, f): f for f in args.files}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    for res in results:
        df = res['data']
        trace_name = res['trace_name']

        # Skip traces with max minute token rate < 1000
        # df['prefill'] and df['decode'] are tokens/sec; sum over 1min gives tokens/min
        minute_tokens = df.set_index('timestamp').resample('1min').sum(numeric_only=True)
        max_tokens_per_minute = (minute_tokens['prefill'] + minute_tokens['decode']).max()

        if max_tokens_per_minute < 1000:
            print(f"Skipping {trace_name}: max minute token rate {max_tokens_per_minute:.2f} < 1000")
            continue

        output_dir = infer_output_dir(res['file_path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save raw per-second data
        df.to_csv(data_dir / f"{trace_name}_throughput_per_second.csv", index=False)
        
        # Plot Time Series - Hourly
        plot_throughput_timeseries(
            df, '1h',
            f"Throughput Over Time (Hourly Average)\n({trace_name})",
            "Tokens / sec",
            output_dir / f"{trace_name}_hourly.png",
            data_dir,
            trace_name
        )
        
        # Plot Time Series - 1 Minute (High resolution)
        plot_throughput_timeseries(
            df, '1min',
            f"Throughput Over Time (1-Minute Average)\n({trace_name})",
            "Tokens / sec",
            output_dir / f"{trace_name}_1min.png",
            data_dir,
            trace_name
        )

if __name__ == "__main__":
    main()

