#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/utils.sh"

# Configuration
export MAX_PARALLEL_JOBS=64  # Number of parallel processes
DATA_FOLDER="/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/"
FULL_TRACE_FILE="/scratch/juncheng/data/prefix_cache/metrics_30day.csv"

# Find the top 20 models by request count
find_top_traces "$DATA_FOLDER" 10 top_traces
find_top_traces "$DATA_FOLDER" -1 all_traces
echo "Found ${#top_traces[@]} top traces to process"
echo "Found ${#all_traces[@]} all traces to process"


for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/*.csv; do
    run_with_pool "python3 /home/juncheng/workspace/prefix_cache/analysis/plot_per_model_io_length_correlation.py $trace" "io_length_correlation_$(basename "$trace")"
done
for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/large/*.csv; do
    run_with_pool "python3 /home/juncheng/workspace/prefix_cache/analysis/plot_per_model_io_length_correlation.py $trace" "io_length_correlation_$(basename "$trace")"
done
wait_for_jobs



