#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/utils.sh"

DATA_FOLDER="/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/"

find_top_traces "$DATA_FOLDER" -1 traces

# Generate inter-arrival time plots (per model and per user) for the top traces
# python3 analysis/plot_per_model_arrival_time.py "${traces[@]}"
# python3 analysis/plot_per_model_arrival_time.py --per-user "${traces[@]}"
# python3 analysis/plot_per_model_arrival_time.py "${traces[@]}" --compare-per-user

# for trace in "${traces[@]}"; do
#     # ttft
#     python3 /home/juncheng/workspace/prefix_cache/analysis/plot_per_model_ttft_vs_tokens.py "$trace" &
# done


for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/*.csv; do
    run_with_pool "python3 /home/juncheng/workspace/prefix_cache/analysis/plot_per_model_ttft_vs_tokens.py \"$trace\"" "ttft_vs_tokens_$(basename "$trace")"
done
for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/large/*.csv; do
    # ttft
    run_with_pool "python3 /home/juncheng/workspace/prefix_cache/analysis/plot_per_model_ttft_vs_tokens.py \"$trace\"" "ttft_vs_tokens_$(basename "$trace")"
done
wait_for_jobs
