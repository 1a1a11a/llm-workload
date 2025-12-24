#!/usr/bin/env bash
set -euo pipefail

# Find the top 20 models by request count 
mapfile -t traces < <(
    wc -l /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/*.csv \
        | grep -v "total" \
        | sort -nr \
        | head -n 20 \
        | awk '{print $2}'
)

if ((${#traces[@]} == 0)); then
    echo "No traces found under /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/large" >&2
    exit 1
fi

echo "Top 20 models by number of requests:"
printf '%s\n' "${traces[@]}"

# Generate inter-arrival time plots (per model and per user) for the top traces
# python3 analysis/plot_per_model_arrival_time.py "${traces[@]}"
# python3 analysis/plot_per_model_arrival_time.py --per-user "${traces[@]}"
python3 analysis/plot_per_model_arrival_time.py "${traces[@]}" --compare-per-user
