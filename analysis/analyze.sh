#!/bin/bash
# Script to run analysis plotting with process pool

set -euo pipefail
source "$(dirname "$0")/utils.sh"

# Configuration
MAX_PARALLEL_JOBS=8  # Number of parallel processes
DATA_FOLDER="/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/"

# Find the top 20 models by request count
find_top_traces "$DATA_FOLDER" -1 top_traces
echo "Found ${#top_traces[@]} traces to process"
for trace in "${top_traces[@]}"; do
    echo "Processing trace: $trace"
done



##### overview of the full trace #####
# full trace
run_with_pool "python3 analysis/overview.py /scratch/juncheng/data/prefix_cache/metrics_30day.csv" "overview_full_trace"

# overview of per-model stats
run_with_pool "python3 analysis/overview.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/" "overview_per_model"

##### user-level analysis #####
# user-level analysis
run_with_pool "python3 analysis/plot_user.py /scratch/juncheng/data/prefix_cache/metrics_30day.csv" "user_level_analysis"

##### per-user analysis #####
# this plots the number of models accessed by each user in a fixed-size window of 1000 requests
run_with_pool "python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/1000k/" "per_user_diversity_1000k"
run_with_pool "python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/100k/" "per_user_diversity_100k"
run_with_pool "python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/large/" "per_user_diversity_large"

##### model-level analysis #####
## inter-arrival time analysis
echo "Starting model-level inter-arrival time analysis..."

# Process each trace file in the main per_model directory
for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/*.csv; do
    if [ -f "$trace" ]; then
        run_with_pool "python3 analysis/plot_per_model_arrival_time.py \"$trace\" --compare-per-user" "arrival_time_$(basename "$trace")"
    fi
done

# Process each trace file in the 100k subdirectory
for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/100k/*.csv; do
    if [ -f "$trace" ]; then
        run_with_pool "python3 analysis/plot_per_model_arrival_time.py \"$trace\" --compare-per-user" "arrival_time_100k_$(basename "$trace")"
    fi
done

## plot top 20 models in a boxplot
run_with_pool "python3 analysis/plot_per_model_arrival_time.py \"${traces[@]}\"" "arrival_time_boxplot"
run_with_pool "python3 analysis/plot_per_model_arrival_time.py --per-user \"${traces[@]}\"" "arrival_time_boxplot_per_user"

## token usage analysis
# token usage analysis, e.g., input and ouptut length distributions
run_with_pool "python3 analysis/plot_per_model_token_distributions.py \"${traces[@]}\"" "token_distributions"

run_with_pool "python3 analysis/plot_per_model_ttft_vs_tokens.py \"${traces[@]}\"" "ttft_vs_tokens"

# Wait for all background jobs to complete
wait_for_jobs

echo "Analysis complete!"

