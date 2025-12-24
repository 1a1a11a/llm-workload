#!/bin/bash
# Script to run analysis plotting with process pool

set -euo pipefail
source "$(dirname "$0")/utils.sh"

# Configuration
export MAX_PARALLEL_JOBS=8  # Number of parallel processes
DATA_FOLDER="/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/"
FULL_TRACE_FILE="/scratch/juncheng/data/prefix_cache/metrics_30day.csv"

# Find the top 20 models by request count
find_top_traces "$DATA_FOLDER" 10 top_traces
find_top_traces "$DATA_FOLDER" -1 all_traces
echo "Found ${#top_traces[@]} top traces to process"
echo "Found ${#all_traces[@]} all traces to process"

##### overview of the full trace #####
## full trace
## model-level analysis
run_with_pool "python3 analysis/overview.py $FULL_TRACE_FILE" "overview_full_trace"
# user-level analysis
run_with_pool "python3 analysis/plot_user.py $FULL_TRACE_FILE" "user_level_analysis"


##### model-level analysis #####
## inter-arrival time analysis
echo "Starting model-level inter-arrival time analysis..."
for trace in ${all_traces[@]}; do
    # overview of per-model stats
    run_with_pool "python3 analysis/overview.py $trace" "overview_$(basename "$trace")"

    # compare per-user and all-user inter-arrival time
    run_with_pool "python3 analysis/plot_per_model_arrival_time.py $trace --compare-per-user" "arrival_time_$(basename "$trace")_compare_per_user"

    # TTFT vs input tokens
    run_with_pool "python3 analysis/plot_per_model_ttft_vs_tokens.py $trace" "ttft_vs_tokens_$(basename "$trace")"
done

## plot top 20 models in a boxplot
# plot all-user inter-arrival time
run_with_pool "python3 analysis/plot_per_model_arrival_time.py ${top_traces[@]}" "arrival_time_boxplot"
# plot per-user inter-arrival time
run_with_pool "python3 analysis/plot_per_model_arrival_time.py --per-user ${top_traces[@]}" "arrival_time_boxplot_per_user"

## token usage analysis
# token usage analysis, e.g., input and ouptut length distributions
run_with_pool "python3 analysis/plot_per_model_token_distributions.py ${top_traces[@]}" "token_distributions"









##### per-user analysis #####
##### user-level analysis #####



# this plots the number of models accessed by each user in a fixed-size window of 1000 requests
run_with_pool "python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/1000k/" "per_user_diversity_1000k"
run_with_pool "python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/100k/" "per_user_diversity_100k"
run_with_pool "python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/large/" "per_user_diversity_large"


# Wait for all background jobs to complete
wait_for_jobs
echo "Analysis complete!"
