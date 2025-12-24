#!/bin/bash
# Script to run analysis plotting

##### find top 20 models #####
mapfile -t traces < <(
    wc -l /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/large/*.csv \
        | grep -v "total" \
        | sort -nr \
        | head -n 20 \
        | awk '{print $2}'
)

if ((${#traces[@]} == 0)); then
    echo "No traces found under /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/large/" >&2
    exit 1
fi

echo "Top 20 models by number of requests:"
printf '%s\n' "${traces[@]}"




##### overview of the full trace #####
# full trace
python3 analysis/overview.py /scratch/juncheng/data/prefix_cache/metrics_30day.csv
# overview of per-model stats
python3 analysis/overview.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/

##### user-level analysis #####
# user-level analysis
python3 analysis/plot_user.py /scratch/juncheng/data/prefix_cache/metrics_30day.csv



##### per-user analysis #####
# this plots the number of models accessed by each user in a fixed-size window of 1000 requests
python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/1000k/
python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/100k/
python3 analysis/plot_per_user_model_diversity.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_user/large/



##### model-level analysis #####
## inter-arrival time analysis
for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/*.csv; do
    # python3 analysis/plot_per_model_arrival_time.py "$trace" &
    # python3 analysis/plot_per_model_arrival_time.py "$trace" --per-user &
    python3 analysis/plot_per_model_arrival_time.py "${traces[@]}" --compare-per-user &
done

for trace in /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/100k/*.csv; do
    # python3 analysis/plot_per_model_arrival_time.py "$trace" &
    # python3 analysis/plot_per_model_arrival_time.py "$trace" --per-user &
    python3 analysis/plot_per_model_arrival_time.py "${traces[@]}" --compare-per-user &
done

## plot top 20 models in a boxplot
python3 analysis/plot_per_model_arrival_time.py "${traces[@]}" &
python3 analysis/plot_per_model_arrival_time.py --per-user "${traces[@]}" &


## token usage analysis
# token usage analysis, e.g., input and ouptut length distributions
# # first find the ten largest models by number of requests
# traces=$(wc -l /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/*.csv | grep -v "total" | sort -nr | head -n 10 | awk '{print $2}')
# echo "Top 10 models by number of requests:"
# echo "$traces"
# then use the output to plot token distributions for these models
python3 analysis/plot_per_model_token_distributions.py $traces


#
