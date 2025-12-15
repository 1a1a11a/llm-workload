#!/bin/bash
# Script to run analysis plotting

##### overview of the full trace #####
# full trace
python3 analysis/overview.py metrics_30day.csv
# overview of per-model stats
python3 analysis/overview.py data/metrics_30day/per_model/

##### user-level analysis #####
# user-level analysis
python3 analysis/plot_user.py metrics_30day.csv

##### token-level analysis #####
# token usage analysis, e.g., input and ouptut length distributions
# first find the ten largest models by number of requests
traces=$(wc -l data/metrics_30day/per_model/*.csv | grep -v "total" | sort -nr | head -n 10 | awk '{print $2}')
echo "Top 10 models by number of requests:"
echo "$traces"
# then use the output to plot token distributions for these models
python3 analysis/plot_per_model_token_distributions.py $traces



##### inter-arrival time analysis #####
for trace in data/metrics_30day/per_model/*.csv; do
    python3 analysis/plot_per_model_arrival_time.py "$trace"
done
for trace in data/metrics_30day/per_model/100k/*.csv; do
    python3 analysis/plot_per_model_arrival_time.py "$trace"
done

##### per-user analysis #####
# this plots the number of models accessed by each user in a fixed-size window of 1000 requests
python3 analysis/plot_per_user_model_diversity.py data/metrics_30day/per_user/1000k/
python3 analysis/plot_per_user_model_diversity.py data/metrics_30day/per_user/100k/
python3 analysis/plot_per_user_model_diversity.py data/metrics_30day/per_user/

# 