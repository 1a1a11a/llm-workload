#! /bin/bash


traces=$(wc -l data/metrics_30day/per_model/*.csv | grep -v "total" | sort -nr | head -n 10 | awk '{print $2}')
echo "Top 10 models by number of requests:"
echo "$traces"
# then use the output to plot token distributions for these models
python3 analysis/plot_per_model_token_distributions.py $traces
