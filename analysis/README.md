# LLM Workload Analysis Toolkit

This repository contains a collection of scripts for analyzing LLM workload metrics data. The analysis focuses on understanding user behavior, model usage patterns, and request timing characteristics.

## Data Structure

The analysis expects data to be located at:
```
/scratch/juncheng/data/prefix_cache/
├── metrics_30day.csv
├── metrics_1day.csv
├── data/
│   └── metrics_30day/
│       └── per_model/
│           ├── model1.csv
│           ├── model2.csv
│           └── ...
```

## Analysis Scripts

### 1. Overview Analysis (`overview.py`)
- Analyzes overall metrics distribution
- Generates CDF plots for unique values
- Creates categorical distributions (pie charts, bar charts)
- Shows numerical distributions (token usage, timing)
- Temporal analysis of request patterns

### 2. User-Level Analysis (`plot_user.py`)
- CDF of requests per user
- Rank plot of requests per user (power-law distribution)
- CDF of models accessed per user
- Rank plot of models per user (model diversity)

### 3. Per-Model Analysis (`plot_per_model_token_distributions.py`)
- Input token distribution CDF
- Output token distribution CDF
- Output/Input ratio distribution CDF
- Individual model plots
- Combined plots for multiple models

### 4. Inter-Arrival Time Analysis (`plot_per_model_arrival_time.py`)
- CDF of inter-arrival times
- Hourly boxplot analysis
- Daily boxplot analysis
- Correlation between consecutive inter-arrival times
- Probability heatmap of current vs next inter-arrival times

### 5. Per-User Model Diversity (`plot_per_user_model_diversity.py`)
- Analysis of model diversity per user
- Plots showing number of models accessed by each user
- Works with different data window sizes (100k, 1000k, large)

## Usage

### Running Individual Scripts

```bash
# Overview analysis
python3 analysis/overview.py /scratch/juncheng/data/prefix_cache/metrics_30day.csv

# User analysis
python3 analysis/plot_user.py /scratch/juncheng/data/prefix_cache/metrics_30day.csv

# Token distribution analysis (for top models)
traces=$(wc -l /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/*.csv | grep -v "total" | sort -nr | head -n 10 | awk '{print $2}')
python3 analysis/plot_per_model_token_distributions.py $traces

# Inter-arrival time analysis
python3 analysis/plot_per_model_arrival_time.py /scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/DeepSeek-R1.csv
```

### Running Full Analysis Pipeline

```bash
# Run all analysis scripts in sequence
./analysis/analyze.sh
```

## Output Structure

All generated plots and figures are saved to:
```
figures/
├── overview/
├── user_analysis/
├── per_model_token_distributions/
├── per_model_arrival_time/
├── per_model_arrival_time_per_user/
├── per_user_model_diversity/
└── ...
```

## Key Features

- **Parallel Processing**: Many scripts support parallel processing for performance
- **Caching**: CDF calculations are cached to speed up repeated runs
- **Flexible Input**: Scripts can handle both single files and directories
- **Comprehensive Visualization**: Multiple plot types for different analysis needs
- **Statistical Summary**: Automatic generation of statistical summaries in plots

## Dependencies

- Python 3 with pandas, matplotlib, numpy
- Required Python modules in `readers/` and `utils/` directories
- Bash shell with standard utilities

## Data Requirements

The scripts expect specific column names in the CSV files:
- `user_id`: Unique identifier for users
- `chute_id` or `model_id`: Model identifier
- `input_tokens`, `output_tokens`: Token usage metrics
- `started_at`, `completed_at`: Timestamps for requests
- `ttft`, `duration`: Timing metrics

## Caching Behavior

- CDF data is cached in CSV files to avoid recomputation
- Cache files are stored in the same directory as output plots
- Use `--force-recompute` flag to refresh cached data
