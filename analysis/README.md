# Analysis Scripts

This directory contains Python scripts for analyzing metrics data and generating statistical visualizations.

## Files Overview

### `overview.py`
**Comprehensive metrics analysis script**
This is the initial script to run for a broad overview of metrics data. This can be done per-model or across all models. 
The input is hardcoded in the script, it takes either a single CSV file or a directory of CSV files. 

Analyzes metrics data (CSV format) and generates statistical plots:
- **CDF plots** of unique values per column
- **Categorical distributions** with pie charts and bar charts
- **Numerical distributions** with CDF plots for token metrics
- **Temporal analysis** with hourly/daily distribution plots

**Key Features:**
- Loads data using `load_metrics_dataframe` utility
- Excludes unique identifiers (invocation_id, timestamps)
- Supports multiprocessing for batch file processing
- Generates high-resolution plots (300 DPI)
- Outputs to organized directory structure

**Usage:**
```bash
python overview.py
```

---

### `plot_arrival_time_model.py`
**Inter-arrival time analysis per-model across all users**
Analyzes request arrival patterns and timing distributions:
- **CDF plot** of inter-arrival times (log-scale)
- **Boxplot by hour** showing temporal patterns with mean lines
- **Daily boxplot** showing day-to-day variation in arrival patterns with mean lines
- **Correlation analysis** with prominent Pearson and Spearman coefficient display between consecutive arrival times
- **Probability heatmap** showing joint distribution of current vs next inter-arrival times with log-scale millisecond display and outlier filtering (excludes bottom/top 0.01% extreme values to focus on 99.98% of main distribution)

**Key Features:**
- Calculates time differences between consecutive requests
- Uses log-scale visualization for wide value ranges
- Provides statistical summaries (mean, median, std, etc.)
- Generates scatter plots for correlation analysis
- Heatmap includes outlier filtering to focus visualization on main distribution
- Shows both parametric (Pearson) and non-parametric (Spearman) correlation coefficients
- Automatic timestamp column detection (handles both 'timestamp' and 'started_at' formats)

**Usage:**
```bash
python plot_arrival_time_model.py /path/to/metrics.csv
```

---

### `plot_arrival_time_user.py`
**Per-user inter-arrival time analysis**

Analyzes arrival patterns for individual users:
- **Boxplot by user** showing user-specific timing patterns
- **Summary statistics** across all users
- **Frequency analysis** (requests per hour per user)
- **Variability metrics** (coefficient of variation)

**Key Features:**
- Groups data by user_id for individual analysis
- Configurable minimum requests per user threshold
- Selects top users by request volume for visualization
- Supports summary plot generation with `--summary` flag

**Usage:**
```bash
python plot_arrival_time_user.py /path/to/metrics.csv --max-users 15 --min-requests 10
```

---

### `plot_token_distributions.py`
**Token usage distribution analysis**

Analyzes token consumption patterns across different models:
- **Input tokens CDF** distribution analysis
- **Output tokens CDF** distribution analysis  
- **Output/Input ratio** analysis across models
- **Multi-model comparison** capabilities

**Key Features:**
- Extracts token data using model_name or chute_id filtering
- Creates CDF plots for multiple models simultaneously
- Configurable model selection (top-K by frequency or specific models)
- Supports ratio calculations for efficiency analysis

**Usage:**
```bash
python plot_token_distributions.py /path/to/metrics.csv --models DeepSeek-R1 GPT-4 --top-k 8
```

---

## Dependencies

All scripts require:
- pandas
- matplotlib
- numpy
- pathlib (standard library)
- argparse (standard library)

Additional dependencies for some scripts:
- `readers.data_loader` module
- `utils.plot` module for styling

## Output Structure

Plots are organized in the `figures/` directory:
```
figures/
├── metrics_analysis/           # overview.py output
├── arrival_time/small/         # plot_arrival_time_model.py output  
├── arrival_time/               # plot_arrival_time_user.py output
└── token_distributions/        # plot_token_distributions.py output
```

## Common Patterns

All scripts:
- Use high-resolution output (300 DPI)
- Support log-scale visualization for wide data ranges
- Provide statistical summaries in plot annotations
- Generate timestamped output directories
- Handle missing/invalid data gracefully