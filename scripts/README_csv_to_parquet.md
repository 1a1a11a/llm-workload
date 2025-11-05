# CSV to Parquet Converter

This script converts CSV files to Parquet format for efficient storage and processing.

## Features

- Convert individual CSV files to Parquet format
- Batch convert all CSV files in a directory
- Recursive directory processing
- Automatic encoding detection (tries UTF-8, UTF-8 with BOM, Latin1, CP1252)
- **Column name normalization**: Automatically renames columns to match MetricsRecord dataclass
- Validation of required columns (chute_id, input_tokens, output_tokens)
- Detection and logging of extra columns not in standard schema
- Memory-efficient processing using pandas
- Comprehensive logging and error handling

## Usage

### Convert a single file
```bash
python scripts/csv_to_parquet.py path/to/file.csv
# Creates path/to/file.parquet
```

### Convert with custom output path
```bash
python scripts/csv_to_parquet.py input.csv output.parquet
```

### Convert all CSV files in a directory
```bash
python scripts/csv_to_parquet.py --dir data/
```

### Convert all CSV files recursively
```bash
python scripts/csv_to_parquet.py --dir data/ --recursive
```

### Show help
```bash
python scripts/csv_to_parquet.py --help
```

## Dependencies

- pandas
- pyarrow (recommended) or fastparquet

## Benefits of Parquet

- **Compression**: Typically 70-90% smaller than CSV
- **Speed**: Much faster to read/write than CSV
- **Schema**: Includes column type information
- **Compatibility**: Works well with Apache Spark, pandas, and other big data tools

## Column Normalization

The script automatically normalizes column names to match the MetricsRecord dataclass schema using shared mappings from `metrics_record.py`:

### Column Mappings
- `timestamp` → `started_at` (1day format compatibility)
- `chuteid` → `chute_id`
- `invocationid` → `invocation_id`
- `function`/`functionname` → `function_name`
- `userid` → `user_id`
- `inputtokens` → `input_tokens`
- `outputtokens` → `output_tokens`
- `startedat` → `started_at`
- `completedat` → `completed_at`

### Validation
- **Required columns**: `chute_id`, `input_tokens`, `output_tokens`
- **Extra columns**: Logged but preserved (e.g., `invocation_id` in 30day format)

### Supported Formats
- **1day format**: `chute_id`, `started_at`, `input_tokens`, `output_tokens`, `ttft`, `duration`, `completion_tps`
- **30day format**: `invocation_id`, `chute_id`, `function_name`, `user_id`, `started_at`, `completed_at`, `input_tokens`, `output_tokens`, `ttft`

## Examples

Convert the metrics files in your workspace:
```bash
# Convert individual files
python scripts/csv_to_parquet.py metrics_1day.csv
python scripts/csv_to_parquet.py metrics_30day.csv

# Convert all CSV files in the data directory
python scripts/csv_to_parquet.py --dir data/
```

## File Structure

After conversion, you'll have both CSV and Parquet files:
```
workspace/
├── metrics_1day.csv
├── metrics_1day.parquet      # ← New file
├── metrics_30day.csv
├── metrics_30day.parquet     # ← New file
└── data/
    ├── metrics.csv
    ├── metrics.parquet       # ← New file
    └── ...
```
