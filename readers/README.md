# Trace Reader

A Python module for reading and parsing metrics data from CSV files into unified `MetricsRecord` objects.

## File Structure

- `metrics_record.py`: Contains the `MetricsRecord` dataclass definition
- `trace_reader.py`: Contains the `TraceReader` class and CSV reading logic
- `example_usage.py`: Example usage and demonstration script
- `README.md`: This documentation

## Features

- **Memory Efficient**: Uses streaming CSV reading to process large files without loading them entirely into memory
- **Unified Data Model**: Convert different CSV formats into consistent `MetricsRecord` objects
- **Flexible Column Mapping**: Uses whatever columns are available in the CSV file and maps them to MetricsRecord fields
- **No Format Detection**: No need to specify or detect file formats - just read the columns that exist
- **Flexible Reading**: Read individual files or multiple files at once
- **DataFrame Conversion**: Easily convert records to pandas DataFrames for analysis
- **Validation**: Built-in validation and error handling for malformed data

## File Formats Supported

### 1day Format
Columns: `chute_id`, `timestamp`, `input_tokens`, `output_tokens`, `ttft`, `duration`, `completion_tps`

### 30day Format
Columns: `invocation_id`, `chute_id`, `function_name`, `user_id`, `started_at`, `completed_at`, `input_tokens`, `output_tokens`, `ttft`

## Usage

### Basic Usage

```python
from trace_reader import TraceReader, read_both_head_files
from metrics_record import MetricsRecord

# Read both head files using convenience function
records = read_both_head_files()

# Or use TraceReader class directly (no format detection needed)
reader = TraceReader()
records_1day = reader.read_file('metrics_1day.head.csv')
records_30day = reader.read_file('metrics_30day_head.csv')

# Read multiple files at once
all_records = reader.read_multiple_files([
    'metrics_1day.head.csv',
    'metrics_30day_head.csv'
])
```

### Working with MetricsRecord Objects

```python
# Access record properties
record = records[0]
print(f"Chute ID: {record.chute_id}")
print(f"Input tokens: {record.input_tokens}")
print(f"Output tokens: {record.output_tokens}")
print(f"Total tokens: {record.total_tokens}")  # Computed property
print(f"TTFT: {record.ttft}")
print(f"Duration: {record.effective_duration}")  # Handles both formats
```

### Converting to DataFrame

```python
from trace_reader import create_metrics_dataframe

df = create_metrics_dataframe(records)
print(df.head())
```

### Running the Example

```bash
cd /home/juncheng/workspace/prefix_cache
python readers/example_usage.py
```

## API Reference

### MetricsRecord

A dataclass representing a unified metrics record.

**Properties:**
- `chute_id`: Model identifier (required)
- `input_tokens`: Number of input tokens (required)
- `output_tokens`: Number of output tokens (required)
- `ttft`: Time to first token (optional)
- `total_tokens`: Computed total tokens (property)
- `effective_duration`: Duration in seconds (property, handles both formats)

**Format-specific fields:**
- 1day format: `timestamp`, `duration`, `completion_tps`
- 30day format: `invocation_id`, `function_name`, `user_id`, `started_at`, `completed_at`

### TraceReader

Main class for reading CSV files.

**Methods:**
- `read_file(file_path, format_hint=None)`: Read a single file
- `read_multiple_files(file_paths, format_hints=None)`: Read multiple files

### Convenience Functions

- `read_metrics_1day_head()`: Read metrics_1day.head.csv
- `read_metrics_30day_head()`: Read metrics_30day_head.csv
- `read_both_head_files()`: Read both head files

### Utility Functions

- `create_metrics_dataframe(records)`: Convert records to pandas DataFrame
