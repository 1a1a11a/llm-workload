#!/usr/bin/env python3
"""
Trace Reader for Metrics Data

This module provides functionality to read and parse metrics data from CSV files
in different formats (1day and 30day traces) into unified MetricsRecord objects.
"""

import pandas as pd
from typing import Optional, List
import logging
import csv

try:
    from .metrics_record import MetricsRecord
except ImportError:
    # Fallback for direct execution
    from metrics_record import MetricsRecord

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TraceReader:
    """
    Reader for metrics trace files.

    Reads CSV files and maps available columns to MetricsRecord fields.
    No format detection - uses whatever columns are present in the file.
    """

    def __init__(self):
        pass

    def read_file(self, file_path: str, format_hint: Optional[str] = None) -> List[MetricsRecord]:
        """
        Read a metrics CSV file and return a list of MetricsRecord objects.

        Args:
            file_path: Path to the CSV file
            format_hint: Optional hint for the format (deprecated, now ignored)

        Returns:
            List of MetricsRecord objects
        """
        try:
            logger.info(f"Reading {file_path}")
            return self._read_csv_file(file_path)

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise


    def _read_csv_file(self, file_path: str) -> List[MetricsRecord]:
        """
        Read a CSV file and map available columns to MetricsRecord fields.

        Uses streaming CSV reading to process files without loading them entirely into memory.
        Uses whatever columns are present in the file, mapping them to MetricsRecord
        fields if they match known field names.
        """
        # Read the file row by row to handle streaming and skip empty lines
        records = []
        header = None
        available_cols = set()

        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row_num, row in enumerate(reader, start=1):
                # Skip empty rows
                if not row or not any(field.strip() for field in row):
                    continue

                # First non-empty row is the header
                if header is None:
                    header = row
                    available_cols = set(header)

                    # Validate required columns
                    required_cols = ['chute_id', 'input_tokens', 'output_tokens']
                    missing_cols = [col for col in required_cols if col not in available_cols]
                    if missing_cols:
                        raise ValueError(f"Missing required columns: {missing_cols}")

                    logger.info(f"Reading CSV with columns: {sorted(available_cols)}")
                    continue

                # Process data rows
                try:
                    row_dict = dict(zip(header, row))
                    record_kwargs = {}

                    # Required fields
                    record_kwargs['chute_id'] = str(row_dict['chute_id'])
                    record_kwargs['input_tokens'] = int(row_dict['input_tokens'])
                    record_kwargs['output_tokens'] = int(row_dict['output_tokens'])

                    # Optional fields - map if column exists and value is not empty/null
                    if 'ttft' in available_cols and row_dict.get('ttft', '').strip():
                        try:
                            record_kwargs['ttft'] = float(row_dict['ttft'])
                        except (ValueError, TypeError):
                            pass  # Skip invalid ttft values

                    # 1day format specific fields
                    if 'timestamp' in available_cols and row_dict.get('timestamp', '').strip():
                        try:
                            record_kwargs['timestamp'] = pd.to_datetime(row_dict['timestamp'])
                        except (ValueError, TypeError):
                            pass  # Skip invalid timestamp values

                    if 'duration' in available_cols and row_dict.get('duration', '').strip():
                        try:
                            record_kwargs['duration'] = float(row_dict['duration'])
                        except (ValueError, TypeError):
                            pass  # Skip invalid duration values

                    if 'completion_tps' in available_cols and row_dict.get('completion_tps', '').strip():
                        try:
                            record_kwargs['completion_tps'] = float(row_dict['completion_tps'])
                        except (ValueError, TypeError):
                            pass  # Skip invalid completion_tps values

                    # 30day format specific fields
                    if 'invocation_id' in available_cols and row_dict.get('invocation_id', '').strip():
                        record_kwargs['invocation_id'] = str(row_dict['invocation_id'])

                    if 'function_name' in available_cols and row_dict.get('function_name', '').strip():
                        record_kwargs['function_name'] = str(row_dict['function_name'])

                    if 'user_id' in available_cols and row_dict.get('user_id', '').strip():
                        record_kwargs['user_id'] = str(row_dict['user_id'])

                    if 'started_at' in available_cols and row_dict.get('started_at', '').strip():
                        try:
                            record_kwargs['started_at'] = pd.to_datetime(row_dict['started_at'])
                        except (ValueError, TypeError):
                            pass  # Skip invalid started_at values

                    if 'completed_at' in available_cols and row_dict.get('completed_at', '').strip():
                        try:
                            record_kwargs['completed_at'] = pd.to_datetime(row_dict['completed_at'])
                        except (ValueError, TypeError):
                            pass  # Skip invalid completed_at values

                    record = MetricsRecord(**record_kwargs)
                    records.append(record)

                except Exception as e:
                    logger.warning(f"Skipping invalid row {row_num}: {e}")
                    continue

        logger.info(f"Successfully read {len(records)} records from CSV file")
        return records


    def read_multiple_files(self, file_paths: List[str]) -> List[MetricsRecord]:
        """
        Read multiple CSV files and combine their records.

        Args:
            file_paths: List of file paths to read

        Returns:
            Combined list of MetricsRecord objects from all files
        """
        all_records = []
        for file_path in file_paths:
            records = self.read_file(file_path)
            all_records.extend(records)

        logger.info(f"Successfully read {len(all_records)} total records from {len(file_paths)} files")
        return all_records


def create_metrics_dataframe(records: List[MetricsRecord]) -> pd.DataFrame:
    """
    Convert a list of MetricsRecord objects to a pandas DataFrame.

    This is useful for analysis and compatibility with existing code.
    """
    data = []
    for record in records:
        row = {
            'chute_id': record.chute_id,
            'input_tokens': record.input_tokens,
            'output_tokens': record.output_tokens,
            'total_tokens': record.total_tokens,
            'ttft': record.ttft,
            'effective_duration': record.effective_duration,
        }

        # Add format-specific fields if available
        if record.timestamp:
            row['timestamp'] = record.timestamp
        if record.duration:
            row['duration'] = record.duration
        if record.completion_tps:
            row['completion_tps'] = record.completion_tps
        if record.invocation_id:
            row['invocation_id'] = record.invocation_id
        if record.function_name:
            row['function_name'] = record.function_name
        if record.user_id:
            row['user_id'] = record.user_id
        if record.started_at:
            row['started_at'] = record.started_at
        if record.completed_at:
            row['completed_at'] = record.completed_at

        data.append(row)

    df = pd.DataFrame(data)
    logger.info(f"Converted {len(records)} records to DataFrame with shape {df.shape}")
    return df


# Convenience functions for common use cases
def read_metrics_1day_head() -> List[MetricsRecord]:
    """Read the metrics_1day.head.csv file"""
    reader = TraceReader()
    return reader.read_file('/home/juncheng/workspace/prefix_cache/metrics_1day.head.csv')


def read_metrics_30day_head() -> List[MetricsRecord]:
    """Read the metrics_30day_head.csv file"""
    reader = TraceReader()
    return reader.read_file('/home/juncheng/workspace/prefix_cache/metrics_30day_head.csv')


def read_both_head_files() -> List[MetricsRecord]:
    """Read both head files and combine the records"""
    reader = TraceReader()
    return reader.read_multiple_files([
        '/home/juncheng/workspace/prefix_cache/metrics_1day.head.csv',
        '/home/juncheng/workspace/prefix_cache/metrics_30day_head.csv'
    ])


if __name__ == "__main__":
    # Example usage
    reader = TraceReader()

    # Read individual files
    print("Reading 1day format file...")
    records_1day = reader.read_file('/home/juncheng/workspace/prefix_cache/metrics_1day.head.csv')
    print(f"Loaded {len(records_1day)} records from 1day format")

    print("\nReading 30day format file...")
    records_30day = reader.read_file('/home/juncheng/workspace/prefix_cache/metrics_30day_head.csv')
    print(f"Loaded {len(records_30day)} records from 30day format")

    # Read both files combined
    print("\nReading both files combined...")
    all_records = reader.read_multiple_files([
        '/home/juncheng/workspace/prefix_cache/metrics_1day.head.csv',
        '/home/juncheng/workspace/prefix_cache/metrics_30day_head.csv'
    ])
    print(f"Loaded {len(all_records)} total records")

    # Convert to DataFrame
    print("\nConverting to DataFrame...")
    df = create_metrics_dataframe(all_records)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Show sample record
    if all_records:
        print(f"\nSample record: {all_records[0]}")
