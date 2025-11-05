#!/usr/bin/env python3
"""Minimal trace reader that streams metrics data."""

from __future__ import annotations

import csv
from dataclasses import asdict, fields as dataclass_fields
from typing import Dict, Iterable, Iterator, List, Optional
from datetime import datetime
import pandas as pd

try:
    from .metrics_record import COLUMN_MAPPINGS, MetricsRecord
except ImportError:  # pragma: no cover - fallback when run as a script
    from metrics_record import COLUMN_MAPPINGS, MetricsRecord


_REQUIRED_COLUMNS = set()  # Made previously required fields optional
_FLOAT_FIELDS = {"ttft", "duration", "completion_tps"}
_TIMESTAMP_FIELDS = {"started_at", "completed_at"}
_OPTIONAL_STR_FIELDS = {"function_name", "user_id"}


class TraceReader:
    """Reader that yields ``MetricsRecord`` objects or pandas frames."""

    def __init__(self, file_path: Optional[str] = None, *, column_mappings: Optional[Dict[str, str]] = None) -> None:
        self.file_path = file_path
        self._metrics_fields = {field.name for field in dataclass_fields(MetricsRecord)}
        lookup: Dict[str, str] = {name.lower(): name for name in self._metrics_fields}
        extras = column_mappings or COLUMN_MAPPINGS
        lookup.update({key.lower(): value for key, value in extras.items()})
        self._column_lookup = lookup

    def _is_parquet_file(self, file_path: str) -> bool:
        """Check if file is a parquet file based on extension."""
        return file_path.lower().endswith('.parquet')

    def __iter__(self) -> Iterator[MetricsRecord]:
        return self.iter_records()

    def iter_records(self) -> Iterator[MetricsRecord]:
        path = self._resolve_path(self.file_path)

        if self._is_parquet_file(path):
            # Handle parquet files
            df = pd.read_parquet(path)
            if df.empty:
                return

            # Rename columns using the same logic as CSV
            df.rename(columns=self._rename_column, inplace=True)

            for _, row in df.iterrows():
                values = {col: row[col] for col in df.columns}
                record = self._row_to_record(values)
                if record is not None:
                    yield record
        else:
            # Handle CSV files (existing logic)
            with open(path, newline='', encoding='utf-8') as handle:
                reader = csv.reader(handle)
                header = self._read_header(reader)
                for row in reader:
                    if not row or not any(cell.strip() for cell in row):
                        continue
                    if len(row) < len(header):
                        row = row + [''] * (len(header) - len(row))
                    values = {column: cell for column, cell in zip(header, row)}
                    record = self._row_to_record(values)
                    if record is not None:
                        yield record

    def read_dataframe(self) -> pd.DataFrame:
        path = self._resolve_path(self.file_path)

        if self._is_parquet_file(path):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, skip_blank_lines=True)

        if df.empty and df.columns.empty:
            return df
        df.rename(columns=self._rename_column, inplace=True)
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        df = df.copy()
        for column in {'input_tokens', 'output_tokens'} & set(df.columns):
            df[column] = pd.to_numeric(df[column], errors='coerce')
        for column in _FLOAT_FIELDS & set(df.columns):
            df[column] = pd.to_numeric(df[column], errors='coerce')
        if {'started_at', 'completed_at'}.issubset(df.columns):
            df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
            df['completed_at'] = pd.to_datetime(df['completed_at'], errors='coerce')
        for column in _OPTIONAL_STR_FIELDS & set(df.columns):
            df[column] = df[column].astype('string').str.strip()
        return df

    def _read_header(self, reader: Iterable[List[str]]) -> List[str]:
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            mapped = [self._rename_column(cell) for cell in row]
            missing = _REQUIRED_COLUMNS - set(mapped)
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")
            return mapped
        raise ValueError('Trace file does not contain a header row')

    def _rename_column(self, column: str) -> str:
        key = (column or '').strip()
        if not key:
            raise ValueError('Encountered empty column name')
        return self._column_lookup.get(key.lower(), key)

    def _row_to_record(self, row: Dict[str, str]) -> MetricsRecord:
        data: Dict[str, object] = {}
        # Previously required fields are now optional with -1 as default
        data['chute_id'] = self._optional_string(row.get('chute_id')) or "-1"
        data['input_tokens'] = self._optional_int_with_default(row.get('input_tokens'), -1)
        data['output_tokens'] = self._optional_int_with_default(row.get('output_tokens'), -1)

        for field in _FLOAT_FIELDS & self._metrics_fields:
            value = self._optional_float(row.get(field))
            if value is not None:
                data[field] = value

        for field in _TIMESTAMP_FIELDS & self._metrics_fields:
            value = self._optional_timestamp(row.get(field))
            if value is not None:
                data[field] = value

        for field in _OPTIONAL_STR_FIELDS & self._metrics_fields:
            value = self._optional_string(row.get(field))
            if value is not None:
                data[field] = value

        return MetricsRecord(**data)

    @staticmethod
    def _require_string(value: Optional[str]) -> str:
        text = TraceReader._optional_string(value)
        if text is None:
            raise ValueError('Missing required string value')
        return text

    @staticmethod
    def _require_int(value: Optional[str]) -> int:
        if value is None or not str(value).strip():
            raise ValueError('Missing required integer value')
        try:
            numeric = int(float(str(value).strip()))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid integer value: {value}') from exc
        return numeric

    @staticmethod
    def _optional_string(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Optional[str]) -> Optional[float]:
        if value is None or not str(value).strip():
            return None
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int_with_default(value: Optional[str], default: int = -1) -> int:
        """Parse optional integer value, returning default if missing or invalid"""
        if value is None or not str(value).strip():
            return default
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _optional_timestamp(value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        return pd.to_datetime(value, errors='coerce')

    def _resolve_path(self, file_path: Optional[str]) -> str:
        path = file_path or self.file_path
        if not path:
            raise ValueError('A file path must be provided')
        return path


def create_metrics_dataframe(records: Iterable[MetricsRecord]) -> pd.DataFrame:
    rows = [asdict(record) for record in records]
    df = pd.DataFrame(rows)
    if not df.empty:
        df['total_tokens'] = df['input_tokens'] + df['output_tokens']
    return df


def iter_metrics_1day_head() -> Iterator[MetricsRecord]:
    return TraceReader('/home/juncheng/workspace/prefix_cache/metrics_1day.head.csv')


def iter_metrics_30day_head() -> Iterator[MetricsRecord]:
    return TraceReader('/home/juncheng/workspace/prefix_cache/metrics_30day_head.csv')

if __name__ == '__main__':
    reader = TraceReader('metrics_1day.head.csv')
    for record in reader:
        print(record)

    reader = TraceReader('metrics_30day.head.csv')
    print(reader.read_dataframe())
