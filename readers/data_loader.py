#!/usr/bin/env python3
"""Utility functions for loading and transforming metrics data."""

import os
import pandas as pd
from typing import Optional, Dict
import psutil
import warnings
import pyarrow.parquet as pq
import pyarrow as pa
import json

import sys

# Add the parent directory to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Load chutes models mapping from JSON file
def load_chutes_models_map(json_path: str = None) -> Dict[str, str]:
    """
    Load the mapping from chute_id to model name from a JSON file.

    Args:
        json_path: Path to the chutes_models.json file. If None, looks for it in the parent directory.

    Returns:
        Dictionary mapping chute_id to model name
    """
    if json_path is None:
        # Default path is the parent directory of the readers folder
        json_path = os.path.join(parent_dir, "chutes_models.json")

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Create mapping from chute_id to name
        chutes_map = {}
        for item in data.get("items", []):
            chute_id = item.get("chute_id")
            name = item.get("name")
            if chute_id and name:
                chutes_map[chute_id] = name

        return chutes_map
    except Exception as e:
        warnings.warn(f"Failed to load chutes models mapping from {json_path}: {e}")
        return {}


# Load the mapping at module level
CHUTES_MODELS_MAP = load_chutes_models_map()

# Column mappings for standardizing column names from different data sources
COLUMN_MAPPINGS = {
    # Timestamp columns
    "timestamp": "started_at",
    "start_time": "started_at",
    "start": "started_at",
    "begin_time": "started_at",
    "end_time": "completed_at",
    "end": "completed_at",
    "finish_time": "completed_at",
    # Token columns
    "prompt_tokens": "input_tokens",
    "input": "input_tokens",
    "request_tokens": "input_tokens",
    "generation_tokens": "output_tokens",
    "output": "output_tokens",
    "response_tokens": "output_tokens",
    "completion_tokens": "output_tokens",
    # Performance metrics
    "tokens_per_second": "completion_tps",
    "tps": "completion_tps",
    "throughput": "completion_tps",
    # ID columns
    "id": "chute_id",
    "model_id": "chute_id",
    "request_id": "chute_id",
    # Model name columns
    "model": "model_name",
    "model_display_name": "model_name",
}


def load_metrics_dataframe(
    file_path: str,
    apply_transforms: bool = True,
    max_memory_gb: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load CSV or Parquet file into a DataFrame with proper column mapping and transformations.

    Args:
        file_path: Path to CSV or Parquet file
        apply_transforms: Whether to apply MetricsRecord-style transformations
        max_memory_gb: Maximum memory usage in GB. If None, uses 80% of available RAM
        max_rows: If specified, load only the first this many rows. If None, load all data
                  (subject to memory constraints)

    Returns:
        DataFrame with renamed columns and applied transformations
    """
    # Determine memory limits
    if max_memory_gb is None:
        # Use 80% of available RAM as default
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        max_memory_gb = available_memory * 0.8

    # Load file based on extension with memory consideration
    file_size_gb = os.path.getsize(file_path) / (1024**3)  # GB

    if file_path.lower().endswith(".parquet"):
        df = _load_parquet_with_memory_check(
            file_path, file_size_gb, max_memory_gb, max_rows
        )
    elif file_path.lower().endswith(".csv"):
        df = _load_csv_with_memory_check(
            file_path, file_size_gb, max_memory_gb, max_rows
        )
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. Only .csv and .parquet are supported."
        )

    # Apply column renaming
    df = df.rename(columns=COLUMN_MAPPINGS)

    if apply_transforms:
        df = _apply_metrics_transforms(df)

    return df


def _load_parquet_with_memory_check(
    file_path: str, file_size_gb: float, max_memory_gb: float, max_rows: Optional[int]
) -> pd.DataFrame:
    """Load parquet file with memory usage estimation and optional row limiting."""
    # Estimate memory usage for parquet (typically 2-5x file size when loaded)
    estimated_memory_gb = file_size_gb * 3  # Conservative estimate

    if max_rows is not None:
        # Use PyArrow to read only first max_rows rows efficiently
        df = _read_parquet_first_n_rows(file_path, max_rows)
        warnings.warn(
            f"Loading only first {max_rows} rows from parquet file due to max_rows parameter"
        )
    elif estimated_memory_gb > max_memory_gb:
        # For memory limits, we need to estimate row count that would fit
        # Use PyArrow to read in batches and accumulate until we hit memory limit
        df = _read_parquet_with_memory_limit(file_path, max_memory_gb)
    else:
        df = pd.read_parquet(file_path)

    return df


def _read_parquet_first_n_rows(file_path: str, n_rows: int) -> pd.DataFrame:
    """Read first N rows from parquet file efficiently using PyArrow."""
    try:
        # Use PyArrow to read only the first n_rows
        parquet_file = pq.ParquetFile(file_path)
        table = parquet_file.read_row_group(0, columns=None)  # Read first row group

        # If the first row group has enough rows, slice it
        if table.num_rows >= n_rows:
            table = table.slice(0, n_rows)
        else:
            # Need to read multiple row groups
            rows_read = 0
            tables = []
            for i in range(parquet_file.num_row_groups):
                if rows_read >= n_rows:
                    break
                rg_table = parquet_file.read_row_group(i, columns=None)
                remaining_rows = n_rows - rows_read
                if rg_table.num_rows <= remaining_rows:
                    tables.append(rg_table)
                    rows_read += rg_table.num_rows
                else:
                    tables.append(rg_table.slice(0, remaining_rows))
                    rows_read += remaining_rows
                    break

            if tables:
                table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

        return table.to_pandas()
    except Exception as e:
        warnings.warn(f"PyArrow reading failed: {e}. Falling back to pandas.")
        # Fallback to pandas
        df = pd.read_parquet(file_path)
        return df.head(n_rows)


def _read_parquet_with_memory_limit(
    file_path: str, max_memory_gb: float
) -> pd.DataFrame:
    """Read parquet file while monitoring memory usage."""
    try:
        parquet_file = pq.ParquetFile(file_path)
        tables = []
        total_memory_mb = 0
        max_memory_mb = max_memory_gb * 1024

        for i in range(parquet_file.num_row_groups):
            rg_table = parquet_file.read_row_group(i, columns=None)
            rg_df = rg_table.to_pandas()
            rg_memory_mb = estimate_dataframe_memory_usage(rg_df) * 1024

            # Check if adding this row group would exceed memory limit
            if total_memory_mb + rg_memory_mb > max_memory_mb and tables:
                warnings.warn(
                    f"Memory limit reached. Loaded {len(tables)} row groups "
                    f"using approximately {total_memory_mb:.1f}MB."
                )
                break

            tables.append(rg_table)
            total_memory_mb += rg_memory_mb

        if tables:
            combined_table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
            return combined_table.to_pandas()
        else:
            # Fallback to reading just first row group
            return parquet_file.read_row_group(0).to_pandas()

    except Exception as e:
        warnings.warn(
            f"PyArrow memory-limited reading failed: {e}. Falling back to pandas."
        )
        # Fallback to pandas approach
        df_full = pd.read_parquet(file_path)
        actual_memory_gb = estimate_dataframe_memory_usage(df_full)

        if actual_memory_gb > max_memory_gb:
            memory_per_row_mb = (actual_memory_gb * 1024) / len(df_full)
            max_rows = int((max_memory_gb * 1024) / memory_per_row_mb)
            warnings.warn(
                f"File uses {actual_memory_gb:.1f}GB. "
                f"Loading first {max_rows} rows to stay within {max_memory_gb:.1f}GB limit."
            )
            return df_full.head(max_rows)
        else:
            return df_full


def _load_csv_with_memory_check(
    file_path: str, file_size_gb: float, max_memory_gb: float, max_rows: Optional[int]
) -> pd.DataFrame:
    """Load CSV file with memory usage estimation and optional row limiting."""
    # Estimate memory usage for CSV (typically 5-10x file size when loaded due to string processing)
    estimated_memory_gb = file_size_gb * 8  # Conservative estimate for CSV

    if max_rows is not None:
        # Load only first max_rows rows
        df = pd.read_csv(file_path, nrows=max_rows)
        warnings.warn(
            f"Loading only first {max_rows} rows from CSV file due to max_rows parameter"
        )
    elif estimated_memory_gb > max_memory_gb:
        # Load full file first and check actual memory usage
        df_full = pd.read_csv(file_path)
        actual_memory_gb = estimate_dataframe_memory_usage(df_full)

        if actual_memory_gb > max_memory_gb:
            # Load only first N rows to fit in memory
            memory_per_row_mb = (actual_memory_gb * 1024) / len(df_full)
            max_rows = int((max_memory_gb * 1024) / memory_per_row_mb)
            df = df_full.head(max_rows)
            warnings.warn(
                f"File uses {actual_memory_gb:.1f}GB. "
                f"Loading first {max_rows} rows to stay within {max_memory_gb:.1f}GB limit."
            )
        else:
            df = df_full
    else:
        df = pd.read_csv(file_path)

    return df


def estimate_dataframe_memory_usage(df: pd.DataFrame) -> float:
    """
    Estimate memory usage of a DataFrame in GB.

    Args:
        df: DataFrame to estimate memory usage for

    Returns:
        Memory usage in GB
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_gb = memory_bytes / (1024**3)
    return memory_gb


def get_system_memory_info() -> Dict[str, float]:
    """
    Get system memory information.

    Returns:
        Dictionary with memory information in GB
    """
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percentage": mem.percent,
    }


def _apply_metrics_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations

    This includes:
    - Calculating duration from started_at and completed_at
    - Calculating completion_tps from tokens/duration
    - Setting model_name from chute_id mapping
    """
    # df = df.copy()

    # Drop invocation_id column to save memory if it exists
    if "invocation_id" in df.columns:
        df = df.drop(columns=["invocation_id"])

    # Convert timestamp columns to datetime if they exist
    if "started_at" in df.columns:
        df["started_at"] = pd.to_datetime(df["started_at"], format='mixed', errors="coerce")
    if "completed_at" in df.columns:
        df["completed_at"] = pd.to_datetime(df["completed_at"], format='mixed', errors="coerce")

    # Calculate duration where both timestamps exist
    if "started_at" in df.columns and "completed_at" in df.columns:
        # Create a mask for rows where both timestamps are valid
        valid_mask = df["started_at"].notna() & df["completed_at"].notna()
        df.loc[valid_mask, "duration"] = (
            df.loc[valid_mask, "completed_at"] - df.loc[valid_mask, "started_at"]
        ).dt.total_seconds()

    # Calculate completion_tps where possible
    if "completion_tps" in df.columns:
        # Only calculate where completion_tps is missing/null
        missing_tps_mask = df["completion_tps"].isna()

        # Need duration, input_tokens, and output_tokens to be valid
        valid_calc_mask = (
            missing_tps_mask
            & df["duration"].notna()
            & df["input_tokens"].notna()
            & df["output_tokens"].notna()
            & (df["input_tokens"] != -1)
            & (df["output_tokens"] != -1)
            & (df["duration"] > 0)
        )

        # Calculate total tokens and completion_tps
        df.loc[valid_calc_mask, "total_tokens"] = (
            df.loc[valid_calc_mask, "input_tokens"]
            + df.loc[valid_calc_mask, "output_tokens"]
        )
        df.loc[valid_calc_mask, "completion_tps"] = (
            df.loc[valid_calc_mask, "total_tokens"]
            / df.loc[valid_calc_mask, "duration"]
        )

    # Set model_name from chute_id mapping
    if "chute_id" in df.columns:
        df["model_name"] = df["chute_id"].map(CHUTES_MODELS_MAP)
        # Fill unmapped values with chute_id as fallback
        df["model_name"] = df["model_name"].fillna(df["chute_id"])

    return df


if __name__ == "__main__":
    df = load_metrics_dataframe(
        "/scratch/juncheng/data/prefix_cache/metrics_1day.head.csv"
    )
    print(df)
    df = load_metrics_dataframe(
        "/scratch/juncheng/data/prefix_cache/metrics_1day.parquet"
    )
    print(df)
