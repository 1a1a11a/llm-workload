#!/usr/bin/env python3
"""
CSV to Parquet Converter

This script converts CSV files to Parquet format for efficient storage and processing.
It can handle individual files or process all CSV files in a directory.

Usage:
    python csv_to_parquet.py input.csv [output.parquet]
    python csv_to_parquet.py --dir /path/to/csv/directory [--recursive]
    python csv_to_parquet.py --help

Dependencies:
    pandas
    pyarrow or fastparquet
"""

import sys
import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from readers.data_loader import COLUMN_MAPPINGS

logger = logging.getLogger(__name__)


def csv_to_parquet(csv_path: str, parquet_path: Optional[str] = None) -> bool:
    """
    Convert a single CSV file to Parquet format.

    Args:
        csv_path: Path to the input CSV file
        parquet_path: Path to the output Parquet file (optional)

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False

        # Generate output path if not provided
        if parquet_path is None:
            csv_path_obj = Path(csv_path)
            parquet_path = str(csv_path_obj.with_suffix(".parquet"))

        logger.info(f"Converting {csv_path} -> {parquet_path}")

        # Read CSV file
        # Try different encodings if UTF-8 fails
        encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        df = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                if encoding != "utf-8":
                    logger.info(f"Used encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {csv_path}: {e}")
                return False

        if df is None:
            logger.error(f"Could not read {csv_path} with any supported encoding")
            return False

        # Normalize column names based on MetricsRecord dataclass
        # Apply column renaming
        df = df.rename(columns=COLUMN_MAPPINGS)

        # Validate that we have the expected columns (at minimum the required ones from MetricsRecord)
        required_columns = ["chute_id", "input_tokens", "output_tokens"]
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            logger.error(f"Missing required columns: {missing_required}")
            return False

        # Log column information
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Log any columns that are not in the standard MetricsRecord fields
        standard_fields = {
            "chute_id",
            "input_tokens",
            "output_tokens",
            "ttft",
            "function_name",
            "user_id",
            "started_at",
            "completed_at",
            "duration",
            "completion_tps",
        }
        extra_columns = [col for col in df.columns if col not in standard_fields]
        if extra_columns:
            logger.info(f"Extra columns (not in MetricsRecord): {extra_columns}")

        # Convert to Parquet
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Successfully converted to {parquet_path}")

        return True

    except Exception as e:
        logger.error(f"Error converting {csv_path}: {e}")
        return False


def find_csv_files(directory: str, recursive: bool = False) -> List[str]:
    """
    Find all CSV files in a directory.

    Args:
        directory: Directory path to search
        recursive: Whether to search subdirectories

    Returns:
        List of CSV file paths
    """
    if recursive:
        pattern = os.path.join(directory, "**", "*.csv")
    else:
        pattern = os.path.join(directory, "*.csv")

    return glob.glob(pattern, recursive=recursive)


def convert_directory(directory: str, recursive: bool = False) -> int:
    """
    Convert all CSV files in a directory to Parquet.

    Args:
        directory: Directory to process
        recursive: Whether to process subdirectories

    Returns:
        Number of files successfully converted
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return 0

    csv_files = find_csv_files(directory, recursive)
    if not csv_files:
        logger.warning(f"No CSV files found in {directory}")
        return 0

    logger.info(f"Found {len(csv_files)} CSV files to convert")

    success_count = 0
    for csv_file in csv_files:
        if csv_to_parquet(csv_file):
            success_count += 1

    logger.info(f"Successfully converted {success_count}/{len(csv_files)} files")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python csv_to_parquet.py data/metrics.csv

  # Convert single file with custom output
  python csv_to_parquet.py data/metrics.csv output/metrics.parquet

  # Convert all CSV files in directory
  python csv_to_parquet.py --dir data/

  # Convert all CSV files recursively
  python csv_to_parquet.py --dir data/ --recursive
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("input_file", nargs="?", help="Input CSV file path")
    group.add_argument("--dir", help="Directory containing CSV files to convert")

    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output Parquet file path (optional, defaults to input_file.parquet)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process subdirectories recursively when using --dir",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dir:
        # Directory mode
        success_count = convert_directory(args.dir, args.recursive)
        if success_count == 0:
            sys.exit(1)
    else:
        # Single file mode
        if not csv_to_parquet(args.input_file, args.output_file):
            sys.exit(1)

    logger.info("Conversion completed successfully")


if __name__ == "__main__":
    main()
