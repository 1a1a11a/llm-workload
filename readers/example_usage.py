#!/usr/bin/env python3
"""
Example usage of the TraceReader module.

This script demonstrates how to use the TraceReader to read metrics data
from CSV files and work with MetricsRecord objects.
"""

from trace_reader import TraceReader, create_metrics_dataframe, read_both_head_files

def main():
    print("=== Trace Reader Example ===\n")

    # Method 1: Use the convenience function to read both files
    print("1. Reading both head files using convenience function:")
    records = read_both_head_files()
    print(f"   Loaded {len(records)} total records\n")

    # Method 2: Use TraceReader class directly
    print("2. Using TraceReader class directly:")
    reader = TraceReader()

    # Read individual files
    records_1day = reader.read_file('/home/juncheng/workspace/prefix_cache/metrics_1day.head.csv')
    records_30day = reader.read_file('/home/juncheng/workspace/prefix_cache/metrics_30day_head.csv')

    print(f"   1day records: {len(records_1day)}")
    print(f"   30day records: {len(records_30day)}")
    print(f"   Total: {len(records_1day) + len(records_30day)}\n")

    # Method 3: Read multiple files at once
    print("3. Reading multiple files at once:")
    all_records = reader.read_multiple_files([
        '/home/juncheng/workspace/prefix_cache/metrics_1day.head.csv',
        '/home/juncheng/workspace/prefix_cache/metrics_30day_head.csv'
    ])
    print(f"   Combined records: {len(all_records)}\n")

    # Work with individual MetricsRecord objects
    print("4. Working with MetricsRecord objects:")
    if records:
        sample = records[0]
        print(f"   Sample record from {sample.chute_id}:")
        print(f"     Input tokens: {sample.input_tokens}")
        print(f"     Output tokens: {sample.output_tokens}")
        print(f"     Total tokens: {sample.total_tokens}")
        print(f"     TTFT: {sample.ttft}")
        if sample.effective_duration:
            print(f"     Duration: {sample.effective_duration:.3f}s")
        if sample.completion_tps:
            print(f"     TPS: {sample.completion_tps:.2f}")
        print()

    # Convert to DataFrame for analysis
    print("5. Converting to pandas DataFrame:")
    df = create_metrics_dataframe(records)
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print()

    # Basic analysis example
    print("6. Basic analysis:")
    print(f"   Average input tokens: {df['input_tokens'].mean():.1f}")
    print(f"   Average output tokens: {df['output_tokens'].mean():.1f}")
    print(f"   Average total tokens: {df['total_tokens'].mean():.1f}")

    if 'ttft' in df.columns:
        ttft_data = df['ttft'].dropna()
        if len(ttft_data) > 0:
            print(f"   Average TTFT: {ttft_data.mean():.3f}s")

    if 'effective_duration' in df.columns:
        duration_data = df['effective_duration'].dropna()
        if len(duration_data) > 0:
            print(f"   Average duration: {duration_data.mean():.3f}s")

    print("\n   Chute ID distribution:")
    print(df['chute_id'].value_counts().head())
    print()

    # Show format differences
    print("7. Format comparison:")
    format_1day = [r for r in records if r.timestamp is not None]
    format_30day = [r for r in records if r.invocation_id is not None]
    print(f"   1day format records: {len(format_1day)}")
    print(f"   30day format records: {len(format_30day)}")
    print()

    print("=== Example Complete ===")

if __name__ == "__main__":
    main()