#!/usr/bin/env python3
"""
Replace chute_id with model name in CSV files (parallel version).

This script reads the chutes_models.json file to create a mapping from chute_id to model name,
then processes all CSV files in the per_user directory in parallel to replace chute_id values 
with model names.
"""

import json
import csv
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# Paths
BASE_DIR = Path("/home/juncheng/workspace/prefix_cache")
JSON_FILE = BASE_DIR / "data/chutes_models.json"
CSV_DIR = BASE_DIR / "data/metrics_30day/per_user/1000k"

def load_chute_mapping(json_file):
    """Load chute_id to model name mapping from JSON file."""
    print(f"Loading chute mapping from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    for item in data['items']:
        chute_id = item['chute_id']
        name = item['name']
        mapping[chute_id] = name
    
    print(f"Loaded {len(mapping)} chute_id to model name mappings")
    return mapping

def process_csv_file(csv_file, chute_mapping):
    """Process a single CSV file to replace chute_id with model name."""
    print(f"Processing {csv_file.name}...")
    
    # Create temporary output file
    temp_file = csv_file.with_suffix('.tmp')
    
    # Read and write with replacement
    with open(csv_file, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        # Get fieldnames and replace chute_id with model_name
        fieldnames = list(reader.fieldnames)
        if 'chute_id' not in fieldnames:
            print(f"Warning: No chute_id column found in {csv_file.name}")
            temp_file.unlink()
            return csv_file.name, 0, []
        
        chute_id_idx = fieldnames.index('chute_id')
        fieldnames[chute_id_idx] = 'model_name'
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process rows
        rows_processed = 0
        unmapped_chutes = set()
        
        for row in reader:
            chute_id = row['chute_id']
            model_name = chute_mapping.get(chute_id, chute_id)
            
            if model_name == chute_id:
                unmapped_chutes.add(chute_id)
            
            # Replace chute_id with model_name
            del row['chute_id']
            row['model_name'] = model_name
            
            writer.writerow(row)
            rows_processed += 1
            
            if rows_processed % 1000000 == 0:
                print(f"  {csv_file.name}: {rows_processed:,} rows...")
    
    # Replace original file with temp file
    temp_file.replace(csv_file)
    print(f"✓ Completed {csv_file.name}: {rows_processed:,} rows")
    
    return csv_file.name, rows_processed, list(unmapped_chutes)

def main():
    # Load chute mapping
    chute_mapping = load_chute_mapping(JSON_FILE)
    
    # Get all CSV files (excluding subdirectories)
    csv_files = sorted([f for f in CSV_DIR.glob("*.csv") if f.is_file()])
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    print(f"Using {cpu_count()} CPU cores\n")
    
    # Process files in parallel
    process_func = partial(process_csv_file, chute_mapping=chute_mapping)
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_func, csv_files)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total_rows = 0
    all_unmapped = set()
    
    for filename, rows, unmapped in results:
        total_rows += rows
        all_unmapped.update(unmapped)
        print(f"  {filename}: {rows:,} rows")
    
    print(f"\nTotal rows processed: {total_rows:,}")
    if all_unmapped:
        print(f"Unmapped chute_ids ({len(all_unmapped)}): {list(all_unmapped)[:10]}")
    
    print("\n✓ All files processed successfully!")

if __name__ == "__main__":
    main()

