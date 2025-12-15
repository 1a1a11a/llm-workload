#!/usr/bin/env python3
"""
Split metrics_30day.csv into per-user CSV files.

Each user's data is saved as user_id.csv in data/metrics_30day/per_user/
Only users with at least 1000 requests are saved.
"""

import csv
import os
from collections import defaultdict

def split_csv_by_user(input_file, output_dir, min_requests=1000, buffer_size=1000):
    """Split CSV file by user_id column, only saving users with min_requests."""
    os.makedirs(output_dir, exist_ok=True)

    # Buffer rows per user in memory (up to buffer_size)
    user_buffers = defaultdict(list)
    user_counts = defaultdict(int)
    user_files = {}
    user_writers = {}

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)

        # Read header
        header = next(reader)

        # Find user_id column index
        user_id_idx = header.index('user_id')

        for row in reader:
            if len(row) <= user_id_idx:
                continue  # Skip malformed rows

            user_id = row[user_id_idx]
            user_counts[user_id] += 1
            user_buffers[user_id].append(row)

            # Flush buffer when it reaches buffer_size
            if len(user_buffers[user_id]) >= buffer_size:
                if user_id not in user_files:
                    filename = f"{user_id}.csv"
                    filepath = os.path.join(output_dir, filename)
                    user_files[user_id] = open(filepath, 'w', newline='', encoding='utf-8')
                    user_writers[user_id] = csv.writer(user_files[user_id])
                    user_writers[user_id].writerow(header)
                
                user_writers[user_id].writerows(user_buffers[user_id])
                user_buffers[user_id].clear()

    # Flush remaining buffers for users with at least min_requests
    users_written = 0
    users_ignored = 0

    for user_id, count in user_counts.items():
        if count >= min_requests:
            if user_id not in user_files:
                filename = f"{user_id}.csv"
                filepath = os.path.join(output_dir, filename)
                user_files[user_id] = open(filepath, 'w', newline='', encoding='utf-8')
                user_writers[user_id] = csv.writer(user_files[user_id])
                user_writers[user_id].writerow(header)
            
            if user_buffers[user_id]:
                user_writers[user_id].writerows(user_buffers[user_id])
            
            users_written += 1
        else:
            users_ignored += 1
            # Remove file if it was created but user has less than min_requests
            if user_id in user_files:
                user_files[user_id].close()
                filename = f"{user_id}.csv"
                filepath = os.path.join(output_dir, filename)
                os.remove(filepath)

    # Close all remaining files
    for user_id, f in user_files.items():
        if user_counts[user_id] >= min_requests:
            f.close()

    print(f"Split complete. Created {users_written} user files in {output_dir}")
    print(f"Ignored {users_ignored} users with less than {min_requests} requests")

if __name__ == "__main__":
    input_file = "/home/juncheng/workspace/prefix_cache/metrics_30day.csv"
    output_dir = "/home/juncheng/workspace/prefix_cache/data/metrics_30day/per_user"

    split_csv_by_user(input_file, output_dir)

