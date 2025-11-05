#!/bin/bash

# Script to split metrics.csv into separate files per chute_id
# Creates one CSV file per chute in the requests/ directory

# Default metrics file location
METRICS_FILE="${1:-data/metrics.csv}"

# Check if metrics file exists
if [ ! -f "$METRICS_FILE" ]; then
    echo "Error: $METRICS_FILE not found"
    echo "Usage: $0 [metrics_file]"
    echo "Default: data/metrics.csv"
    exit 1
fi

# Check if requests directory exists, create if not
if [ ! -d "requests" ]; then
    echo "Creating requests/ directory..."
    mkdir -p requests
fi

# Count total lines in metrics file for progress indication
total_lines=$(wc -l < "$METRICS_FILE")
echo "Processing $total_lines lines from $METRICS_FILE..."

# Split the CSV by chute_id (first column)
# Use awk to group by first field and write to separate files
awk -F',' '
NR==1 {
    header=$0
    print "Header: " header
    next
}
{
    # Detect which column contains chute_id based on header
    if (NR==2) {
        for (i=1; i<=NF; i++) {
            if (header ~ /^chute_id,/ || header ~ /,chute_id,/ || header ~ /,chute_id$/) {
                # Find which column number contains "chute_id"
                split(header, headers, ",")
                for (j=1; j<=length(headers); j++) {
                    if (headers[j] == "chute_id") {
                        chuteid_col = j
                        print "Using column " chuteid_col " for chute_id"
                        break
                    }
                }
                break
            }
        }
        if (chuteid_col == 0) {
            chuteid_col = 1  # Default to first column if not found
            print "chute_id column not found, defaulting to column 1"
        }
    }
    file="requests/" $chuteid_col ".csv"
    if (!(file in seen)) {
        print header > file
        seen[file]
        file_count++
    }
    print >> file
}
END {
    print "Created " file_count " chute files in requests/ directory"
}
' "$METRICS_FILE"

echo "Split complete! Files are in the requests/ directory."
echo "Each file is named after its chute_id and contains all requests for that chute."
