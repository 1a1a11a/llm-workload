#!/bin/bash

# Script to rename chute CSV files from chute_id to chute name (part after /)
# Handles duplicate names by adding suffixes

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq first."
    exit 1
fi

# Create temporary files
MAPPING_FILE=$(mktemp)
FINAL_MAPPING_FILE=$(mktemp)

# Extract chute_id -> name mapping from chutes_models.json
jq -r '.items[] | "\(.chute_id),\(.name)"' chutes_model.json > "$MAPPING_FILE"

# Create final mapping with suffixes for duplicates
echo "Processing chute names and handling duplicates..."

# Process all entries to handle duplicates
while IFS=',' read -r chute_id full_name; do
    if [ -n "$chute_id" ] && [ -n "$full_name" ]; then
        # Extract part after the last "/"
        short_name=$(basename "$full_name")

        # Replace any slashes in short_name with underscores to avoid path issues
        safe_name=$(echo "$short_name" | tr '/' '_')

        echo "$chute_id,$safe_name" >> "$FINAL_MAPPING_FILE.tmp"
    fi
done < "$MAPPING_FILE"

# Now group by safe_name and add suffixes where needed
# Sort by name first, then process
sort -t',' -k2 "$FINAL_MAPPING_FILE.tmp" | awk -F',' '
{
    name = $2
    id = $1
    if (name in count) {
        count[name]++
        suffix = "-" count[name]
    } else {
        count[name] = 1
        suffix = ""
    }
    print id "," name suffix
}' > "$FINAL_MAPPING_FILE"

# Clean up temp file
rm "$FINAL_MAPPING_FILE.tmp"

# Change to requests directory
cd requests/

# Process each CSV file
for file in *.csv; do
    # Extract chute_id from filename (remove .csv extension)
    chute_id="${file%.csv}"

    # Look up the name for this chute_id (now includes suffix if needed)
    final_name=$(grep "^${chute_id}," "$FINAL_MAPPING_FILE" | cut -d',' -f2)

    if [ -n "$final_name" ]; then
        # Rename the file
        if [ "$file" != "${final_name}.csv" ]; then
            echo "Renaming $file to ${final_name}.csv"
            mv "$file" "${final_name}.csv"
        fi
    else
        line_count=$(wc -l < "$file")
        echo "Warning: No name found for chute_id $chute_id, keeping original filename ($line_count lines)"
    fi
done

# Clean up
rm "$MAPPING_FILE" "$FINAL_MAPPING_FILE"

echo "Renaming complete!"
# move small files to small/ directory
# mkdir small/; find . -maxdepth 1 -type f -exec sh -c 'lines=$(wc -l < "$1"); if [ "$lines" -lt 1000000 ]; then mv "$1" small/; echo "Moved $1 ($lines lines)"; fi' _ {} \;

