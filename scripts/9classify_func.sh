#!/bin/bash

# Script to move traces to corresponding directories based on dominant function name
# 

DATA_DIR="data/metrics_30day/per_model/non_chat/"
NONCHAT_DIR="data/metrics_30day/per_model/non_chat/"
TOTAL_TRACES=0
TOTAL_CHAT_TRACES=0
TOTAL_FILES=0
FILTERED_FILES=0

# Create non_chat directory if it doesn't exist
mkdir -p "$NONCHAT_DIR"

echo "Filtering traces in $DATA_DIR..."
echo "================================"

# Process each CSV file
for file in "$DATA_DIR"/*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        TOTAL_FILES=$((TOTAL_FILES + 1))
        
        # Count total lines (excluding header)
        total_lines=$(($(wc -l < "$file") - 1))
        TOTAL_TRACES=$((TOTAL_TRACES + total_lines))
        
        # Count chat/completion traces with valid token data
        # Filter for function_name being 'chat' or 'chat_stream' AND having non-empty input_tokens
        chat_lines=$(awk -F',' '
        NR > 1 && 
        ($3 == "chat" || $3 == "chat_stream") && 
        $7 != "" && $7 != "input_tokens" {count++} 
        END {print count+0}' "$file")
        
        TOTAL_CHAT_TRACES=$((TOTAL_CHAT_TRACES + chat_lines))
        
        echo "File: $filename"
        echo "  Total lines: $total_lines"
        echo "  Chat/Completion lines: $chat_lines"
        
        # If no valid chat traces found, move to non_chat directory organized by dominant function
        if [ "$chat_lines" -eq 0 ]; then
            # Find the dominant function name in this file
            dominant_func=$(awk -F',' '
            NR > 1 && $3 != "" && $3 != "function_name" {
                func_count[$3]++
            }
            END {
                max_count = 0
                dominant = ""
                for (func in func_count) {
                    if (func_count[func] > max_count) {
                        max_count = func_count[func]
                        dominant = func
                    }
                }
                print dominant
            }' "$file")
            
            # Create subdirectory for the dominant function
            func_dir="$NONCHAT_DIR/$dominant_func"
            mkdir -p "$func_dir"
            
            echo "  -> Moving to non_chat/$dominant_func/ (no valid chat traces, dominant function: $dominant_func)"
            mv "$file" "$func_dir/"
            FILTERED_FILES=$((FILTERED_FILES + 1))
        else
            echo "  -> Keeping (has valid traces)"
        fi
        echo ""
    fi
done

echo "================================"
echo "SUMMARY:"
echo "Total files processed: $TOTAL_FILES"
echo "Files moved to non_chat/: $FILTERED_FILES"
echo "Files kept: $((TOTAL_FILES - FILTERED_FILES))"
echo "Total lines across all files: $TOTAL_TRACES"
echo "Total chat/completion lines: $TOTAL_CHAT_TRACES"
echo "Percentage of valid lines: $(echo "scale=2; $TOTAL_CHAT_TRACES * 100 / $TOTAL_TRACES" | bc -l 2>/dev/null || echo "N/A")%"
echo ""
echo "Non-chat files organized in: $NONCHAT_DIR/<dominant_function>/"
