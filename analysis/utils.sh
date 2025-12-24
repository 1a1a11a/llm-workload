#!/bin/bash

MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-64}  # Default to number of CPU cores if not set

##### find top 20 traces #####
find_top_traces() {
    local folder="$1"
    local topK="${2:-20}"
    local output_var="$3"
    
    # Clear traces array
    local traces=()
    
    if [ "$topK" -eq -1 ]; then
        # Just list all CSV files without counting
        while IFS= read -r line; do
            traces+=("$line")
        done < <(ls "$folder"/*.csv 2>/dev/null)
    else
        # Get top K by line count
        while IFS= read -r line; do
            traces+=("$line")
        done < <(
            wc -l "$folder"/*.csv \
                | grep -v "total" \
                | sort -nr \
                | head -n "$topK" \
                | awk '{print $2}'
        )
    fi

    if ((${#traces[@]} == 0)); then
        echo "No traces found under $folder/" >&2
        return 1
    fi

    if [ -n "$output_var" ]; then
        # Set the output variable to the traces array
        # Use a loop to properly set the array
        eval "$output_var=()"
        for trace in "${traces[@]}"; do
            eval "$output_var+=(\"\$trace\")"
        done
    else
        # Print to stdout
        if [ "$topK" -eq -1 ]; then
            echo "All models:"
        else
            echo "Top $topK models by number of requests:"
        fi
        printf '%s\n' "${traces[@]}"
    fi
}


# Function to run command with process pool management
run_with_pool() {
    local cmd="$1"
    local job_name="${2:-job}"
    echo "MAX_PARALLEL_JOBS: $MAX_PARALLEL_JOBS"
    # Wait if we've reached the max parallel jobs
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
        sleep 1
    done
    
    # Run the command in background
    echo "Starting: $job_name"
    eval "$cmd" &
}

# Function to wait for all background jobs to complete
wait_for_jobs() {
    echo "Waiting for all background jobs to complete..."
    wait
    echo "All jobs completed."
}

#### Example usage ####
# find_top_traces "/path/to/folder" 10 top_traces
# echo "Found ${#top_traces[@]} traces to process"
# for trace in "${top_traces[@]}"; do
#     echo "Processing trace: $trace"
# done

# for i in $(seq 1 20); do
#     run_with_pool "sleep 8; echo 'Completed sleep $i'" "sleep_job_$i"
# done
# wait_for_jobs
