#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Assign the directory path to a variable
directory=$1

# Check if the provided argument is a directory
if [ ! -d "$directory" ]; then
    echo "Error: $directory is not a directory"
    exit 1
fi

# Iterate over every CSV file in the directory
for file in "$directory"/*.csv; do
    if [ -f "$file" ]; then
        # Get the file name
        filename=$(basename "$file")
        # Get the number of lines, words, and bytes in the file
        details=$(wc "$file")
        # Print the file name and details
        echo "File: $filename, $details"
    fi
done

