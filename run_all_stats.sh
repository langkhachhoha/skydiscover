#!/bin/bash

# Demo script to run statistics calculation on all JSON files

echo "========================================="
echo "Running Statistics Calculation on All Files"
echo "========================================="
echo ""

# Array of JSON files
files=("math_kimi.json" "math_gpt.json" "system_kimi.json" "system_gpt.json")

for file in "${files[@]}"
do
    if [ -f "$file" ]; then
        echo "Processing: $file"
        echo "========================================="
        python calculate_stats.py "$file"
        echo ""
        echo "Press Enter to continue to next file..."
        read
    else
        echo "File not found: $file"
    fi
done

echo "========================================="
echo "All files processed!"
echo "========================================="
