#!/bin/bash

# This script finds files larger than 100MB in the current directory and subdirectories.

echo "Searching for files larger than 100MB..."
echo "----------------------------------------"

find . -type f -size +100M -exec du -h {} + | sort -hr

echo "----------------------------------------"
echo "Done."
