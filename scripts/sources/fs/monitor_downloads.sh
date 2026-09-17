#!/bin/bash
# Monitor and organize FactSet fundamentals downloads
# Run periodically (every 5 minutes) to move files from ~/Downloads to data/raw/fs/fundamentals/

DATA_DIR="/Users/goviedb/Development/thesis/data/raw/fs/fundamentals"
DOWNLOADS="$HOME/Downloads"
PROGRESS_FILE="$DATA_DIR/_progress.json"

# Move any new downloads
count=0
for file in "$DOWNLOADS"/fs_fundamentals_*.json; do
  if [ -f "$file" ]; then
    basename=$(basename "$file")
    # Move to fundamentals directory
    mv "$file" "$DATA_DIR/${basename}"
    ((count++))
  fi
done

if [ $count -gt 0 ]; then
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Moved $count files"

  # Count total files
  total=$(ls "$DATA_DIR"/fs_fundamentals_*.json 2>/dev/null | wc -l)
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Total files: $total"
fi

# Optionally check browser progress
if command -v osascript &>/dev/null; then
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Checking browser progress..."
fi
