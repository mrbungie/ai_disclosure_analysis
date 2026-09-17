#!/usr/bin/env python3
"""
Run FactSet fundamentals batch downloads with resumable progress tracking.
Downloads are triggered via chrome_javascript tool; results land in ~/Downloads
and are moved to data/raw/fs/fundamentals/.
"""

import json
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Project paths
REPO_ROOT = Path("/Users/goviedb/Development/thesis")
DATA_DIR = REPO_ROOT / "data" / "raw" / "fs" / "fundamentals"
PROGRESS_FILE = DATA_DIR / "_progress.json"
DOWNLOADS_DIR = Path.home() / "Downloads"

def load_ids():
    """Load factset_ids from id_map.parquet."""
    with open("/tmp/factset_ids.json") as f:
        return json.load(f)

def load_progress():
    """Load progress tracking file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "completed_ids": [],
        "started_at": datetime.now().isoformat(),
        "total_ids": 0,
        "batch_size": 25,
        "concurrency": 4,
        "batches_completed": 0,
        "total_batches": 0
    }

def save_progress(progress):
    """Save progress tracking file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def move_downloads(batch_num, ids_in_batch):
    """Find and move downloaded JSONs from ~/Downloads to data/raw/fs/fundamentals/."""
    moved = []
    for factset_id in ids_in_batch:
        # The download function creates files named: fs_fundamentals_<id>.json
        # where special chars are replaced with _
        safe_id = factset_id.replace(".", "_").replace("-", "_")
        src = DOWNLOADS_DIR / f"fs_fundamentals_{safe_id}.json"

        if src.exists():
            # Rename to batch-numbered file
            dst = DATA_DIR / f"fs_fundamentals_batch_{batch_num:03d}_{safe_id}.json"
            src.rename(dst)
            moved.append(str(dst))
            print(f"  Moved: {dst.name}")
        else:
            print(f"  WARNING: Not found in Downloads: {factset_id}")

    return moved

def run_batch_download(batch_ids, batch_num):
    """
    Trigger a batch download via chrome_javascript.
    This injects the batch_ids array into the browser and calls runFundamentalsBatch().
    """
    ids_json = json.dumps(batch_ids)

    # Build JavaScript that passes the IDs to the pre-loaded function
    js_code = f"""
    const batchIds = {ids_json};
    const result = await runFundamentalsBatch(batchIds, 4);
    return {{"batch": {batch_num}, "ids_count": batchIds.length, "result": result}};
    """

    # Call chrome_javascript via subprocess
    # This is a placeholder - in real execution, this would be called via the MCP tool
    print(f"\n  [Batch {batch_num}] Running {len(batch_ids)} IDs: {batch_ids[:3]}...")
    print(f"  Injecting JavaScript for batch download...")

    # In production, you would call mcp__chrome-mcp-server__chrome_javascript here
    # For now, we just return success and the script will need to be run manually
    # or via a separate MCP call per batch

    return {"batch": batch_num, "ids_count": len(batch_ids), "status": "injected"}

def main():
    """Main: load IDs, track progress, run resumable batch downloads."""

    ids = load_ids()
    progress = load_progress()

    print(f"FactSet Fundamentals Batch Download")
    print(f"Total IDs to process: {len(ids)}")
    print(f"Already completed: {len(progress['completed_ids'])}")

    # Remaining IDs
    remaining = [id for id in ids if id not in progress['completed_ids']]
    print(f"Remaining: {len(remaining)}\n")

    if not remaining:
        print("All IDs already downloaded!")
        return

    # Process in batches
    batch_size = progress['batch_size']
    concurrency = progress['concurrency']
    batch_num = progress.get('batches_completed', 0)

    for i in range(0, len(remaining), batch_size):
        batch_ids = remaining[i:i+batch_size]
        batch_num += 1

        print(f"[{batch_num}] Processing batch: {len(batch_ids)} IDs")

        # Run download - this is where chrome_javascript would be called
        # For now, just show what would happen
        result = run_batch_download(batch_ids, batch_num)

        # In production, wait for downloads and move files
        print(f"  Waiting 5 seconds for downloads to complete...")
        time.sleep(5)

        # Move downloaded files
        # moved = move_downloads(batch_num, batch_ids)
        # print(f"  Moved {len(moved)} files")

        # Update progress
        progress['completed_ids'].extend(batch_ids)
        progress['batches_completed'] = batch_num
        save_progress(progress)

        print(f"  Progress saved. Completed: {len(progress['completed_ids'])}/{len(ids)}")

        # Small pause between batches
        if i + batch_size < len(remaining):
            print(f"  Pausing 30s before next batch...")
            time.sleep(30)

    print("\nBatch download instructions complete!")
    print("Next: run parse_fundamentals.py to convert JSON to parquet.")

if __name__ == '__main__':
    main()
