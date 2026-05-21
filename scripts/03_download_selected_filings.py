import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def download_filing(url, local_path, headers, max_retries=3):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            # Sleep 0.15s before request to respect SEC rate limits (max 10 reqs/sec)
            time.sleep(0.15)
            
            response = requests.get(url, headers=headers)
            if response.status_code == 403:
                print(f"\nForbidden error (403) downloading {url}. Checking User-Agent header settings...")
                return False, "403 Forbidden"
            
            response.raise_for_status()
            
            # Save contents
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            return True, None
        except Exception as e:
            print(f"\nAttempt {attempt+1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                return False, str(e)
    return False, "Max retries exceeded"

def main():
    # Load configuration
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    user_agent = config["sec"]["user_agent"]
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}. Run 02_select_download_batch.py first.")
        return
        
    df = pd.read_parquet(manifest_path)
    
    selected_mask = df["download_status"] == "selected"
    selected_filings = df[selected_mask]
    
    if len(selected_filings) == 0:
        print("No filings selected for download. Run 02_select_download_batch.py or verify manifest status.")
        return
        
    headers = {"User-Agent": user_agent}
    print(f"Starting download of {len(selected_filings)} selected filings...")
    
    success_count = 0
    fail_count = 0
    
    # Save updates incrementally to the manifest
    for idx, row in tqdm(selected_filings.iterrows(), total=len(selected_filings)):
        url = row["sec_url"]
        local_path = Path(row["local_path"])
        
        # Check if already downloaded (incremental run capability)
        if local_path.exists() and local_path.stat().st_size > 0:
            df.at[idx, "download_status"] = "completed"
            df.at[idx, "updated_at"] = datetime.now()
            success_count += 1
            continue
            
        success, err = download_filing(url, local_path, headers)
        if success:
            df.at[idx, "download_status"] = "completed"
            df.at[idx, "updated_at"] = datetime.now()
            success_count += 1
        else:
            df.at[idx, "download_status"] = f"failed: {err}"
            df.at[idx, "updated_at"] = datetime.now()
            fail_count += 1
            
        # Incremental save to keep state up to date
        df.to_parquet(manifest_path, index=False)
        
    print(f"\nDownload finished. Successes: {success_count}, Failures: {fail_count}")

if __name__ == "__main__":
    main()
