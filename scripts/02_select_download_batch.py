import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    # Load configuration
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}. Run 01_build_filing_manifest.py first.")
        return
        
    df = pd.read_parquet(manifest_path)
    
    # Define batch selection parameters for MVP (10-Ks between 2022 and 2025)
    start_year = 2022
    end_year = 2025
    batch_id = f"batch_{start_year}_{end_year}"
    
    selected_count = 0
    for idx, row in df.iterrows():
        filing_date = row["filing_date"]
        # Handle string or date/timestamp objects
        if hasattr(filing_date, "year"):
            year = filing_date.year
        else:
            try:
                # Convert string if necessary
                dt = datetime.strptime(str(filing_date).split()[0], "%Y-%m-%d")
                year = dt.year
            except Exception:
                continue
                
        if start_year <= year <= end_year and row["download_status"] == "pending":
            df.at[idx, "download_status"] = "selected"
            df.at[idx, "batch_id"] = batch_id
            df.at[idx, "updated_at"] = datetime.now()
            selected_count += 1
            
    if selected_count > 0:
        df.to_parquet(manifest_path, index=False)
        print(f"Successfully selected {selected_count} filings for download in batch '{batch_id}'")
    else:
        print("No new pending filings matched the batch criteria (2022-2025).")

if __name__ == "__main__":
    main()
