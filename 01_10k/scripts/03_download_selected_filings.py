import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for common/

try:
    import pipeline_logger
except ImportError:
    from common import pipeline_logger

def download_filing(metadata, url, local_path, headers, max_retries=3):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            # Sleep 0.15s before request to respect SEC rate limits (max 10 reqs/sec)
            time.sleep(0.15)
            
            response = requests.get(url, headers=headers)
            duration = time.time() - start_time
            
            if response.status_code == 403:
                err_msg = "403 Forbidden"
                pipeline_logger.log_event(
                    pipeline_step="download",
                    level="ERROR",
                    message="Forbidden 403 downloading filing",
                    ticker=metadata.get("ticker"),
                    cik=metadata.get("cik"),
                    accession_number=metadata.get("accession_number"),
                    duration_seconds=duration,
                    details={"attempt": attempt, "error_message": err_msg, "response_code": 403}
                )
                return False, err_msg
            
            response.raise_for_status()
            
            # Save contents
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            pipeline_logger.log_event(
                pipeline_step="download",
                level="SUCCESS",
                message="Filing downloaded successfully",
                ticker=metadata.get("ticker"),
                cik=metadata.get("cik"),
                accession_number=metadata.get("accession_number"),
                duration_seconds=duration,
                details={"attempt": attempt, "response_code": response.status_code, "response_bytes": len(response.text)}
            )
            return True, None
            
        except Exception as e:
            err_msg = str(e)
            duration = time.time() - start_time
            response_code = None
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                response_code = e.response.status_code
                
            pipeline_logger.log_event(
                pipeline_step="download",
                level="WARNING" if attempt < max_retries else "ERROR",
                message=f"Attempt {attempt} failed: {err_msg}",
                ticker=metadata.get("ticker"),
                cik=metadata.get("cik"),
                accession_number=metadata.get("accession_number"),
                duration_seconds=duration,
                details={"attempt": attempt, "error_message": err_msg, "response_code": response_code}
            )
            
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                return False, err_msg
                
    return False, "Max retries exceeded"

def main():
    # Load configuration
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    user_agent = config["sec"]["user_agent"]
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    
    if not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step="download",
            level="ERROR",
            message=f"Manifest file not found at {manifest_path}. Run 02_select_download_batch.py first."
        )
        return
        
    df = pd.read_parquet(manifest_path)
    
    selected_mask = df["download_status"] == "selected"
    selected_filings = df[selected_mask]
    
    if len(selected_filings) == 0:
        pipeline_logger.log_event(
            pipeline_step="download",
            level="INFO",
            message="No filings selected for download. Run 02_select_download_batch.py or verify manifest status."
        )
        return
        
    headers = {"User-Agent": user_agent}
    pipeline_logger.log_event(
        pipeline_step="download",
        level="INFO",
        message=f"Starting download of {len(selected_filings)} selected filings"
    )
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    # Save updates incrementally to the manifest
    for idx, row in tqdm(selected_filings.iterrows(), total=len(selected_filings)):
        url = row["sec_url"]
        local_path = Path(row["local_path"])
        
        # Check if already downloaded (incremental run capability)
        if local_path.exists() and local_path.stat().st_size > 0:
            df.at[idx, "download_status"] = "completed"
            df.at[idx, "updated_at"] = datetime.now()
            success_count += 1
            skipped_count += 1
            
            pipeline_logger.log_event(
                pipeline_step="download",
                level="INFO",
                message="Filing already downloaded, skipping",
                ticker=row.get("ticker"),
                cik=row.get("cik"),
                accession_number=row.get("accession_number"),
                details={"status": "skipped_completed", "response_bytes": int(local_path.stat().st_size)}
            )
            continue
            
        metadata = {
            "ticker": row.get("ticker"),
            "cik": row.get("cik"),
            "accession_number": row.get("accession_number"),
            "form_type": row.get("form_type"),
            "filing_date": row.get("filing_date")
        }
        
        success, err = download_filing(metadata, url, local_path, headers)
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
        
    pipeline_logger.log_event(
        pipeline_step="download",
        level="SUCCESS" if fail_count == 0 else "WARNING",
        message=f"Download batch finished. Total: {len(selected_filings)}, Successes: {success_count} (Skipped: {skipped_count}), Failures: {fail_count}",
        details={"total_selected": len(selected_filings), "success_count": success_count, "skipped_count": skipped_count, "fail_count": fail_count}
    )

if __name__ == "__main__":
    main()
