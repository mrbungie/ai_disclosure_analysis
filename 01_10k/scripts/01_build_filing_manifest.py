import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for common/

try:
    import pipeline_logger
except ImportError:
    from common import pipeline_logger

def main():
    # Load configuration
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    user_agent = config["sec"]["user_agent"]
    start_year = config["pipeline"]["start_year"]
    end_year = config["pipeline"]["end_year"]
    form_types = config["pipeline"]["form_types"]
    
    universe_path = Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet"
    if not universe_path.exists():
        pipeline_logger.log_event(
            pipeline_step="manifest_build",
            level="ERROR",
            message=f"Universe file not found at {universe_path}. Run 00_build_firm_universe.py first."
        )
        return
        
    universe_df = pd.read_parquet(universe_path)
    
    manifest_output = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    existing_manifest_df = None
    if manifest_output.exists():
        try:
            existing_manifest_df = pd.read_parquet(manifest_output)
            pipeline_logger.log_event(
                pipeline_step="manifest_build",
                level="INFO",
                message=f"Loaded existing manifest with {len(existing_manifest_df)} filings."
            )
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="manifest_build",
                level="WARNING",
                message=f"Could not load existing manifest: {e}"
            )
            
    headers = {"User-Agent": user_agent}
    manifest_data = []
    updated_universe = []
    
    import sys
    force_refresh = "--force" in sys.argv
    submissions_dir = Path(config["paths"]["raw_submissions"])
    submissions_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_logger.log_event(
        pipeline_step="manifest_build",
        level="INFO",
        message="Building filing manifest from SEC submissions..."
    )
    for idx, row in universe_df.iterrows():
        ticker = row["ticker"]
        cik = row["cik"]
        
        cache_file = submissions_dir / f"CIK{cik}.json"
        data = None
        
        if cache_file.exists() and not force_refresh:
            pipeline_logger.log_event(
                pipeline_step="manifest_build",
                level="INFO",
                message=f"Loading submissions for {ticker} (CIK: {cik}) from local cache: {cache_file.name}",
                ticker=ticker,
                cik=cik
            )
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                pipeline_logger.log_event(
                    pipeline_step="manifest_build",
                    level="ERROR",
                    message=f"Error loading cached submissions for {ticker}: {e}",
                    ticker=ticker,
                    cik=cik
                )
                
        if data is None:
            pipeline_logger.log_event(
                pipeline_step="manifest_build",
                level="INFO",
                message=f"Querying submissions for {ticker} (CIK: {cik}) from SEC EDGAR...",
                ticker=ticker,
                cik=cik
            )
            # Respect SEC rate limits (max 10 requests per second, we do 1 per 0.15s)
            time.sleep(0.15)
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            start_time = time.time()
            try:
                response = requests.get(url, headers=headers)
                duration = time.time() - start_time
                response.raise_for_status()
                data = response.json()
                
                # Write JSON response to cache file
                with open(cache_file, "w") as f:
                    json.dump(data, f)
                pipeline_logger.log_event(
                    pipeline_step="manifest_build",
                    level="SUCCESS",
                    message=f"Successfully fetched and cached submissions for {ticker}",
                    ticker=ticker,
                    cik=cik,
                    duration_seconds=duration
                )
            except Exception as e:
                duration = time.time() - start_time
                pipeline_logger.log_event(
                    pipeline_step="manifest_build",
                    level="ERROR",
                    message=f"Error querying submissions for {ticker}: {e}",
                    ticker=ticker,
                    cik=cik,
                    duration_seconds=duration,
                    details={"error": str(e)}
                )
                updated_universe.append(row.to_dict())
                continue
            
        # Update SIC code and description in universe
        firm_info = row.to_dict()
        firm_info["sic"] = str(data.get("sic", ""))
        firm_info["industry_group"] = data.get("sicDescription", "Technology/Software")
        updated_universe.append(firm_info)
        
        # Collect all filing pages: recent + any overflow pages the SEC paginates into extra files
        filing_pages = []
        recent_filings = data.get("filings", {}).get("recent", {})
        if recent_filings:
            filing_pages.append(recent_filings)

        for extra in data.get("filings", {}).get("files", []):
            extra_name = extra.get("name", "")
            if not extra_name:
                continue
            # Only fetch if the page could contain filings in our year range
            filing_from = extra.get("filingFrom", "")
            filing_to = extra.get("filingTo", "")
            try:
                page_max_year = int(filing_to[:4]) if filing_to else end_year
                page_min_year = int(filing_from[:4]) if filing_from else start_year
            except ValueError:
                page_max_year, page_min_year = end_year, start_year
            if page_max_year < start_year or page_min_year > end_year:
                continue  # entire page is outside our range, skip

            extra_cache = submissions_dir / extra_name
            extra_data = None
            if extra_cache.exists() and not force_refresh:
                try:
                    with open(extra_cache) as f:
                        extra_data = json.load(f)
                except Exception:
                    pass

            if extra_data is None:
                time.sleep(0.15)
                extra_url = f"https://data.sec.gov/submissions/{extra_name}"
                try:
                    resp = requests.get(extra_url, headers=headers)
                    resp.raise_for_status()
                    extra_data = resp.json()
                    with open(extra_cache, "w") as f:
                        json.dump(extra_data, f)
                    pipeline_logger.log_event(
                        pipeline_step="manifest_build",
                        level="INFO",
                        message=f"Fetched overflow filing page {extra_name} for {ticker}",
                        ticker=ticker, cik=cik
                    )
                except Exception as e:
                    pipeline_logger.log_event(
                        pipeline_step="manifest_build",
                        level="WARNING",
                        message=f"Could not fetch overflow page {extra_name} for {ticker}: {e}",
                        ticker=ticker, cik=cik
                    )
                    continue

            if extra_data:
                filing_pages.append(extra_data)

        if not filing_pages:
            continue

        for filing_page in filing_pages:
            acc_nums = filing_page.get("accessionNumber", [])
            filing_dates = filing_page.get("filingDate", [])
            report_dates = filing_page.get("reportDate", [])
            forms = filing_page.get("form", [])
            primary_docs = filing_page.get("primaryDocument", [])

            for i in range(len(forms)):
                form = forms[i]
                filing_date_str = filing_dates[i]

                # Filter form type and dates
                if form not in form_types:
                    continue

                try:
                    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
                    year = filing_date.year
                except Exception:
                    continue

                if year < start_year or year > end_year:
                    continue

                acc_num = acc_nums[i]
                acc_num_no_dashes = acc_num.replace("-", "")
                primary_doc = primary_docs[i]

                # Construct standard SEC archive URL
                sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num_no_dashes}/{primary_doc}"

                # Local path format: descriptive and human-readable filenames
                filename = f"{ticker}_{year}_{form}_{acc_num}.html"
                local_path = f"data/raw/filings_html/{filename}"

                # Check if file is already downloaded (new or old location)
                new_path_obj = Path(local_path)
                old_path_obj = Path(f"data/raw/filings_html/{cik}/{acc_num}.html")

                # Migrate file from old format to new format
                if old_path_obj.exists() and old_path_obj.stat().st_size > 0:
                    new_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        old_path_obj.rename(new_path_obj)
                        # Try to clean up empty CIK directory
                        try:
                            old_path_obj.parent.rmdir()
                        except Exception:
                            pass
                        pipeline_logger.log_event(
                            pipeline_step="manifest_build",
                            level="INFO",
                            message=f"Migrated filing layout: {old_path_obj.name} -> {new_path_obj.name}",
                            ticker=ticker,
                            cik=cik,
                            accession_number=acc_num
                        )
                    except Exception as e:
                        pipeline_logger.log_event(
                            pipeline_step="manifest_build",
                            level="ERROR",
                            message=f"Error migrating filing layout {old_path_obj}: {e}",
                            ticker=ticker,
                            cik=cik,
                            accession_number=acc_num
                        )

                # Check if we already have this filing in the manifest
                existing_row = None
                if existing_manifest_df is not None:
                    match = existing_manifest_df[existing_manifest_df["accession_number"] == acc_num]
                    if not match.empty:
                        existing_row = match.iloc[0]

                if existing_row is not None:
                    download_status = existing_row["download_status"]
                    parse_status = existing_row["parse_status"]
                    prefilter_status = existing_row["prefilter_status"]
                    llm_status = existing_row.get("llm_status", "pending")
                    priority_score = existing_row.get("priority_score", 1.0)
                    batch_id = existing_row.get("batch_id", "")
                    created_at = existing_row.get("created_at", datetime.now())
                else:
                    download_status = "pending"
                    parse_status = "pending"
                    prefilter_status = "pending"
                    llm_status = "pending"
                    priority_score = 1.0
                    batch_id = ""
                    created_at = datetime.now()

                # Ensure download_status is marked completed if file exists on disk
                if new_path_obj.exists() and new_path_obj.stat().st_size > 0:
                    download_status = "completed"

                manifest_data.append({
                    "document_id": acc_num,
                    "cik": cik,
                    "ticker": ticker,
                    "form_type": form,
                    "filing_date": filing_date,
                    "period_end_date": datetime.strptime(report_dates[i], "%Y-%m-%d").date() if report_dates[i] else None,
                    "accession_number": acc_num,
                    "sec_url": sec_url,
                    "local_path": local_path,
                    "download_status": download_status,
                    "parse_status": parse_status,
                    "prefilter_status": prefilter_status,
                    "llm_status": llm_status,
                    "priority_score": priority_score,
                    "batch_id": batch_id,
                    "created_at": created_at,
                    "updated_at": datetime.now()
                })
            
    # Save updated universe
    pd.DataFrame(updated_universe).to_parquet(universe_path, index=False)
    pipeline_logger.log_event(
        pipeline_step="manifest_build",
        level="INFO",
        message=f"Updated firm universe with SIC codes at {universe_path}"
    )
    
    if manifest_data:
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_parquet(manifest_output, index=False)
        pipeline_logger.log_event(
            pipeline_step="manifest_build",
            level="SUCCESS",
            message=f"Successfully created/updated filing manifest with {len(manifest_df)} filings at {manifest_output}",
            details={"filing_count": len(manifest_df), "output_file": str(manifest_output)}
        )
    else:
        pipeline_logger.log_event(
            pipeline_step="manifest_build",
            level="WARNING",
            message="No matching filings found to build manifest."
        )

if __name__ == "__main__":
    main()
