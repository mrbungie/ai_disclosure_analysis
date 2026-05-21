import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

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
        print(f"Error: Universe file not found at {universe_path}. Run 00_build_firm_universe.py first.")
        return
        
    universe_df = pd.read_parquet(universe_path)
    
    headers = {"User-Agent": user_agent}
    manifest_data = []
    updated_universe = []
    
    print("Building filing manifest from SEC submissions...")
    for idx, row in universe_df.iterrows():
        ticker = row["ticker"]
        cik = row["cik"]
        print(f"Querying submissions for {ticker} (CIK: {cik})...")
        
        # Respect SEC rate limits (max 10 requests per second, we do 1 per 0.15s)
        time.sleep(0.15)
        
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error querying {ticker}: {e}")
            updated_universe.append(row.to_dict())
            continue
            
        # Update SIC code and description in universe
        firm_info = row.to_dict()
        firm_info["sic"] = str(data.get("sic", ""))
        firm_info["industry_group"] = data.get("sicDescription", "Technology/Software")
        updated_universe.append(firm_info)
        
        # Parse filings
        recent_filings = data.get("filings", {}).get("recent", {})
        if not recent_filings:
            continue
            
        acc_nums = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        report_dates = recent_filings.get("reportDate", [])
        forms = recent_filings.get("form", [])
        primary_docs = recent_filings.get("primaryDocument", [])
        
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
            
            # Local path format
            local_path = f"data/raw/filings_html/{cik}/{acc_num}.html"
            
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
                "download_status": "pending",
                "parse_status": "pending",
                "prefilter_status": "pending",
                "llm_status": "pending",
                "priority_score": 1.0,
                "batch_id": "",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
            
    # Save updated universe
    pd.DataFrame(updated_universe).to_parquet(universe_path, index=False)
    print(f"Updated firm universe with SIC codes at {universe_path}")
    
    if manifest_data:
        manifest_df = pd.DataFrame(manifest_data)
        manifest_output = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
        manifest_df.to_parquet(manifest_output, index=False)
        print(f"Successfully created filing manifest with {len(manifest_df)} filings at {manifest_output}")
    else:
        print("No matching filings found to build manifest.")

if __name__ == "__main__":
    main()
