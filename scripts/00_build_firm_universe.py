import os
import json
import requests
import pandas as pd
from pathlib import Path

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

def main():
    # Load configuration
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    user_agent = config["sec"]["user_agent"]
    tickers = config["pipeline"]["tickers"]
    output_dir = Path(config["paths"]["interim_manifests"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": user_agent}
    
    pipeline_logger.log_event(
        pipeline_step="universe_build",
        level="INFO",
        message=f"Starting firm universe build for {len(tickers)} configured tickers"
    )
    
    cache_path = Path("data/sec_company_tickers.json")
    sec_data = None
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                sec_data = json.load(f)
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="INFO",
                message=f"Loaded SEC CIK mappings from local cache: {cache_path}"
            )
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="WARNING",
                message=f"Error loading SEC CIK mappings cache: {e}"
            )
            
    if sec_data is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        pipeline_logger.log_event(
            pipeline_step="universe_build",
            level="INFO",
            message=f"Fetching CIK mappings from SEC EDGAR: {url}"
        )
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            sec_data = response.json()
            # Cache the file
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(sec_data, f)
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="SUCCESS",
                message=f"Successfully downloaded and cached CIK mapping to {cache_path}"
            )
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="WARNING",
                message=f"Error fetching CIK mapping from SEC: {e}. Using fallback map."
            )
            # Fallback list for the MVP in case SEC is down or blocks request
            sec_data = {
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "APPLE INC"},
                "1": {"cik_str": 796343, "ticker": "ADBE", "title": "ADOBE INC."},
                "2": {"cik_str": 2487, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
                "3": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
                "4": {"cik_str": 1108524, "ticker": "CRM", "title": "Salesforce, Inc."},
                "5": {"cik_str": 858877, "ticker": "CSCO", "title": "CISCO SYSTEMS INC"},
                "6": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
                "7": {"cik_str": 51143, "ticker": "IBM", "title": "INTERNATIONAL BUSINESS MACHINES CORP"},
                "8": {"cik_str": 50863, "ticker": "INTC", "title": "INTEL CORP"},
                "9": {"cik_str": 311094, "ticker": "INTU", "title": "INTUIT INC"},
                "10": {"cik_str": 1326801, "ticker": "META", "title": "Meta Platforms, Inc."},
                "11": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
                "12": {"cik_str": 1065280, "ticker": "NFLX", "title": "NETFLIX INC"},
                "13": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
                "14": {"cik_str": 939211, "ticker": "ORCL", "title": "ORACLE CORP"},
                "15": {"cik_str": 1577552, "ticker": "PANW", "title": "Palo Alto Networks Inc."},
                "16": {"cik_str": 1321655, "ticker": "PLTR", "title": "Palantir Technologies Inc."},
                "17": {"cik_str": 804328, "ticker": "QCOM", "title": "QUALCOMM INC/DE"},
                "18": {"cik_str": 1640147, "ticker": "SNOW", "title": "Snowflake Inc."},
                "19": {"cik_str": 1318605, "ticker": "TSLA", "title": "TESLA, INC."}
            }
    
    # Parse SEC data
    sec_lookup = {}
    for entry in sec_data.values():
        sec_lookup[entry["ticker"].upper()] = {
            "cik": str(entry["cik_str"]).zfill(10),
            "company_name": entry["title"]
        }
        
    # Build universe
    universe_data = []
    missing_tickers = []
    for ticker in tickers:
        ticker_upper = ticker.upper()
        if ticker_upper in sec_lookup:
            info = sec_lookup[ticker_upper]
            universe_data.append({
                "ticker": ticker_upper,
                "cik": info["cik"],
                "company_name": info["company_name"],
                "sic": "",
                "industry_group": "Technology/Software"
            })
        else:
            missing_tickers.append(ticker_upper)
            pipeline_logger.log_event(
                pipeline_step="universe_build",
                level="WARNING",
                message=f"Ticker {ticker_upper} not found in SEC CIK registry",
                ticker=ticker_upper
            )
            
    df = pd.DataFrame(universe_data)
    
    # Save to parquet
    output_file = output_dir / "firm_universe.parquet"
    df.to_parquet(output_file, index=False)
    
    pipeline_logger.log_event(
        pipeline_step="universe_build",
        level="SUCCESS",
        message=f"Successfully built firm universe with {len(df)} companies (missing: {len(missing_tickers)})",
        details={
            "company_count": len(df),
            "missing_count": len(missing_tickers),
            "output_file": str(output_file)
        }
    )

if __name__ == "__main__":
    main()
