import os
import json
import requests
import pandas as pd
from pathlib import Path

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
    
    print("Fetching CIK mapping from SEC...")
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        sec_data = response.json()
    except Exception as e:
        print(f"Error fetching from SEC: {e}")
        # Fallback list for the MVP in case SEC is down or blocks request
        print("Using static fallback for tickers...")
        sec_data = {
            "0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
            "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
            "2": {"cik_str": 320193, "ticker": "AAPL", "title": "APPLE INC"},
            "3": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
            "4": {"cik_str": 1318605, "ticker": "TSLA", "title": "TESLA, INC."},
            "5": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
            "6": {"cik_str": 1326801, "ticker": "META", "title": "Meta Platforms, Inc."},
            "7": {"cik_str": 1108524, "ticker": "CRM", "title": "Salesforce, Inc."},
            "8": {"cik_str": 796343, "ticker": "ADBE", "title": "ADOBE INC."},
            "9": {"cik_str": 93410, "ticker": "CSCO", "title": "CISCO SYSTEMS, INC."},
            "10": {"cik_str": 1065280, "ticker": "NFLX", "title": "NETFLIX INC"},
            "11": {"cik_str": 939211, "ticker": "ORCL", "title": "ORACLE CORP"},
            "12": {"cik_str": 858877, "ticker": "CSCO", "title": "CISCO SYSTEMS INC"},
            "13": {"cik_str": 2487, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
            "14": {"cik_str": 804328, "ticker": "QCOM", "title": "QUALCOMM INC/DE"},
            "15": {"cik_str": 311094, "ticker": "INTU", "title": "INTUIT INC"},
            "16": {"cik_str": 50863, "ticker": "INTC", "title": "INTEL CORP"},
            "17": {"cik_str": 51143, "ticker": "IBM", "title": "INTERNATIONAL BUSINESS MACHINES CORP"},
            "18": {"cik_str": 1640147, "ticker": "SNOW", "title": "Snowflake Inc."},
            "19": {"cik_str": 1321655, "ticker": "PLTR", "title": "Palantir Technologies Inc."},
            "20": {"cik_str": 1577552, "ticker": "PANW", "title": "Palo Alto Networks Inc."}
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
            print(f"Warning: Ticker {ticker} not found in SEC lookup.")
            
    df = pd.DataFrame(universe_data)
    
    # Save to parquet
    output_file = output_dir / "firm_universe.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Successfully wrote {len(df)} firms to {output_file}")

if __name__ == "__main__":
    main()
