"""
Execute OECD (2025) AI Patent Identification Pipeline on BigQuery.
Downloads the firm-year panel and saves locally to data/raw/patents/ and BigQuery.

DO NOT RUN: this makes network/BigQuery calls, which are forbidden in this
environment. Kept runnable for a machine with BigQuery credentials.
"""

import socket
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import time
from pathlib import Path
import pandas as pd
from google.cloud import bigquery
from build_oecd_patents_panel import get_oecd_taxonomies, compile_keywords_regex, generate_sql

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = "skillforge-dev-502010"
DATASET_ID = "thesis_patents"
OUT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.sp500_firm_year_ai_patents_oecd2025"

OUT_PARQUET = REPO_ROOT / "data" / "raw" / "patents" / "sp500_firm_year_ai_patents_oecd2025.parquet"
OUT_CSV = REPO_ROOT / "data" / "raw" / "patents" / "sp500_firm_year_ai_patents_oecd2025.csv"

def main():
    print("Initializing OECD 2025 AI Patent Pipeline...")
    core_cpc, related_cpc, keywords = get_oecd_taxonomies()
    print(f"OECD Taxonomies: {len(core_cpc)} Core CPC, {len(related_cpc)} Related CPC, {len(keywords)} Keywords.")

    regex_chunks = compile_keywords_regex(keywords)
    sql = generate_sql(core_cpc, related_cpc, regex_chunks)

    client = bigquery.Client(project=PROJECT_ID)

    print("Running query against patents-public-data and saving to BigQuery...")
    t0 = time.time()
    job_config = bigquery.QueryJobConfig(
        destination=OUT_TABLE,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    query_job = client.query(sql, job_config=job_config)
    query_job.result()
    elapsed = time.time() - t0
    print(f"Query completed in {elapsed:.1f}s. Saved to {OUT_TABLE}.")

    print("Fetching results locally...")
    df = client.query(f"SELECT * FROM `{OUT_TABLE}` ORDER BY year DESC, ai_patents_oecd DESC").to_dataframe()

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} firm-year rows to {OUT_PARQUET} and {OUT_CSV}")

    print("\n" + "="*50)
    print("ANNUAL CORPUS SUMMARY (OECD 2025 AI PATENTS)")
    print("="*50)
    yearly = df.groupby("year").agg({
        "total_patents": "sum",
        "ai_patents_oecd": "sum",
        "ai_patents_core": "sum",
        "ai_patents_related": "sum"
    }).reset_index()
    yearly["ai_share_pct"] = (yearly["ai_patents_oecd"] / yearly["total_patents"] * 100).round(2)
    print(yearly.to_string(index=False))

    print("\n" + "="*50)
    print("TOP 15 S&P 500 FIRMS BY TOTAL AI PATENTS (2020-2025)")
    print("="*50)
    top_firms = df[df["year"].between(2020, 2025)].groupby(["ticker", "company_name"]).agg({
        "total_patents": "sum",
        "ai_patents_oecd": "sum",
        "ai_patents_core": "sum",
        "ai_patents_related": "sum"
    }).reset_index()
    top_firms["ai_share_pct"] = (top_firms["ai_patents_oecd"] / top_firms["total_patents"] * 100).round(2)
    top_firms = top_firms.sort_values(by="ai_patents_oecd", ascending=False).head(15)
    print(top_firms.to_string(index=False))

if __name__ == "__main__":
    main()
