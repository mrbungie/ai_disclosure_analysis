#!/usr/bin/env python3
"""
Compute firm-level pre-SEC shock patent capacity using BigQuery and OECD (2025) AI patent taxonomy.
Uses strictly published patents (publication_date < shock_date) to prevent any look-ahead bias.

DO NOT RUN: this makes network/BigQuery calls, which are forbidden in this
environment. Kept runnable for a machine with BigQuery credentials.

Shock dates / Cutoffs:
  1. SEC Enforcement Action: March 18, 2024 (2024-03-18) [Actions against Delphia & Global Predictions]
  2. SEC Warning Speech: December 5, 2023 (2023-12-05) [Gary Gensler AI washing speech]
  3. Pre-2024Q1 calendar boundary: January 1, 2024 (2024-01-01)
  4. Pre-2023 / Pre-GenAI clean boundary: January 1, 2023 (2023-01-01)
"""

import socket
_orig_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses
socket.getaddrinfo = _patched_getaddrinfo

from pathlib import Path
import pandas as pd
from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_SQL = REPO_ROOT / "scripts" / "raw_ingestion" / "patents" / "sql" / "oecd_ai_patents_query.sql"
PROJECT_ID = "skillforge-dev-502010"
DATASET_ID = "thesis_patents"

def build_query():
    raw_sql = BASE_SQL.read_text()
    
    # In candidate_patents, add p.family_id
    sql = raw_sql.replace(
        "p.publication_number,\n    p.filing_date,",
        "p.family_id,\n    p.publication_number,\n    p.filing_date,"
    )
    # In deduped_patents, add family_id and publication_date
    sql = sql.replace(
        "deduped_patents AS (\n  SELECT\n    ticker,\n    cik,\n    company_name,\n    publication_number,\n    filing_year,\n    publication_year,",
        "deduped_patents AS (\n  SELECT\n    ticker,\n    family_id,\n    cik,\n    company_name,\n    publication_number,\n    publication_date,\n    filing_date,\n    filing_year,\n    publication_year,"
    )
    # In patent_evaluation, add family_id and publication_date
    sql = sql.replace(
        "patent_evaluation AS (\n  SELECT\n    ticker,\n    cik,\n    company_name,\n    publication_number,\n    filing_year,\n    publication_year,",
        "patent_evaluation AS (\n  SELECT\n    ticker,\n    family_id,\n    cik,\n    company_name,\n    publication_number,\n    publication_date,\n    filing_date,\n    filing_year,\n    publication_year,"
    )
    # In classified_patents, add family_id and publication_date
    sql = sql.replace(
        "classified_patents AS (\n  SELECT\n    ticker,\n    cik,\n    company_name,\n    publication_number,\n    filing_year,\n    publication_year,",
        "classified_patents AS (\n  SELECT\n    ticker,\n    family_id,\n    cik,\n    company_name,\n    publication_number,\n    publication_date,\n    filing_date,\n    filing_year,\n    publication_year,"
    )
    
    # Replace the final SELECT
    idx = sql.find("-- Aggregate by firm and filing year")
    cte_part = sql[:idx].rstrip()
    if cte_part.endswith(")"):
        cte_part = cte_part + ",\n"
    
    final_query = cte_part + """
firm_capacity AS (
  SELECT
    m.ticker,
    m.cik,
    m.company_name,
    
    -- Pre-enforcement shock (publication_date < 20240318)
    COUNT(DISTINCT IF(c.publication_date < 20240318, c.family_id, NULL)) AS total_families_pre_enforce,
    COUNT(DISTINCT IF(c.publication_date < 20240318 AND c.is_ai_oecd, c.family_id, NULL)) AS ai_families_pre_enforce,
    COUNT(DISTINCT IF(c.publication_date < 20240318, c.publication_number, NULL)) AS total_pubs_pre_enforce,
    COUNT(DISTINCT IF(c.publication_date < 20240318 AND c.is_ai_oecd, c.publication_number, NULL)) AS ai_pubs_pre_enforce,

    -- Pre-warning shock (publication_date < 20231205)
    COUNT(DISTINCT IF(c.publication_date < 20231205, c.family_id, NULL)) AS total_families_pre_warning,
    COUNT(DISTINCT IF(c.publication_date < 20231205 AND c.is_ai_oecd, c.family_id, NULL)) AS ai_families_pre_warning,
    COUNT(DISTINCT IF(c.publication_date < 20231205, c.publication_number, NULL)) AS total_pubs_pre_warning,
    COUNT(DISTINCT IF(c.publication_date < 20231205 AND c.is_ai_oecd, c.publication_number, NULL)) AS ai_pubs_pre_warning,

    -- Pre-2024Q1 (publication_date < 20240101)
    COUNT(DISTINCT IF(c.publication_date < 20240101, c.family_id, NULL)) AS total_families_pre_2024q1,
    COUNT(DISTINCT IF(c.publication_date < 20240101 AND c.is_ai_oecd, c.family_id, NULL)) AS ai_families_pre_2024q1,
    COUNT(DISTINCT IF(c.publication_date < 20240101, c.publication_number, NULL)) AS total_pubs_pre_2024q1,
    COUNT(DISTINCT IF(c.publication_date < 20240101 AND c.is_ai_oecd, c.publication_number, NULL)) AS ai_pubs_pre_2024q1,

    -- Pre-2023 / Pre-GenAI clean baseline (publication_date < 20230101)
    COUNT(DISTINCT IF(c.publication_date < 20230101, c.family_id, NULL)) AS total_families_pre_2023,
    COUNT(DISTINCT IF(c.publication_date < 20230101 AND c.is_ai_oecd, c.family_id, NULL)) AS ai_families_pre_2023,
    COUNT(DISTINCT IF(c.publication_date < 20230101, c.publication_number, NULL)) AS total_pubs_pre_2023,
    COUNT(DISTINCT IF(c.publication_date < 20230101 AND c.is_ai_oecd, c.publication_number, NULL)) AS ai_pubs_pre_2023

  FROM (SELECT DISTINCT ticker, cik, company_name FROM matched_assignees) m
  LEFT JOIN classified_patents c ON m.ticker = c.ticker
  GROUP BY 1, 2, 3
)
SELECT * FROM firm_capacity
ORDER BY ai_families_pre_enforce DESC
"""
    return final_query

def main():
    print("Building SQL query with pre-2023 cutoffs...")
    sql = build_query()
    out_sql = REPO_ROOT / "scripts" / "raw_ingestion" / "patents" / "sql" / "preshock_patents_query.sql"
    out_sql.write_text(sql)

    print("Submitting query to Google BigQuery...")
    client = bigquery.Client(project=PROJECT_ID)

    job_config = bigquery.QueryJobConfig(
        destination=f"{PROJECT_ID}.{DATASET_ID}.sp500_firm_preshock_patent_capacity",
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )

    query_job = client.query(sql, job_config=job_config)
    query_job.result()
    print("Query finished successfully!")

    res_df = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.sp500_firm_preshock_patent_capacity`").to_dataframe()
    print(f"Downloaded {len(res_df)} firms from BigQuery.")

    # Calculate non-AI counts
    res_df["non_ai_families_pre_enforce"] = res_df["total_families_pre_enforce"] - res_df["ai_families_pre_enforce"]
    res_df["non_ai_families_pre_warning"] = res_df["total_families_pre_warning"] - res_df["ai_families_pre_warning"]
    res_df["non_ai_families_pre_2024q1"] = res_df["total_families_pre_2024q1"] - res_df["ai_families_pre_2024q1"]
    res_df["non_ai_families_pre_2023"] = res_df["total_families_pre_2023"] - res_df["ai_families_pre_2023"]

    out_parquet = REPO_ROOT / "data" / "raw" / "patents" / "sp500_firm_preshock_patent_capacity.parquet"
    res_df.to_parquet(out_parquet, index=False)
    print(f"Saved results to {out_parquet}")

if __name__ == "__main__":
    main()
