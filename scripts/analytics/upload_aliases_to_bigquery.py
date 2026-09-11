"""
Upload S&P 500 patent aliases dataset to BigQuery.
Project: skillforge-dev-502010
Dataset: thesis_patents
"""

import socket
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

from pathlib import Path
from google.cloud import bigquery
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_WIDE_PATH = REPO_ROOT / "data" / "raw" / "sp500_patent_aliases_seed.csv"
SEED_LONG_PATH = REPO_ROOT / "data" / "raw" / "sp500_patent_assignees_long.csv"

PROJECT_ID = "skillforge-dev-502010"
DATASET_ID = "thesis_patents"

def main():
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"

    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {PROJECT_ID}.{DATASET_ID} already exists.")
    except Exception:
        client.create_dataset(dataset, exists_ok=True)
        print(f"Created dataset {PROJECT_ID}.{DATASET_ID}.")

    # 1. Upload Wide Table
    df_wide = pd.read_csv(SEED_WIDE_PATH, dtype={"cik": str})
    table_wide_id = f"{PROJECT_ID}.{DATASET_ID}.sp500_patent_aliases_seed"
    job_config_wide = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("cik", "STRING"),
            bigquery.SchemaField("company_name", "STRING"),
            bigquery.SchemaField("cleaned_name", "STRING"),
            bigquery.SchemaField("company_stem", "STRING"),
            bigquery.SchemaField("num_aliases", "INTEGER"),
            bigquery.SchemaField("aliases_list", "STRING"),
            bigquery.SchemaField("assignee_regex", "STRING"),
            bigquery.SchemaField("active_status", "STRING"),
        ]
    )
    job_wide = client.load_table_from_dataframe(df_wide, table_wide_id, job_config=job_config_wide)
    job_wide.result()
    print(f"Uploaded {len(df_wide)} rows to {table_wide_id}")

    # 2. Upload Long Table
    df_long = pd.read_csv(SEED_LONG_PATH, dtype={"cik": str})
    table_long_id = f"{PROJECT_ID}.{DATASET_ID}.sp500_patent_assignees_long"
    job_config_long = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("cik", "STRING"),
            bigquery.SchemaField("company_name", "STRING"),
            bigquery.SchemaField("assignee_alias", "STRING"),
            bigquery.SchemaField("is_known_subsidiary", "BOOLEAN"),
            bigquery.SchemaField("active_status", "STRING"),
        ]
    )
    job_long = client.load_table_from_dataframe(df_long, table_long_id, job_config=job_config_long)
    job_long.result()
    print(f"Uploaded {len(df_long)} rows to {table_long_id}")

if __name__ == "__main__":
    main()
