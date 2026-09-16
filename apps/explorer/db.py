"""
apps/explorer/db.py — Internal DuckDB connection manager with dynamic view registration.

Protocol-agnostic: Supports local filesystem paths (data/) as well as remote protocols
(s3://, hf://, https://, http://, file://) configured via DATA_URI / DATA_PATH / DATA_DIR.
No external database file is required.
"""

from __future__ import annotations

import os
import re
import threading
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from apps.explorer.config import (
    DATA_URI,
    HF_TOKEN,
    S3_ACCESS_KEY_ID,
    S3_ENDPOINT,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
    S3_URL_STYLE,
    is_remote_uri,
)

COLLECTION_MIN = 200
PART_SUFFIX = re.compile(r"__(?:part|session|run)=")
PART_FILE = re.compile(r"^part-\d+\.parquet$")

_LOCK = threading.Lock()
_CONN: duckdb.DuckDBPyConnection | None = None
_INIT_STATS: dict[str, Any] = {
    "status": "not_initialized",
    "views_count": 0,
    "failed_count": 0,
    "elapsed_s": 0.0,
    "data_uri": str(DATA_URI),
    "protocol": "local",
}


def _ident(parts: list[str]) -> str:
    name = "__".join(parts)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    return name or "root"


def group_datasets_local(data_dir: Path) -> dict[tuple[str, str], tuple[str, bool, bool]]:
    """Group local parquet files: (schema, view) -> (glob, hive_partitioning, filename_column)."""
    if not data_dir.exists():
        return {}

    files = sorted(p for p in data_dir.rglob("*.parquet") if not p.name.startswith("."))
    plain_by_dir: dict[Path, list[Path]] = defaultdict(list)
    datasets: dict[str, tuple[Path, str, bool, bool]] = {}

    for f in files:
        try:
            rel = f.relative_to(data_dir)
        except ValueError:
            continue

        hive_idx = next((i for i, c in enumerate(rel.parts[:-1]) if "=" in c), None)
        if hive_idx is not None:
            root = data_dir.joinpath(*rel.parts[:hive_idx])
            datasets.setdefault(f"{root}/**/*.parquet", (root, "", True, False))
        elif m := PART_SUFFIX.search(f.name):
            prefix = f.name[: m.start()]
            datasets.setdefault(f"{f.parent}/{f.name[: m.end()]}*.parquet", (f.parent, prefix, False, False))
        elif PART_FILE.match(f.name):
            datasets.setdefault(f"{f.parent}/part-*.parquet", (f.parent, "", False, False))
        else:
            plain_by_dir[f.parent].append(f)

    for d, fs in plain_by_dir.items():
        if len(fs) >= COLLECTION_MIN:
            datasets[f"{d}/*.parquet"] = (d, "", False, True)
        else:
            for f in fs:
                datasets[str(f)] = (f.parent, f.stem, False, False)

    views: dict[tuple[str, str], tuple[str, bool, bool]] = {}
    for glob, (root, stem, hive, filename) in sorted(datasets.items()):
        try:
            rel_parts = list(root.relative_to(data_dir).parts) + ([stem] if stem else [])
        except ValueError:
            rel_parts = [stem] if stem else ["data"]

        if len(rel_parts) == 1 and stem:
            schema, name_parts = "main", rel_parts
        else:
            schema, name_parts = _ident(rel_parts[:1]), rel_parts[1:]

        name, n = _ident(name_parts), 2
        while (schema, name) in views:
            name = f"{_ident(name_parts)}_{n}"
            n += 1
        views[(schema, name)] = (glob, hive, filename)

    return views


def group_datasets_remote(data_uri: str, file_urls: list[str]) -> dict[tuple[str, str], tuple[str, bool, bool]]:
    """Group remote parquet URLs (s3://, https://, hf://): (schema, view) -> (url, hive, filename)."""
    clean_prefix = data_uri.rstrip("/") + "/"
    plain_by_dir: dict[str, list[tuple[tuple[str, ...], str, str]]] = defaultdict(list)
    datasets: dict[str, tuple[tuple[str, ...], str, bool, bool]] = {}

    for url in sorted(file_urls):
        if not url.startswith(clean_prefix):
            continue
        rel = url[len(clean_prefix):]
        parts = rel.split("/")
        parent_url = "/".join(url.split("/")[:-1])
        filename = parts[-1]

        hive_idx = next((i for i, c in enumerate(parts[:-1]) if "=" in c), None)
        if hive_idx is not None:
            root_url = clean_prefix + "/".join(parts[:hive_idx])
            root_parts = tuple(parts[:hive_idx])
            datasets.setdefault(f"{root_url}/**/*.parquet", (root_parts, "", True, False))
        elif m := PART_SUFFIX.search(filename):
            prefix = filename[: m.start()]
            datasets.setdefault(f"{parent_url}/{filename[: m.end()]}*.parquet", (tuple(parts[:-1]), prefix, False, False))
        elif PART_FILE.match(filename):
            datasets.setdefault(f"{parent_url}/part-*.parquet", (tuple(parts[:-1]), "", False, False))
        else:
            plain_by_dir[parent_url].append((tuple(parts[:-1]), filename, url))

    for parent_url, entries in plain_by_dir.items():
        if len(entries) >= COLLECTION_MIN:
            dir_parts = entries[0][0]
            datasets[f"{parent_url}/*.parquet"] = (dir_parts, "", False, True)
        else:
            for dir_parts, filename, url in entries:
                stem = filename[:-8] if filename.endswith(".parquet") else filename
                datasets[url] = (dir_parts, stem, False, False)

    views: dict[tuple[str, str], tuple[str, bool, bool]] = {}
    for glob_or_url, (dir_parts, stem, hive, filename_col) in sorted(datasets.items()):
        rel_parts = list(dir_parts) + ([stem] if stem else [])
        if len(rel_parts) == 1 and stem:
            schema, name_parts = "main", rel_parts
        else:
            schema, name_parts = _ident(rel_parts[:1]), rel_parts[1:]
        name, n = _ident(name_parts), 2
        while (schema, name) in views:
            name = f"{_ident(name_parts)}_{n}"
            n += 1
        views[(schema, name)] = (glob_or_url, hive, filename_col)

    return views


def _configure_duckdb_extensions(conn: duckdb.DuckDBPyConnection) -> None:
    """Install and configure extensions for HTTP/S3/HF remote storage."""
    try:
        conn.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        pass

    # S3 / B2 configuration
    if S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY:
        try:
            conn.execute(f"SET s3_access_key_id = '{S3_ACCESS_KEY_ID}'")
            conn.execute(f"SET s3_secret_access_key = '{S3_SECRET_ACCESS_KEY}'")
            if S3_ENDPOINT:
                conn.execute(f"SET s3_endpoint = '{S3_ENDPOINT}'")
            if S3_REGION:
                conn.execute(f"SET s3_region = '{S3_REGION}'")
            if S3_URL_STYLE:
                conn.execute(f"SET s3_url_style = '{S3_URL_STYLE}'")
        except Exception:
            pass

    # Hugging Face token authorization
    if HF_TOKEN:
        try:
            conn.execute(
                f"CREATE OR REPLACE SECRET hf_token (TYPE HTTP, EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {HF_TOKEN}'}})"
            )
        except Exception:
            pass


def init_db(data_uri: str | None = None, force_refresh: bool = False) -> duckdb.DuckDBPyConnection:
    """Initialize internal in-memory DuckDB and register views over local or remote data_uri."""
    global _CONN, _INIT_STATS

    target_uri = (data_uri or DATA_URI).strip()

    with _LOCK:
        if _CONN is not None and not force_refresh and _INIT_STATS.get("data_uri") == target_uri:
            return _CONN

        t0 = time.time()
        conn = duckdb.connect(":memory:")
        conn.execute("SET preserve_insertion_order = false;")

        # Configure network/cloud storage support
        _configure_duckdb_extensions(conn)

        protocol = urllib.parse.urlparse(target_uri).scheme.lower() or "local"

        if is_remote_uri(target_uri):
            # Remote protocol: use DuckDB's glob to list files
            clean_uri = target_uri.rstrip("/")
            try:
                res = conn.execute(f"SELECT file FROM glob('{clean_uri}/**/*.parquet')").fetchall()
                file_urls = [r[0] for r in res]
                views = group_datasets_remote(target_uri, file_urls)
            except Exception:
                views = {}
        else:
            # Local filesystem
            local_path = Path(target_uri.replace("file://", "")).resolve()
            views = group_datasets_local(local_path)

        success = 0
        failed = 0
        schemas_created = set()

        for (schema, name), (glob_or_url, hive, filename) in views.items():
            opts = ["union_by_name = true"]
            if hive:
                opts.append("hive_partitioning = true")
            if filename:
                opts.append("filename = true")
            safe_target = glob_or_url.replace("'", "''")

            if schema not in schemas_created:
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                schemas_created.add(schema)

            try:
                conn.execute(
                    f'CREATE OR REPLACE VIEW "{schema}"."{name}" AS '
                    f'SELECT * FROM read_parquet(\'{safe_target}\', {", ".join(opts)})'
                )
                success += 1
            except Exception:
                failed += 1

        # Register thesis sector mapping views
        ensure_sector_views(conn)

        elapsed = time.time() - t0
        _CONN = conn
        _INIT_STATS = {
            "status": "ready",
            "views_count": success,
            "failed_count": failed,
            "elapsed_s": round(elapsed, 2),
            "data_uri": target_uri,
            "protocol": protocol,
            "schemas": sorted(list(schemas_created)),
        }
        return _CONN


def get_db() -> duckdb.DuckDBPyConnection:
    """Return the active DuckDB connection, initializing if necessary."""
    if _CONN is None:
        return init_db()
    return _CONN


def get_init_stats() -> dict[str, Any]:
    """Return current connection stats."""
    return dict(_INIT_STATS)


def run_query(sql: str, limit: int = 1000) -> tuple[pd.DataFrame | None, str | None, float, int]:
    """Run SQL safely and return (df, error_message, elapsed_seconds, total_rows)."""
    conn = get_db()
    t0 = time.time()
    try:
        cleaned_sql = sql.strip().rstrip(";")
        upper = cleaned_sql.upper()
        if limit and limit > 0 and "LIMIT" not in upper:
            query_to_run = f"{cleaned_sql} LIMIT {limit}"
        else:
            query_to_run = cleaned_sql

        with _LOCK:
            cur = conn.cursor()
            res = cur.execute(query_to_run)
            df = res.fetchdf()
            elapsed = time.time() - t0
            row_count = len(df)
            return df, None, round(elapsed, 3), row_count
    except Exception as e:
        elapsed = time.time() - t0
        return None, str(e), round(elapsed, 3), 0


def list_tables_by_schema() -> dict[str, list[str]]:
    """Return map of schema -> list of table/view names."""
    conn = get_db()
    try:
        with _LOCK:
            df = conn.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') "
                "ORDER BY table_schema, table_name"
            ).fetchdf()
        out: dict[str, list[str]] = defaultdict(list)
        for _, row in df.iterrows():
            out[row["table_schema"]].append(row["table_name"])
        return dict(out)
    except Exception:
        return {}


def get_columns_for_table(schema: str, table_name: str) -> list[dict[str, str]]:
    """Return column names and types for a specific table."""
    conn = get_db()
    try:
        with _LOCK:
            df = conn.execute(f'DESCRIBE "{schema}"."{table_name}"').fetchdf()
        return [{"name": str(r["column_name"]), "type": str(r["column_type"])} for _, r in df.iterrows()]
    except Exception:
        return []


def get_gold_hierarchy() -> dict[str, dict[str, list[str]]]:
    """Return organized hierarchy for gold layer: grain -> { 'datasets': [...], 'spines': [...], ... }."""
    tables = list_tables_by_schema().get("gold", [])
    grains = ["firm", "firm_year", "firm_quarter", "call", "document", "activity"]
    hierarchy: dict[str, dict[str, list[str]]] = {
        g: {"datasets": [], "spines": [], "covariates": [], "targets": []} for g in grains
    }

    for t in tables:
        parts = t.split("__")
        if len(parts) >= 2:
            kind, grain = parts[0], parts[1]
            if grain in hierarchy and kind in hierarchy[grain]:
                hierarchy[grain][kind].append(t)

    return hierarchy


def get_tickers_list() -> list[tuple[str, str]]:
    """Return list of (display_label, ticker_symbol) for available companies."""
    conn = get_db()
    try:
        with _LOCK:
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'silver' AND table_name = 'firm_universe'"
            ).fetchdf()
            if not tables.empty:
                df = conn.execute(
                    "SELECT ticker, company_name, sic, industry_group "
                    "FROM silver.firm_universe "
                    "ORDER BY ticker"
                ).fetchdf()
                return [(f"{row['ticker']} - {row['company_name'] or 'N/A'}", str(row["ticker"])) for _, row in df.iterrows()]

            df = conn.execute(
                "SELECT DISTINCT ticker FROM gold.spines__firm__firm ORDER BY ticker"
            ).fetchdf()
            return [(str(row["ticker"]), str(row["ticker"])) for _, row in df.iterrows()]
    except Exception:
        return [("AAPL - Apple Inc.", "AAPL"), ("MSFT - Microsoft Corp", "MSFT"), ("NVDA - NVIDIA Corp", "NVDA")]


def get_firm_overview(ticker: str) -> dict[str, Any]:
    """Return overview details for a single firm card."""
    conn = get_db()
    data: dict[str, Any] = {"ticker": ticker}
    try:
        with _LOCK:
            firm_df = conn.execute(
                f"SELECT * FROM silver.firm_universe WHERE ticker = '{ticker}' LIMIT 1"
            ).fetchdf()
            if not firm_df.empty:
                r = firm_df.iloc[0]
                data.update({
                    "company_name": r.get("company_name", "N/A"),
                    "cik": r.get("cik", "N/A"),
                    "sic": r.get("sic", "N/A"),
                    "industry_group": r.get("industry_group", "N/A"),
                    "active_status": r.get("active_status", "N/A"),
                    "delisted": bool(r.get("delisted", False)),
                })
            else:
                data.update({"company_name": ticker, "cik": "N/A", "industry_group": "N/A", "delisted": False})

            posture_df = conn.execute(
                f"SELECT * FROM gold.datasets__firm__firm WHERE ticker = '{ticker}' LIMIT 1"
            ).fetchdf()
            if not posture_df.empty:
                pr = posture_df.iloc[0]
                data.update({
                    "archetype": pr.get("archetype", "Unclassified"),
                    "archetype_stability": round(float(pr.get("archetype_stability", 0.0)), 3) if pd.notnull(pr.get("archetype_stability")) else None,
                    "n_frames_total": int(pr.get("n_frames", 0)) if pd.notnull(pr.get("n_frames")) else 0,
                    "promotional_posture": round(float(pr.get("promotional_posture", 0.0)), 3) if pd.notnull(pr.get("promotional_posture")) else None,
                    "risk_orientation": round(float(pr.get("risk_orientation", 0.0)), 3) if pd.notnull(pr.get("risk_orientation")) else None,
                    "governance_orientation": round(float(pr.get("governance_orientation", 0.0)), 3) if pd.notnull(pr.get("governance_orientation")) else None,
                    "disclosure_intensity": round(float(pr.get("disclosure_intensity", 0.0)), 3) if pd.notnull(pr.get("disclosure_intensity")) else None,
                })
            else:
                data.update({
                    "archetype": "N/A",
                    "archetype_stability": None,
                    "n_frames_total": 0,
                })
    except Exception as e:
        data["error"] = str(e)
    return data


SIC2_NAMES_THESIS = {
    "00": "Unclassified", "01": "Agricultural Production - Crops", "10": "Metal Mining",
    "13": "Oil & Gas Extraction", "14": "Mining & Quarrying Nonmetallic Minerals",
    "15": "Building Construction - General Contractors", "16": "Heavy Construction Non-Building",
    "17": "Construction - Special Trade Contractors", "20": "Food & Kindred Products",
    "21": "Tobacco Products", "22": "Textile Mill Products", "23": "Apparel & Other Finished Products",
    "24": "Lumber & Wood Products", "25": "Furniture & Fixtures", "26": "Paper & Allied Products",
    "27": "Printing & Publishing", "28": "Chemicals & Pharmaceuticals", "29": "Petroleum Refining",
    "30": "Rubber & Misc Plastics Products", "31": "Leather & Leather Products",
    "33": "Primary Metal Industries", "34": "Fabricated Metal Products",
    "35": "Industrial Machinery & Computer Equipment", "36": "Electronic & Other Electrical Equipment",
    "37": "Transportation Equipment / Aerospace", "38": "Measuring & Optical Instruments / Medical Devices",
    "39": "Misc Manufacturing Industries", "40": "Railroad Transportation",
    "42": "Motor Freight Transportation & Warehousing", "44": "Water Transportation",
    "45": "Transportation by Air", "47": "Transportation Services", "48": "Communications",
    "49": "Electric, Gas & Sanitary Services", "50": "Wholesale Trade - Durable Goods",
    "51": "Wholesale Trade - Nondurable Goods", "52": "Building Materials & Garden Supplies",
    "53": "General Merchandise Stores", "54": "Food Stores", "55": "Automotive Dealers & Service Stations",
    "56": "Apparel & Accessory Stores", "57": "Home Furniture & Furnishings Stores",
    "58": "Eating & Drinking Places", "59": "Miscellaneous Retail",
    "60": "Depository Institutions / Commercial Banking", "61": "Nondepository Credit Institutions",
    "62": "Security & Commodity Brokers / Asset Mgmt", "63": "Insurance Carriers",
    "64": "Insurance Agents, Brokers & Service", "65": "Real Estate Operators & Lessors",
    "67": "Holding & Other Investment Offices / REITs", "70": "Hotels, Rooming Houses & Camps",
    "73": "Business Services / Computer Software & Services", "78": "Motion Pictures",
    "79": "Amusement & Recreation Services", "80": "Health Services",
    "87": "Engineering, Accounting, Research & Management",
}

THESIS_ECONOMIC_SECTORS = [
    "Technology",
    "Financial Services",
    "Healthcare & Pharma",
    "Industrials & Mfg",
    "Retail & Wholesale",
    "Utilities",
    "Transportation",
    "Energy & Mining",
    "Business Services",
    "Communications",
    "Other / Diversified",
]

THESIS_BROAD_DIVISIONS = [
    "Manufacturing",
    "Finance, Ins. & Real Estate",
    "Services",
    "Transport & Utilities",
    "Wholesale & Retail",
    "Mining & Construction",
    "Other / Diversified",
]


def ensure_sector_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Register internal_firm_sectors view matching thesis.qmd's exact three-tier taxonomy."""
    try:
        conn.execute("""
        CREATE OR REPLACE VIEW internal_firm_sectors AS
        SELECT 
            ticker,
            company_name,
            sic,
            LPAD(CAST(sic AS VARCHAR), 4, '0') AS sic4,
            SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS sic2,
            CASE 
                WHEN TRY_CAST(sic AS INTEGER) < 2000 THEN 'Mining & Construction'
                WHEN TRY_CAST(sic AS INTEGER) < 4000 THEN 'Manufacturing'
                WHEN TRY_CAST(sic AS INTEGER) < 5000 THEN 'Transport & Utilities'
                WHEN TRY_CAST(sic AS INTEGER) < 6000 THEN 'Wholesale & Retail'
                WHEN TRY_CAST(sic AS INTEGER) < 6800 THEN 'Finance, Ins. & Real Estate'
                WHEN TRY_CAST(sic AS INTEGER) < 9000 THEN 'Services'
                ELSE 'Other / Diversified'
            END AS division,
            CASE 
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) IN (73, 35, 36) THEN 'Technology'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) = 48 THEN 'Communications'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) IN (28, 38, 80) THEN 'Healthcare & Pharma'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) BETWEEN 60 AND 67 THEN 'Financial Services'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) BETWEEN 20 AND 39 THEN 'Industrials & Mfg'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) BETWEEN 40 AND 47 THEN 'Transportation'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) = 49 THEN 'Utilities'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) BETWEEN 10 AND 14 
                  OR TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) = 29 THEN 'Energy & Mining'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) BETWEEN 50 AND 59 THEN 'Retail & Wholesale'
                WHEN TRY_CAST(SUBSTRING(LPAD(CAST(sic AS VARCHAR), 4, '0'), 1, 2) AS INTEGER) BETWEEN 70 AND 89 THEN 'Business Services'
                ELSE 'Other / Diversified'
            END AS agg_sector
        FROM silver.firm_universe;
        """)
    except Exception:
        pass


def get_sic_filter_options() -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Return choices for: (1) Broad SIC Divisions, (2) 2-digit SIC Groups, (3) Aggregated Economic Sectors."""
    conn = get_db()
    sic2_choices = []

    try:
        with _LOCK:
            ensure_sector_views(conn)
            df2 = conn.execute(
                "SELECT sic2, COUNT(*) AS n_firms "
                "FROM internal_firm_sectors "
                "WHERE sic2 IS NOT NULL AND sic2 != '' "
                "GROUP BY sic2 ORDER BY n_firms DESC"
            ).fetchdf()
            for _, r in df2.iterrows():
                c = str(r["sic2"])
                name = SIC2_NAMES_THESIS.get(c, f"SIC Group {c}")
                cnt = int(r["n_firms"])
                sic2_choices.append((f"{c}: {name} ({cnt} firms)", c))
    except Exception:
        pass

    return THESIS_BROAD_DIVISIONS, sic2_choices, THESIS_ECONOMIC_SECTORS


