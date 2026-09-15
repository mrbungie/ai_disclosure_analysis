"""Build a per-ticker roster of SEC-registered corporate officers from Form 3/4/5 data.

Deterministic, no LLM, no free-text parsing of transcripts or proxies.
Downloads SEC's quarterly Form 3/4/5 structured bulk datasets
(https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets),
which is the SAME data every officer's insider stock filing (Form 4) is
built from -- `RPTOWNER_RELATIONSHIP` and `RPTOWNER_TITLE` are fields the
officer (or their filing agent) fills in under Section 16 liability, not
free text anyone wrote for a transcript.

Why this exists: `build_earnings_call_speaker_roles.py`'s cross-quarter
`prepared`-turn roster and operator hand-off signal both need a person to
either speak in `prepared` at least once, or be introduced by the operator,
to be identified. Neither fires for an executive who is on every call ONLY
to field Q&A and is never announced by name (ServiceNow's Amit Zavery,
President/CPO/COO, confirmed via this exact dataset: Form 4 accession
0001781064-26-000018 lists `RPTOWNERNAME="Zavery Amit"`,
`RPTOWNER_RELATIONSHIP="Officer"`, `RPTOWNER_TITLE="President, CPO and
COO"`). This roster closes that gap with a name-independent, SEC-of-record
source of truth: if someone filed a Form 4 as an officer of a ticker at any
point in the sample window, they are that company's officer, regardless of
what any call transcript's `section` field says.

Writes data/interim/sec_officer_roster/officer_roster.parquet:
one row per (ticker, officer_name_raw, title) triple, deduplicated across
every quarter fetched.
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw" / "sec_form345"
OUT_DIR = REPO_ROOT / "data" / "interim" / "sec_officer_roster"
OUT_PATH = OUT_DIR / "officer_roster.parquet"

# Covers the full earnings-call sample window (calls span 2021Q1-2026Q4 by
# fiscal label; officers typically file many Form 4s across a multi-year
# tenure, so any one quarter in a term usually catches them, and pulling
# every quarter maximizes coverage rather than relying on that).
QUARTERS = [(y, q) for y in range(2021, 2027) for q in range(1, 5) if not (y == 2026 and q == 4)]

USER_AGENT = "thesis-research contact@example.com"
BASE_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"


def fetch_quarter(y: int, q: int) -> tuple[Path, Path] | None:
    """Download (if needed) and extract REPORTINGOWNER.tsv + SUBMISSION.tsv for one quarter."""
    qdir = RAW_DIR / f"{y}q{q}"
    owner_tsv = qdir / "REPORTINGOWNER.tsv"
    sub_tsv = qdir / "SUBMISSION.tsv"
    if owner_tsv.exists() and sub_tsv.exists():
        return owner_tsv, sub_tsv

    url = BASE_URL.format(y=y, q=q)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    if resp.status_code == 404:
        print(f"  {y}Q{q}: not published yet, skipping")
        return None
    resp.raise_for_status()

    qdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extract("REPORTINGOWNER.tsv", qdir)
        zf.extract("SUBMISSION.tsv", qdir)
    print(f"  {y}Q{q}: downloaded and extracted")
    return owner_tsv, sub_tsv


def process_quarter(y: int, q: int, owner_tsv: Path, sub_tsv: Path) -> Path:
    """Filter one quarter's officer rows into its own small parquet (cached, resumable)."""
    quarter_out = owner_tsv.parent / "officer_rows.parquet"
    if quarter_out.exists():
        return quarter_out
    # A fresh connection per quarter -- reusing one connection across many
    # read_csv+join calls in a loop was observed to degrade badly (a single
    # quarter's join alone runs in ~0.15s standalone, but the loop stalled
    # for 10+ minutes after accumulating ~15 registered results on one
    # connection).
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                sub.ISSUERTRADINGSYMBOL AS ticker,
                own.RPTOWNERNAME AS officer_name_raw,
                own.RPTOWNER_TITLE AS title
            FROM read_csv('{owner_tsv.as_posix()}', delim='\t', header=True, quote='',
                           all_varchar=True, strict_mode=False) own
            JOIN read_csv('{sub_tsv.as_posix()}', delim='\t', header=True, quote='',
                           all_varchar=True, strict_mode=False) sub
                ON sub.ACCESSION_NUMBER = own.ACCESSION_NUMBER
            WHERE own.RPTOWNER_RELATIONSHIP ILIKE '%Officer%'
              AND sub.ISSUERTRADINGSYMBOL IS NOT NULL AND sub.ISSUERTRADINGSYMBOL <> ''
        ) TO '{quarter_out.as_posix()}' (FORMAT PARQUET)
    """)
    con.close()
    return quarter_out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    quarter_parquets = []

    for y, q in QUARTERS:
        try:
            paths = fetch_quarter(y, q)
        except requests.RequestException as exc:
            print(f"  {y}Q{q}: fetch failed ({exc}), skipping", file=sys.stderr)
            continue
        if paths is None:
            continue
        owner_tsv, sub_tsv = paths
        quarter_parquets.append(process_quarter(y, q, owner_tsv, sub_tsv))
        print(f"  {y}Q{q}: filtered")
        # be polite to SEC's servers between quarter downloads
        time.sleep(0.2)

    if not quarter_parquets:
        print("no quarters fetched, nothing to write", file=sys.stderr)
        sys.exit(1)

    glob_pattern = str(RAW_DIR / "*" / "officer_rows.parquet")
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE officer_roster AS
        SELECT DISTINCT ticker, officer_name_raw, title
        FROM read_parquet('{glob_pattern}')
    """)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY officer_roster TO '{OUT_PATH}' (FORMAT PARQUET)")

    n_rows, n_tickers = con.execute(
        "SELECT count(*), count(DISTINCT ticker) FROM officer_roster"
    ).fetchone()
    print(f"officer_roster written: {OUT_PATH} ({n_rows} rows, {n_tickers} tickers)")


if __name__ == "__main__":
    main()
