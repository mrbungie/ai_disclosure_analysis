"""
scripts/us/earnings_calls/01_fetch_transcripts.py — earnings call transcripts
for the US universe, from the public `kurry/sp500_earnings_transcripts`
dataset on Hugging Face.

WHY THIS DOCUMENT TYPE MATTERS MORE THAN ITS PHASE NUMBER SUGGESTS.
docs/document_expansion_plan.md files earnings calls as Fase 4, blocked on
"fuente no resuelta". But docs/problemas_academicos.md #5 is blocked on the
same thing, and that one is about whether the instrument measures what the
thesis claims: Welltower, the only firm in the corpus with an SEC comment
letter about its AI claims, is NOT flagged by the promotional score. That
is not a bug — the SEC did not object to the 10-K. It objected to
"industry-leading" said on the EARNINGS CALL and in the press release
without proportional support in the 10-K. The score measures excess
promotion WITHIN a filing; the regulator pursues a GAP BETWEEN CHANNELS.
Without the calls there is no second channel to measure the gap against.

WHY THIS SOURCE. The plan assumed the alternative to a paid provider was
"517 separate scrapers". It isn't: this dataset is 33,362 transcripts over
685 companies, 2005-2025, MIT-licensed, one download. Measured against
this project's universe it covers 481 of 517 tickers (93.0%) inside the
2021-2025 window. A second candidate (`glopardo/sp500-earnings-transcripts`)
was checked and is a strict subset — it adds zero companies — so there is
nothing to union. `Bose345/sp500_earnings_transcripts` is the same 33,362
rows mirrored.

TWO THINGS THAT ARE NOT FIXED BY CHOOSING THIS SOURCE, and that any
analysis has to carry:

  1. ~3.2 transcripts per company-year, not 4. Firms skip quarters. The
     call panel is unbalanced the same way the Italian one is, and a
     quarter with no call is NOT "this firm said nothing about AI".
  2. Provenance. MIT is the licence the dataset's author applied; it is not
     a claim about rights in the underlying transcript text, which came
     from some transcription publisher. For a thesis that chain has to be
     stated. It is checkable: a sampled transcript can be cross-read
     against the Item 2.02 8-K press release of the same date, which IS a
     primary source in EDGAR.

Stores the RAW transcript record per call, gzip-compressed, and a manifest
in the shared shape. Same fetch/parse split as every other document type:
turning speaker turns into paragraphs is 02_extract_sections.py's job.

Usage:
    uv run python scripts/us/earnings_calls/01_fetch_transcripts.py [--limit N]
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))

import pipeline_logger

DATASET = os.environ.get("EARNINGS_CALLS_DATASET", "kurry/sp500_earnings_transcripts")
FORM_TYPE = "Earnings call transcript"


def _normalize_ticker(value: str) -> str:
    """`BRK.B` in one source is `BRK-B` in the other. Normalising both sides
    to the dash form is what turns an apparent 35-ticker coverage gap into
    mostly renamed/acquired companies."""
    return str(value).upper().strip().replace(".", "-")


def _plain(value):
    """numpy/pandas scalars and arrays -> plain JSON types.

    `structured_content` is a numpy array of dicts (the speaker turns).
    Handing that to json.dump with default=str stringifies the whole array
    instead of encoding it, so the turns survive as text that cannot be
    read back — which is exactly what happened."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def _is_readable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    turns = payload.get("structured_content")
    return isinstance(turns, list) and (not turns or isinstance(turns[0], dict))


def load_dataset() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get("HF_TOKEN")
    frames = []
    for name in list_repo_files(DATASET, repo_type="dataset", token=token):
        if name.endswith(".parquet"):
            frames.append(pd.read_parquet(
                hf_hub_download(DATASET, name, repo_type="dataset", token=token)))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--from-year", type=int, default=2021)
    parser.add_argument("--to-year", type=int, default=2025)
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / "configs" / "us" / "config.yaml").read_text())
    raw_dir = REPO_ROOT / "data" / "raw" / "earnings_calls"
    manifest_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "filing_manifest_earnings_calls.parquet"

    universe = pd.read_csv(REPO_ROOT / "configs" / "us" / "universe.csv")
    wanted = {_normalize_ticker(t): t for t in universe["ticker"]}
    cik_by_ticker = dict(zip(universe["ticker"].map(_normalize_ticker), universe["cik"]))

    pipeline_logger.log_event(pipeline_step="us_fetch_earnings_calls", level="INFO",
                              message=f"Loading {DATASET}...", log_dir=manifest_dir)
    df = load_dataset()
    df["_ticker"] = df["symbol"].map(_normalize_ticker)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    scoped = df[df["_ticker"].isin(wanted) & df["year"].between(args.from_year, args.to_year)].copy()
    if args.limit:
        scoped = scoped.head(args.limit)

    pipeline_logger.log_event(
        pipeline_step="us_fetch_earnings_calls", level="INFO",
        message=(f"{len(df):,} transcripts in dataset; {len(scoped):,} in universe and "
                 f"[{args.from_year}, {args.to_year}] across {scoped['_ticker'].nunique()} tickers"),
        log_dir=manifest_dir)

    rows = []
    for record in scoped.to_dict("records"):
        ticker = record["_ticker"]
        # Natural key: ticker + fiscal period. The dataset has no filing
        # accession of its own, and the call is identified by which quarter
        # it reports on, not by when it happened.
        document_id = f"{ticker}_{int(record['year'])}Q{int(record['quarter'])}"
        path = raw_dir / ticker / f"{document_id}.json.gz"
        # Rewritten when absent OR unreadable. An earlier version of this
        # script serialised `structured_content` with json's default=str,
        # which turned the numpy array of speaker turns into a Python REPR
        # string — valid text, unparseable JSON, and invisible until
        # extraction choked on it. Skipping purely on "the file exists"
        # would leave those files broken forever, since the fetch is
        # otherwise idempotent.
        if not _is_readable(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: _plain(record[k]) for k in
                       ("symbol", "quarter", "year", "date", "content",
                        "structured_content", "company_name", "company_id")
                       if k in record}
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        turns = record.get("structured_content")
        rows.append({
            "document_id": document_id,
            "ticker": wanted[ticker],
            "cik": cik_by_ticker.get(ticker, ""),
            "source": f"huggingface:{DATASET}",
            "form_type": FORM_TYPE,
            "filing_type": "earnings_call",
            "filing_date": str(record.get("date") or "")[:10],
            "period_end_date": f"{int(record['year'])}Q{int(record['quarter'])}",
            "local_path": str(path),
            "format": "json",
            "download_status": "completed",
            "n_bytes": path.stat().st_size,
            "n_turns": len(turns) if turns is not None else 0,
            "n_chars": len(str(record.get("content") or "")),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(manifest_path, index=False)
    pipeline_logger.log_event(
        pipeline_step="us_fetch_earnings_calls", level="SUCCESS",
        message=(f"{len(manifest):,} transcripts, {manifest.ticker.nunique()} tickers, "
                 f"{manifest.n_bytes.sum()/1e9:.2f} GB -> {manifest_path}"),
        log_dir=manifest_dir)


if __name__ == "__main__":
    main()
