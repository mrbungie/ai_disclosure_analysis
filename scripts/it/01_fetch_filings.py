"""
scripts/it/01_fetch_filings.py — fetches every Italian ESEF annual financial
report from filings.xbrl.org and mirrors the raw XHTML locally.

Saves RAW BYTES, gzip-compressed, not extracted text — download and
extraction are two stages on purpose, the same contract as
scripts/cl/01_fetch_filings.py and scripts/us/edgar_fetch.py: a later
change to extraction re-reads the local mirror instead of re-hitting the
aggregator, and the slow network stage can finish independently of however
long the parsing rules take to get right.

WHAT A ROW IS. One row per FILING, keyed by the source's own `fxo_id`, not
one row per issuer-year. Issuers do file more than once for the same period
— Nexi has two 2024-12-31 filings, one carrying the full annual report and
one carrying only the financial statements — and choosing between them is a
judgement about content. Raw storage records both and lets extraction (or a
query) decide, which is the same reason the manifest keeps `form_type`
verbatim alongside the normalized `filing_type`.

`has_narrative` is computed here from bytes already in memory (see
xbrl_filings_client.probe_report). It is a marker heuristic, not a parse:
roughly 10% of Italian ESEF filings contain only the tagged financial
statements, with the management report published separately as a PDF the
mandate does not reach. Recording that at fetch time is what keeps it a
visible number instead of a silent hole in the corpus.

Idempotent and resumable: a filing whose local .xhtml.gz already exists is
skipped, so re-running after widening the window only fetches what is new.

Usage:
    uv run python scripts/it/01_fetch_filings.py [--limit N] [--lei LEI]
"""

import argparse
import gzip
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))

import pipeline_logger
from xbrl_filings_client import download_report, iter_filings, probe_report

CHECKPOINT_EVERY = 25
#: Verbatim regulator label, kept next to the normalized `filing_type` so a
#: later remap of the normalization is a view change, never a re-fetch
#: (docs/international_expansion_plan.md).
FORM_TYPE = "Relazione finanziaria annuale"


def _manifest_row(**kwargs) -> dict:
    """The shared manifest shape. No `country_code` column: build_duckdb.py
    adds its own literal one per country in the UNION, and a manifest that
    also carried one collides under UNION BY NAME — the same trap Chile's
    manifest documents having hit."""
    row = {
        "document_id": "", "lei": "", "name": "", "source": "filings.xbrl.org",
        "form_type": FORM_TYPE, "filing_type": "annual", "period_end_date": "",
        # The aggregator records when IT ingested a filing, not when the
        # issuer published it. Kept under its own name rather than passed
        # off as `filing_date`, which this source simply does not have.
        "date_added": "", "report_url": "", "local_path": "", "format": "xhtml",
        "download_status": "pending", "n_bytes": 0, "n_words": 0,
        "has_narrative": False, "sha256": "", "package_sha256": "",
        "created_at": datetime.now(), "updated_at": datetime.now(),
    }
    row.update(kwargs)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Fetch only the first N pending filings (smoke-testing)")
    parser.add_argument("--lei", default=None, help="Only this issuer's filings")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / "configs" / "it" / "config.yaml").read_text())
    raw_dir = REPO_ROOT / config["storage"]["raw_documents"]
    manifest_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "filing_manifest.parquet"
    window = config["corpus"]["filings"]["period_end"]

    existing = (pd.read_parquet(manifest_path).set_index("document_id").to_dict("index")
                if manifest_path.exists() else {})

    pipeline_logger.log_event(
        pipeline_step="it_fetch_filings", level="INFO",
        message=f"Listing {config['source']['country_filter']} ESEF filings "
                f"in [{window['from']}, {window['to']}]...",
        log_dir=manifest_dir)

    targets = []
    for attributes, name, identifier in iter_filings(config["source"]["country_filter"]):
        period_end = attributes.get("period_end") or ""
        if not (window["from"] <= period_end <= window["to"]):
            continue
        lei = identifier or (attributes.get("fxo_id") or "").split("-")[0]
        if args.lei and lei != args.lei:
            continue
        targets.append((attributes, name, lei, period_end))

    rows = dict(existing)
    pending = []
    for attributes, name, lei, period_end in targets:
        document_id = attributes["fxo_id"]
        local_path = raw_dir / lei / f"{document_id}.xhtml.gz"
        # Idempotent on the FILE existing, not on the manifest agreeing —
        # the file is only written after a successful download, so it is
        # the source of truth even if the manifest was lost.
        if local_path.exists():
            continue
        pending.append((document_id, attributes, name, lei, period_end, local_path))

    if args.limit:
        pending = pending[:args.limit]

    pipeline_logger.log_event(
        pipeline_step="it_fetch_filings", level="INFO",
        message=f"{len(targets)} filings in window, {len(pending)} to download",
        log_dir=manifest_dir)

    def flush():
        pd.DataFrame(list(rows.values())).to_parquet(manifest_path, index=False)

    done = 0
    for document_id, attributes, name, lei, period_end, local_path in tqdm(pending):
        base = _manifest_row(
            document_id=document_id, lei=lei, name=name, period_end_date=period_end,
            date_added=attributes.get("date_added") or "",
            report_url=attributes.get("report_url") or "",
            package_sha256=attributes.get("sha256") or "")
        try:
            xhtml = download_report(attributes["report_url"])
            probe = probe_report(xhtml)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(local_path, "wb") as fh:
                fh.write(xhtml)
            rows[document_id] = {
                **base, "local_path": str(local_path), "download_status": "completed",
                # Our own hash of what we stored. `package_sha256` is the
                # aggregator's hash of the ZIP package, a different object;
                # conflating them would make an integrity check meaningless.
                "sha256": hashlib.sha256(xhtml).hexdigest(),
                "n_bytes": probe["n_bytes"], "n_words": probe["n_words"],
                "has_narrative": probe["has_narrative"],
                "updated_at": datetime.now(),
            }
            if not probe["has_narrative"]:
                pipeline_logger.log_event(
                    pipeline_step="it_fetch_filings", level="WARNING",
                    message=f"no management-report marker ({probe['n_words']:,} words) — "
                            f"financial statements only?",
                    ticker=document_id, log_dir=manifest_dir)
        except Exception as error:  # noqa: BLE001 — one bad filing must not end the run
            rows[document_id] = {**base, "download_status": f"failed: {error}",
                                 "updated_at": datetime.now()}
            pipeline_logger.log_event(
                pipeline_step="it_fetch_filings", level="ERROR",
                message=f"download failed: {error}", ticker=document_id, log_dir=manifest_dir)

        done += 1
        if done % CHECKPOINT_EVERY == 0:
            flush()

    flush()
    manifest = pd.DataFrame(list(rows.values()))
    ok = manifest[manifest.download_status == "completed"]
    pipeline_logger.log_event(
        pipeline_step="it_fetch_filings", level="SUCCESS",
        message=(f"{len(ok)}/{len(manifest)} downloaded, "
                 f"{int(ok.has_narrative.sum()) if len(ok) else 0} with a management report, "
                 f"{ok.n_bytes.sum() / 1e9 if len(ok) else 0:.1f} GB of XHTML -> {manifest_path}"),
        log_dir=manifest_dir)


if __name__ == "__main__":
    main()
