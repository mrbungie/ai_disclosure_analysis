"""
scripts/it/01b_fetch_xbrl_facts.py — the iXBRL facts that ship inside each
Italian annual report, fetched as the aggregator's already-extracted
xBRL-JSON.

WHY THIS IS A SEPARATE SCRIPT AND NOT PART OF 01. Italy is the only country
in this project where the accounting data and the disclosure narrative
arrive in ONE document: an ESEF report is XHTML with inline XBRL, so the
IFRS-tagged facts are physically inside the file 01_fetch_filings.py
already mirrors. The US needs a separate XBRL fetch (data/raw/xbrl_facts),
Chile another (data/raw/xbrl_cl) — Italy would need none if the facts were
read straight out of the XHTML.

They are fetched separately anyway, for two reasons. Parsing inline XBRL
out of a 25-140 MB XHTML means resolving contexts, units, scaling and sign
flips, which is exactly the work the aggregator has already done and
published as xBRL-JSON (OIM). And keeping it a second, idempotent pass
means the narrative download — the long one — never has to be restarted to
add it.

WHAT IS IN THERE. Measured on real Italian filings: 400-600 facts each,
180-290 distinct concepts, 330-450 of them numeric, on the standard IFRS
taxonomy (ifrs-full:Equity, ProfitLoss, ComprehensiveIncome, Cash,
DividendsPaid). No TextBlock facts in any file sampled — ESEF block tagging
of the notes does not show up here, so this does NOT help segment the
narrative; it is the accounting side only.

Raw JSON is stored, not a flattened table. Same fetch/parse split as
everywhere else in this project: turning facts into a tidy
(document_id, concept, period, value) parquet is a later step that must be
re-runnable against untouched raw data.

Usage:
    uv run python scripts/it/01b_fetch_xbrl_facts.py [--limit N]
"""

import argparse
import gzip
import json
import re
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
from xbrl_filings_client import BASE, USER_AGENT, iter_filings

CHECKPOINT_EVERY = 50
_NUMERIC_RE = re.compile(r"^-?[\d.]+(?:[eE]-?\d+)?$")


def _fetch(url: str) -> bytes:
    import time
    import urllib.request

    from xbrl_filings_client import REQUEST_DELAY

    time.sleep(REQUEST_DELAY)
    request = urllib.request.Request(BASE + url if url.startswith("/") else url,
                                     headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def _summarize(payload: bytes) -> dict:
    """Counts computed from bytes already in memory, so "which filings came
    back with no usable accounting data" is answerable without re-reading
    2.6 GB of JSON. Not a parse — no concept is interpreted here."""
    facts = json.loads(payload).get("facts", {}) or {}
    concepts, numeric = set(), 0
    for fact in facts.values():
        concept = (fact.get("dimensions") or {}).get("concept", "")
        if concept:
            concepts.add(concept)
        value = fact.get("value")
        if isinstance(value, (int, float)) or (isinstance(value, str) and _NUMERIC_RE.match(value)):
            numeric += 1
    return {"n_facts": len(facts), "n_concepts": len(concepts), "n_numeric_facts": numeric}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / "configs" / "it" / "config.yaml").read_text())
    xbrl_dir = REPO_ROOT / config["storage"]["raw_xbrl"]
    manifest_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "xbrl_facts_manifest.parquet"
    window = config["corpus"]["filings"]["period_end"]

    # Append-only, same reason as the filing manifest (see manifest_store):
    # a whole-file rewrite loses any concurrent writer's rows and leaves
    # "re-run with a change" no option but deleting what is there. What is
    # already done is read off the FILES, not this manifest, so the manifest
    # is never consulted to decide work.
    written: list[dict] = []

    pending = []
    for attributes, _name, identifier in iter_filings(config["source"]["country_filter"]):
        period_end = attributes.get("period_end") or ""
        if not (window["from"] <= period_end <= window["to"]):
            continue
        if not attributes.get("json_url"):
            continue
        lei = identifier or (attributes.get("fxo_id") or "").split("-")[0]
        local_path = xbrl_dir / lei / f"{attributes['fxo_id']}.facts.json.gz"
        if local_path.exists():
            continue
        pending.append((attributes, lei, period_end, local_path))
    if args.limit:
        pending = pending[:args.limit]

    pipeline_logger.log_event(
        pipeline_step="it_fetch_xbrl", level="INFO",
        message=f"{len(pending)} filings without local xBRL-JSON facts", log_dir=manifest_dir)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    part_num = 0

    def flush():
        nonlocal written, part_num
        if written:
            pd.DataFrame(written).to_parquet(
                manifest_dir / f"xbrl_facts_manifest__run={run_id}__part={part_num:04d}.parquet",
                index=False)
            part_num += 1
            written = []
        # Derived snapshot, rebuildable from the parts at any time.
        frames = [pd.read_parquet(p) for p in
                  sorted(manifest_dir.glob("xbrl_facts_manifest__run=*__part=*.parquet"))]
        if manifest_path.exists() and not frames:
            return
        if frames:
            snap = pd.concat(frames, ignore_index=True)
            snap.drop_duplicates(subset="document_id", keep="last").to_parquet(
                manifest_path, index=False)

    for i, (attributes, lei, period_end, local_path) in enumerate(tqdm(pending), 1):
        document_id = attributes["fxo_id"]
        row = {"document_id": document_id, "lei": lei, "period_end_date": period_end,
               "local_path": "", "format": "xbrl_json", "download_status": "pending",
               "n_bytes": 0, "n_facts": 0, "n_concepts": 0, "n_numeric_facts": 0,
               "updated_at": datetime.now()}
        try:
            payload = _fetch(attributes["json_url"])
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(local_path, "wb") as fh:
                fh.write(payload)
            row.update(local_path=str(local_path), download_status="completed",
                       n_bytes=len(payload), **_summarize(payload))
        except Exception as error:  # noqa: BLE001 — one bad filing must not end the run
            row["download_status"] = f"failed: {error}"
            pipeline_logger.log_event(
                pipeline_step="it_fetch_xbrl", level="ERROR",
                message=f"facts download failed: {error}", ticker=document_id, log_dir=manifest_dir)
        written.append(row)
        if i % CHECKPOINT_EVERY == 0:
            flush()

    flush()
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else pd.DataFrame()
    ok = manifest[manifest.download_status == "completed"] if len(manifest) else manifest
    pipeline_logger.log_event(
        pipeline_step="it_fetch_xbrl", level="SUCCESS",
        message=(f"{len(ok)}/{len(manifest)} fact sets, "
                 f"{ok.n_facts.sum() if len(ok) else 0:,} facts, "
                 f"{ok.n_bytes.sum() / 1e9 if len(ok) else 0:.2f} GB -> {manifest_path}"),
        log_dir=manifest_dir)


if __name__ == "__main__":
    main()
