"""
scripts/it/02_extract_sections.py — paragraphs out of every downloaded
Italian ESEF annual report.

WHOLE DOCUMENT, no item-level narrowing — the same call
scripts/cl/02_extract_text.py makes and for the same reason. A 10-K has a
rigid Item 1/1A/7/8 structure where Item 8 is pure financials, so
scripts/us/ narrows. A Relazione finanziaria annuale IS the disclosure
document; there is no equivalent split to make, and these files carry no
`<h1>`, no anchors and no table of contents to make it with (see
scripts/common/pdf/backends/html_typography.py for what they carry
instead).

The backend is `html_typography`, not markdownify. That is the whole point
of this step: these XHTML files are rendered PDFs — half of them literally
pdf2htmlEX output — whose structure survives only as font sizes in the
stylesheet. Converting them to markdown recovers ZERO headings; resolving
the CSS recovers the document's own section titles ("Relazione sulla
gestione", "BILANCIO CONSOLIDATO", "Cariche sociali").

Emits the same paragraph contract every other country does
(content_type / paragraph_index / page / paragraph_text — see
build_duckdb.py), so Italy needs no new downstream code, plus `block_type`
and the typography that produced it for auditing.

Usage:
    uv run python scripts/it/02_extract_sections.py [--limit N] [--comment "..."]
"""

import argparse
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))

import pipeline_logger
from section_extraction import safe_slug

from scripts.common.pdf import blocks as B
from scripts.common.pdf.backends.html_typography import HtmlTypographyBackend

CHECKPOINT_EVERY = 25
_BACKEND: HtmlTypographyBackend | None = None


def _init_worker() -> None:
    global _BACKEND
    _BACKEND = HtmlTypographyBackend()


def _process_one(row: dict) -> dict:
    document_id, local_path, idx = row["document_id"], row["local_path"], row["_idx"]
    if not local_path or not Path(local_path).exists():
        return {"idx": idx, "document_id": document_id, "paragraphs": [],
                "parse_status": "failed: missing local file", "error": None}
    try:
        with gzip.open(local_path, "rt", encoding="utf-8", errors="ignore") as fh:
            html = fh.read()
        paragraphs = B.blocks_to_paragraphs(_BACKEND.parse_html(html))
        for paragraph in paragraphs:
            paragraph["document_id"] = document_id
            paragraph["lei"] = row["lei"]
            paragraph["name"] = row["name"]
            paragraph["form_type"] = row["form_type"]
            paragraph["filing_type"] = row["filing_type"]
            paragraph["period_end_date"] = row["period_end_date"]
        status = "completed" if paragraphs else "failed: no paragraphs extracted"
        return {"idx": idx, "document_id": document_id, "paragraphs": paragraphs,
                "parse_status": status, "error": None}
    except Exception as error:  # noqa: BLE001 — one bad filing must not end the run
        return {"idx": idx, "document_id": document_id, "paragraphs": [],
                "parse_status": f"failed: {error}", "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--comment", default="run")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / "configs" / "it" / "config.yaml").read_text())
    manifest_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    sections_dir = REPO_ROOT / config["storage"]["interim_sections"]
    sections_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "filing_manifest.parquet"

    if not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step="it_extract_sections", level="ERROR",
            message=f"No manifest at {manifest_path}; run scripts/it/01_fetch_filings.py first",
            log_dir=manifest_dir)
        return

    # What is already extracted is derived from the OUTPUT PARTS, not from
    # a parse_status column in filing_manifest.parquet.
    #
    # This matters because 01_fetch_filings.py owns that file and rewrites
    # it wholesale every 25 downloads. Running extraction alongside a live
    # fetch — which is the point, so filings get processed as they land —
    # would have the two processes clobbering each other's writes: the
    # fetcher would drop every parse_status, the extractor would drop every
    # filing downloaded since it last read. Deriving state from the parts
    # already on disk is the same "the file is the source of truth"
    # contract the fetcher itself uses for downloads, and it needs no lock.
    # A per-run LEDGER of every filing attempted, not just the ones that
    # produced rows. Deriving `already` from the paragraph parts alone
    # looked equivalent and is not: a filing that yields zero paragraphs
    # never appears in them, so a loop re-parsing whatever is pending would
    # retry those same failures on every pass, forever.
    df = pd.read_parquet(manifest_path)
    already = set()
    for ledger in sections_dir.glob("extracted_documents__run=*.parquet"):
        try:
            already.update(pd.read_parquet(ledger, columns=["document_id"]).document_id)
        except Exception:  # noqa: BLE001 — a ledger still being written is not fatal
            continue
    for part in sections_dir.glob("filing_paragraphs__run=*__part=*.parquet"):
        # Parts written before the ledger existed still count as done.
        try:
            already.update(pd.read_parquet(part, columns=["document_id"]).document_id.unique())
        except Exception:  # noqa: BLE001
            continue
    pending = df[(df.download_status == "completed") & (~df.document_id.isin(already))]
    if args.limit:
        pending = pending.head(args.limit)
    if pending.empty:
        pipeline_logger.log_event(
            pipeline_step="it_extract_sections", level="INFO",
            message=f"Nothing pending ({len(already):,} filings already extracted).",
            log_dir=manifest_dir)
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    comment_slug = safe_slug(args.comment)
    # Parsing a 68 MB document peaks near 1 GB of RSS, so the pool is sized
    # against MEMORY, not core count — the default os.cpu_count() workers
    # would try to hold ~24 of those at once.
    workers = args.workers or max(1, min(6, os.cpu_count() or 1))
    pipeline_logger.log_event(
        pipeline_step="it_extract_sections", level="INFO",
        message=f"Extracting {len(pending)} filings (run_id={run_id}, workers={workers})",
        log_dir=manifest_dir)

    jobs = [{**row.to_dict(), "_idx": idx} for idx, row in pending.iterrows()]
    buffer, part_num, since_checkpoint = [], 0, 0
    ledger: list[dict] = []
    ledger_path = sections_dir / f"extracted_documents__run={run_id}.parquet"

    def flush_ledger():
        if ledger:
            pd.DataFrame(ledger).to_parquet(ledger_path, index=False)

    def flush():
        nonlocal buffer, part_num
        if not buffer:
            return
        part = pd.DataFrame(buffer)
        path = sections_dir / (f"filing_paragraphs__run={run_id}__part={part_num:04d}"
                               f"__{comment_slug}.parquet")
        part["run_id"], part["run_comment"] = run_id, args.comment
        part.to_parquet(path, index=False)
        pipeline_logger.log_event(
            pipeline_step="it_extract_sections", level="INFO",
            message=f"Flushed {path.name}: {len(part):,} paragraph rows", log_dir=manifest_dir)
        buffer, part_num = [], part_num + 1

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            buffer.extend(result["paragraphs"])
            ledger.append({"document_id": result["document_id"],
                           "parse_status": result["parse_status"],
                           "n_paragraphs": len(result["paragraphs"]),
                           "run_id": run_id, "extracted_at": datetime.now()})
            if result["error"]:
                pipeline_logger.log_event(
                    pipeline_step="it_extract_sections", level="ERROR",
                    message=result["parse_status"], ticker=result["document_id"],
                    log_dir=manifest_dir)
            since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                # Paragraphs first, then the ledger — a document is only
                # recorded as done once its text is actually on disk.
                flush()
                flush_ledger()
                since_checkpoint = 0

    flush()
    flush_ledger()
    pipeline_logger.log_event(
        pipeline_step="it_extract_sections", level="SUCCESS",
        message=f"Finished run_id={run_id}; {len(already) + len(jobs):,} filings extracted "
                f"in total; parts in {sections_dir}", log_dir=manifest_dir)


if __name__ == "__main__":
    main()
