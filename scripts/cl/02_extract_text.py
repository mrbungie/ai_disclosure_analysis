"""
scripts/cl/02_extract_text.py — extracts paragraphs from every completed CMF
filing (Memoria Anual / Análisis Razonado) via scripts/cl/cmf_pdf_paragraphs.py.

Whole-document unit, not an Item like scripts/us/*/02_extract_sections.py —
the Memoria/Análisis Razonado are ALREADY the narrower disclosure documents
this project analyzes (see 01_fetch_filings.py's docstring: Estados
Financieros, Declaración de responsabilidad and Hechos Relevantes are
skipped at fetch time), so there's no further Item-level narrowing to do
here — every paragraph PyMuPDF's block segmentation finds is kept.

Different manifest shape than scripts/common/section_extraction.py's
run_extraction() expects (document_id/rut/nemo, not accession_number/
ticker/cik; no item_key or fixed target_items — one manifest row in, a
variable number of paragraph rows out) — so this is its own checkpoint/
part-file runner rather than a caller of that shared module. Same
conventions though (append-only globbable parts, safe_slug, a
parse_status column only ever set to "completed" once the paragraphs are
actually on disk), reused via import rather than duplicated.

Multi-part documents (large Memorias split across N pdf.gz files — see
01_fetch_filings.py's _save_pdf_parts): each part is decompressed and run
through extract_paragraphs SEPARATELY (a part is its own standalone PDF,
own page numbering from 0), then the parts' paragraph lists are
concatenated in part order and paragraph_index renumbered densely across
the whole document; `source_part` is kept alongside `page` so a page
number can still be traced back to which physical PDF part produced it.

Usage:
    uv run python scripts/cl/02_extract_text.py [--limit N] [--comment "..."]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/cl/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # scripts/common/

import pipeline_logger
from section_extraction import safe_slug
from cmf_pdf_paragraphs import extract_paragraphs

CHECKPOINT_EVERY = 50


def _process_one(row: dict) -> dict:
    """Runs in a worker process: extract one document's paragraphs, return
    a result dict. No shared state (df, logger file handle) touched here —
    same isolation contract as section_extraction.py's _process_one."""
    document_id = row["document_id"]
    local_path = row["local_path"]
    idx = row["_idx"]

    if not local_path:
        return {"idx": idx, "document_id": document_id, "parse_status": "failed: no local_path",
                "paragraphs": [], "error": None}

    part_paths = [p for p in local_path.split(";") if p]
    missing = [p for p in part_paths if not Path(p).exists()]
    if missing:
        return {"idx": idx, "document_id": document_id,
                "parse_status": f"failed: missing file(s) {missing}", "paragraphs": [], "error": None}

    try:
        all_paragraphs = []
        for part_num, path in enumerate(part_paths):
            with gzip.open(path, "rb") as f:
                pdf_bytes = f.read()
            for para in extract_paragraphs(pdf_bytes):
                para["source_part"] = part_num
                all_paragraphs.append(para)

        for new_idx, para in enumerate(all_paragraphs):
            para["paragraph_index"] = new_idx
            para["document_id"] = document_id
            para["rut"] = row["rut"]
            para["nemo"] = row["nemo"]
            para["form_type"] = row["form_type"]
            para["filing_type"] = row["filing_type"]
            para["period_end_date"] = row["period_end_date"]

        parse_status = "completed" if all_paragraphs else "failed: no paragraphs extracted"
        return {"idx": idx, "document_id": document_id, "parse_status": parse_status,
                "paragraphs": all_paragraphs, "error": None}
    except Exception as e:
        return {"idx": idx, "document_id": document_id, "parse_status": f"failed: {e}",
                "paragraphs": [], "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only the first N pending documents (smoke-testing)")
    parser.add_argument("--comment", default="run", help="Short label for this run, embedded in output filenames/columns.")
    args = parser.parse_args()

    with open("configs/cl/config.yaml") as f:
        config = yaml.safe_load(f)

    manifest_dir = Path(config["storage"]["interim_manifests"])
    manifest_path = manifest_dir / "filing_manifest.parquet"
    sections_dir = Path(config["storage"]["interim_sections"])
    sections_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step="cl_extract_text", level="ERROR",
            message=f"Manifest not found at {manifest_path}. Run scripts/cl/01_fetch_filings.py first.",
            log_dir=manifest_dir,
        )
        return

    df = pd.read_parquet(manifest_path)
    if "parse_status" not in df.columns:
        df["parse_status"] = "pending"
    df["parse_status"] = df["parse_status"].fillna("pending")

    pending = df[(df["download_status"] == "completed") & (df["parse_status"] == "pending")]
    if args.limit:
        pending = pending.head(args.limit)

    if len(pending) == 0:
        pipeline_logger.log_event(
            pipeline_step="cl_extract_text", level="INFO", message="No pending documents to extract.",
            log_dir=manifest_dir,
        )
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_date = datetime.now().isoformat()
    comment_slug = safe_slug(args.comment)

    pipeline_logger.log_event(
        pipeline_step="cl_extract_text", level="INFO",
        message=f"Extracting paragraphs from {len(pending)} documents (run_id={run_id}, comment={args.comment})...",
        log_dir=manifest_dir,
    )

    jobs = [{**row.to_dict(), "_idx": idx} for idx, row in pending.iterrows()]

    buffer = []
    part_num = 0
    completed_since_checkpoint = 0

    def flush_buffer():
        nonlocal buffer, part_num
        if not buffer:
            return
        part_df = pd.DataFrame(buffer)
        part_path = sections_dir / f"filing_paragraphs__run={run_id}__part={part_num:04d}__{comment_slug}.parquet"
        part_df["run_id"] = run_id
        part_df["run_date"] = run_date
        part_df["run_comment"] = args.comment
        part_df["part_num"] = part_num
        part_df["part_file"] = part_path.name
        part_df.to_parquet(part_path, index=False)
        pipeline_logger.log_event(
            pipeline_step="cl_extract_text", level="INFO",
            message=f"Flushed part {part_num:04d} ({part_path.name}): {len(part_df)} paragraph rows",
            log_dir=manifest_dir,
        )
        buffer = []
        part_num += 1

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            idx = result["idx"]

            df.at[idx, "parse_status"] = result["parse_status"]
            df.at[idx, "updated_at"] = datetime.now()
            buffer.extend(result["paragraphs"])

            if result["parse_status"] == "completed":
                pipeline_logger.log_event(
                    pipeline_step="cl_extract_text", level="SUCCESS",
                    message=f"Extracted {len(result['paragraphs'])} paragraphs",
                    ticker=result["document_id"], log_dir=manifest_dir,
                )
            elif result["error"] is not None:
                pipeline_logger.log_event(
                    pipeline_step="cl_extract_text", level="ERROR",
                    message=f"Exception extracting: {result['error']}",
                    ticker=result["document_id"], log_dir=manifest_dir,
                )
            else:
                pipeline_logger.log_event(
                    pipeline_step="cl_extract_text", level="WARNING",
                    message=result["parse_status"], ticker=result["document_id"], log_dir=manifest_dir,
                )

            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                # Paragraphs FIRST, then manifest — a row is only ever
                # marked "completed" once its text is actually on disk.
                flush_buffer()
                df.to_parquet(manifest_path, index=False)
                completed_since_checkpoint = 0

    flush_buffer()
    df.to_parquet(manifest_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="cl_extract_text", level="SUCCESS",
        message=(
            f"Finished paragraph extraction run_id={run_id} ({args.comment}). "
            f"Parts written to {sections_dir}/filing_paragraphs__run={run_id}__part=*.parquet — "
            f"read the whole corpus with a wildcard glob over "
            f"{sections_dir}/filing_paragraphs__run=*__part=*.parquet"
        ),
        log_dir=manifest_dir,
    )


if __name__ == "__main__":
    main()
