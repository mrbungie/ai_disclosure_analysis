import argparse
import os
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for common/

from section_segmenter import clean_html_to_lines, general_segment  # same dir, no path fix needed

try:
    import pipeline_logger
except ImportError:
    from common import pipeline_logger

# Manifest is rewritten to disk every this-many completions instead of
# every single row (progress is still safe against a crash/kill, but
# doesn't pay a full parquet write per filing).
CHECKPOINT_EVERY = 100

# The 3 items this pipeline stage narrows the corpus to (see
# verif_scripts/section_audit for why: ~88% of AI/ML mentions in the full
# 10-K fall inside these 3; the rest is deliberately left out, not missed).
# Segmentation itself is general (any Item N) — see section_segmenter.py.
TARGET_ITEMS = {"1": "Item 1", "1A": "Item 1A", "7": "Item 7"}


def process_one(row):
    """Runs in a worker process: parse one filing, return a result dict.
    No shared state (df, logger file handle) is touched here."""
    html_path = Path(row["local_path"])
    acc_num = row["accession_number"]
    ticker = row["ticker"]
    filing_date = row["filing_date"]
    idx = row["_idx"]

    if not html_path.exists():
        return {
            "idx": idx, "acc_num": acc_num, "ticker": ticker, "cik": row.get("cik"),
            "parse_status": "failed: raw file missing", "sections": [], "error": None,
        }

    try:
        lines = clean_html_to_lines(html_path)
        all_segments = general_segment(lines)
        sections = []
        extracted_details = {}
        for item_key, sec_name in TARGET_ITEMS.items():
            sec_text = all_segments.get(item_key, "")
            if sec_text:
                char_len = len(sec_text)
                word_count = len(sec_text.split())
                sections.append({
                    "accession_number": acc_num,
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "section_name": sec_name,
                    "section_text": sec_text,
                    "char_len": char_len,
                    "word_count": word_count
                })
                extracted_details[sec_name] = {"char_len": char_len, "word_count": word_count}

        parse_status = "completed" if sections else "failed: no sections extracted"
        return {
            "idx": idx, "acc_num": acc_num, "ticker": ticker, "cik": row.get("cik"),
            "parse_status": parse_status, "sections": sections, "error": None,
            "details": extracted_details,
        }
    except Exception as e:
        return {
            "idx": idx, "acc_num": acc_num, "ticker": ticker, "cik": row.get("cik"),
            "parse_status": f"failed: {str(e)}", "sections": [], "error": str(e),
        }


def safe_slug(text):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-") or "run"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="run",
                         help="Short label for this run, embedded in the output filenames/columns.")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_date = datetime.now().isoformat()
    run_comment = args.comment

    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    sections_dir = Path(config["paths"]["interim_sections"])
    sections_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step="extract_sections",
            level="ERROR",
            message=f"Manifest file not found at {manifest_path}. Run download script first."
        )
        return

    df = pd.read_parquet(manifest_path)

    pending_parse = df[(df["download_status"] == "completed") & (df["parse_status"] == "pending")]

    if len(pending_parse) == 0:
        pipeline_logger.log_event(
            pipeline_step="extract_sections",
            level="INFO",
            message="No pending filings to parse."
        )
        return

    pipeline_logger.log_event(
        pipeline_step="extract_sections",
        level="INFO",
        message=(
            f"Extracting narrative sections as Markdown from {len(pending_parse)} filings "
            f"(run_id={run_id}, comment={run_comment})..."
        )
    )

    # One row dict per pending filing, tagged with its original df index so
    # results can be written back without relying on ordering.
    jobs = []
    for idx, row in pending_parse.iterrows():
        row_dict = row.to_dict()
        row_dict["_idx"] = idx
        jobs.append(row_dict)

    # Sections are flushed to a NEW part-file every CHECKPOINT_EVERY completions
    # (append-only, glob-readable: data/interim/sections/filing_sections__run=*.parquet).
    # A kill/crash mid-run only loses the current unflushed buffer, never
    # previously-flushed parts, and never silently drops rows that the manifest
    # already marked "completed" (that's what caused the prior data loss).
    comment_slug = safe_slug(run_comment)
    buffer = []
    part_num = 0
    completed_since_checkpoint = 0

    def flush_buffer():
        nonlocal buffer, part_num
        if not buffer:
            return
        part_df = pd.DataFrame(buffer)
        part_df["run_id"] = run_id
        part_df["run_date"] = run_date
        part_df["run_comment"] = run_comment
        part_path = sections_dir / f"filing_sections__run={run_id}__part={part_num:04d}__{comment_slug}.parquet"
        part_df.to_parquet(part_path, index=False)
        pipeline_logger.log_event(
            pipeline_step="extract_sections",
            level="INFO",
            message=f"Flushed {len(part_df)} sections to {part_path.name}"
        )
        buffer = []
        part_num += 1

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            idx = result["idx"]

            df.at[idx, "parse_status"] = result["parse_status"]
            df.at[idx, "updated_at"] = datetime.now()
            buffer.extend(result["sections"])

            if result["parse_status"] == "completed":
                pipeline_logger.log_event(
                    pipeline_step="extract_sections",
                    level="SUCCESS",
                    message=f"Successfully extracted {len(result['sections'])} sections",
                    ticker=result["ticker"],
                    cik=result["cik"],
                    accession_number=result["acc_num"],
                    details=result.get("details", {})
                )
            elif result["error"] is not None:
                pipeline_logger.log_event(
                    pipeline_step="extract_sections",
                    level="ERROR",
                    message=f"Exception extracting sections: {result['error']}",
                    ticker=result["ticker"],
                    cik=result["cik"],
                    accession_number=result["acc_num"],
                    details={"error": result["error"]}
                )
            else:
                pipeline_logger.log_event(
                    pipeline_step="extract_sections",
                    level="WARNING",
                    message=f"{result['parse_status']} for filing",
                    ticker=result["ticker"],
                    cik=result["cik"],
                    accession_number=result["acc_num"]
                )

            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                # Flush sections FIRST, then the manifest — so a row is only ever
                # marked "completed" in the manifest once its text is actually
                # on disk, never the other way around.
                flush_buffer()
                df.to_parquet(manifest_path, index=False)
                completed_since_checkpoint = 0

    # Final flush of whatever's left, then final manifest save.
    flush_buffer()
    df.to_parquet(manifest_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="extract_sections",
        level="SUCCESS",
        message=(
            f"Finished section extraction run_id={run_id} ({run_comment}). "
            f"Parts written to {sections_dir}/filing_sections__run={run_id}__part=*.parquet — "
            f"read the whole corpus with a wildcard glob over "
            f"{sections_dir}/filing_sections__run=*__part=*.parquet"
        )
    )

if __name__ == "__main__":
    main()
