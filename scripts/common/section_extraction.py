"""
scripts/common/section_extraction.py — shared extraction-run logic used by
every country's 10-K/10-Q extraction scripts (currently scripts/us/10k/
02_extract_sections.py and scripts/us/10q/02_extract_sections.py). This
module is deliberately COUNTRY-AGNOSTIC: it holds the run/checkpoint/
traceability/multiprocessing plumbing only. The actual heading-parsing
logic (clean_html_to_lines, general_segment, known_issue) is sensitive to
local filing-format idiosyncrasies (SEC's "Item N" convention for the US;
a different country's filings would need a different segmenter entirely)
— so the caller imports ITS OWN country-specific segmenter and passes the
three functions in, rather than this module hardcoding a US-specific
import. Only TARGET_ITEMS, the manifest path, the output filename prefix,
and the segmenter functions differ per country/form; the checkpointing
and part-file design is one source of truth so extraction runs never
drift apart in behavior.
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/common/

import pipeline_logger

# Manifest is rewritten to disk every this-many completions instead of
# every single row (progress is still safe against a crash/kill, but
# doesn't pay a full parquet write per filing).
CHECKPOINT_EVERY = 100


def _empty_rows(target_items, acc_num, ticker, filing_date, note):
    """One found=False row per target item — used when the filing can't be
    parsed at all (missing file, exception), so every filing always has
    exactly len(target_items) rows in the trace, never zero."""
    return [{
        "accession_number": acc_num, "ticker": ticker, "filing_date": filing_date,
        "item_key": item_key, "section_name": sec_name, "found": False,
        "section_text": "", "char_len": 0, "word_count": 0, "note": note,
    } for item_key, sec_name in target_items.items()]


def _process_one(row, target_items, form, clean_html_to_lines, general_segment, known_issue):
    """Runs in a worker process: parse one filing, return a result dict.
    No shared state (df, logger file handle) is touched here.
    clean_html_to_lines/general_segment/known_issue are the CALLER's
    country-specific segmenter functions, passed through rather than
    imported here — see this module's docstring."""
    html_path = Path(row["local_path"])
    acc_num = row["accession_number"]
    ticker = row["ticker"]
    filing_date = row["filing_date"]
    idx = row["_idx"]

    if not html_path.exists():
        return {
            "idx": idx, "acc_num": acc_num, "ticker": ticker, "cik": row.get("cik"),
            "parse_status": "failed: raw file missing",
            "sections": _empty_rows(target_items, acc_num, ticker, filing_date, "raw file missing"),
            "n_found": 0, "error": None,
        }

    try:
        lines = clean_html_to_lines(html_path)
        all_segments = general_segment(lines, form=form, ticker=ticker)
        sections = []
        extracted_details = {}
        n_found = 0
        # One row per TARGET item, ALWAYS — found or not. A missing item is
        # then an explicit found=False row, not a silently absent one you'd
        # have to infer by checking what ISN'T in the table.
        for item_key, sec_name in target_items.items():
            sec_text = all_segments.get(item_key, "")
            found = bool(sec_text)
            char_len = len(sec_text)
            word_count = len(sec_text.split()) if sec_text else 0
            note = None
            if not found:
                # known_issue: scripts/us/known_segmenter_issues.yaml — a
                # documented, already-investigated limitation for this
                # ticker/form/item (see that file's header). Tagging it
                # here means a "found=False" row reads as "understood
                # gap" instead of a fresh mystery when someone audits the
                # trace later.
                issue = known_issue(ticker, form, item_key)
                note = f"not found by segmenter (known issue: {issue['category']})" if issue else "not found by segmenter"
            sections.append({
                "accession_number": acc_num,
                "ticker": ticker,
                "filing_date": filing_date,
                "item_key": item_key,
                "section_name": sec_name,
                "found": found,
                "section_text": sec_text,
                "char_len": char_len,
                "word_count": word_count,
                "note": note,
            })
            if found:
                n_found += 1
                extracted_details[sec_name] = {"char_len": char_len, "word_count": word_count}

        parse_status = "completed" if n_found > 0 else "failed: no sections extracted"
        return {
            "idx": idx, "acc_num": acc_num, "ticker": ticker, "cik": row.get("cik"),
            "parse_status": parse_status, "sections": sections, "n_found": n_found, "error": None,
            "details": extracted_details,
        }
    except Exception as e:
        return {
            "idx": idx, "acc_num": acc_num, "ticker": ticker, "cik": row.get("cik"),
            "parse_status": f"failed: {str(e)}",
            "sections": _empty_rows(target_items, acc_num, ticker, filing_date, f"exception: {e}"),
            "n_found": 0, "error": str(e),
        }


def safe_slug(text):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-") or "run"


def run_extraction(
    pipeline_step: str,
    manifest_path: Path,
    sections_dir: Path,
    target_items: dict[str, str],
    output_prefix: str,
    run_comment: str,
    clean_html_to_lines: Callable,
    general_segment: Callable,
    known_issue: Callable,
    form: str = "10-K",
) -> None:
    """Parses every (download_status==completed, parse_status==pending) row
    of `manifest_path` with the CALLER's country-specific segmenter
    (clean_html_to_lines/general_segment/known_issue — see this module's
    docstring for why they're passed in rather than imported here),
    extracting `target_items` only. Writes append-only, wildcard-globbable
    part files under `sections_dir` named "{output_prefix}__run={run_id}
    __part={part_num}__{comment}.parquet" — `output_prefix` is what keeps
    a 10-K and a 10-Q extraction run's outputs distinguishable when they
    land in the same directory (never pooled — see the project's
    data-scope decision). `form` ("10-K" or "10-Q") is passed through to
    general_segment — see section_segmenter.ITEM_ALIASES' docstring for
    why it matters there."""
    sections_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step=pipeline_step, level="ERROR",
            message=f"Manifest file not found at {manifest_path}. Run the fetch script first.",
        )
        return

    df = pd.read_parquet(manifest_path)
    pending_parse = df[(df["download_status"] == "completed") & (df["parse_status"] == "pending")]

    if len(pending_parse) == 0:
        pipeline_logger.log_event(
            pipeline_step=pipeline_step, level="INFO", message="No pending filings to parse.",
        )
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_date = datetime.now().isoformat()

    pipeline_logger.log_event(
        pipeline_step=pipeline_step, level="INFO",
        message=(
            f"Extracting {', '.join(target_items.values())} as Markdown from "
            f"{len(pending_parse)} filings (run_id={run_id}, comment={run_comment})..."
        ),
    )

    jobs = []
    for idx, row in pending_parse.iterrows():
        row_dict = row.to_dict()
        row_dict["_idx"] = idx
        jobs.append(row_dict)

    comment_slug = safe_slug(run_comment)
    buffer = []
    part_num = 0
    completed_since_checkpoint = 0

    def flush_buffer():
        nonlocal buffer, part_num
        if not buffer:
            return
        part_df = pd.DataFrame(buffer)
        part_path = sections_dir / f"{output_prefix}__run={run_id}__part={part_num:04d}__{comment_slug}.parquet"
        part_df["run_id"] = run_id
        part_df["run_date"] = run_date
        part_df["run_comment"] = run_comment
        part_df["part_num"] = part_num
        part_df["part_file"] = part_path.name
        part_df.to_parquet(part_path, index=False)
        n_found_in_part = int(part_df["found"].sum())
        pipeline_logger.log_event(
            pipeline_step=pipeline_step, level="INFO",
            message=(
                f"Flushed part {part_num:04d} ({part_path.name}): "
                f"{len(part_df)} rows, {n_found_in_part} sections found"
            ),
        )
        buffer = []
        part_num += 1

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(_process_one, job, target_items, form, clean_html_to_lines, general_segment, known_issue)
            for job in jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            idx = result["idx"]

            df.at[idx, "parse_status"] = result["parse_status"]
            df.at[idx, "updated_at"] = datetime.now()
            buffer.extend(result["sections"])

            if result["parse_status"] == "completed":
                found_items = sorted(result.get("details", {}).keys())
                pipeline_logger.log_event(
                    pipeline_step=pipeline_step, level="SUCCESS",
                    message=(
                        f"Extracted {result['n_found']}/{len(target_items)} target items "
                        f"({', '.join(found_items) or 'none'})"
                    ),
                    ticker=result["ticker"], cik=result["cik"], accession_number=result["acc_num"],
                    details=result.get("details", {}),
                )
            elif result["error"] is not None:
                pipeline_logger.log_event(
                    pipeline_step=pipeline_step, level="ERROR",
                    message=f"Exception extracting sections: {result['error']}",
                    ticker=result["ticker"], cik=result["cik"], accession_number=result["acc_num"],
                    details={"error": result["error"]},
                )
            else:
                pipeline_logger.log_event(
                    pipeline_step=pipeline_step, level="WARNING",
                    message=f"{result['parse_status']} for filing",
                    ticker=result["ticker"], cik=result["cik"], accession_number=result["acc_num"],
                )

            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                # Sections FIRST, then manifest — a row is only ever marked
                # "completed" once its text is actually on disk.
                flush_buffer()
                df.to_parquet(manifest_path, index=False)
                completed_since_checkpoint = 0

    flush_buffer()
    df.to_parquet(manifest_path, index=False)

    pipeline_logger.log_event(
        pipeline_step=pipeline_step, level="SUCCESS",
        message=(
            f"Finished section extraction run_id={run_id} ({run_comment}). "
            f"Parts written to {sections_dir}/{output_prefix}__run={run_id}__part=*.parquet — "
            f"read the whole corpus with a wildcard glob over "
            f"{sections_dir}/{output_prefix}__run=*__part=*.parquet"
        ),
    )
