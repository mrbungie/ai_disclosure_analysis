"""
scripts/verif/section_audit/section_audit.py — verification, not pipeline.

Of the filings where scripts/us/10k/02_extract_sections.py found no text
for one of its 3 target items (Item 1 Business, Item 1A Risk Factors,
Item 7 MD&A), how many are (a) genuinely absent / incorporated by
reference / structurally exceptional vs (b) present in the filing but
missed by extraction — a real bug? Classified by running the SAME general
item segmenter the pipeline itself uses (section_segmenter.py) and
checking whether it found the item that 02_extract_sections.py missed.

It recognizes any "Item N[A-C]" heading (Item 1 through Item 16), not
just the 3 target items, and tells a real heading from a table-of-
contents row structurally (a TOC row is a markdown hyperlink; a real
heading is the link's plain-text target) rather than by a length guess.
This is a heuristic, not a guarantee; a handful of segments may still be
misattributed on filings with unusual TOC/cross-reference formatting.

Output (gitignored, cheap to regenerate):
    data/interim/audits/section_audit/missing_section_audit.parquet
    data/interim/audits/section_audit/summary.json

Usage:
    uv run python scripts/verif/section_audit/section_audit.py
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "audits" / "section_audit"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "us"))
from section_segmenter import clean_html_to_lines, general_segment, load_filing_sections  # noqa: E402

# The 3 items scripts/us/10k/02_extract_sections.py actually extracts today.
TARGET_ITEMS = ["1", "1A", "7"]


def process_one(row: dict) -> dict:
    """Runs in a worker process. Returns per-filing missing-section
    classifications for the 3 target items not found by
    02_extract_sections.py."""
    html_path = Path(row["local_path"])
    acc_num = row["accession_number"]
    ticker = row["ticker"]
    extractor_found = set(row["extractor_found_items"])

    if not html_path.exists():
        return {"accession_number": acc_num, "ticker": ticker, "error": "raw file missing"}

    try:
        lines = clean_html_to_lines(html_path)
        segments = general_segment(lines)

        missing_rows = []
        for item in TARGET_ITEMS:
            if item in extractor_found:
                continue  # 02_extract_sections.py already got this one
            cls = "found_by_general_segmenter" if item in segments else "not_found_by_either"
            missing_rows.append({
                "accession_number": acc_num, "ticker": ticker, "item": item,
                "classification": cls,
                "general_segment_char_len": len(segments.get(item, "")),
            })

        return {
            "accession_number": acc_num, "ticker": ticker, "error": None,
            "missing_rows": missing_rows,
        }
    except Exception as e:
        return {"accession_number": acc_num, "ticker": ticker, "error": str(e)}


def main():
    with open(REPO_ROOT / "configs" / "us" / "config.yaml") as f:
        config = yaml.safe_load(f)

    manifest = pd.read_parquet(
        Path(config["storage"]["interim_manifests"]) / "filing_manifest.parquet"
    )
    manifest = manifest[manifest["parse_status"] == "completed"].copy()

    sections = load_filing_sections(config)
    found_by_extractor = (
        sections.groupby("accession_number")["section_name"]
        .apply(lambda s: set(x.replace("Item ", "") for x in s))
        .to_dict()
    )
    manifest["extractor_found_items"] = manifest["accession_number"].map(
        lambda acc: list(found_by_extractor.get(acc, set()))
    )

    jobs = manifest.to_dict("records")
    print(f"Auditing {len(jobs)} filings ({os.cpu_count()} workers)...")

    missing_rows_all = []
    errors = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result.get("error"):
                errors.append(result)
                continue
            missing_rows_all.extend(result["missing_rows"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    missing_df = pd.DataFrame(missing_rows_all)
    missing_df["run_id"] = run_id
    missing_df.to_parquet(OUTPUT_DIR / "missing_section_audit.parquet", index=False)

    summary = {"run_id": run_id, "run_date": datetime.now().isoformat(), "n_filings": len(jobs), "n_errors": len(errors)}
    if not missing_df.empty:
        summary["missing_section_classification"] = (
            missing_df.groupby(["item", "classification"]).size().unstack(fill_value=0).to_dict("index")
        )
    else:
        summary["missing_section_classification"] = {}

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
