"""
scripts/us/6k/02_extract_sections.py — extracts the full text of every
completed 6-K in filing_manifest_6k.parquet, as a single item_key="0"
"section" (see segmenter_6k.py for why there's no real per-Item
segmentation). Same shared runner as 8-K/10-K/10-Q/proxy
(scripts/common/section_extraction.py).

Usage:
    uv run python scripts/us/6k/02_extract_sections.py
"""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))  # scripts/common/
from section_extraction import run_extraction

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/us/6k/
from segmenter_6k import TARGET_ITEMS, clean_html_to_lines, general_segment, known_issue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="run",
                         help="Short label for this run, embedded in the output filenames/columns.")
    args = parser.parse_args()

    with open(Path("configs/us/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    run_extraction(
        pipeline_step="extract_sections_6k",
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_6k.parquet",
        sections_dir=Path(config["storage"]["interim_sections"]),
        target_items=TARGET_ITEMS,
        output_prefix="filing_sections_6k",
        run_comment=args.comment,
        form="6-K",
        clean_html_to_lines=clean_html_to_lines,
        general_segment=general_segment,
        known_issue=known_issue,
    )


if __name__ == "__main__":
    main()
