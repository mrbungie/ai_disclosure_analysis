"""
scripts/us/20f/02_extract_sections.py — extracts the full text of every
completed 20-F in filing_manifest_20f.parquet, as a single item_key="0"
"section" (see 20f_segmenter.py for why there's no real per-heading
segmentation yet). Same shared runner as 10-K/proxy/8-K
(scripts/common/section_extraction.py).

Usage:
    uv run python scripts/us/20f/02_extract_sections.py
"""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))  # scripts/common/
from section_extraction import run_extraction

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/us/20f/
from segmenter_20f import TARGET_ITEMS, clean_html_to_lines, general_segment, known_issue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="run",
                         help="Short label for this run, embedded in the output filenames/columns.")
    args = parser.parse_args()

    with open(Path("configs/us/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    run_extraction(
        pipeline_step="extract_sections_20f",
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_20f.parquet",
        sections_dir=Path(config["storage"]["interim_sections"]),
        target_items=TARGET_ITEMS,
        output_prefix="filing_sections_20f",
        run_comment=args.comment,
        form="20-F",
        clean_html_to_lines=clean_html_to_lines,
        general_segment=general_segment,
        known_issue=known_issue,
    )


if __name__ == "__main__":
    main()
