"""
scripts/us/8k/02_extract_sections.py — extracts the full text of every
completed 8-K in filing_manifest_8k.parquet, as a single item_key="0"
"section" (see 8k_segmenter.py for why there's no real per-Item
segmentation yet). Same shared runner as 10-K/10-Q/proxy
(scripts/common/section_extraction.py) -- output is just as safe against
partial-write truncation (it merges into the full manifest by row index,
not by overwriting with a partial subset).

Usage:
    uv run python scripts/us/8k/02_extract_sections.py
"""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # scripts/common/
from section_extraction import run_extraction

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/us/8k/
from segmenter_8k import TARGET_ITEMS, clean_html_to_lines, general_segment, known_issue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="run",
                         help="Short label for this run, embedded in the output filenames/columns.")
    args = parser.parse_args()

    with open(Path("configs/us/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    run_extraction(
        pipeline_step="extract_sections_8k",
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_8k.parquet",
        sections_dir=Path(config["storage"]["interim_sections"]),
        target_items=TARGET_ITEMS,
        output_prefix="filing_sections_8k",
        run_comment=args.comment,
        form="8-K",
        clean_html_to_lines=clean_html_to_lines,
        general_segment=general_segment,
        known_issue=known_issue,
    )


if __name__ == "__main__":
    main()
