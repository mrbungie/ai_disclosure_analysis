"""
scripts/us/proxy/02_extract_sections.py — extracts the full text of every
completed DEF 14A in filing_manifest_proxy.parquet, as a single
item_key="0" "section" (see proxy_segmenter.py for why there's no real
per-heading segmentation yet). Same shared runner as 10-K/10-Q
(scripts/common/section_extraction.py) -- output is just as safe against
partial-write truncation (it merges into the full manifest by row index,
not by overwriting with a partial subset).

Usage:
    uv run python scripts/us/proxy/02_extract_sections.py
"""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # scripts/common/
from section_extraction import run_extraction

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/us/proxy/
from proxy_segmenter import TARGET_ITEMS, clean_html_to_lines, general_segment, known_issue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="run",
                         help="Short label for this run, embedded in the output filenames/columns.")
    args = parser.parse_args()

    with open(Path("configs/us/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    run_extraction(
        pipeline_step="extract_sections_proxy",
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_proxy.parquet",
        sections_dir=Path(config["storage"]["interim_sections"]),
        target_items=TARGET_ITEMS,
        output_prefix="filing_sections_proxy",
        run_comment=args.comment,
        form="DEF 14A",
        clean_html_to_lines=clean_html_to_lines,
        general_segment=general_segment,
        known_issue=known_issue,
    )


if __name__ == "__main__":
    main()
