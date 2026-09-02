"""
scripts/us/10k/02_extract_sections.py — extracts Item 1/1A/7 (Business,
Risk Factors, MD&A) from every completed 10-K in filing_manifest.parquet.
Thin wrapper around scripts/common/section_extraction.py, which holds the
run/checkpoint/traceability logic shared across every country's extraction
scripts — the actual heading-parsing logic (section_segmenter.py, in this
same scripts/us/ directory) is country-specific and imported here, then
passed into the shared runner rather than the runner importing it itself.
"""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # scripts/common/
from section_extraction import run_extraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/us/
from section_segmenter import clean_html_to_lines, general_segment, known_issue

# The 3 items this pipeline stage narrows the corpus to (see
# verif_scripts/section_audit for why: ~88% of AI/ML mentions in the full
# 10-K fall inside these 3; the rest is deliberately left out, not missed).
# Segmentation itself is general (any Item N) — see section_segmenter.py.
TARGET_ITEMS = {"1": "Item 1", "1A": "Item 1A", "7": "Item 7"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="run",
                         help="Short label for this run, embedded in the output filenames/columns.")
    args = parser.parse_args()

    with open(Path("configs/us/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    run_extraction(
        pipeline_step="extract_sections",
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest.parquet",
        sections_dir=Path(config["storage"]["interim_sections"]),
        target_items=TARGET_ITEMS,
        output_prefix="filing_sections",
        run_comment=args.comment,
        form="10-K",
        clean_html_to_lines=clean_html_to_lines,
        general_segment=general_segment,
        known_issue=known_issue,
    )


if __name__ == "__main__":
    main()
