"""
scripts/us/10q/02_extract_sections.py — extracts Item 2 (MD&A) and Item 1A
(Part II — Risk Factor updates) from every completed 10-Q in
filing_manifest_10q.parquet. Thin wrapper around
scripts/common/section_extraction.py, which holds the run/checkpoint/
traceability logic shared with the 10-K extraction script — same
segmenter (section_segmenter.general_segment already handles 10-Q's
Part-restart item numbering), same run-partitioned parquet design.

Item 2 (10-Q's MD&A — the direct analog of the 10-K's Item 7) is the
primary target: the 10-Q panel exists as a quarterly "shock series" (see
the project's data-scope decision — never pooled with the 10-K panel),
and MD&A is the narrative section most likely to carry an incremental,
quarter-over-quarter AI-disclosure update, mirroring why the 10-K stage
narrows to Item 1/1A/7 rather than the full filing.

Item 1A (Part II — updated Risk Factors) is ALSO targeted: unlike Item 1
(Financial Statements — XBRL-tagged numbers/notes, not narrative) or
Items 3/4 (near-always boilerplate — see configs/known_segmenter_issues.yaml),
Item 1A only appears when the filer reports a MATERIAL CHANGE to risk
factors since the 10-K — exactly the kind of "shock" signal this panel
exists to capture (e.g. a company adding a new AI-related risk factor
mid-year). Legitimately empty/not-found in most quarters — that's
correct, not a segmenter failure — see item_key "1A" in the trace: found
== False there means "no update reported," not "extraction missed it."
"""

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # scripts/common/
from section_extraction import run_extraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/us/
from section_segmenter import clean_html_to_lines, general_segment, known_issue

TARGET_ITEMS = {"2": "Item 2", "1A": "Item 1A"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="run",
                         help="Short label for this run, embedded in the output filenames/columns.")
    args = parser.parse_args()

    with open(Path("configs/us/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    run_extraction(
        pipeline_step="extract_sections_10q",
        manifest_path=Path(config["storage"]["interim_manifests"]) / "filing_manifest_10q.parquet",
        sections_dir=Path(config["storage"]["interim_sections"]),
        target_items=TARGET_ITEMS,
        output_prefix="filing_sections_10q",
        run_comment=args.comment,
        form="10-Q",
        clean_html_to_lines=clean_html_to_lines,
        general_segment=general_segment,
        known_issue=known_issue,
    )


if __name__ == "__main__":
    main()
