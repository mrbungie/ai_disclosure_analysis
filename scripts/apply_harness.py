"""
apply_harness.py — Run the two ACTIVE (frozen) candidates over the whole
parsed corpus, composed exactly as designed: the detection harness
pre-classifies every paragraph; the classification harness tags the ones
that pass. These are the deterministic programs whose agreement with the
reward labels was measured on the eval set's test split, applied to the
population.

Output: data/processed/classified_paragraphs.parquet
    paragraph metadata (ticker, accession, filing_date, section, index, text)
    + is_ai_related + the six dimension tags (False when not AI-related)
    + detection_candidate / classification_candidate (provenance).

Firm-year aggregation and any windowing/chunking for reading are analysis,
downstream of this file.

Usage:
    uv run python scripts/apply_harness.py [--detection NAME] [--classification NAME]
    (defaults: each task's ACTIVE)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import harness_fit, pipeline_logger

OUTPUT_PATH = Path("data/processed/classified_paragraphs.parquet")


# Dimension tags for paragraphs the detection harness rejects: NA, not False.
# The classification task is only DEFINED on AI text — "not evaluated" must
# stay distinguishable from "evaluated and negative", or naive corpus-wide
# averages of a dimension would silently include the non-AI denominator.
NA_TAGS = {f: pd.NA for f in ["is_substantive", "is_promotional", "is_risk_related",
                              "is_governance_related", "is_use_case_specific", "is_quantified"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", default=None, help="Detection candidate (default: its ACTIVE)")
    parser.add_argument("--classification", default=None, help="Classification candidate (default: its ACTIVE)")
    args = parser.parse_args()

    det_name = args.detection or harness_fit.active_candidate("detection")
    cls_name = args.classification or harness_fit.active_candidate("classification")
    detect = harness_fit.load_candidate("detection", det_name)
    tag = harness_fit.load_candidate("classification", cls_name)
    print(f"Applying detection={det_name}, classification={cls_name}")

    with open("configs/config.json") as f:
        config = json.load(f)
    manifest = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet")
    sections = pd.read_parquet(Path(config["paths"]["interim_sections"]) / "filing_sections.parquet")
    completed = set(manifest.loc[manifest["parse_status"] == "completed", "accession_number"])
    sections = sections[sections["accession_number"].isin(completed)]

    rows = []
    for _, row in tqdm(sections.iterrows(), total=len(sections)):
        paragraphs = [p.strip() for p in row["section_text"].split("\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            ai = bool(detect(p))
            tags = tag(p) if ai else NA_TAGS
            rows.append({
                "ticker": row["ticker"],
                "accession_number": row["accession_number"],
                "filing_date": row["filing_date"],
                "section_name": row["section_name"],
                "paragraph_index": i,
                "paragraph_text": p,
                "is_ai_related": ai,
                **tags,
            })
    df = pd.DataFrame(rows)
    for f in NA_TAGS:
        df[f] = df[f].astype("boolean")  # nullable: True/False on AI rows, NA elsewhere
    df["detection_candidate"] = det_name
    df["classification_candidate"] = cls_name

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    n_ai = int(df["is_ai_related"].sum())
    print(f"\nClassified {len(df)} paragraphs ({n_ai} AI-related, {n_ai/len(df)*100:.2f}%) -> {OUTPUT_PATH}")

    pipeline_logger.log_event(pipeline_step="apply_harness", level="SUCCESS",
                              message=f"Applied det={det_name} cls={cls_name}: {len(df)} paragraphs, {n_ai} AI-related.",
                              details={"detection": det_name, "classification": cls_name,
                                       "n_paragraphs": len(df), "n_ai": n_ai})


if __name__ == "__main__":
    main()
