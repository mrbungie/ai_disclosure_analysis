"""
apply_harness.py — Run the two frozen (ACTIVE) candidates over the whole
parsed corpus, composed exactly as the cascade is designed (docs/
distillation_map.html §1): the detection harness pre-classifies every
paragraph, producing the candidate frame; the classification harness tags
chunks built around the admitted paragraphs, and those tags are broadcast
back to every member paragraph — so the final output stays one row per
paragraph, matching the eval-set unit of analysis for stage 1 even though
stage 2 labels at chunk granularity.

Two things this script writes:
  data/processed/candidate_frame.parquet
      Paragraph-level rows the detection ACTIVE admits. This IS the
      population scripts/build_eval_set.py --stage classification samples
      from — the map's "deploy the prefilter to produce the candidate
      frame" step.
  data/processed/classified_paragraphs.parquet
      paragraph metadata (ticker, accession, filing_date, section, index,
      text) + is_ai_related + the six dimension tags (NA when not
      AI-related — not False; the classification task is only DEFINED on
      AI text) + detection_candidate / classification_candidate / chunk_id
      (provenance).

Firm-year aggregation and any further windowing/chunking for reading are
analysis, downstream of this file.

Usage:
    uv run python scripts/apply_harness.py [--detection NAME] [--classification NAME]
    (defaults: each task's ACTIVE)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import harness_fit, pipeline_logger

CANDIDATE_FRAME_PATH = Path("data/processed/candidate_frame.parquet")
OUTPUT_PATH = Path("data/processed/classified_paragraphs.parquet")

DIMENSION_FIELDS = ["is_substantive", "is_promotional", "is_risk_related",
                    "is_governance_related", "is_use_case_specific", "is_quantified"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection", default=None, help="Detection candidate (default: its ACTIVE)")
    parser.add_argument("--classification", default=None,
                        help="Classification candidate (default: its ACTIVE, if one has been frozen)")
    parser.add_argument("--chunk-window", type=int, default=None,
                        help="Paragraphs of context each side of an admitted paragraph "
                             "(default: configs/config.json seed_screen.chunk_window)")
    parser.add_argument("--detection-only", action="store_true",
                        help="Write the candidate frame and stop — for freezing stage 2, "
                             "which needs the frame to sample from before it has a classifier.")
    args = parser.parse_args()

    with open("configs/config.json") as f:
        config = json.load(f)
    chunk_window = args.chunk_window if args.chunk_window is not None else config["seed_screen"]["chunk_window"]

    det_name = args.detection or harness_fit.active_candidate("detection")
    detect = harness_fit.load_candidate("detection", det_name)
    print(f"Applying detection={det_name}")

    paragraphs = harness_fit.flatten_corpus_paragraphs(config)
    paragraphs["is_ai_related"] = [bool(detect(t)) for t in paragraphs["paragraph_text"]]

    CANDIDATE_FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate_frame = paragraphs[paragraphs["is_ai_related"]].drop(columns=["is_ai_related"])
    candidate_frame.to_parquet(CANDIDATE_FRAME_PATH, index=False)
    n_ai = len(candidate_frame)
    print(f"Candidate frame: {n_ai}/{len(paragraphs)} paragraphs admitted "
          f"({n_ai / len(paragraphs) * 100:.2f}%) -> {CANDIDATE_FRAME_PATH}")

    if args.detection_only:
        pipeline_logger.log_event(pipeline_step="apply_harness", level="SUCCESS",
                                  message=f"Wrote candidate frame with det={det_name}: {n_ai} paragraphs.",
                                  details={"detection": det_name, "n_ai": n_ai})
        return

    cls_name = args.classification or harness_fit.active_candidate("classification")
    tag = harness_fit.load_candidate("classification", cls_name)
    print(f"Applying classification={cls_name} (chunk_window={chunk_window})")

    chunks, membership = harness_fit.build_chunks(paragraphs, admit_col="is_ai_related", window=chunk_window)
    chunk_tags = pd.DataFrame([tag(t) for t in chunks["chunk_text"]])
    chunk_tags["chunk_id"] = chunks["chunk_id"].to_numpy()

    # Broadcast each chunk's six tags to every member paragraph — the
    # rollout that keeps the final panel paragraph-indexed even though the
    # classification decision was made at chunk granularity.
    paragraph_tags = membership.merge(chunk_tags, on="chunk_id", how="left")

    df = paragraphs.merge(paragraph_tags, on="paragraph_id", how="left")
    for f in DIMENSION_FIELDS:
        df[f] = df[f].where(df["is_ai_related"], pd.NA).astype("boolean")
    df["chunk_id"] = df["chunk_id"].where(df["is_ai_related"])
    df["detection_candidate"] = det_name
    df["classification_candidate"] = cls_name

    df = df[["ticker", "accession_number", "filing_date", "section_name", "paragraph_index",
            "paragraph_text", "is_ai_related", *DIMENSION_FIELDS, "chunk_id",
            "detection_candidate", "classification_candidate"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    n_tagged = int(df["is_ai_related"].sum())
    print(f"\nClassified {len(df)} paragraphs ({n_tagged} AI-related, tagged from {len(chunks)} chunks) "
          f"-> {OUTPUT_PATH}")

    pipeline_logger.log_event(pipeline_step="apply_harness", level="SUCCESS",
                              message=f"Applied det={det_name} cls={cls_name}: {len(df)} paragraphs, "
                                      f"{n_tagged} AI-related via {len(chunks)} chunks.",
                              details={"detection": det_name, "classification": cls_name,
                                       "n_paragraphs": len(df), "n_ai": n_tagged, "n_chunks": len(chunks)})


if __name__ == "__main__":
    main()
