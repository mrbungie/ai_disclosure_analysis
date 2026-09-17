"""
scripts/us/earnings_calls/02_extract_sections.py — speaker turns out of the
downloaded transcripts, as paragraphs.

ONE TURN = ONE PARAGRAPH. Measured before deciding: turns run 315 chars at
the median and 1,088 at p90, so a turn already IS paragraph-sized, and only
5.4% run past 1,500 characters (a CEO reading prepared remarks). Splitting
those on some length rule would cut a single continuous statement at an
arbitrary point; leaving them whole keeps the unit meaningful — the thing a
named person said in one go.

WHY THE SPEAKER AND THE SECTION ARE KEPT, and are the reason this document
type is worth having at all. A 10-K has one voice and one register. A call
has two things a filing cannot give:

  - WHO said it. An analyst asking "how is AI helping margins?" is not the
    company making an AI claim. Attributing the analyst's framing to the
    firm would manufacture disclosure that never happened — and analysts
    speak in roughly half the turns.
  - WHERE in the call. Prepared remarks are scripted and lawyered, close to
    filing register. The Q&A is improvised, which is where an executive
    actually overreaches. `docs/problemas_academicos.md` #5 turns on
    exactly this: the SEC objected to "industry-leading" SAID ON THE CALL
    without proportional support in the 10-K.

The Q&A boundary is the operator's own handoff ("question-and-answer
session", "first question"), matched on the operator's turns only. When no
marker is found the whole call is labelled `unknown` rather than guessed at
— a wrong boundary silently mislabels every turn after it, and a
conservative `unknown` is filterable while a wrong `prepared` is not.

Usage:
    uv run python scripts/us/earnings_calls/02_extract_sections.py [--limit N]
"""

import argparse
import gzip
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))

import pipeline_logger
from section_extraction import safe_slug

CHECKPOINT_EVERY = 500
#: Bumped whenever the segmentation rules change. It goes in the part
#: FILENAME and decides what counts as already-extracted, so a rule change
#: produces a new run ALONGSIDE the old one instead of requiring the old
#: output to be deleted first. v1 mislabelled 95% of every call as Q&A (see
#: _QA_HANDOFF); its parts are still on disk and still readable.
EXTRACTOR_VERSION = "3"

#: The operator says "question-and-answer" TWICE per call, and only the
#: second one is the boundary. The greeting describes the agenda in future
#: tense — "After the speakers' remarks, there WILL BE a question-and-answer
#: session" — and matching that flips the whole call to Q&A at turn 0, which
#: is exactly what v1 did: 494,597 turns labelled qa against 24,129
#: prepared. The real handoff is imminent and names the first analyst:
#: "Your first question comes from the line of...", "We will now begin the
#: question-and-answer session".
#: v2's version demanded "first question" be followed immediately by the
#: verb, and 28% of calls phrase it with words in between ("our first
#: question FOR THE CALL TODAY comes from...") or without the noun at all
#: ("we'll go first this afternoon to Vijay Kumar"). Loosened accordingly.
_QA_HANDOFF = re.compile(
    r"first\s+question|"
    r"we(?:'ll| will)?\s+(?:now\s+)?(?:go|take)\s+first|"
    r"we(?:'ll| will)\s+now\s+(?:begin|open|take)|"
    r"now\s+begin\s+the\s+question|"
    r"the\s+floor\s+is\s+now\s+open|"
    r"open\s+(?:it\s+|the\s+line\s+)?(?:up\s+)?(?:for|to)\s+questions|"
    r"at\s+this\s+time.{0,40}\bquestion",
    re.IGNORECASE)
_OPERATOR = re.compile(r"^\s*operator\b", re.IGNORECASE)


def extract_turns(payload: dict) -> list[dict]:
    turns = payload.get("structured_content") or []
    rows, section, index = [], "prepared", 0
    seen_operator, operator_turns = False, 0
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = (turn.get("speaker") or "").strip()
        text = re.sub(r"\s+", " ", (turn.get("text") or "")).strip()
        if not text:
            continue
        is_operator = bool(_OPERATOR.match(speaker))
        # The operator's FIRST turn is the greeting; its agenda sentence is
        # never the handoff, however it is worded.
        first_operator_turn = is_operator and not seen_operator
        if is_operator:
            seen_operator = True
            operator_turns += 1
        # Two signals, because neither alone covers this corpus. The lexical
        # one misses phrasings nobody enumerated; the STRUCTURAL one — the
        # operator's second turn is the handoff — held on every call
        # inspected, including the ones no wording matched, because the
        # operator only speaks twice before the Q&A: to greet, and to hand
        # over. It is used only as a fallback, since a call where the
        # operator interjects mid-remarks would break it.
        structural = is_operator and operator_turns == 2 and any(
            not r["block_type"] == "operator" for r in rows)
        if (section == "prepared" and is_operator and not first_operator_turn
                and (_QA_HANDOFF.search(text) or structural)):
            # The handoff turn itself belongs to neither half; everything
            # after it is Q&A.
            section = "qa"
            rows.append({"content_type": "prose", "block_type": "operator",
                         "section": "handoff", "speaker": speaker,
                         "paragraph_index": index, "paragraph_text": text})  # noqa: E501
            index += 1
            continue
        rows.append({
            "content_type": "prose",
            "block_type": "operator" if is_operator else "speaker_turn",
            "section": section, "speaker": speaker,
            "paragraph_index": index, "paragraph_text": text,
        })
        index += 1
    if not any(r["section"] == "qa" for r in rows):
        # No operator handoff found: do not pretend to know where the Q&A
        # began. See the module docstring.
        for row in rows:
            row["section"] = "unknown"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--comment", default="run")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / "configs" / "us" / "config.yaml").read_text())
    manifest_dir = REPO_ROOT / config["storage"]["interim_manifests"]
    sections_dir = REPO_ROOT / config["storage"]["interim_sections"]
    sections_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "filing_manifest_earnings_calls.parquet"
    equibles_manifest_path = manifest_dir / "filing_manifest_earnings_calls_eq.parquet"
    stockanalysis_manifest_path = manifest_dir / "filing_manifest_earnings_calls_sa.parquet"
    backfill_manifest_path = manifest_dir / "filing_manifest_earnings_calls_eq_backfill.parquet"
    other_paths = [equibles_manifest_path, stockanalysis_manifest_path, backfill_manifest_path]
    if not manifest_path.exists() and not any(p.exists() for p in other_paths):
        pipeline_logger.log_event(
            pipeline_step="us_extract_earnings_calls", level="ERROR",
            message="No manifest; run 01_fetch_transcripts.py first", log_dir=manifest_dir)
        return

    # Four independent sources, four independent manifest files (never one
    # mutating another's file): the Hugging Face bulk dataset here,
    # 03_fill_gaps_eq.py's per-ticker gap-filling,
    # 04_fill_gaps_sa.py's headless-Chrome gap-filling and
    # 05_register_eq_backfill.py's Equibles backfill, each in its own
    # parquet. 01_fetch_transcripts.py REWRITES
    # filing_manifest_earnings_calls.parquet wholesale on every run
    # (`manifest.to_parquet(..., index=False)` with a freshly-built
    # DataFrame) — concatenating at read time here, rather than merging the
    # three into one file on disk, is what survives that rewrite instead of
    # getting silently wiped by it.
    manifests = [pd.read_parquet(manifest_path)] if manifest_path.exists() else []
    for path in other_paths:
        if path.exists():
            manifests.append(pd.read_parquet(path))
    manifest = pd.concat(manifests, ignore_index=True).drop_duplicates("document_id", keep="first")
    # Already-extracted set comes from the output parts, not a parse_status
    # column — same rule as scripts/it/, so this can run beside a fetch.
    already = set()
    for part in sections_dir.glob(f"earnings_call_paragraphs__v={EXTRACTOR_VERSION}"
                                  f"__run=*__part=*.parquet"):
        try:
            already.update(pd.read_parquet(part, columns=["document_id"]).document_id.unique())
        except Exception:  # noqa: BLE001 — a part mid-write is not fatal
            continue
    pending = manifest[(manifest.download_status == "completed")
                       & (~manifest.document_id.isin(already))]
    if args.limit:
        pending = pending.head(args.limit)
    if pending.empty:
        pipeline_logger.log_event(
            pipeline_step="us_extract_earnings_calls", level="INFO",
            message=f"Nothing pending ({len(already):,} already extracted).", log_dir=manifest_dir)
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    comment_slug = safe_slug(args.comment)
    buffer, part_num = [], 0

    def flush():
        nonlocal buffer, part_num
        if not buffer:
            return
        part = pd.DataFrame(buffer)
        path = sections_dir / (f"earnings_call_paragraphs__v={EXTRACTOR_VERSION}"
                               f"__run={run_id}__part={part_num:04d}__{comment_slug}.parquet")
        part["run_id"], part["extractor_version"] = run_id, EXTRACTOR_VERSION
        part.to_parquet(path, index=False)
        pipeline_logger.log_event(
            pipeline_step="us_extract_earnings_calls", level="INFO",
            message=f"Flushed {path.name}: {len(part):,} rows", log_dir=manifest_dir)
        buffer, part_num = [], part_num + 1

    done = 0
    for row in tqdm(pending.to_dict("records"), total=len(pending)):
        try:
            with gzip.open(row["local_path"], "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as error:
            pipeline_logger.log_event(
                pipeline_step="us_extract_earnings_calls", level="ERROR",
                message=f"unreadable: {error}", ticker=row["document_id"], log_dir=manifest_dir)
            continue
        for paragraph in extract_turns(payload):
            paragraph.update(document_id=row["document_id"], ticker=row["ticker"],
                             cik=row["cik"], form_type=row["form_type"],
                             filing_type=row["filing_type"], filing_date=row["filing_date"],
                             period_end_date=row["period_end_date"])
            buffer.append(paragraph)
        done += 1
        if done % CHECKPOINT_EVERY == 0:
            flush()

    flush()
    pipeline_logger.log_event(
        pipeline_step="us_extract_earnings_calls", level="SUCCESS",
        message=f"Finished run_id={run_id}; parts in {sections_dir}", log_dir=manifest_dir)


if __name__ == "__main__":
    main()
