"""
verif_scripts/section_audit/section_audit.py — verification, not pipeline.

Two questions raised about scripts/04_extract_sections.py's fixed 3-item
scope (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A):

1. MISSING-SECTION AUDIT: of the filings where 04_extract_sections.py found
   no text for one of its 3 target items, how many are (a) genuinely absent
   / incorporated by reference / structurally exceptional vs (b) present in
   the filing but missed by that script's specific start/end regex pair
   (a real extraction bug)? Classified by running a GENERAL item segmenter
   (any "Item N[A-C]", not just the 3 hardcoded ones) and checking whether
   it found the item that 04_extract_sections.py missed.

2. AI-MENTION LEAKAGE AUDIT: of all AI/ML keyword hits (same seed_screen
   construct: configs/config.json: seed_screen.ai_keywords/false_positives)
   across the FULL 10-K text, what fraction falls inside Item 1/1A/7 vs
   elsewhere — broken out by item, with Item 1C (Cybersecurity) and Item 8
   (Financial Statements) called out specifically, since those are the two
   candidates most likely to carry AI-relevant language the 3-item corpus
   would miss.

Both audits share the ONE general item segmenter scripts/04_extract_sections.py
itself uses (scripts/section_segmenter.py) — no duplicated regex to drift
out of sync. It recognizes any "Item N[A-C]" heading (Item 1 through Item
16), not just the 3 target items, and tells a real heading from a
table-of-contents row structurally (a TOC row is a markdown hyperlink; a
real heading is the link's plain-text target) rather than by a length
guess. This is a heuristic, not a guarantee; a handful of segments may
still be misattributed on filings with unusual TOC/cross-reference
formatting.

Outputs (gitignored, cheap to regenerate — no LLM involved):
    data/interim/audits/section_audit/missing_section_audit.parquet
    data/interim/audits/section_audit/ai_leakage_by_item.parquet
    data/interim/audits/section_audit/summary.json

Usage:
    uv run python verif_scripts/section_audit/section_audit.py
"""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "audits" / "section_audit"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from section_segmenter import clean_html_to_lines, general_segment  # noqa: E402

# The 3 items scripts/04_extract_sections.py actually extracts today.
TARGET_ITEMS = ["1", "1A", "7"]


def build_ai_regex(ai_keywords: list[str]) -> re.Pattern:
    patterns = [r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b" for kw in ai_keywords]
    return re.compile("|".join(patterns), re.IGNORECASE)


def clean_false_positives(text: str, false_positives: list[str]) -> str:
    for fp in false_positives:
        text = re.sub(r"\b" + re.escape(fp) + r"\b", "", text, flags=re.IGNORECASE)
    return text


# Set once per worker process via initializer (avoids re-pickling the
# compiled regex + keyword lists on every task).
_AI_REGEX = None
_FALSE_POSITIVES = None


def _worker_init(ai_keywords, false_positives):
    global _AI_REGEX, _FALSE_POSITIVES
    _AI_REGEX = build_ai_regex(ai_keywords)
    _FALSE_POSITIVES = false_positives


def process_one(row: dict) -> dict:
    """Runs in a worker process. Returns per-filing missing-section
    classifications (for the 3 target items not found by
    04_extract_sections.py's own PATTERNS) and per-item AI-hit counts."""
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
                continue  # 04_extract_sections.py already got this one
            if item in segments and len(segments[item]) > 1000:
                cls = "found_by_general_segmenter"  # likely a real extraction-rule gap
            elif item in segments:
                cls = "found_but_below_length_threshold"
            else:
                cls = "not_found_by_either"  # absent / incorporated by reference / exceptional format
            missing_rows.append({
                "accession_number": acc_num, "ticker": ticker, "item": item,
                "classification": cls,
                "general_segment_char_len": len(segments.get(item, "")),
            })

        leakage_rows = []
        for item_key, text in segments.items():
            cleaned = clean_false_positives(text, _FALSE_POSITIVES)
            hit_count = len(_AI_REGEX.findall(cleaned))
            if hit_count > 0:
                leakage_rows.append({
                    "accession_number": acc_num, "ticker": ticker, "item": item_key,
                    "ai_hit_count": hit_count, "char_len": len(text),
                })

        return {
            "accession_number": acc_num, "ticker": ticker, "error": None,
            "missing_rows": missing_rows, "leakage_rows": leakage_rows,
        }
    except Exception as e:
        return {"accession_number": acc_num, "ticker": ticker, "error": str(e)}


def main():
    with open(REPO_ROOT / "configs" / "config.json") as f:
        config = json.load(f)

    manifest = pd.read_parquet(
        Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    )
    manifest = manifest[manifest["parse_status"] == "completed"].copy()

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import harness_fit  # noqa: E402

    sections = harness_fit.load_filing_sections(config)
    found_by_extractor = (
        sections.groupby("accession_number")["section_name"]
        .apply(lambda s: set(x.replace("Item ", "") for x in s))
        .to_dict()
    )
    manifest["extractor_found_items"] = manifest["accession_number"].map(
        lambda acc: list(found_by_extractor.get(acc, set()))
    )

    ai_keywords = config["seed_screen"]["ai_keywords"]
    false_positives = config["seed_screen"]["false_positives"]

    jobs = manifest.to_dict("records")
    print(f"Auditing {len(jobs)} filings ({os.cpu_count()} workers)...")

    missing_rows_all = []
    leakage_rows_all = []
    errors = []

    with ProcessPoolExecutor(
        max_workers=os.cpu_count(), initializer=_worker_init, initargs=(ai_keywords, false_positives)
    ) as executor:
        futures = [executor.submit(process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result.get("error"):
                errors.append(result)
                continue
            missing_rows_all.extend(result["missing_rows"])
            leakage_rows_all.extend(result["leakage_rows"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    missing_df = pd.DataFrame(missing_rows_all)
    missing_df["run_id"] = run_id
    missing_df.to_parquet(OUTPUT_DIR / "missing_section_audit.parquet", index=False)

    leakage_df = pd.DataFrame(leakage_rows_all)
    leakage_df["run_id"] = run_id
    leakage_df.to_parquet(OUTPUT_DIR / "ai_leakage_by_item.parquet", index=False)

    # ---- summarize ----
    summary = {"run_id": run_id, "run_date": datetime.now().isoformat(), "n_filings": len(jobs), "n_errors": len(errors)}

    if not missing_df.empty:
        summary["missing_section_classification"] = (
            missing_df.groupby(["item", "classification"]).size().unstack(fill_value=0).to_dict("index")
        )
    else:
        summary["missing_section_classification"] = {}

    if not leakage_df.empty:
        total_hits = int(leakage_df["ai_hit_count"].sum())
        by_item = leakage_df.groupby("item")["ai_hit_count"].sum().sort_values(ascending=False)
        core_items = {"1", "1A", "7"}
        in_core = int(leakage_df[leakage_df["item"].isin(core_items)]["ai_hit_count"].sum())
        summary["ai_leakage"] = {
            "total_ai_mentions_full_filing": total_hits,
            "mentions_in_core_1_1A_7": in_core,
            "pct_in_core_1_1A_7": round(100 * in_core / total_hits, 2) if total_hits else None,
            "mentions_outside_core": total_hits - in_core,
            "pct_outside_core": round(100 * (total_hits - in_core) / total_hits, 2) if total_hits else None,
            "by_item_all": {k: int(v) for k, v in by_item.items()},
            "item_1C_cybersecurity_mentions": int(by_item.get("1C", 0)),
            "item_8_financial_statements_mentions": int(by_item.get("8", 0)),
            "n_filings_with_any_ai_mention_outside_core": int(
                leakage_df[~leakage_df["item"].isin(core_items)]["accession_number"].nunique()
            ),
        }
    else:
        summary["ai_leakage"] = {}

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
