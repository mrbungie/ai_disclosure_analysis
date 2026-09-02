"""
scripts/verif/section_audit/ai_leakage_check.py — standalone re-verification
of the claim "most AI/ML keyword mentions in the full 10-K fall inside
Item 1+1A+7" against the CURRENT corpus (fetched via edgartools, this
session, not the earlier requests-based corpus the original claim was
measured on). Not part of the pipeline; run on demand.

Same AI-keyword construct the old (now-removed) seed_screen used, recovered
from git history — see configs/config.json's history before the YAML
migration.

Usage:
    uv run python scripts/verif/section_audit/ai_leakage_check.py
"""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "us"))
from section_segmenter import clean_html_to_lines, general_segment  # noqa: E402

AI_KEYWORDS = [
    "ai", "artificial intelligence", "generative ai", "gen ai", "machine learning",
    "large language model", "llm", "deep learning", "natural language processing",
    "predictive analytics", "algorithmic", "automation", "computer vision",
    "neural network", "openai", "anthropic", "claude", "deepseek", "chatgpt",
    "copilot", "gemini", "aip", "aiops",
]
FALSE_POSITIVES = ["adobe illustrator", "appreciation", "said", "paid"]
CORE_ITEMS = {"1", "1A", "7"}

_AI_RE = re.compile(
    "|".join(r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b" for kw in AI_KEYWORDS),
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    for fp in FALSE_POSITIVES:
        text = re.sub(r"\b" + re.escape(fp) + r"\b", "", text, flags=re.IGNORECASE)
    return text


def process_one(local_path: str) -> dict:
    try:
        lines = clean_html_to_lines(Path(local_path))
        segments = general_segment(lines)
        counts = {}
        for item_key, text in segments.items():
            n = len(_AI_RE.findall(_clean(text)))
            if n:
                counts[item_key] = n
        return {"local_path": local_path, "counts": counts, "error": None}
    except Exception as e:
        return {"local_path": local_path, "counts": {}, "error": str(e)}


def main():
    with open(REPO_ROOT / "configs" / "us" / "config.yaml") as f:
        config = yaml.safe_load(f)
    manifest = pd.read_parquet(Path(config["storage"]["interim_manifests"]) / "filing_manifest.parquet")
    manifest = manifest[manifest["download_status"] == "completed"]
    paths = manifest["local_path"].tolist()

    print(f"Scanning {len(paths)} 10-K filings for AI/ML keyword mentions across ALL items...")

    total_by_item = {}
    errors = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_one, p) for p in paths]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result["error"]:
                errors += 1
                continue
            for item_key, n in result["counts"].items():
                total_by_item[item_key] = total_by_item.get(item_key, 0) + n

    total = sum(total_by_item.values())
    in_core = sum(n for k, n in total_by_item.items() if k in CORE_ITEMS)

    summary = {
        "n_filings": len(paths),
        "n_errors": errors,
        "total_ai_mentions": total,
        "mentions_in_core_1_1A_7": in_core,
        "pct_in_core": round(100 * in_core / total, 2) if total else None,
        "pct_outside_core": round(100 * (total - in_core) / total, 2) if total else None,
        "by_item": dict(sorted(total_by_item.items(), key=lambda kv: -kv[1])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
