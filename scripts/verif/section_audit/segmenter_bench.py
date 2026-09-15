"""
scripts/verif/section_audit/segmenter_bench.py — one-command benchmark for
section_segmenter.py against a FIXED, repeatable sample of real filings.
Built to support iterating on the segmenter without re-diagnosing one
ticker at a time by hand: run this, see the failure rate per item per
form type, change the segmenter, run again.

Not part of the pipeline. Samples are seeded (same filings every run) so
before/after comparisons are apples to apples.

Usage:
    uv run python scripts/verif/section_audit/segmenter_bench.py
    uv run python scripts/verif/section_audit/segmenter_bench.py --form 10-Q --n 60
    uv run python scripts/verif/section_audit/segmenter_bench.py --show-failures
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "raw_processing" / "us"))
from section_segmenter import clean_html_to_lines, general_segment, known_issue  # noqa: E402

# What "plausible" means per item, per form — a real company's content for
# that item is essentially never shorter than this. Calibrated from direct
# inspection of real filings (see git history / journals for the specific
# examples): TOC-row leftovers and stub pointers land far below these
# floors; real content lands far above.
MIN_PLAUSIBLE_CHARS = {
    "10-K": {"1": 3000, "1A": 3000, "7": 3000, "8": 3000},
    "10-Q": {"1": 3000, "2": 1000, "3": 200, "4": 200},  # Part I only, for now
}


def load_sample(form: str, n: int, seed: int) -> pd.DataFrame:
    with open(REPO_ROOT / "configs" / "us" / "config.yaml") as f:
        config = yaml.safe_load(f)
    manifest_path = (
        Path(config["storage"]["interim_manifests"]) / "filing_manifest.parquet"
        if form == "10-K"
        else Path(config["storage"]["interim_manifests"]) / "filing_manifest_10q.parquet"
    )
    manifest = pd.read_parquet(REPO_ROOT / manifest_path)
    manifest = manifest[manifest["download_status"] == "completed"]
    return manifest.sample(min(n, len(manifest)), random_state=seed)


def bench(form: str, n: int, seed: int, show_failures: bool) -> None:
    sample = load_sample(form, n, seed)
    floors = MIN_PLAUSIBLE_CHARS[form]

    results = {item: {"ok": 0, "fail": 0, "known": 0, "failures": []} for item in floors}
    for _, row in sample.iterrows():
        lines = clean_html_to_lines(Path(row["local_path"]))
        segs = general_segment(lines, form=form)
        for item, floor in floors.items():
            size = len(segs.get(item, ""))
            if size >= floor:
                results[item]["ok"] += 1
            else:
                results[item]["fail"] += 1
                issue = known_issue(row["ticker"], form, item)
                if issue:
                    results[item]["known"] += 1
                results[item]["failures"].append((row["ticker"], row["accession_number"], size, issue))

    print(f"\n=== {form}  (n={len(sample)}, seed={seed}) ===")
    for item, r in results.items():
        total = r["ok"] + r["fail"]
        pct = 100 * r["ok"] / total if total else 0
        unexplained = r["fail"] - r["known"]
        known_note = f", {r['known']} known" if r["known"] else ""
        print(
            f"  Item {item:<3} {r['ok']:>3}/{total:<3} ok  ({pct:5.1f}%)"
            f"  [{unexplained} unexplained fail{known_note}]"
        )
        if show_failures and r["failures"]:
            for ticker, acc, size, issue in r["failures"][:15]:
                tag = f"  [known: {issue['category']}]" if issue else ""
                print(f"      FAIL {ticker:8} {acc}  size={size}{tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", choices=["10-K", "10-Q", "both"], default="both")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    forms = ["10-K", "10-Q"] if args.form == "both" else [args.form]
    for form in forms:
        bench(form, args.n, args.seed, args.show_failures)


if __name__ == "__main__":
    main()
