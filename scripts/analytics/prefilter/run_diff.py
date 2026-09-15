"""
scripts/analytics/prefilter/run_diff.py — run-over-run diff of the prefilter's
positive set, moved out of `scripts/enrichment/ai_prefilter_apply_frozen.py`
(E-M4): counting new positives between two runs is a report, not a row of
enrichment output.

Compares two `data/interim/prefilter_predictions_unique/
prefilter_predictions__run=*.parquet` files (read-only, additive — never
written to) and counts how many unique texts flip from not-positive to
`is_ai_prefiltered` between them.

Output: data/results/prefilter/run_diff.json

Usage:
    uv run python scripts/analytics/prefilter/run_diff.py
    uv run python scripts/analytics/prefilter/run_diff.py --prior-run 20260910T050100Z --current-run 20260912T183947Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

PRED_DIR = L.INTERIM / "prefilter_predictions_unique"


def _predictions_path(run_id: str) -> Path:
    p = PRED_DIR / f"prefilter_predictions__run={run_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def latest_two_runs() -> tuple[str, str]:
    runs = sorted(p.name.removeprefix("prefilter_predictions__run=").removesuffix(".parquet")
                  for p in PRED_DIR.glob("prefilter_predictions__run=*.parquet"))
    if len(runs) < 2:
        raise FileNotFoundError(f"need at least two prediction runs under {PRED_DIR}, found {len(runs)}")
    return runs[-2], runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prior-run", default=None)
    parser.add_argument("--current-run", default=None)
    args = parser.parse_args()

    prior_run, current_run = args.prior_run, args.current_run
    if prior_run is None or current_run is None:
        default_prior, default_current = latest_two_runs()
        prior_run = prior_run or default_prior
        current_run = current_run or default_current

    prior = pd.read_parquet(_predictions_path(prior_run), columns=["text_hash", "is_ai_prefiltered"])
    current = pd.read_parquet(_predictions_path(current_run), columns=["text_hash", "is_ai_prefiltered"])

    prior_positive = set(prior.loc[prior["is_ai_prefiltered"], "text_hash"])
    current_positive = set(current.loc[current["is_ai_prefiltered"], "text_hash"])

    result = {
        "prior_run": prior_run, "current_run": current_run,
        "prior_positive": len(prior_positive), "current_positive": len(current_positive),
        "new_positive_since_prior_run": len(current_positive - prior_positive),
        "dropped_since_prior_run": len(prior_positive - current_positive),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = L.results_path("prefilter", "run_diff.json")
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"-> {out_path}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
