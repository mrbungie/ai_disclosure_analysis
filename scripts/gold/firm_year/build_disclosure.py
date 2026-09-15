"""Firm-year disclosure volume and disclosed activities, filings only.

Sums of the document families over the firm-year's filings (10-K, 10-Q,
DEF 14A, 8-K published in the calendar year):

  covariates/firm_year/disclosure_volume  n_docs, n_paragraphs, n_words, AI
      frame counts by concept, `*_per_1k` (per 1,000 words), any_ai and
      `*_rate` (share of the year's frames; null without frames), as
      defined by `ai_intensity.aggregate`
  covariates/firm_year/activities         activity instances by family and
      grounding marker, n_activities, and their `*_per_1k` over the same
      filing words (call activities are excluded: they have no word
      denominator here)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "document"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from ai_intensity import FILING_FORMS, aggregate  # noqa: E402
from build_activities import COUNTS  # noqa: E402
from build_document import gold_document_table  # noqa: E402

import layers as L  # noqa: E402

BUILDER = "scripts/gold/firm_year/build_disclosure.py"
VOLUME_COLUMNS = [
    "n_docs", "n_paragraphs", "n_words", "n_frames", "n_promo", "n_quant", "n_spec", "n_risk", "n_gov", "n_hyp",
    "n_realized", "n_deployed", "n_revenue_outcome", "n_cost_outcome", "n_ai_investment", "n_ai_infrastructure",
    "frames_per_1k", "promo_per_1k", "quant_per_1k", "spec_per_1k", "risk_per_1k", "gov_per_1k", "hyp_per_1k",
    "realized_per_1k", "deployed_per_1k", "revenue_outcome_per_1k", "cost_outcome_per_1k",
    "ai_investment_per_1k", "ai_infrastructure_per_1k", "any_ai",
    "promo_rate", "quant_rate", "spec_rate", "risk_rate", "gov_rate", "hyp_rate", "realized_rate",
]
ACTIVITY_COLUMNS = COUNTS + ["n_activities"]


def main() -> None:
    spine = L.read_gold("firm_year")
    docs = gold_document_table()
    docs = docs[docs["form"].isin(FILING_FORMS)].copy()
    docs["year"] = docs["fecha"].dt.year
    volume = aggregate(docs, ["ticker", "year"])[["ticker", "year"] + VOLUME_COLUMNS]
    out = spine.merge(volume, on=["ticker", "year"], how="left", validate="one_to_one")
    L.write_gold("covariates", "firm_year", "disclosure_volume", out, builder=BUILDER)

    acts = L.read_gold("document", ("covariates", "activities"))
    acts = acts[acts["accession_number"].isin(docs["accession_number"])]
    acts = acts.assign(year=acts["fecha"].dt.year).groupby(["ticker", "year"])[ACTIVITY_COLUMNS].sum().reset_index()
    out = (spine.merge(volume[["ticker", "year", "n_words"]], on=["ticker", "year"], how="left")
           .merge(acts, on=["ticker", "year"], how="left", validate="one_to_one"))
    for c in ACTIVITY_COLUMNS:
        out[f"{c}_per_1k"] = 1000.0 * out[c] / out["n_words"]
    L.write_gold("covariates", "firm_year", "activities", out.drop(columns="n_words"), builder=BUILDER)


if __name__ == "__main__":
    main()
