"""Materializes data/gold/covariates/firm_year/disclosure_volume.parquet:
raw counts, per-1k-word intensities, and rates of AI-text disclosure by
concept (promo/quant/spec/risk/gov/hyp/realized/deployed/...), one row
per (ticker, fiscal_year). The semantic-frame volume signal, orthogonal
to posture (HOW a firm talks) and financials/market (what it IS).

Source: `scripts/gold/posture/ai_intensity.py` (`document_table()` +
`aggregate()`), filings only, grouped by (ticker, calendar filing year) --
not `data/processed/clusters/firm_year_master_v2.parquet` (G-H6a): that
panel is itself built from the same `ai_intensity.document_table()`, so
this reads the same source one hop earlier instead of through the washing
group's consolidated panel.

Usage:
    uv run python scripts/gold/consolidate/disclosure_volume/build_covariates_firm_year.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from ai_intensity import FILING_FORMS, aggregate, document_table  # noqa: E402
from posture_features import write_gold  # noqa: E402

import layers as L  # noqa: E402

OUT = L.gold_path("covariates", "firm_year", "disclosure_volume")

COLUMNS = [
    "n_docs", "n_paragraphs", "n_words", "n_frames", "n_promo", "n_quant", "n_spec", "n_risk", "n_gov", "n_hyp",
    "n_realized", "n_deployed", "n_revenue_outcome", "n_cost_outcome", "n_ai_investment", "n_ai_infrastructure",
    "frames_per_1k", "promo_per_1k", "quant_per_1k", "spec_per_1k", "risk_per_1k", "gov_per_1k", "hyp_per_1k",
    "realized_per_1k", "deployed_per_1k", "revenue_outcome_per_1k", "cost_outcome_per_1k",
    "ai_investment_per_1k", "ai_infrastructure_per_1k", "any_ai",
    "promo_rate", "quant_rate", "spec_rate", "risk_rate", "gov_rate", "hyp_rate", "realized_rate",
]


def main() -> None:
    docs = document_table()
    docs = docs[docs["form"].isin(FILING_FORMS)].copy()
    docs["year"] = docs["fecha"].dt.year
    fy = aggregate(docs, ["ticker", "year"])
    df = fy[["ticker", "year"] + COLUMNS].copy()
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(int).astype(str))
    write_gold(OUT, df, keys=["id"], inputs=[],
              builder="scripts/gold/consolidate/disclosure_volume/build_covariates_firm_year.py")
    print(f"-> {OUT} ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
