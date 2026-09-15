"""External benchmark for @tbl-b1-patents-controls: does disclosed AI activity
predict AI patenting, controlling for firm size and sector?

Two OLS specifications, each on universe firms matched to a patent assignee
with a posture-archetype assignment:

    log(1 + AI patents, 2021-2023) ~ log(1 + disclosed activities) + log(1 + market cap) + C(sic2)
    log(1 + AI patents, 2021-2023) ~ log(1 + grounded activities)  + log(1 + market cap) + C(sic2)

"Grounded" activities are those with evidence_type in {named, metric} (a
named entity or a quantified metric backs the claim, rather than a bare
verbal assertion). AI patents are Google Patents / OECD (2025) taxonomy
matches, restricted to 2021-2023 publications (the window where all filings
have cleared the 18-month statutory confidentiality period, so the count is
un-truncated).

Sources: gold/covariates/firm_year/patents.parquet, gold/covariates/firm_year/market.parquet,
gold/covariates/firm/posture_archetype_static.parquet (universe of firms with a
posture-archetype assignment), silver.firm_universe (sic2), silver.ai_activities +
silver.filing_manifest[_10q] (disclosed/grounded activity counts, ticker resolved
per document -- filings by accession_number, earnings calls by document_id; a text
appearing in both a call and a filing is counted once, attributed to the filing).

Usage:
    uv run python scripts/analytics/appendix/patents_controls.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

PATENT_YEARS = (2021, 2023)
CALL_FORM_TYPE = "Earnings call transcript"


def sic2_by_ticker() -> dict[str, str]:
    fu = pl.scan_parquet(L.path("silver.firm_universe")).select("ticker", pl.col("sic").str.slice(0, 2).alias("sic2"))
    df = fu.collect().to_pandas()
    return dict(zip(df["ticker"], df["sic2"]))


def market_cap_by_ticker() -> dict[str, float]:
    mkt = pl.read_parquet(L.gold_path("covariates", "firm_year", "market")).select("ticker", "market_cap")
    df = mkt.group_by("ticker").agg(pl.col("market_cap").mean()).to_pandas()
    return dict(zip(df["ticker"], df["market_cap"]))


def ai_patents_2021_2023(universe_tickers: list[str]) -> pd.DataFrame:
    pt = pl.read_parquet(L.gold_path("covariates", "firm_year", "patents")).to_pandas()
    pt = pt[pt["ticker"].isin(universe_tickers) & pt["year"].between(*PATENT_YEARS)]
    # Gold keeps one row per spine firm-year with null counts where no patent
    # filing was matched; firms without any matched filing stay out of the benchmark.
    pt = pt.dropna(subset=["ai_patents_oecd"])
    return pt.groupby("ticker").agg(ai_pat=("ai_patents_oecd", "sum")).reset_index()


def key_to_ticker() -> pl.LazyFrame:
    """accession_number -> ticker for filings; document_id -> ticker for earnings
    calls (silver.filing_manifest keys a call by document_id, not accession_number)."""
    filings = (pl.scan_parquet(L.path("silver.filing_manifest"))
              .filter(pl.col("form_type") != CALL_FORM_TYPE)
              .select(pl.col("accession_number").alias("key"), "ticker"))
    calls = (pl.scan_parquet(L.path("silver.filing_manifest"))
            .filter(pl.col("form_type") == CALL_FORM_TYPE)
            .select(pl.col("document_id").alias("key"), "ticker"))
    tenq = pl.scan_parquet(L.path("silver.filing_manifest_10q")).select(pl.col("accession_number").alias("key"), "ticker")
    return pl.concat([filings, calls, tenq], how="vertical_relaxed").unique()


def activity_counts_by_ticker() -> pd.DataFrame:
    """n_act (disclosed activities) and n_grounded (evidence_type in
    {named, metric}) per ticker. A text_hash/activity_id shared by a call
    transcript and a filing (identical text) is attributed to the filing."""
    acts = (pl.scan_parquet(L.path("silver.ai_activities"))
           .filter(pl.col("has_activity"))
           .select(pl.col("accession_number").alias("key"), "evidence_type", "text_hash", "activity_id", "form")
           .join(key_to_ticker(), on="key", how="left")
           .filter(pl.col("ticker").is_not_null())
           .with_columns((pl.col("form") == "Earnings call").cast(pl.Int8).alias("_channel_rank"))
           .sort(["ticker", "text_hash", "activity_id", "_channel_rank"])
           .unique(subset=["ticker", "text_hash", "activity_id"], keep="first")
           .collect().to_pandas())
    acts["grounded"] = acts["evidence_type"].isin(["named", "metric"])
    return acts.groupby("ticker").agg(n_act=("text_hash", "size"), n_grounded=("grounded", "sum")).reset_index()


def build_bench() -> pd.DataFrame:
    archetypes = L.read_gold("firm", ("covariates", "posture_archetype_static", ["archetype"]))
    universe_tickers = archetypes.dropna(subset=["archetype"])["ticker"].unique().tolist()
    bench = pd.DataFrame({"ticker": universe_tickers})
    bench = bench.merge(activity_counts_by_ticker(), on="ticker", how="left").fillna({"n_act": 0, "n_grounded": 0})
    bench = bench.merge(ai_patents_2021_2023(universe_tickers), on="ticker", how="left")
    bench["sic2"] = bench["ticker"].map(sic2_by_ticker())
    bench["market_cap"] = bench["ticker"].map(market_cap_by_ticker())
    bench = bench.dropna(subset=["ai_pat", "market_cap", "sic2"])
    bench["log_ai_pat"] = np.log1p(bench["ai_pat"])
    bench["log_mktcap"] = np.log1p(bench["market_cap"].clip(lower=0))
    bench["log_n_act"] = np.log1p(bench["n_act"])
    bench["log_n_grounded"] = np.log1p(bench["n_grounded"])
    return bench


def main() -> None:
    bench = build_bench()
    n = len(bench)
    n_sic = bench["sic2"].nunique()

    mod_act = smf.ols("log_ai_pat ~ log_n_act + log_mktcap + C(sic2)", data=bench).fit()
    mod_grd = smf.ols("log_ai_pat ~ log_n_grounded + log_mktcap + C(sic2)", data=bench).fit()

    rows = []
    for label, mod, var in [("Disclosed AI activities (log A)", mod_act, "log_n_act"),
                            ("Grounded activities (log G)", mod_grd, "log_n_grounded")]:
        rows.append({
            "regressor": label,
            "coefficient": mod.params[var],
            "se": mod.bse[var],
            "p": mod.pvalues[var],
            "n": n,
            "n_sic": n_sic,
            "r2": mod.rsquared,
        })
    out = pd.DataFrame(rows)
    out_path = L.results_path("appendix", "patents_controls.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
