"""Sector representation by posture archetype (thesis.qmd
`fig-sector-archetype-heatmap`, ~1408-1500): each of the 10 aggregated
economic sectors' share of firms in each archetype, relative to that
sector's baseline share of the whole firm universe (observed / expected
ratio) -- the sectoral-sorting evidence in Chapter 4.

Reads silver.firm_universe (sic -> the same 10-sector SIC2 mapping used
throughout the thesis) and the gold archetype prediction
(covariates/firm/posture_archetype_static.parquet). Writes
data/results/posture/archetype_sector.parquet: one row per
(sector, archetype) with n_firms, pct_of_archetype (column-normalized
share), baseline_share (sector's share of the whole universe) and
relative_representation (pct_of_archetype / baseline_share, the ratio the
qmd's heatmap colors).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

ARCHETYPES = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers", "No AI"]


def map_sector(sic2: str) -> str:
    try:
        s = int(sic2)
    except ValueError:
        return "Other / Diversified"
    if s in (73, 35, 36):
        return "Technology"
    if s == 48:
        return "Communications"
    if s in (28, 38, 80):
        return "Healthcare & Pharma"
    if 60 <= s <= 67:
        return "Financial Services"
    if 20 <= s <= 39:
        return "Industrials & Mfg"
    if 40 <= s <= 47:
        return "Transportation"
    if s == 49:
        return "Utilities"
    if 10 <= s <= 14 or s == 29:
        return "Energy & Mining"
    if 50 <= s <= 59:
        return "Retail & Wholesale"
    if 70 <= s <= 89:
        return "Business Services"
    return "Other / Diversified"


def main() -> None:
    fu = L.scan("silver.firm_universe").filter(pl.col("country_code") == "us") \
        .select("ticker", "sic").collect().to_pandas()
    fu["sic2"] = fu["sic"].astype(str).str.zfill(4).str[:2]
    fu["sector"] = fu["sic2"].apply(map_sector)

    df_strat = L.read_gold("firm", ("covariates", "posture_archetype_static"))
    df_sec = df_strat.merge(fu, on="ticker", how="left")

    ct = pd.crosstab(df_sec["sector"], df_sec["archetype"])
    baseline = ct.sum(axis=1) / ct.values.sum()
    pct_col = ct.div(ct.sum(axis=0), axis=1)
    rel_rep = pct_col.div(baseline, axis=0)[ARCHETYPES]

    rows = []
    for sector in ct.index:
        for a in ARCHETYPES:
            rows.append({"sector": sector, "archetype": a, "n_firms": int(ct.loc[sector, a]),
                        "pct_of_archetype": float(pct_col.loc[sector, a]),
                        "baseline_share": float(baseline.loc[sector]),
                        "relative_representation": float(rel_rep.loc[sector, a])})
    out_df = pd.DataFrame(rows)
    out = L.results_path("posture", "archetype_sector.parquet")
    out_df.to_parquet(out, index=False)
    print(out_df.sort_values(["archetype", "relative_representation"], ascending=[True, False]).to_string())
    print(f"\n-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
