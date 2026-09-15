"""Nearest firms to each posture archetype's own centroid
(thesis.qmd `tbl-representative-firms`, ~1340-1390): the standardized
(z-scored) 8 posture dimensions, centroid = mean z-score of the firms
already assigned to that archetype (NOT the Archetypal Analysis vertex
itself), ranked by Euclidean distance to that centroid within archetype.
The qmd shows the closest 4 per archetype; this keeps every ranked firm so
a different N (or a different consumer) doesn't need to rebuild it.

Reads data/gold/covariates/firm/posture_archetype_static.parquet (the
pooled fit's own population and labels) and silver.firm_universe for
company_name. Writes data/results/posture/representative_firms.parquet
(ticker, company_name, archetype, rank, distance_to_centroid), rank 1 =
closest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

FEAT_COLS = ["promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation", "temporal_posture",
            "ai_positioning", "specificity", "disclosure_intensity"]


def main() -> None:
    fu = L.scan("silver.firm_universe").filter(__import__("polars").col("country_code") == "us") \
        .select("ticker", "company_name", "sic", "active_status").collect().to_pandas()
    df_strat = L.read_gold("firm", ("covariates", "posture_archetype_static"))
    df_sec = df_strat.merge(fu, on="ticker", how="left")

    fit = df_sec[df_sec["archetype"] != "No AI"].copy()
    zz = (fit[FEAT_COLS] - fit[FEAT_COLS].mean()) / fit[FEAT_COLS].std()

    rows = []
    for a in sorted(fit["archetype"].unique()):
        idx = fit.index[fit["archetype"] == a]
        c = zz.loc[idx].mean().values
        dist = np.linalg.norm(zz.loc[idx].values - c, axis=1)
        order = np.argsort(dist)
        ranked = idx[order]
        for rank, i in enumerate(ranked, start=1):
            rows.append({"ticker": fit.loc[i, "ticker"], "company_name": fit.loc[i, "company_name"],
                        "archetype": a, "rank": rank, "distance_to_centroid": float(dist[order[rank - 1]])})
    out_df = pd.DataFrame(rows)
    out = L.results_path("posture", "representative_firms.parquet")
    out_df.to_parquet(out, index=False)

    # Sanity print matching the qmd's own top-4 wording.
    for a in sorted(fit["archetype"].unique()):
        top4 = out_df[(out_df["archetype"] == a) & (out_df["rank"] <= 4)]
        print(f"[{a}] " + ", ".join(f"{r.company_name} ({r.ticker})" for r in top4.itertuples()))
    print(f"\n-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
