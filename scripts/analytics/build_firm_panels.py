"""Re-merges the firm-year panel with the financial/market tables.

`build_firm_clusters.py` produces the TEXT side (archetypes, behavior
shares, the firm-year panel). This joins that to the accounting and market
side, producing the four label-carrying tables every downstream analytics
doc reads:

  firm_year_full_crosscheck  panel + raw XBRL levels + next-year growth +
                             filing-window return  (docs 02, 03)
  firm_year_master_v2        panel + accounting ratios + market factors
                             (docs 04, 05)
  cohort_2021_crosscheck     the subset of firms already present in 2021
                             (doc 03)
  segment_financials         one row per firm: mean ratios/market metrics
                             carrying both cluster labels  (docs 07, 08)

Why this is a separate script from the clustering: the financial inputs
(`firm_year_financials*`, `firm_year_market_factors`,
`firm_year_filing_returns`, `firm_year_roic_wacc`) did NOT change when
DEF 14A and 8-K entered the corpus — no new prices, no new XBRL. Only the
LABELS moved. Keeping the join separate makes that explicit and means a
re-label never risks rewriting the financial tables themselves; this
script only ever reads them.

Like the clustering script, this reconstructs a step whose original code
was never versioned, from the column schemas of the stored outputs and the
prose of docs/analytics/02-08. Deterministic, no LLM, no API.

Usage:
    uv run python scripts/analytics/build_firm_panels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"

PANEL_TEXT_COLUMNS = [
    "ticker", "cik", "year", "n_frames", "archetype", "archetype_dist",
    "specificity_index", "quantified_rate", "promotional_rate", "strategic_rate",
    "realized_share", "hypothetical_share", "risk_share", "gov_share",
    "firm_subject_share",
]
BEHAVIOR_SHARE_PREFIX = "behavior_share_"
DOMAIN_SHARE_COLUMNS = ["domain_share_customer_facing", "domain_share_internal",
                        "domain_share_unspecified"]
MENTION_COLUMNS = ["n_entity_mentions", "entities_named"]

GROWTH_COLUMNS = ["next_revenue_yoy", "next_rd_expense_yoy", "next_capex_yoy",
                  "next_sga_expense_yoy"]
LEVEL_COLUMNS = ["revenue", "rd_expense", "capex", "sga_expense"]
RATIO_COLUMNS = ["gross_margin", "operating_margin", "net_margin", "roa", "roe",
                 "current_ratio", "debt_to_equity", "asset_turnover",
                 "rd_intensity", "capex_intensity", "sic2"]
MARKET_COLUMNS = ["market_cap", "pe_ratio", "ps_ratio", "pb_ratio", "ev_revenue",
                  "ev_ebitda", "beta", "vol_pre_60d", "vol_post_60d",
                  "momentum_12_1", "car_m1_p5"]
# docs/analytics/07 aggregates these per firm; sic2 is categorical and
# entities_named is text, so neither belongs in a mean.
SEGMENT_COLUMNS = ["gross_margin", "operating_margin", "net_margin", "roa", "roe",
                   "current_ratio", "debt_to_equity", "rd_intensity", "beta",
                   "vol_pre_60d", "vol_post_60d", "momentum_12_1", "car_m1_p5",
                   "pe_ratio", "ps_ratio", "market_cap"]


def read(name: str, directory: Path) -> pd.DataFrame:
    return pd.read_parquet(directory / f"{name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-dir", type=Path, default=CLUSTERS)
    args = parser.parse_args()

    panel = read("firm_year_archetype_behaviors", args.clusters_dir)
    behavior_columns = [c for c in panel.columns if c.startswith(BEHAVIOR_SHARE_PREFIX)]
    text_side = panel[PANEL_TEXT_COLUMNS + behavior_columns
                      + DOMAIN_SHARE_COLUMNS + MENTION_COLUMNS]

    financials = read("firm_year_financials", args.clusters_dir)
    ratios = read("firm_year_financials_ratios", args.clusters_dir)
    market = read("firm_year_market_factors", args.clusters_dir)
    returns = read("firm_year_filing_returns", args.clusters_dir)
    crossed = read("voice_x_behavior", args.clusters_dir)
    print(f"panel: {len(panel):,} filas, {panel['ticker'].nunique():,} empresas | "
          f"financieros: {len(ratios):,} filas, {ratios['ticker'].nunique():,} empresas")

    # --- full crosscheck: niveles crudos + crecimiento + retorno de ventana ---
    # LEFT desde el panel: una empresa-año sin contrapartida financiera se
    # conserva con NaN en vez de desaparecer, porque "no tenemos XBRL de
    # esta empresa" es un dato distinto de "esta empresa no divulgó IA".
    full = (text_side
            .merge(financials[["ticker", "year"] + LEVEL_COLUMNS + GROWTH_COLUMNS],
                   on=["ticker", "year"], how="left")
            .merge(returns[["ticker", "year", "ret_m1_p5"]], on=["ticker", "year"], how="left"))
    coverage = full[GROWTH_COLUMNS[0]].notna().mean()
    print(f"firm_year_full_crosscheck: {len(full):,} filas "
          f"({coverage*100:.0f}% con crecimiento t+1)")

    # --- master v2: ratios contables + factores de mercado ---
    master = (text_side
              .merge(ratios[["ticker", "year"] + RATIO_COLUMNS + GROWTH_COLUMNS],
                     on=["ticker", "year"], how="left")
              .merge(market[["ticker", "year"] + MARKET_COLUMNS],
                     on=["ticker", "year"], how="left"))
    print(f"firm_year_master_v2: {len(master):,} filas "
          f"({master['operating_margin'].notna().mean()*100:.0f}% con ratios)")

    # --- cohorte 2021: empresas ya presentes en el primer año del panel ---
    cohort_tickers = set(panel.loc[panel["year"] == 2021, "ticker"])
    cohort = full[full["ticker"].isin(cohort_tickers)].copy()
    print(f"cohort_2021_crosscheck: {len(cohort):,} filas, "
          f"{cohort['ticker'].nunique():,} empresas de la cohorte 2021")

    # --- segmentos: una fila por empresa, MEDIANA de sus años (robusta a un
    # año atípico; es lo que 07_...md documenta y lo que usan las demás tablas) ---
    per_firm = (master.groupby("ticker")[SEGMENT_COLUMNS].median().reset_index()
                .merge(crossed, on="ticker", how="inner"))
    print(f"segment_financials: {len(per_firm):,} empresas con ambas etiquetas y financieros")

    for name, table in (("firm_year_full_crosscheck", full),
                        ("firm_year_master_v2", master),
                        ("cohort_2021_crosscheck", cohort),
                        ("segment_financials", per_firm)):
        destination = args.clusters_dir / f"{name}.parquet"
        table.to_parquet(destination, index=False)
        print(f"-> {destination} ({len(table):,} filas, {len(table.columns)} cols)")


if __name__ == "__main__":
    main()
