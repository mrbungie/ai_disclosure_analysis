"""DEPRECATED (item 2, gold/analytics migration wave): every column this
script produced now lives in data/gold/datasets/firm_year/firm_year.parquet
(covariates: posture, disclosure_volume, financial_ratios, market;
targets: firm_year) plus predictions/firm_year/posture_archetype_static.parquet
(archetype label) and covariates/firm_year/value_creation.parquet
(roic_minus_wacc) -- see scripts/gold/consolidate/firm_year/build_spine.py
for the spine those are all keyed on. No script reads
firm_year_master_v2/firm_year_full_crosscheck/cohort_2021_crosscheck/
segment_financials.parquet any more (cohort_2021_crosscheck and
segment_financials never had a consumer). Kept here for history only; not
run by anything.

Re-merges the firm-year panel with the financial/market tables.

`build_strategy_dimensions.py` produces the TEXT side (the k=3 posture
archetype and its 8 dimensions, the firm-year panel). This joins that to
the accounting and market side, producing the four label-carrying tables
every downstream analytics doc reads:

  firm_year_full_crosscheck  panel + raw XBRL levels + next-year growth +
                             filing-window return  (docs 02, 03)
  firm_year_master_v2        panel + accounting ratios + market factors
                             (docs 04, 05)
  cohort_2021_crosscheck     the subset of firms already present in 2021
                             (doc 03)
  segment_financials         one row per firm: median ratios/market metrics
                             carrying its (pooled) archetype label
                             (docs 07, 08)

Why this is a separate script from the clustering: the financial inputs
(`firm_year_financials*`, `firm_year_market_factors`,
`firm_year_filing_returns`, `firm_year_roic_wacc`) did NOT change when
DEF 14A and 8-K entered the corpus — no new prices, no new XBRL. Only the
LABELS moved. Keeping the join separate makes that explicit and means a
re-label never risks rewriting the financial tables themselves; this
script only ever reads them.

2026-09-13 (docs/migration_v1_to_v2_analytics.md): this used to read
`firm_year_archetype_behaviors.parquet`/`voice_x_behavior.parquet`, the
outputs of the now-deprecated `build_firm_clusters.py` (the old Chapter 4
A/B/C/D archetypes + 0-3 behavior clusters, already superseded in
`thesis.qmd` by `build_strategy_dimensions.py`'s k=3 posture archetype).
That meant `firm_year_master_v2.parquet`'s `archetype` column was silently
a DIFFERENT construct from the one the rest of the thesis calls
`archetype` -- `thesis.qmd` had grown a defensive comment ("the year-level
letter codes in the master panel ... are not used here") to route around
it. Fixed by sourcing the text side from `firm_year_strategy_dimensions.
parquet`/`firm_strategy_dimensions.parquet` directly, so there is exactly
one `archetype` construct everywhere. The old adoption-concept behavior
shares, domain shares and entity-mention counts have no equivalent in the
posture-dimension framework and are not reconstructed here (not used by
`thesis.qmd`); `segment_financials` now carries the pooled archetype label
instead of the old voice/behavior cross-cluster label.

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

REPO_ROOT = Path(__file__).resolve().parents[3]
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"

PANEL_TEXT_COLUMNS = [
    "ticker", "year", "n_frames", "archetype", "cluster",
    "promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
    "temporal_posture", "ai_positioning", "specificity", "disclosure_intensity",
]

GROWTH_COLUMNS = ["next_revenue_yoy", "next_rd_expense_yoy", "next_capex_yoy",
                  "next_sga_expense_yoy"]
LEVEL_COLUMNS = ["revenue", "rd_expense", "capex", "sga_expense"]
RATIO_COLUMNS = ["gross_margin", "operating_margin", "net_margin", "roa", "roe",
                 "current_ratio", "debt_to_equity", "asset_turnover",
                 "rd_intensity", "capex_intensity", "sic2"]
MARKET_COLUMNS = ["market_cap", "pe_ratio", "ps_ratio", "pb_ratio", "ev_revenue",
                  "ev_ebitda", "beta", "idio_vol_252d", "vol_pre_60d", "vol_post_60d",
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

    panel = read("firm_year_strategy_dimensions", args.clusters_dir)
    text_side = panel[PANEL_TEXT_COLUMNS]

    financials = read("firm_year_financials", args.clusters_dir)
    ratios = read("firm_year_financials_ratios", args.clusters_dir)
    market = read("firm_year_market_factors", args.clusters_dir)
    returns = read("firm_year_filing_returns", args.clusters_dir)
    # Pooled (not panel) archetype: one label per firm, for `segment_financials`
    # below -- the same construct `firm_strategy_dimensions.parquet` uses
    # everywhere else, not the old voice/behavior cross-cluster label.
    firm_archetype = read("firm_strategy_dimensions", args.clusters_dir)[["ticker", "archetype"]]
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
    # año atípico; es lo que 04_perfiles_economicos.md documenta y lo que usan las demás tablas) ---
    per_firm = (master.groupby("ticker")[SEGMENT_COLUMNS].median().reset_index()
                .merge(firm_archetype, on="ticker", how="inner"))
    print(f"segment_financials: {len(per_firm):,} empresas con arquetipo y financieros")

    # --- MODO FINAL: el panel son TODAS las empresas-año con filings, con ceros ---
    # Cada empresa-año con al menos un filing puntuable entra, con intensidad de
    # IA por 1.000 palabras (cero si no habla) desde ai_intensity.py. Las tasas
    # de texto (posture dims) y las etiquetas de
    # arquetipo vienen del panel condicionado y quedan NaN donde la empresa no
    # habló de IA: son propiedades de CÓMO se habla, no existen para el cero.
    # `year` = año de presentación, igual que los financieros.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "posture"))
    from ai_intensity import document_table, aggregate, FILING_FORMS
    docs = document_table()
    docs = docs[docs["form"].isin(FILING_FORMS)].assign(year=lambda d: d["fecha"].dt.year)
    base = aggregate(docs, ["ticker", "year"])
    base = base.merge(text_side.drop(columns=["n_frames"]), on=["ticker", "year"], how="left")
    base["in_text_panel"] = base["archetype"].notna()
    full = (base.merge(financials[["ticker", "year"] + LEVEL_COLUMNS + GROWTH_COLUMNS], on=["ticker", "year"], how="left")
                .merge(returns[["ticker", "year", "ret_m1_p5"]], on=["ticker", "year"], how="left"))
    master = (base.merge(ratios[["ticker", "year"] + RATIO_COLUMNS + GROWTH_COLUMNS], on=["ticker", "year"], how="left")
                  .merge(market[["ticker", "year"] + MARKET_COLUMNS], on=["ticker", "year"], how="left"))
    print(f"firm_year_master_v2: {len(master):,} empresas-año con filings, "
          f"{int(master['any_ai'].sum()):,} con algún frame de IA, "
          f"{int(master['in_text_panel'].sum()):,} con etiquetas del panel condicionado "
          f"({master['operating_margin'].notna().mean()*100:.0f}% con ratios)")
    cohort = full[full["ticker"].isin(cohort_tickers)].copy()
    per_firm = (master[master["in_text_panel"]].groupby("ticker")[SEGMENT_COLUMNS].median().reset_index()
                .merge(firm_archetype, on="ticker", how="inner"))

    for name, table in (("firm_year_full_crosscheck", full),
                        ("firm_year_master_v2", master),
                        ("cohort_2021_crosscheck", cohort),
                        ("segment_financials", per_firm)):
        destination = args.clusters_dir / f"{name}.parquet"
        table.to_parquet(destination, index=False)
        print(f"-> {destination} ({len(table):,} filas, {len(table.columns)} cols)")


if __name__ == "__main__":
    main()
