"""Construye el lado CONTABLE del panel empresa-año desde XBRL crudo.

Produce dos parquets que hasta ahora existían como datos sin código que los
generara (`docs/analytics/05_senal_incremental.md` y
`04_perfiles_economicos.md` los describían en prosa; el script
original nunca se versionó y se perdió):

  firm_year_financials.parquet         revenue / R&D / capex / SG&A del FY
                                       disclosed en cada 10-K + crecimiento
                                       YoY del FY SIGUIENTE  (docs 02, 03)
  firm_year_financials_ratios.parquet  lo anterior + balance, márgenes,
                                       ROA/ROE, liquidez, apalancamiento,
                                       EBITDA y sic2                (doc 04)

Regla de alineamiento temporal (docs/analytics/leakage-checking.md — esta es
la parte que NO se puede simplificar):

  1. Sólo hechos XBRL de duración anual (340-380 días) cuentan como "el FY".
  2. Para cada 10-K de un ticker, `disclosed_period_end` = el `period_end`
     anual más reciente <= `filing_date` (+10 días de margen). Es el FY que
     el documento efectivamente reporta: dato YA CONOCIDO al momento del
     filing, válido como covariable contemporánea, nunca como outcome.
  3. `next_period_end` = el `period_end` anual inmediatamente siguiente de
     ESE MISMO ticker. Es el FY que todavía no había terminado cuando se
     presentó el documento: el único válido como outcome futuro.

  Nunca se usa aritmética de año calendario. El mes de cierre fiscal varía
  por empresa (NVDA cierra en enero, KO en diciembre) y un shift fijo de -1
  año mezcla horizontes de predicción distintos entre empresas — el bug que
  documenta leakage-checking.md.

Diferencias deliberadas respecto de la versión perdida (rige lo nuevo):

  a. `next_*_yoy` queda en NULL cuando el gap entre `disclosed_period_end` y
     `next_period_end` cae fuera de [340, 380] días. Son huecos de XBRL que
     ponen el "FY siguiente" a 2-4 años de distancia (3 de 2.325 filas en la
     versión anterior, que leakage-checking.md pedía filtrar y nadie
     filtraba). `next_gap_days` queda en la tabla para poder auditarlo.
  b. Cadenas de fallback por métrica (ver `DURATION_METRICS`): un filer que
     reporta `RevenueFromContractWithCustomerExcludingAssessedTax` y no
     `Revenues` ya no queda sin revenue. Sube cobertura sin cambiar la
     definición.
  c. Denominadores <= 0 producen NULL en vez de un ratio absurdo (equity
     negativo -> ROE/PB/deuda-equity nulos). La versión anterior los dejaba
     entrar y después los winsorizaba al 2-98%.

Determinístico, sin LLM, sin costo de API: lee `data/raw/xbrl_facts/us/` y
`filing_manifest` y no escribe nada fuera de `data/processed/clusters/`.

Uso:
    uv run python scripts/analytics/build_firm_financials.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
XBRL_GLOB = REPO_ROOT / "data" / "raw" / "xbrl_facts" / "us" / "*.parquet"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"

ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 340, 380
# Margen entre el cierre fiscal y el filing: el 10-K se presenta semanas
# después del cierre, así que el FY disclosed es el último que cerró ANTES
# del filing. Los 10 días cubren el caso raro de un filing casi simultáneo.
DISCLOSED_SLACK_DAYS = 10
# Balance: los hechos "instant" del cierre fiscal caen exactamente en
# `period_end`; 20 días cubren diferencias de convención (último día hábil).
INSTANT_TOLERANCE_DAYS = 20
# Portada del 10-K: `dei:EntityCommonStockSharesOutstanding` se fecha cerca
# del FILING, no del cierre fiscal — semanas después. Se busca por separado.
SHARES_TOLERANCE_DAYS = 75

# Cadenas de fallback: se toma el primer concepto con dato para ese
# (ticker, period_end). El orden es de definición más estrecha a más amplia.
DURATION_METRICS: dict[str, list[str]] = {
    "revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueNet",
        "us-gaap:SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": [
        "us-gaap:CostOfRevenue",
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfGoodsSold",
        "us-gaap:CostOfServices",
    ],
    "rd_expense": [
        "us-gaap:ResearchAndDevelopmentExpense",
        "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "sga_expense": [
        "us-gaap:SellingGeneralAndAdministrativeExpense",
        "us-gaap:GeneralAndAdministrativeExpense",
    ],
    "capex": [
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        "us-gaap:PaymentsToAcquireProductiveAssets",
    ],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "net_income": [
        "us-gaap:NetIncomeLoss",
        "us-gaap:ProfitLoss",
    ],
    "da": [
        "us-gaap:DepreciationDepletionAndAmortization",
        "us-gaap:DepreciationAndAmortization",
    ],
    "interest_expense": [
        "us-gaap:InterestExpense",
        "us-gaap:InterestExpenseNonoperating",
        "us-gaap:InterestExpenseDebt",
    ],
    "pretax_income": [
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax_expense": ["us-gaap:IncomeTaxExpenseBenefit"],
    "eps_diluted": ["us-gaap:EarningsPerShareDiluted"],
}

INSTANT_METRICS: dict[str, list[str]] = {
    "total_assets": ["us-gaap:Assets"],
    "equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "current_assets": ["us-gaap:AssetsCurrent"],
    "current_liabilities": ["us-gaap:LiabilitiesCurrent"],
    "long_term_debt": [
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:LongTermDebt",
    ],
    "cash": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
}
SHARES_METRIC = {"shares_out": ["dei:EntityCommonStockSharesOutstanding"]}

GROWTH_METRICS = ["revenue", "rd_expense", "capex", "sga_expense"]
LEVEL_COLUMNS = ["revenue", "rd_expense", "capex", "sga_expense"]


def _concept_priority(metrics: dict[str, list[str]]) -> pd.DataFrame:
    rows = [(concept, metric, rank)
            for metric, concepts in metrics.items()
            for rank, concept in enumerate(concepts)]
    return pd.DataFrame(rows, columns=["concept", "metric", "priority"])


def load_facts(con: duckdb.DuckDBPyConnection, glob: Path) -> pd.DataFrame:
    """Hechos XBRL crudos, ya deduplicados a un valor por
    (ticker, concepto, period_type, period_end).

    XBRL repite cada cifra anual en 2-3 filings distintos (los comparativos
    del año anterior), y las repeticiones pueden diferir por reexpresiones.
    Se toma la MEDIANA de las repeticiones, que es robusta a una reexpresión
    aislada — misma regla que documenta 05_senal_incremental.md."""
    return con.execute(f"""
        SELECT ticker, concept, period_type, period_start, period_end,
               median(numeric_value) AS value
        FROM read_parquet('{glob}', union_by_name=True)
        WHERE numeric_value IS NOT NULL AND ticker IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
    """).fetchdf()


def pivot_metrics(facts: pd.DataFrame, metrics: dict[str, list[str]],
                  period_type: str) -> pd.DataFrame:
    """(ticker, period_end) x métrica, resolviendo la cadena de fallback:
    para cada celda gana el concepto de menor `priority` que tenga dato."""
    priority = _concept_priority(metrics)
    df = facts[facts["period_type"] == period_type].merge(priority, on="concept")
    if period_type == "duration":
        days = (df["period_end"] - df["period_start"]).dt.days
        df = df[(days >= ANNUAL_MIN_DAYS) & (days <= ANNUAL_MAX_DAYS)]
    df = (df.sort_values(["ticker", "period_end", "metric", "priority"])
            .drop_duplicates(["ticker", "period_end", "metric"], keep="first"))
    wide = df.pivot_table(index=["ticker", "period_end"], columns="metric",
                          values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    for metric in metrics:
        if metric not in wide.columns:
            wide[metric] = np.nan
    return wide.sort_values(["ticker", "period_end"]).reset_index(drop=True)


def align_to_filings(filings: pd.DataFrame, annual: pd.DataFrame,
                     metrics: list[str]) -> pd.DataFrame:
    """El corazón del script: para cada 10-K, el FY que reporta y el que
    todavía no había terminado. Ver el docstring del módulo."""
    out = []
    by_ticker = {t: g.reset_index(drop=True) for t, g in annual.groupby("ticker")}
    for row in filings.itertuples(index=False):
        group = by_ticker.get(row.ticker)
        if group is None or group.empty:
            continue
        ends = group["period_end"].values
        cutoff = np.datetime64(row.filing_date + pd.Timedelta(days=DISCLOSED_SLACK_DAYS))
        idx = int(np.searchsorted(ends, cutoff, side="right")) - 1
        if idx < 0:
            continue
        disclosed = group.iloc[idx]
        record = {
            "ticker": row.ticker,
            "year": row.filing_date.year,
            "filing_date": row.filing_date,
            "accession_number": row.accession_number,
            "disclosed_period_end": disclosed["period_end"],
        }
        record.update({m: disclosed.get(m, np.nan) for m in metrics})
        if idx + 1 < len(group):
            following = group.iloc[idx + 1]
            gap = (following["period_end"] - disclosed["period_end"]).days
            record["next_period_end"] = following["period_end"]
            record["next_gap_days"] = gap
            usable = ANNUAL_MIN_DAYS <= gap <= ANNUAL_MAX_DAYS
            for metric in GROWTH_METRICS:
                base, nxt = disclosed.get(metric, np.nan), following.get(metric, np.nan)
                record[f"next_{metric}_yoy"] = (
                    nxt / base - 1 if usable and pd.notna(base) and pd.notna(nxt)
                    and base > 0 else np.nan)
        else:
            record["next_period_end"] = pd.NaT
            record["next_gap_days"] = np.nan
            for metric in GROWTH_METRICS:
                record[f"next_{metric}_yoy"] = np.nan
        out.append(record)
    return pd.DataFrame(out)


def attach_asof(panel: pd.DataFrame, wide: pd.DataFrame, metrics: list[str],
                left_on: str, tolerance_days: int) -> pd.DataFrame:
    """Pega hechos 'instant' al panel por CERCANÍA de fecha, por ticker.

    Dos anclas distintas y ese es el punto: el balance se ancla al cierre
    fiscal (`disclosed_period_end`), pero `shares_out` es la cifra de portada
    del 10-K y se fecha cerca del FILING. Mergear todo por `period_end`
    exacto hacía que la búsqueda agarrara la fila de portada en vez de la del
    cierre fiscal y perdiera el balance entero (bug documentado en 04_perfiles_economicos.md)."""
    left = panel.sort_values(left_on).copy()
    right = (wide[["ticker", "period_end"] + metrics]
             .dropna(subset=["period_end"]).sort_values("period_end"))
    merged = pd.merge_asof(
        left, right.rename(columns={"period_end": f"_{left_on}_matched"}),
        left_on=left_on, right_on=f"_{left_on}_matched", by="ticker",
        direction="nearest", tolerance=pd.Timedelta(days=tolerance_days))
    return merged.drop(columns=[f"_{left_on}_matched"])


def safe_div(numerator: pd.Series, denominator: pd.Series,
             positive_denominator: bool = True) -> pd.Series:
    """División que devuelve NULL en vez de un ratio sin sentido.

    `positive_denominator` importa para equity y revenue: una empresa con
    patrimonio contable negativo produce un ROE positivo espurio (pérdida
    dividida por equity negativo), y ese número no significa nada."""
    den = denominator.astype(float).copy()
    den = den.where(den > 0) if positive_denominator else den.where(den != 0)
    return numerator.astype(float) / den


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--xbrl-glob", type=Path, default=XBRL_GLOB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        filings = con.execute("""
            SELECT ticker, cik, filing_date, accession_number
            FROM filing_manifest
            WHERE country_code = 'us' AND form_type = '10-K' AND ticker IS NOT NULL
            ORDER BY ticker, filing_date
        """).fetchdf()
        universe = con.execute("""
            SELECT ticker, sic FROM firm_universe
            WHERE country_code = 'us' AND ticker IS NOT NULL
        """).fetchdf()
        facts = load_facts(con, args.xbrl_glob)
    finally:
        con.close()
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    print(f"{len(filings):,} filings 10-K | {len(facts):,} hechos XBRL "
          f"({facts['ticker'].nunique():,} tickers)")

    duration = pivot_metrics(facts, DURATION_METRICS, "duration")
    instants = pivot_metrics(facts, INSTANT_METRICS, "instant")
    shares = pivot_metrics(facts, SHARES_METRIC, "instant")
    print(f"anuales: {len(duration):,} (ticker, period_end) | "
          f"instantáneos: {len(instants):,} | shares: {len(shares):,}")

    panel = align_to_filings(filings, duration, list(DURATION_METRICS))
    print(f"alineados: {len(panel):,} filas ({panel['ticker'].nunique():,} tickers)")
    gap = panel["next_gap_days"].dropna()
    print(f"gap al FY siguiente dentro de [340,380]: "
          f"{(gap.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)).mean()*100:.1f}% "
          f"({int((~gap.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)).sum())} filas con "
          f"next_*_yoy anulado)")

    financials = panel[["ticker", "year", "filing_date", "accession_number",
                        "disclosed_period_end"] + LEVEL_COLUMNS +
                       ["next_period_end", "next_gap_days"] +
                       [f"next_{m}_yoy" for m in GROWTH_METRICS]].copy()

    ratios = attach_asof(panel, instants, list(INSTANT_METRICS),
                         "disclosed_period_end", INSTANT_TOLERANCE_DAYS)
    ratios = attach_asof(ratios, shares, ["shares_out"],
                         "filing_date", SHARES_TOLERANCE_DAYS)

    ratios["gross_margin"] = safe_div(ratios["revenue"] - ratios["cost_of_revenue"],
                                      ratios["revenue"])
    ratios["operating_margin"] = safe_div(ratios["operating_income"], ratios["revenue"])
    ratios["net_margin"] = safe_div(ratios["net_income"], ratios["revenue"])
    ratios["roa"] = safe_div(ratios["net_income"], ratios["total_assets"])
    ratios["roe"] = safe_div(ratios["net_income"], ratios["equity"])
    ratios["current_ratio"] = safe_div(ratios["current_assets"], ratios["current_liabilities"])
    ratios["debt_to_equity"] = safe_div(ratios["long_term_debt"], ratios["equity"])
    ratios["asset_turnover"] = safe_div(ratios["revenue"], ratios["total_assets"])
    ratios["rd_intensity"] = safe_div(ratios["rd_expense"], ratios["revenue"])
    ratios["capex_intensity"] = safe_div(ratios["capex"], ratios["revenue"])
    ratios["ebitda"] = ratios["operating_income"] + ratios["da"].fillna(0)
    universe["sic2"] = universe["sic"].astype("string").str.zfill(4).str[:2]
    ratios = ratios.merge(universe[["ticker", "sic2"]].drop_duplicates("ticker"),
                          on="ticker", how="left")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("firm_year_financials", financials),
                        ("firm_year_financials_ratios", ratios)):
        destination = args.output_dir / f"{name}.parquet"
        table.sort_values(["ticker", "year"]).to_parquet(destination, index=False)
        print(f"-> {destination} ({len(table):,} filas, {len(table.columns)} cols)")

    coverage = {c: f"{ratios[c].notna().mean()*100:.0f}%"
                for c in ["revenue", "rd_expense", "capex", "sga_expense",
                          "operating_income", "total_assets", "equity",
                          "long_term_debt", "shares_out", "next_revenue_yoy"]}
    print("cobertura:", coverage)


if __name__ == "__main__":
    main()
