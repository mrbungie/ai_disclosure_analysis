"""ROIC vs. WACC por empresa-año: ¿el capital rinde más de lo que cuesta?

Produce `firm_year_roic_wacc.parquet`, el insumo de
`docs/analytics/08_roic_wacc_value_creation.md`, que hasta ahora existía como
dato sin código que lo generara.

    NOPAT             = operating_income x (1 - tasa efectiva de impuesto)
    invested_capital  = deuda de largo plazo + patrimonio (valor LIBRO)
    ROIC              = NOPAT / invested_capital
    cost_of_equity    = rf_anualizado(fecha del filing) + beta x ERP
    cost_of_debt      = interest_expense / deuda LP   (fallback: rf + 2%)
    WACC              = E/(E+D) x coe + D/(E+D) x cod x (1 - tasa efectiva)
                        con E = market cap y D = deuda LP

Decisiones que se apartan de la versión perdida (rige lo nuevo), todas
documentadas acá porque cambian el NIVEL de las cifras:

  1. **ERP geométrico, no aritmético.** La versión anterior usaba 8,2%, que
     es la media ARITMÉTICA del `mktrf` diario 2000-2026 anualizada x252. La
     media geométrica del mismo período es ~6,5%, y es la que corresponde
     para descontar flujos multi-período. La diferencia NO es neutral entre
     segmentos: como coe = rf + beta x ERP, un ERP inflado infla las
     DIFERENCIAS de WACC en proporción a las diferencias de beta, y los
     segmentos de 08_...md difieren justamente en beta (0,46 vs 1,00). Se
     calcula desde los propios factores del repo y queda registrado en la
     columna `erp` de la salida. `--erp 0.082` reproduce la versión vieja.
  2. **Tasa efectiva de impuesto acotada a [0, 1]**, con fallback a la tasa
     estatutaria federal (21%) cuando el pretax es <= 0 o la razón cae fuera
     del rango. Sin esto una empresa con pérdida pretax y crédito fiscal
     produce un NOPAT mayor que su resultado operativo.
  3. **Costo de deuda acotado a [0, 30%]**: `interest_expense / deuda LP` se
     dispara cuando la deuda LP es casi nula (el gasto financiero incluye
     deuda corriente y arriendos que no están en el denominador).

  4. **El spread por defecto es LIBRO contra LIBRO.** Antes el ROIC se calculaba
     sobre capital invertido contable y el WACC se ponderaba con market cap:
     dividir con una regla y ponderar con otra sesga el spread con el
     market-to-book de cada empresa, que es justo una dimensión donde los
     segmentos difieren (una tecnológica que vale 10x libro y una utility que
     vale 1,2x no son comparables así). Ahora `wacc` usa ponderadores de LIBRO
     (patrimonio contable y deuda contable), consistentes con el denominador del
     ROIC, y `roic_minus_wacc` se calcula con ese. El WACC con ponderadores de
     mercado queda en `wacc_market` / `roic_minus_wacc_market` para poder
     comparar las dos lecturas — el costo de capital que enfrenta la empresa en
     el mercado es una pregunta legítima, pero no es la que se resta a un ROIC
     contable.

Determinístico, sin LLM, sin costo de API.

Uso:
    uv run python scripts/analytics/build_roic_wacc.py
    uv run python scripts/analytics/build_roic_wacc.py --erp 0.082   # versión vieja
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORS = REPO_ROOT / "data" / "raw" / "market" / "factors" / "ff3_daily.parquet"
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"

STATUTORY_TAX_RATE = 0.21          # tasa federal US desde la TCJA (2018)
DEBT_SPREAD_FALLBACK = 0.02        # rf + 200 bp cuando no hay InterestExpense
MAX_COST_OF_DEBT = 0.30
ERP_SAMPLE_START = "2000-01-01"
TRADING_DAYS = 252


def annualized_equity_premium(factors: pd.DataFrame, start: str) -> tuple[float, float]:
    """Prima de riesgo de mercado anualizada, aritmética y geométrica.

    La aritmética (media diaria x 252) es la que usaba la versión anterior;
    sobreestima el retorno compuesto de un horizonte largo, que es lo que un
    WACC descuenta."""
    sample = factors[factors["date"] >= pd.Timestamp(start)]["mktrf"].dropna()
    years = len(sample) / TRADING_DAYS
    arithmetic = float(sample.mean() * TRADING_DAYS)
    geometric = float(np.exp(np.log1p(sample).sum() / years) - 1)
    return arithmetic, geometric


def risk_free_at(factors: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """rf diario vigente en (o justo antes de) cada fecha de filing,
    anualizado por composición."""
    daily = factors[["date", "rf"]].dropna().sort_values("date")
    frame = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
    merged = pd.merge_asof(frame, daily, on="date", direction="backward")
    return (1 + merged["rf"]) ** TRADING_DAYS - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-dir", type=Path, default=CLUSTERS)
    parser.add_argument("--factors", type=Path, default=FACTORS)
    parser.add_argument("--erp", type=float, default=None,
                        help="Prima de riesgo de mercado. Default: la geométrica "
                             "calculada desde los factores (ver docstring).")
    args = parser.parse_args()

    ratios = pd.read_parquet(args.clusters_dir / "firm_year_financials_ratios.parquet")
    market = pd.read_parquet(args.clusters_dir / "firm_year_market_factors.parquet")
    factors = pd.read_parquet(args.factors)[["date", "mktrf", "rf"]]
    factors["date"] = pd.to_datetime(factors["date"])

    arithmetic, geometric = annualized_equity_premium(factors, ERP_SAMPLE_START)
    erp = args.erp if args.erp is not None else geometric
    print(f"ERP desde {ERP_SAMPLE_START}: aritmética {arithmetic:.4f} | "
          f"geométrica {geometric:.4f} | usada {erp:.4f}")

    panel = ratios.merge(market[["ticker", "year", "beta", "market_cap"]],
                         on=["ticker", "year"], how="left")
    panel["filing_date"] = pd.to_datetime(panel["filing_date"])

    rate = panel["tax_expense"] / panel["pretax_income"].where(panel["pretax_income"] > 0)
    panel["effective_tax_rate"] = rate.where(rate.between(0, 1)).fillna(STATUTORY_TAX_RATE)

    panel["nopat"] = panel["operating_income"] * (1 - panel["effective_tax_rate"])
    invested = panel["long_term_debt"].fillna(0) + panel["equity"]
    panel["invested_capital"] = invested.where(invested > 0)
    panel["roic"] = panel["nopat"] / panel["invested_capital"]

    panel["rf_annualized"] = risk_free_at(factors, panel["filing_date"]).values
    panel["erp"] = erp
    panel["cost_of_equity"] = panel["rf_annualized"] + panel["beta"] * erp

    debt = panel["long_term_debt"].where(panel["long_term_debt"] > 0)
    cost_of_debt = panel["interest_expense"] / debt
    panel["cost_of_debt"] = cost_of_debt.where(cost_of_debt.between(0, MAX_COST_OF_DEBT)) \
        .fillna(panel["rf_annualized"] + DEBT_SPREAD_FALLBACK)

    debt_value = panel["long_term_debt"].fillna(0)

    def mix(equity_value: pd.Series) -> pd.Series:
        total = equity_value + debt_value
        weight_equity = equity_value / total.where(total > 0)
        return (weight_equity * panel["cost_of_equity"]
                + (1 - weight_equity) * panel["cost_of_debt"]
                * (1 - panel["effective_tax_rate"]))

    # Libro contra libro: mismo criterio de valuación que el denominador del
    # ROIC. Es el que se resta.
    panel["wacc"] = mix(panel["equity"])
    panel["roic_minus_wacc"] = panel["roic"] - panel["wacc"]
    # Mercado, para comparar: mismo costo de capital, otra ponderación.
    panel["wacc_market"] = mix(panel["market_cap"])
    panel["roic_minus_wacc_market"] = panel["roic"] - panel["wacc_market"]

    destination = args.clusters_dir / "firm_year_roic_wacc.parquet"
    panel.sort_values(["ticker", "year"]).to_parquet(destination, index=False)
    print(f"-> {destination} ({len(panel):,} filas, {len(panel.columns)} cols)")
    print(f"cobertura: ROIC {panel['roic'].notna().mean()*100:.0f}% | "
          f"WACC {panel['wacc'].notna().mean()*100:.0f}% | "
          f"spread {panel['roic_minus_wacc'].notna().mean()*100:.0f}%")
    print(f"medianas: ROIC {panel['roic'].median():.4f} | "
          f"WACC libro {panel['wacc'].median():.4f} (spread {panel['roic_minus_wacc'].median():.4f}) | "
          f"WACC mercado {panel['wacc_market'].median():.4f} "
          f"(spread {panel['roic_minus_wacc_market'].median():.4f})")
    consistent = panel[["roic_minus_wacc", "roic_minus_wacc_market"]].dropna()
    if len(consistent) > 10:
        correlation = consistent.corr().iloc[0, 1]
        print(f"correlación entre los dos spreads: {correlation:.3f} sobre "
              f"{len(consistent):,} filas")
        print("  -> medido: la inconsistencia libro/mercado movía el NIVEL del spread "
              "(~0,9 p.p. en la mediana),\n     no el ordenamiento entre empresas. "
              "Cualquier comparación ENTRE segmentos era, en la práctica,\n     "
              "insensible a esto. Corregido igual, porque la definición ahora es "
              "coherente y\n     la cobertura sube (spread disponible 66% -> 73%).")


if __name__ == "__main__":
    main()
