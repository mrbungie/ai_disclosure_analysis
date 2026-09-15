"""
scripts/analytics/fills.py — analytics-side fills for gold columns that gold
leaves null.

Gold covariates/targets never impute: a value is null when the entity does
not report the input. The analyses below keep their specification by filling
here, explicitly, from the raw components gold writes next to the derived
column.

ROIC - WACC with unreported long-term debt read as zero debt
(`roic_wacc_debt_as_zero`): same formulas as
scripts/gold/financials/build_roic_wacc.py (book-weighted WACC), from the
components covariates/firm_year/financials (firm-year) and
covariates|targets/call/financials (call, `_pre`/`_post` suffix) carry: nopat, equity, long_term_debt,
cost_of_equity, cost_of_debt, effective_tax_rate. `cost_of_debt` already
carries its own rf + spread fallback, which does not depend on debt being
reported.
"""
from __future__ import annotations

import pandas as pd

VALUE_COMPONENTS = ["nopat", "equity", "long_term_debt", "cost_of_equity", "cost_of_debt", "effective_tax_rate"]


def roic_wacc_debt_as_zero(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """`roic`, `wacc` and `roic_minus_wacc` (each + `suffix`) with a missing
    `long_term_debt` read as 0."""
    col = lambda name: df[f"{name}{suffix}"]  # noqa: E731
    debt = col("long_term_debt").fillna(0)
    equity = col("equity")
    invested = debt + equity
    roic = col("nopat") / invested.where(invested > 0)
    total = equity + debt
    weight_equity = equity / total.where(total > 0)
    wacc = (weight_equity * col("cost_of_equity")
            + (1 - weight_equity) * col("cost_of_debt") * (1 - col("effective_tax_rate")))
    return pd.DataFrame({f"roic{suffix}": roic, f"wacc{suffix}": wacc,
                         f"roic_minus_wacc{suffix}": roic - wacc}, index=df.index)
