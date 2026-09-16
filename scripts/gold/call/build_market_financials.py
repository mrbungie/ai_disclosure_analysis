"""Call market and accounting families: price windows around each call and the
10-K snapshots known before (and first filed after) it.

Price windows (`market_windows`, trading days of the firm's own series, idx =
first trading day on or after the call date):
  price_pre             close at idx-1
  return60              adjusted return from idx-60 to idx-1
  beta_pre              one-factor beta over [idx-252, idx), at least 120 days
  beta_post_63/126/252  one-factor beta over [idx, idx+63/126/252), at least
                        60/120/120 days
  ncskew_pre, duvol_pre             market-model residuals over [idx-252, idx-21), at least 90 days
  ncskew_post_105d, duvol_post_105d over [idx+21, idx+126), at least 90 days
  ncskew_pre_63, ncskew_post_63     over [idx-63, idx) and [idx, idx+63), at least 60 days

Accounting snapshots (covariates/firm_year/financials and next_revenue_yoy
from targets/firm_year/financials, one row per filed 10-K):
  pre   last 10-K filed strictly before the call: accession_number,
        filing_date_pt, revenue, operating_income, total_assets,
        operating_margin, asset_turnover, roa, shares_out, and the `_pre`
        fundamentals (rd_intensity, gross_margin, revenue_yoy = the revenue
        growth that 10-K reports over its prior fiscal year, ROIC - WACC and
        its components with rf at that filing date and beta_pre)
  post  first 10-K filed strictly after the call: the `_post` fundamentals
        (ROIC - WACC with beta_post_126 and rf 125 trading days after the call;
        next_revenue_yoy_post = growth of the fiscal year after that 10-K)
  debt_to_equity, liabilities_to_assets   instant XBRL facts of the pre 10-K
Valuation: log_market_cap = log(price_pre x shares_out), ps_ratio_pre; and
log_market_cap_post, ps_ratio_post with the close at idx+125 and the post 10-K.

Outputs: covariates/call/{market, financials}, targets/call/{market, financials}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "financials"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
from build_roic_wacc import (DEBT_SPREAD_FALLBACK, ERP_SAMPLE_START, MAX_COST_OF_DEBT, STATUTORY_TAX_RATE,  # noqa: E402
                             annualized_equity_premium, risk_free_at)
from market_windows import beta, duvol, event_index, load_factors, load_prices, ncskew, one_factor_ols, with_factors  # noqa: E402

BUILDER = "scripts/gold/call/build_market_financials.py"
BETA_MIN_OBS = 120
BETA_MIN_OBS_63 = 60  # same ~95% completeness ratio as BETA_MIN_OBS/126, scaled to a 63-day window
CRASH_GAP, CRASH_PRE, CRASH_POST, CRASH_MIN_OBS = 21, 231, 105, 90
NCSKEW63_MIN_OBS = 60
POST_WINDOW_END_OFFSET = 125  # last trading day of beta_post_126's [idx, idx+126) window
VALUE_COMPONENTS = ["nopat", "equity", "long_term_debt", "cost_of_equity", "cost_of_debt", "effective_tax_rate"]
LEVERAGE_CONCEPTS = {
    "assets": ["us-gaap:Assets"],
    "liabilities": ["us-gaap:Liabilities"],
    "debt": ["us-gaap:LongTermDebtNoncurrent", "us-gaap:LongTermDebt"],
    "equity": ["us-gaap:StockholdersEquity", "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}

COV_MARKET = ["price_pre", "return60", "beta_pre", "ncskew_pre", "duvol_pre", "ncskew_pre_63", "log_market_cap",
              "ps_ratio_pre"]
TGT_MARKET = ["beta_post_63", "beta_post_126", "beta_post_252", "ncskew_post_63", "ncskew_post_105d",
              "duvol_post_105d", "log_market_cap_post", "ps_ratio_post"]
COV_FINANCIALS = (["accession_number", "filing_date_pt", "revenue", "operating_income", "total_assets", "operating_margin",
                   "asset_turnover", "roa", "shares_out", "debt_to_equity", "liabilities_to_assets",
                   "rd_intensity_pre", "gross_margin_pre", "revenue_yoy_pre", "roic_minus_wacc_pre"]
                  + [f"{c}_pre" for c in VALUE_COMPONENTS])
TGT_FINANCIALS = (["rd_intensity_post", "gross_margin_post", "next_revenue_yoy_post", "asset_turnover_post",
                   "roic_minus_wacc_post"] + [f"{c}_post" for c in VALUE_COMPONENTS])


# ---- price windows -------------------------------------------------------------

def price_windows(panel: pd.DataFrame) -> pd.DataFrame:
    """Every price-window measure of a call (one row per call of `panel`)."""
    factors = load_factors()
    prices = load_prices()
    # post-call close: trading days with a close (the adjusted series may be shorter)
    closes = {t: d[["date", "close"]].assign(date=lambda x: pd.to_datetime(x["date"]))
              .dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
              for t, d in L.read("silver.market_prices").to_pandas().groupby("ticker")}
    rows = []
    for row in panel[["call_accession_number", "ticker", "fecha"]].itertuples(index=False):
        rec = {"call_accession_number": row.call_accession_number, "price_post": np.nan}
        pr = prices.get(row.ticker)
        if pr is not None:
            idx = event_index(pr, row.fecha)
            rec.update(beta_windows(pr, factors, idx))
            rec.update(crash_windows(pr, factors, idx))
            rec.update(ncskew63_windows(pr, factors, idx))
        cl = closes.get(row.ticker)
        if cl is not None:
            end = event_index(cl, row.fecha) + POST_WINDOW_END_OFFSET
            if 0 <= end < len(cl):
                rec["price_post"] = float(cl["close"].iloc[end])
        rows.append(rec)
    return panel.merge(pd.DataFrame(rows), on="call_accession_number", how="left", validate="one_to_one")


def beta_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    out = {"beta_pre": np.nan, "beta_post_63": np.nan, "beta_post_126": np.nan, "beta_post_252": np.nan,
           "price_pre": np.nan, "return60": np.nan}
    if 0 <= idx - 1 and idx < len(pr):
        out["price_pre"] = float(pr["close"].iloc[idx - 1])
        if idx - 60 >= 0 and pr["adj_close"].iloc[idx - 60] > 0:
            out["return60"] = float(pr["adj_close"].iloc[idx - 1] / pr["adj_close"].iloc[idx - 60] - 1)
        if idx - 252 >= 0:
            out["beta_pre"] = beta(with_factors(pr.iloc[idx - 252:idx], factors), BETA_MIN_OBS)
        for days, min_obs in ((63, BETA_MIN_OBS_63), (126, BETA_MIN_OBS), (252, BETA_MIN_OBS)):
            if idx + days < len(pr):
                out[f"beta_post_{days}"] = beta(with_factors(pr.iloc[idx:idx + days], factors), min_obs)
    return out


def _residuals(pr: pd.DataFrame, factors: pd.DataFrame, start: int, end: int, min_obs: int) -> np.ndarray | None:
    win = with_factors(pr.iloc[start:end], factors)
    if len(win) < min_obs:
        return None
    return one_factor_ols(win)[2]


def crash_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    out = {"ncskew_pre": np.nan, "duvol_pre": np.nan, "ncskew_post_105d": np.nan, "duvol_post_105d": np.nan}
    pre_start, pre_end = idx - CRASH_GAP - CRASH_PRE, idx - CRASH_GAP
    post_start, post_end = idx + CRASH_GAP, idx + CRASH_GAP + CRASH_POST
    if pre_start >= 0:
        res = _residuals(pr, factors, pre_start, pre_end, CRASH_MIN_OBS)
        if res is not None:
            out["ncskew_pre"], out["duvol_pre"] = ncskew(res, CRASH_MIN_OBS), duvol(res, CRASH_MIN_OBS)
    if post_end <= len(pr):
        res = _residuals(pr, factors, post_start, post_end, CRASH_MIN_OBS)
        if res is not None:
            out["ncskew_post_105d"], out["duvol_post_105d"] = ncskew(res, CRASH_MIN_OBS), duvol(res, CRASH_MIN_OBS)
    return out


def ncskew63_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    out = {"ncskew_pre_63": np.nan, "ncskew_post_63": np.nan}
    if idx - 63 >= 0:
        res = _residuals(pr, factors, idx - 63, idx, NCSKEW63_MIN_OBS)
        out["ncskew_pre_63"] = ncskew(res, NCSKEW63_MIN_OBS) if res is not None else np.nan
    if idx + 63 < len(pr):
        res = _residuals(pr, factors, idx, idx + 63, NCSKEW63_MIN_OBS)
        out["ncskew_post_63"] = ncskew(res, NCSKEW63_MIN_OBS) if res is not None else np.nan
    return out


# ---- accounting snapshots --------------------------------------------------------

def attach_snapshot(panel: pd.DataFrame, filings: pd.DataFrame, direction: str, suffix: str) -> pd.DataFrame:
    """The 10-K of `filings` filed strictly before (backward) or after
    (forward) each call, its columns suffixed; `panel` row order kept."""
    left = panel[["ticker", "fecha"]].assign(fecha=lambda d: pd.to_datetime(d["fecha"]).astype("datetime64[ns]"))
    left = left.sort_values(["fecha", "ticker"])
    right = filings.assign(filing_date=lambda d: pd.to_datetime(d["filing_date"]).astype("datetime64[ns]"))
    right = right.sort_values(["filing_date", "ticker"])
    merged = pd.merge_asof(left, right, left_on="fecha", right_on="filing_date", by="ticker",
                           direction=direction, allow_exact_matches=False)
    merged.index = left.index
    merged = merged.reindex(panel.index).drop(columns=["ticker", "fecha"]).add_suffix(f"_{suffix}")
    return pd.concat([panel, merged], axis=1)


def roic_minus_wacc(rf: pd.Series, beta_: pd.Series, erp: float, operating_income: pd.Series,
                    pretax_income: pd.Series, tax_expense: pd.Series, equity: pd.Series,
                    long_term_debt: pd.Series, interest_expense: pd.Series) -> pd.DataFrame:
    """The formulas of `build_roic_wacc.value_creation` (book-weighted WACC) on
    one snapshot. Null when long-term debt is not reported."""
    rate = tax_expense / pretax_income.where(pretax_income > 0)
    effective_tax_rate = rate.where(rate.between(0, 1)).fillna(STATUTORY_TAX_RATE)
    nopat = operating_income * (1 - effective_tax_rate)
    invested = long_term_debt + equity
    roic = nopat / invested.where(invested > 0)
    cost_of_equity = rf + beta_ * erp
    debt = long_term_debt.where(long_term_debt > 0)
    cost_of_debt_raw = interest_expense / debt
    cost_of_debt = cost_of_debt_raw.where(cost_of_debt_raw.between(0, MAX_COST_OF_DEBT)).fillna(rf + DEBT_SPREAD_FALLBACK)
    total = equity + long_term_debt
    weight_equity = equity / total.where(total > 0)
    wacc = weight_equity * cost_of_equity + (1 - weight_equity) * cost_of_debt * (1 - effective_tax_rate)
    return pd.DataFrame({"roic_minus_wacc": roic - wacc, "nopat": nopat, "equity": equity,
                         "long_term_debt": long_term_debt, "cost_of_equity": cost_of_equity,
                         "cost_of_debt": cost_of_debt, "effective_tax_rate": effective_tax_rate})


def select_instant_values(facts: pd.DataFrame, mappings: dict[str, list[str]]) -> pd.DataFrame:
    """One row per accession: for each metric the first concept of its list with
    a fact, at that concept's latest period_end."""
    rows: list[dict[str, object]] = []
    facts = facts.copy()
    facts["period_end"] = pd.to_datetime(facts["period_end"])
    for accession, group in facts.groupby("accession_number"):
        row: dict[str, object] = {"accession_number": accession}
        for metric, tags in mappings.items():
            eligible = group[group["concept"].isin(tags)].sort_values("period_end", ascending=False)
            for tag in tags:
                values = eligible[eligible["concept"] == tag]
                if not values.empty:
                    row[metric] = values.iloc[0]["numeric_value"]
                    break
        rows.append(row)
    return pd.DataFrame(rows)


def attach_leverage(panel: pd.DataFrame) -> pd.DataFrame:
    accessions = pl.from_pandas(panel[["accession_number"]].dropna().drop_duplicates())
    tags = [tag for tags in LEVERAGE_CONCEPTS.values() for tag in tags]
    facts = (L.scan("bronze.xbrl_facts")
             .filter(pl.col("has_dimensions").not_() & (pl.col("period_type") == "instant")
                     & pl.col("concept").is_in(tags))
             .select("accession_number", "concept", "numeric_value", "period_end")
             .join(accessions.lazy(), on="accession_number", how="semi", maintain_order="left")
             .collect().to_pandas())
    values = select_instant_values(facts, LEVERAGE_CONCEPTS)
    out = panel.merge(values.add_prefix("_lev_").rename(columns={"_lev_accession_number": "accession_number"}),
                      on="accession_number", how="left")
    out["debt_to_equity"] = out["_lev_debt"] / out["_lev_equity"]
    out["liabilities_to_assets"] = out["_lev_liabilities"] / out["_lev_assets"]
    return out.drop(columns=[c for c in out.columns if c.startswith("_lev_")])


def main() -> None:
    panel = L.read_gold("call", spine_columns=L.GOLD_SPINE_COLUMNS["call"])
    panel = price_windows(panel)

    filings = L.read_gold("firm_year",
                          ("covariates", "financials", ["filing_date", "accession_number", "revenue", "gross_margin",
                                                        "rd_intensity", "operating_income", "equity", "long_term_debt",
                                                        "interest_expense", "pretax_income", "tax_expense", "shares_out",
                                                        "total_assets", "operating_margin", "asset_turnover", "roa",
                                                        "revenue_yoy"]),
                          ("targets", "financials", ["next_revenue_yoy"]),
                          spine_columns=["id", "ticker"]).drop(columns="id").dropna(subset=["filing_date"])
    panel = attach_snapshot(panel, filings, "backward", "pre")
    panel = attach_snapshot(panel, filings, "forward", "post")
    panel = panel.rename(columns={f"{c}_pre": c for c in ["accession_number", "revenue", "operating_income", "total_assets",
                                                          "operating_margin", "asset_turnover", "roa", "shares_out"]})
    panel = panel.rename(columns={"filing_date_pre": "filing_date_pt"})
    panel["log_market_cap"] = np.log(panel["price_pre"] * panel["shares_out"])
    panel["log_market_cap_post"] = np.log(panel["price_post"] * panel["shares_out_post"])
    panel["ps_ratio_pre"] = panel["price_pre"] * panel["shares_out"] / panel["revenue"]
    panel["ps_ratio_post"] = panel["price_post"] * panel["shares_out_post"] / panel["revenue_post"]

    factors = load_factors()
    _, erp = annualized_equity_premium(factors, ERP_SAMPLE_START)
    rf_pre = risk_free_at(factors, panel["filing_date_pt"])
    rf_post = risk_free_at(factors, panel["fecha"] + pd.to_timedelta(POST_WINDOW_END_OFFSET * 7 / 5, unit="D"))
    value_pre = roic_minus_wacc(rf_pre, panel["beta_pre"], erp, panel["operating_income"], panel["pretax_income_pre"],
                                panel["tax_expense_pre"], panel["equity_pre"], panel["long_term_debt_pre"],
                                panel["interest_expense_pre"])
    value_post = roic_minus_wacc(rf_post, panel["beta_post_126"], erp, panel["operating_income_post"],
                                 panel["pretax_income_post"], panel["tax_expense_post"], panel["equity_post"],
                                 panel["long_term_debt_post"], panel["interest_expense_post"])
    for c in ["roic_minus_wacc"] + VALUE_COMPONENTS:
        panel[f"{c}_pre"], panel[f"{c}_post"] = value_pre[c], value_post[c]

    panel = attach_leverage(panel)
    spine = L.GOLD_SPINE_COLUMNS["call"]
    L.write_gold("covariates", "call", "market", panel[spine + COV_MARKET], builder=BUILDER)
    L.write_gold("covariates", "call", "financials", panel[spine + COV_FINANCIALS], builder=BUILDER)
    L.write_gold("targets", "call", "market", panel[spine + TGT_MARKET], builder=BUILDER)
    L.write_gold("targets", "call", "financials", panel[spine + TGT_FINANCIALS], builder=BUILDER)
    print(f"ERP geométrico usado: {erp:.4f}")


if __name__ == "__main__":
    main()
