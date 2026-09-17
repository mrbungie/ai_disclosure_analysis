"""Call market and accounting families: price windows around each call and the
10-K snapshots known before (and first filed after) it.

Price windows (`market_windows`, trading days of the firm's own series, idx =
first trading day on or after the call date). Post-call windows carry the gaps
the thesis states (thesis_document/thesis.qmd @tbl-market-outcomes): beta uses
[+21, +N], valuation and return-distribution metrics use [+2, +63]:
  price_pre             close at idx-1
  return60              adjusted return from idx-60 to idx-1
  beta_pre              one-factor beta over [idx-252, idx), at least 120 days
                        (kept for build_roic_wacc / value_creation's cost of
                        equity; not the pre-call regressor of the symmetric
                        beta design below)
  beta_pre_63           one-factor beta over [idx-63, idx-21), at least 40 days
                        (42-day window, same length and announcement-period
                        exclusion as beta_post_63)
  beta_pre_252          one-factor beta over [idx-252, idx-21), at least 219
                        days (231-day window, same length as beta_post_252's
                        [idx+21, idx+252))
  beta_post_63/126/252  one-factor beta over [idx+21, idx+63/126/252), at
                        least 40/100/219 days (95% of the 42/105/231-day window)
  ncskew_pre, duvol_pre          market-model residuals over [idx-252, idx-21), at least 90 days
                                  (pre-call window, not stated by the thesis; kept for comparison
                                  against the mirrored window below, not the headline pre-control)
  ncskew_pre_63                  over [idx-63, idx), at least 60 days (pre-call, unchanged)
  ncskew_pre_61, duvol_pre_61    over [idx-63, idx-2), at least 58 days: the mirror of
                                  ncskew_post_63/duvol_post_63's own window, same 61-day span and
                                  2-day announcement gap, so post-on-pre is the same construct on
                                  symmetric windows
  ncskew_post_63, duvol_post_63  over [idx+2, idx+63), at least 58 days (95% of 61)

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
  debt_to_equity, liabilities_to_assets   from the pre 10-K's fs-sourced
        long_term_debt/equity/total_assets (already attached from
        covariates/firm_year/financials); liabilities_to_assets is
        reconstructed as (total_assets - equity) / total_assets (fs has no
        native total-liabilities row)
Valuation: log_market_cap = log(fs's own market_cap at price_pre's trading
day, NOT price_pre x shares_out -- fixes the pre-split understatement in
data/results/fs_validation/summary.md), ps_ratio_pre = market_cap / revenue;
and log_market_cap_post, ps_ratio_post with market_cap at idx+62 (the last
trading day of the thesis's [+2, +63) valuation window) and the post 10-K's
revenue. log_market_cap_pre2/ps_ratio_pre2 mirror the [+2, +63) window's own
2-day start gap on the pre-call side, at idx-2 (against the same pre-call 10-K
revenue as ps_ratio_pre) -- kept alongside log_market_cap/ps_ratio_pre (idx-1,
unchanged) rather than replacing them.

Classic-literature families (docs/plans/cap5_classic_specs.md has the full
citations and windows):
  beta_shift_pre, beta_shift_delta(_se/_t), beta_shift_n_pre/n_post
        Brenner (1979) pooled beta-shift test: one OLS of daily excess return
        on mktrf, a POST dummy and POST x mktrf over the pooled sample
        [idx-63, idx-21) union [idx+21, idx+63); `beta_shift_pre` is the
        slope on mktrf (the pre-call beta of this design), `beta_shift_delta`
        the slope on POST x mktrf (the shift), min 40 obs on each side.
  car_m1_p1, car_p2_p63
        Brown & Warner (1985) / MacKinlay (1997) cumulative abnormal return:
        a market model (mktrf) estimated over [idx-250, idx-30), at least
        120 days, gives (alpha, beta); CAR is the sum of daily abnormal
        returns (actual excess minus that model's prediction) over
        [idx-1, idx+1] (3 days, all required) and [idx+2, idx+63) (61 days,
        at least 40).
  ncskew_wk_pre/post, duvol_wk_pre/post, sigma_wk_pre, ret_wk_pre,
  avg_volume_wk_pre, avg_volume_wk_prior26, n_weeks_pre/post
        Chen, Hong & Stein (2001) / Kim, Li & Zhang (2011) weekly crash-risk
        design: firm-specific weekly return W = residual of an expanded
        market model (mktrf at lags/leads -2..+2) over 26 TRADING weeks
        (calendar weeks with at least one trading day, in traded order) after
        the call's own week (pre: the 26 weeks before it); NCSKEW/DUVOL,
        SIGMA (sd of W) and RET (100 x mean W) all read this pre-window W.
        `avg_volume_wk_pre`/`avg_volume_wk_prior26` (mean weekly share volume
        over the pre window and the 26 weeks before it) let the analytics
        layer finish KLZ's DTURN control once the pre-call 10-K's shares
        outstanding is known.

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
from market_windows import (beta, duvol, event_index, expanded_market_residuals, load_factors, load_prices,  # noqa: E402
                            ncskew, one_factor_ols, to_weekly, week_index, with_factors)

BUILDER = "scripts/gold/call/build_market_financials.py"
BETA_MIN_OBS = 120
# Post-call beta windows carry a 21-trading-day gap after the call
# (thesis_document/thesis.qmd @tbl-call-beta-paper / @tbl-market-outcomes:
# beta_post_63/126/252 = [+21,+63]/[+21,+126]/[+21,+252]), so each window is
# shorter than its name by 21 days (42/105/231 obs). Min-observation floors
# keep the same ~95% completeness ratio the no-gap windows used.
BETA_POST_GAP = 21
BETA_MIN_OBS_POST_63 = 40    # 95% of 42
BETA_MIN_OBS_POST_126 = 100  # 95% of 105
BETA_MIN_OBS_POST_252 = 219  # 95% of 231
# Symmetric pre-call beta windows, mirrored on the post-call ones so a
# pre/post comparison holds both window length and the same 21-day
# announcement-period exclusion fixed: beta_pre_63 = [-63, -21) matches
# beta_post_63 = [+21, +63) (42 trading days each); beta_pre_252 =
# [-252, -21) matches beta_post_252's 231-day span (`beta_post_252` itself
# keeps the +21 gap, so [+21, +252) is 231 days -- same length as
# [-252, -21)). `beta_pre` (== [-252, 0), no gap, at least 120 obs) is left
# unchanged: build_roic_wacc / value_creation still key their cost of equity
# off it, and CLAUDE.md's "one construct, one formula" rule means it should
# not be redefined to feed one particular regression design.
BETA_PRE_GAP = 21
BETA_MIN_OBS_PRE_63 = 40    # 95% of 42
BETA_MIN_OBS_PRE_252 = 219  # 95% of 231
CRASH_GAP, CRASH_PRE, CRASH_MIN_OBS = 21, 231, 90
# ncskew_post_63/duvol_post_63 carry a 2-trading-day gap after the call
# (thesis @tbl-market-outcomes: return-distribution and valuation metrics use
# [+2,+63]), so the post window is 61 days; the pre window (unspecified by the
# thesis) is left at its original 63-day, no-gap span.
NCSKEW_POST_GAP = 2
NCSKEW_PRE63_MIN_OBS = 60   # 95% of 63
NCSKEW_POST63_MIN_OBS = 58  # 95% of 61
# Last trading day of the [+2,+63) window used for log_market_cap_post /
# ps_ratio_post (thesis @tbl-market-outcomes: valuation metrics = [+2,+63]).
VALUATION_WINDOW_LAST_DAY = 62
# Mirrored pre-call NCSKEW/DUVOL window, [idx-63, idx-2): same 61-day span and
# 2-trading-day announcement gap as ncskew_post_63/duvol_post_63's
# [idx+2, idx+63), so post-on-pre is an ANCOVA on the same construct over
# symmetric windows rather than the old, longer [-252,-21) residual window
# (kept as `ncskew_pre`/`duvol_pre` for comparison -- CLAUDE.md's "one
# construct, one formula" is not violated: the two windows answer different
# questions, one asymmetric-history and one call-mirrored).
NCSKEW_PRE61_MIN_OBS = 58  # 95% of 61, same ratio as NCSKEW_POST63_MIN_OBS
# Mirrored pre-call valuation day for log_market_cap_pre2 / ps_ratio_pre2:
# idx-2, the pre-call mirror of the [+2,+63) window's own start-side gap
# (VALUATION_WINDOW_LAST_DAY reads the END of that window, idx+62; the START
# offset, 2 trading days, is what is mirrored here rather than the window's
# length, since a level is a single point, not an average over a window).
# `log_market_cap`/`ps_ratio_pre` (idx-1, the close immediately before the
# call) are kept unchanged for comparison.
VALUATION_PRE_DAY = 2

# ---- classic-literature families (docs/plans/cap5_classic_specs.md) ------
# Beta-shift (Brenner 1979): pooled pre/post OLS over the same [-63,-21) /
# [+21,+63) windows as beta_pre_63/beta_post_63, min 40 obs on each side.
BETA_SHIFT_PRE_START, BETA_SHIFT_PRE_END = 63, 21     # [idx-63, idx-21)
BETA_SHIFT_POST_START, BETA_SHIFT_POST_END = 21, 63   # [idx+21, idx+63)
BETA_SHIFT_MIN_OBS_SIDE = 40
# CAR (Brown & Warner 1985 / MacKinlay 1997): market-model estimation window
# [idx-250, idx-30), at least 120 days; CAR[-1,+1] requires all 3 days,
# CAR[+2,+63) (61 days) requires at least 40 (~65%, consistent with this
# module's other 61/63-day floors).
CAR_EST_START, CAR_EST_END, CAR_EST_MIN_OBS = 250, 30, 120
CAR_M1P1_MIN_OBS = 3
CAR_P2P63_MIN_OBS = 40
# Weekly crash risk (Chen, Hong & Stein 2001; Kim, Li & Zhang 2011): 26
# TRADING weeks each side of the call's own week, min 20; the "prior 26
# weeks" window for KLZ's DTURN control is the 26 weeks before that.
CRASH_WK_WINDOW = 26
CRASH_WK_MIN_OBS = 20
CRASH_WK_GAP = 1  # post window starts the week AFTER the call's own week
CRASH_WK_PRIOR_MIN_OBS = 15
POST_WINDOW_END_OFFSET = 125  # last trading day of beta_post_126's [idx+21, idx+126) window (unchanged: only the start moved)
TTM_WINDOW_END_OFFSET = 251   # last trading day of beta_post_252's [idx+21, idx+252) window (unchanged: only the start moved)
TTM_QUARTERS = 4
VALUE_COMPONENTS = ["nopat", "equity", "long_term_debt", "cost_of_equity", "cost_of_debt", "effective_tax_rate"]

COV_MARKET = ["price_pre", "return60", "beta_pre", "beta_pre_63", "beta_pre_252", "ncskew_pre", "duvol_pre",
              "ncskew_pre_63", "ncskew_pre_61", "duvol_pre_61", "log_market_cap", "ps_ratio_pre",
              "log_market_cap_pre2", "ps_ratio_pre2",
              "beta_shift_pre", "beta_shift_n_pre", "beta_shift_n_post",
              "ncskew_wk_pre", "duvol_wk_pre", "sigma_wk_pre", "ret_wk_pre",
              "avg_volume_wk_pre", "avg_volume_wk_prior26", "n_weeks_pre"]
TGT_MARKET = ["beta_post_63", "beta_post_126", "beta_post_252", "ncskew_post_63", "duvol_post_63",
              "log_market_cap_post", "ps_ratio_post",
              "beta_shift_delta", "beta_shift_delta_se", "beta_shift_delta_t",
              "car_m1_p1", "car_p2_p63", "ncskew_wk_post", "duvol_wk_post", "n_weeks_post"]
COV_FINANCIALS = (["accession_number", "filing_date_pt", "revenue", "operating_income", "total_assets", "operating_margin",
                   "asset_turnover", "roa", "shares_out", "debt_to_equity", "liabilities_to_assets",
                   "rd_intensity_pre", "gross_margin_pre", "revenue_yoy_pre", "roic_minus_wacc_pre"]
                  + [f"{c}_pre" for c in VALUE_COMPONENTS])
TGT_FINANCIALS = (["rd_intensity_post", "gross_margin_post", "next_revenue_yoy_post", "asset_turnover_post",
                   "roic_minus_wacc_post"] + [f"{c}_post" for c in VALUE_COMPONENTS]
                  + ["gross_margin_ttm_post", "revenue_growth_ttm_post", "roic_minus_wacc_ttm_post"])
COV_FINANCIALS_TTM = ["gross_margin_ttm_pre", "revenue_growth_ttm_pre", "roic_minus_wacc_ttm_pre"]


# ---- price windows -------------------------------------------------------------

def price_windows(panel: pd.DataFrame) -> pd.DataFrame:
    """Every price-window measure of a call (one row per call of `panel`)."""
    factors = load_factors()
    prices = load_prices()
    weekly = {t: to_weekly(pr, factors) for t, pr in prices.items()}
    # post-call close/market_cap: trading days with a close (the adjusted series may be shorter)
    closes = {t: d[["date", "close", "market_cap"]].assign(date=lambda x: pd.to_datetime(x["date"]))
              .dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
              for t, d in L.read("silver.market_prices").to_pandas().groupby("ticker")}
    rows = []
    for row in panel[["call_accession_number", "ticker", "fecha"]].itertuples(index=False):
        # `market_cap_post` is read at idx+62, the last trading day of the
        # thesis's [+2, +63) valuation window; `market_cap_pre2` mirrors the
        # window's start-side gap at idx-2. Nothing else in this builder needs
        # the price/market cap at any other pre- or post-call day.
        rec = {"call_accession_number": row.call_accession_number, "market_cap_post": np.nan, "market_cap_pre2": np.nan}
        pr = prices.get(row.ticker)
        if pr is not None:
            idx = event_index(pr, row.fecha)
            rec.update(beta_windows(pr, factors, idx))
            rec.update(crash_windows(pr, factors, idx))
            rec.update(ncskew63_windows(pr, factors, idx))
            rec.update(beta_shift_windows(pr, factors, idx))
            rec.update(car_windows(pr, factors, idx))
        wk = weekly.get(row.ticker)
        if wk is not None and len(wk):
            rec.update(weekly_crash_windows(wk, week_index(wk, row.fecha)))
        cl = closes.get(row.ticker)
        if cl is not None:
            call_idx = event_index(cl, row.fecha)
            val_end = call_idx + VALUATION_WINDOW_LAST_DAY
            if 0 <= val_end < len(cl):
                mcap = cl["market_cap"].iloc[val_end]
                if pd.notna(mcap):
                    rec["market_cap_post"] = float(mcap)
            pre2 = call_idx - VALUATION_PRE_DAY
            if 0 <= pre2 < len(cl):
                mcap_pre2 = cl["market_cap"].iloc[pre2]
                if pd.notna(mcap_pre2):
                    rec["market_cap_pre2"] = float(mcap_pre2)
        rows.append(rec)
    return panel.merge(pd.DataFrame(rows), on="call_accession_number", how="left", validate="one_to_one")


def beta_shift_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    """Brenner (1979) pooled beta-shift test: one OLS of daily excess return
    on [1, mktrf, POST, POST*mktrf] over [idx-63,idx-21) union [idx+21,idx+63),
    POST=1 on the post-call rows. `beta_shift_pre` is the mktrf slope (the
    pooled design's own pre-call beta), `beta_shift_delta` the POST*mktrf
    slope. SE from the pooled residual variance, classic (non-robust) OLS --
    the point of the Brenner test is exactly that pooled, homoskedastic
    variance, not a firm-clustered one (clustering is not meaningful within
    a single call's own residuals)."""
    out = {"beta_shift_pre": np.nan, "beta_shift_delta": np.nan, "beta_shift_delta_se": np.nan,
           "beta_shift_delta_t": np.nan, "beta_shift_n_pre": 0, "beta_shift_n_post": 0}
    pre_lo, pre_hi = idx - BETA_SHIFT_PRE_START, idx - BETA_SHIFT_PRE_END
    post_lo, post_hi = idx + BETA_SHIFT_POST_START, idx + BETA_SHIFT_POST_END
    if pre_lo < 0 or post_hi > len(pr):
        return out
    pre = with_factors(pr.iloc[pre_lo:pre_hi], factors)
    post = with_factors(pr.iloc[post_lo:post_hi], factors)
    if len(pre) < BETA_SHIFT_MIN_OBS_SIDE or len(post) < BETA_SHIFT_MIN_OBS_SIDE:
        return out
    out["beta_shift_n_pre"], out["beta_shift_n_post"] = len(pre), len(post)
    post_dummy = np.concatenate([np.zeros(len(pre)), np.ones(len(post))])
    mkt = np.concatenate([pre["mktrf"].to_numpy(), post["mktrf"].to_numpy()])
    excess = np.concatenate([pre["ret"].to_numpy() - pre["rf"].to_numpy(), post["ret"].to_numpy() - post["rf"].to_numpy()])
    design = np.column_stack([np.ones(len(excess)), mkt, post_dummy, post_dummy * mkt])
    coef, *_ = np.linalg.lstsq(design, excess, rcond=None)
    resid = excess - design @ coef
    n, k = design.shape
    if n <= k:
        return out
    s2 = float(resid @ resid) / (n - k)
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(s2 * np.diag(xtx_inv))
    out["beta_shift_pre"] = float(coef[1])
    out["beta_shift_delta"] = float(coef[3])
    out["beta_shift_delta_se"] = float(se[3])
    out["beta_shift_delta_t"] = float(coef[3] / se[3]) if se[3] > 0 else np.nan
    return out


def car_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    """Brown & Warner (1985) / MacKinlay (1997) cumulative abnormal return: a
    market model estimated over [idx-250, idx-30) gives (alpha, beta); CAR is
    the sum of daily abnormal returns (actual excess - that model's
    prediction) over [idx-1, idx+1] and [idx+2, idx+63)."""
    out = {"car_m1_p1": np.nan, "car_p2_p63": np.nan}
    est_lo, est_hi = idx - CAR_EST_START, idx - CAR_EST_END
    if est_lo < 0:
        return out
    est = with_factors(pr.iloc[est_lo:est_hi], factors)
    if len(est) < CAR_EST_MIN_OBS:
        return out
    coef = one_factor_ols(est)[0]

    def abnormal_sum(start: int, end: int, min_obs: int) -> float:
        lo, hi = idx + start, idx + end
        if lo < 0 or hi > len(pr) or hi <= lo:
            return np.nan
        win = with_factors(pr.iloc[lo:hi], factors)
        if len(win) < min_obs:
            return np.nan
        excess = win["ret"].to_numpy() - win["rf"].to_numpy()
        predicted = coef[0] + coef[1] * win["mktrf"].to_numpy()
        return float(np.sum(excess - predicted))

    out["car_m1_p1"] = abnormal_sum(-1, 2, CAR_M1P1_MIN_OBS)
    out["car_p2_p63"] = abnormal_sum(2, 63, CAR_P2P63_MIN_OBS)
    return out


def weekly_crash_windows(wk: pd.DataFrame, idx_w: int) -> dict:
    """Chen, Hong & Stein (2001) / Kim, Li & Zhang (2011) weekly crash-risk
    design over the trading weeks of `wk` (market_windows.to_weekly), `idx_w`
    the call's own week (market_windows.week_index): NCSKEW/DUVOL/SIGMA/RET
    all read the SAME expanded-market-model residuals of the pre window
    [idx_w-26, idx_w); the post window [idx_w+1, idx_w+27) gives the
    post-call NCSKEW/DUVOL; `avg_volume_wk_pre`/`avg_volume_wk_prior26` (mean
    weekly share volume, pre window and the 26 weeks before it) let the
    analytics layer finish KLZ's DTURN control once shares outstanding is
    known (a call-grain, not ticker-grain, quantity)."""
    out = {"ncskew_wk_pre": np.nan, "duvol_wk_pre": np.nan, "sigma_wk_pre": np.nan, "ret_wk_pre": np.nan,
           "avg_volume_wk_pre": np.nan, "avg_volume_wk_prior26": np.nan, "n_weeks_pre": 0,
           "ncskew_wk_post": np.nan, "duvol_wk_post": np.nan, "n_weeks_post": 0}
    pre_lo, pre_hi = idx_w - CRASH_WK_WINDOW, idx_w
    post_lo, post_hi = idx_w + CRASH_WK_GAP, idx_w + CRASH_WK_GAP + CRASH_WK_WINDOW
    if pre_lo >= 0:
        out["n_weeks_pre"] = pre_hi - pre_lo
        res_pre = expanded_market_residuals(wk, pre_lo, pre_hi, CRASH_WK_MIN_OBS)
        if res_pre is not None:
            out["ncskew_wk_pre"] = ncskew(res_pre, CRASH_WK_MIN_OBS)
            out["duvol_wk_pre"] = duvol(res_pre, CRASH_WK_MIN_OBS)
            out["sigma_wk_pre"] = float(res_pre.std(ddof=1)) if len(res_pre) > 1 else np.nan
            out["ret_wk_pre"] = float(100.0 * res_pre.mean())
        if pre_lo < len(wk):
            out["avg_volume_wk_pre"] = float(wk["volume_sum"].iloc[pre_lo:pre_hi].mean())
        prior_lo, prior_hi = pre_lo - CRASH_WK_WINDOW, pre_lo
        if prior_lo >= 0 and (prior_hi - prior_lo) >= CRASH_WK_PRIOR_MIN_OBS:
            out["avg_volume_wk_prior26"] = float(wk["volume_sum"].iloc[prior_lo:prior_hi].mean())
    if post_hi <= len(wk):
        out["n_weeks_post"] = post_hi - post_lo
        res_post = expanded_market_residuals(wk, post_lo, post_hi, CRASH_WK_MIN_OBS)
        if res_post is not None:
            out["ncskew_wk_post"] = ncskew(res_post, CRASH_WK_MIN_OBS)
            out["duvol_wk_post"] = duvol(res_post, CRASH_WK_MIN_OBS)
    return out


def beta_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    out = {"beta_pre": np.nan, "beta_pre_63": np.nan, "beta_pre_252": np.nan,
           "beta_post_63": np.nan, "beta_post_126": np.nan, "beta_post_252": np.nan,
           "price_pre": np.nan, "market_cap": np.nan, "return60": np.nan}
    if 0 <= idx - 1 and idx < len(pr):
        out["price_pre"] = float(pr["close"].iloc[idx - 1])
        mcap_pre = pr["market_cap"].iloc[idx - 1]
        if pd.notna(mcap_pre):
            out["market_cap"] = float(mcap_pre)
        if idx - 60 >= 0 and pr["adj_close"].iloc[idx - 60] > 0:
            out["return60"] = float(pr["adj_close"].iloc[idx - 1] / pr["adj_close"].iloc[idx - 60] - 1)
        if idx - 252 >= 0:
            out["beta_pre"] = beta(with_factors(pr.iloc[idx - 252:idx], factors), BETA_MIN_OBS)
        # Symmetric pre-call windows: same [-N, -21) span and gap as the
        # matching beta_post_N's [+21, +N) (see BETA_PRE_GAP above).
        for days, min_obs in ((63, BETA_MIN_OBS_PRE_63), (252, BETA_MIN_OBS_PRE_252)):
            if idx - days >= 0:
                out[f"beta_pre_{days}"] = beta(with_factors(pr.iloc[idx - days:idx - BETA_PRE_GAP], factors), min_obs)
        # Post-call beta windows carry the thesis's [+21, +N] gap
        # (thesis_document/thesis.qmd @tbl-market-outcomes / @tbl-call-beta-paper).
        for days, min_obs in ((63, BETA_MIN_OBS_POST_63), (126, BETA_MIN_OBS_POST_126), (252, BETA_MIN_OBS_POST_252)):
            if idx + days < len(pr):
                out[f"beta_post_{days}"] = beta(with_factors(pr.iloc[idx + BETA_POST_GAP:idx + days], factors), min_obs)
    return out


def _residuals(pr: pd.DataFrame, factors: pd.DataFrame, start: int, end: int, min_obs: int) -> np.ndarray | None:
    win = with_factors(pr.iloc[start:end], factors)
    if len(win) < min_obs:
        return None
    return one_factor_ols(win)[2]


def crash_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    """Pre-call crash-risk residual window only ([idx-252, idx-21)); the
    thesis states no pre-call window for these metrics, so it is kept as-is.
    The post-call crash-risk metrics the thesis does state a window for
    (NCSKEW/DUVOL, [+2, +63]) are computed by `ncskew63_windows` instead."""
    out = {"ncskew_pre": np.nan, "duvol_pre": np.nan}
    pre_start, pre_end = idx - CRASH_GAP - CRASH_PRE, idx - CRASH_GAP
    if pre_start >= 0:
        res = _residuals(pr, factors, pre_start, pre_end, CRASH_MIN_OBS)
        if res is not None:
            out["ncskew_pre"], out["duvol_pre"] = ncskew(res, CRASH_MIN_OBS), duvol(res, CRASH_MIN_OBS)
    return out


def ncskew63_windows(pr: pd.DataFrame, factors: pd.DataFrame, idx: int) -> dict:
    """ncskew_pre_63 over [idx-63, idx) (pre-call, no thesis-stated window,
    left unchanged); ncskew_post_63/duvol_post_63 over [idx+2, idx+63), the
    thesis's [+2, +63] window for return-distribution metrics;
    ncskew_pre_61/duvol_pre_61 over [idx-63, idx-2), the mirror of that
    post-call window (same 61-day span, same 2-day announcement gap)."""
    out = {"ncskew_pre_63": np.nan, "ncskew_pre_61": np.nan, "duvol_pre_61": np.nan,
           "ncskew_post_63": np.nan, "duvol_post_63": np.nan}
    if idx - 63 >= 0:
        res = _residuals(pr, factors, idx - 63, idx, NCSKEW_PRE63_MIN_OBS)
        out["ncskew_pre_63"] = ncskew(res, NCSKEW_PRE63_MIN_OBS) if res is not None else np.nan
        res61 = _residuals(pr, factors, idx - 63, idx - NCSKEW_POST_GAP, NCSKEW_PRE61_MIN_OBS)
        if res61 is not None:
            out["ncskew_pre_61"] = ncskew(res61, NCSKEW_PRE61_MIN_OBS)
            out["duvol_pre_61"] = duvol(res61, NCSKEW_PRE61_MIN_OBS)
    if idx + 63 < len(pr):
        res = _residuals(pr, factors, idx + NCSKEW_POST_GAP, idx + 63, NCSKEW_POST63_MIN_OBS)
        if res is not None:
            out["ncskew_post_63"] = ncskew(res, NCSKEW_POST63_MIN_OBS)
            out["duvol_post_63"] = duvol(res, NCSKEW_POST63_MIN_OBS)
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


def ttm_quarterly() -> pd.DataFrame:
    """Trailing-twelve-month aggregates per (ticker, fiscal quarter), from the
    firm-quarter panel.

    The annual family measures an outcome at the first 10-K filed after the
    call, so the distance between the call and the figure runs from months to
    more than a year, and `next_revenue_yoy` reaches the fiscal year AFTER that
    10-K. This family fixes the horizon instead: the four fiscal quarters that
    FOLLOW the call's own quarter. Matching on fiscal period, not on filing
    date, is what keeps the window strictly ahead of the call -- the quarter a
    call discusses is usually reported only after that call, so a filing-date
    match lands on the period the call itself was about. The quarterly panel
    carries Q4 (build_financials reconstructs it as FY - YTD_Q3 from the 10-K),
    so no quarter is skipped by going through 10-Q filings only.

    `_pre` and `_post` are the SAME construction on two windows, q-3..q and
    q+1..q+4, so a change between them is not a change of definition.
    """
    q = L.read_gold("firm_quarter", ("covariates", "financials"))
    q["pq"] = pd.PeriodIndex(q["quarter"].astype(str), freq="Q")
    q["ord"] = q["pq"].astype("int64")
    q = q.sort_values(["ticker", "ord"]).reset_index(drop=True)
    q["invested_capital"] = q["equity"].fillna(0) + q["debt"].fillna(0)
    grp = q.groupby("ticker", group_keys=False)

    def window_sum(col):
        return grp[col].apply(lambda s: s.rolling(TTM_QUARTERS, min_periods=TTM_QUARTERS).sum())

    def window_mean(col):
        return grp[col].apply(lambda s: s.where(s > 0).rolling(TTM_QUARTERS, min_periods=TTM_QUARTERS).mean())

    for col in ["revenue", "cogs", "operating_income"]:
        q[f"{col}_ttm"] = window_sum(col)
    q["invested_capital_ttm"] = window_mean("invested_capital")

    # Rolling sums end at the row's own quarter, so `_pre` reads straight off
    # them and `_post` is the same aggregate shifted four quarters forward.
    out = pd.DataFrame({"ticker": q["ticker"], "ord": q["ord"]})
    gross_profit = q["revenue_ttm"] - q["cogs_ttm"]
    out["gross_margin_ttm_pre"] = np.where(q["revenue_ttm"] > 0, gross_profit / q["revenue_ttm"], np.nan)
    prior_revenue = grp["revenue_ttm"].apply(lambda s: s.shift(TTM_QUARTERS))
    out["revenue_growth_ttm_pre"] = np.where(prior_revenue > 0, q["revenue_ttm"] / prior_revenue - 1, np.nan)
    out["operating_income_ttm_pre"] = q["operating_income_ttm"]
    out["invested_capital_ttm_pre"] = q["invested_capital_ttm"]
    out["equity_ttm_pre"], out["debt_ttm_pre"] = q["equity"], q["debt"]
    for col in [c for c in out.columns if c.endswith("_pre")]:
        out[col.replace("_pre", "_post")] = out.groupby(q["ticker"])[col].shift(-TTM_QUARTERS)
    out["revenue_growth_ttm_post"] = out.groupby(q["ticker"])["revenue_growth_ttm_pre"].shift(-TTM_QUARTERS)
    return out


def ttm_spread(rf: pd.Series, beta_: pd.Series, erp: float, operating_income_ttm: pd.Series,
               invested_capital_ttm: pd.Series, equity: pd.Series, debt: pd.Series,
               effective_tax_rate: pd.Series, cost_of_debt: pd.Series) -> pd.Series:
    """ROIC - WACC on a TTM window: NOPAT over the window's AVERAGE invested
    capital, minus a book-weighted WACC valued at the window's end.

    This is an ESTIMATED spread, and three of its inputs are approximations the
    quarterly XBRL cannot supply. The effective tax rate and the cost of debt
    come from the last 10-K filed before the call (quarterly filings tag no
    interest expense, pretax income or tax expense), and the WACC is weighted
    on BOOK equity and debt rather than market values.

    It is also not a purely operational measure: the cost of equity is
    rf + beta * ERP, so the firm's own market risk enters the outcome by
    construction and this spread cannot be read as independent of the beta
    results. The `rf` date must therefore be the end of the same window the
    `beta_` argument was estimated over."""
    nopat = operating_income_ttm * (1 - effective_tax_rate)
    roic = nopat / invested_capital_ttm.where(invested_capital_ttm > 0)
    cost_of_equity = rf + beta_ * erp
    total = equity.fillna(0) + debt.fillna(0)
    weight_equity = equity / total.where(total > 0)
    wacc = weight_equity * cost_of_equity + (1 - weight_equity) * cost_of_debt * (1 - effective_tax_rate)
    return roic - wacc


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


def attach_leverage(panel: pd.DataFrame) -> pd.DataFrame:
    """debt_to_equity, liabilities_to_assets as of the PRE-call 10-K.

    fs (FactSet) replacement for the old bronze.xbrl_facts LEVERAGE_CONCEPTS
    extraction (docs/plans/fs_gold_replacement.md): `long_term_debt_pre` and
    `equity_pre` are already attached from covariates/firm_year/financials
    (fs-sourced, same fallback-free STND fields build_firm_financials.annual_panel_fs
    reads) -- reusing them here instead of a second, independent XBRL
    extraction is "one construct, one formula" (CLAUDE.md), not a
    duplicate definition of long_term_debt/equity that could silently
    disagree with the firm_year figure. fs has no native total-liabilities
    row, so `liabilities_to_assets` is reconstructed as
    (total_assets - equity) / total_assets, the same reconstruction
    documented in docs/sources/fs.md for invested capital / EV."""
    equity = panel["equity_pre"].where(panel["equity_pre"] > 0)
    assets = panel["total_assets"].where(panel["total_assets"] > 0)
    panel["debt_to_equity"] = panel["long_term_debt_pre"] / equity
    panel["liabilities_to_assets"] = (panel["total_assets"] - panel["equity_pre"]) / assets
    return panel


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
    # log_market_cap/ps_ratio read fs's own market_cap directly (not
    # price * shares_out) -- fixes the pre-split understatement documented
    # in data/results/fs_validation/summary.md.
    panel["log_market_cap"] = np.log(panel["market_cap"].where(panel["market_cap"] > 0))
    panel["log_market_cap_post"] = np.log(panel["market_cap_post"].where(panel["market_cap_post"] > 0))
    panel["ps_ratio_pre"] = panel["market_cap"] / panel["revenue"]
    panel["ps_ratio_post"] = panel["market_cap_post"] / panel["revenue_post"]
    # Mirrored pre-call valuation level, at idx-2 (VALUATION_PRE_DAY), against
    # the same pre-call 10-K revenue as `ps_ratio_pre`.
    panel["log_market_cap_pre2"] = np.log(panel["market_cap_pre2"].where(panel["market_cap_pre2"] > 0))
    panel["ps_ratio_pre2"] = panel["market_cap_pre2"] / panel["revenue"]

    # ERP/rf stay on the repo's own Ken French factors (kept from existing
    # source, docs/plans/fs_gold_replacement.md S0), NOT market_windows'
    # fs-based load_factors() used above for the beta regressor -- using
    # the fs-bounded (2015+) series here would silently disagree with
    # build_roic_wacc.value_creation's ERP (used by firm_year) on the same
    # construct, exactly what CLAUDE.md's "one construct, one formula"
    # rule forbids.
    rf_factors = L.read("bronze.market_factors_daily").select(["date", "mktrf", "rf"]).to_pandas()
    rf_factors["date"] = pd.to_datetime(rf_factors["date"])
    _, erp = annualized_equity_premium(rf_factors, ERP_SAMPLE_START)
    rf_pre = risk_free_at(rf_factors, panel["filing_date_pt"])
    rf_post = risk_free_at(rf_factors, panel["fecha"] + pd.to_timedelta(POST_WINDOW_END_OFFSET * 7 / 5, unit="D"))
    value_pre = roic_minus_wacc(rf_pre, panel["beta_pre"], erp, panel["operating_income"], panel["pretax_income_pre"],
                                panel["tax_expense_pre"], panel["equity_pre"], panel["long_term_debt_pre"],
                                panel["interest_expense_pre"])
    value_post = roic_minus_wacc(rf_post, panel["beta_post_126"], erp, panel["operating_income_post"],
                                 panel["pretax_income_post"], panel["tax_expense_post"], panel["equity_post"],
                                 panel["long_term_debt_post"], panel["interest_expense_post"])
    for c in ["roic_minus_wacc"] + VALUE_COMPONENTS:
        panel[f"{c}_pre"], panel[f"{c}_post"] = value_pre[c], value_post[c]

    # ---- TTM family: same construction on q-3..q and q+1..q+4 ----------------
    tag = panel["id"].str.extract(r"_(\d{4}Q[1-4])$")[0]
    panel["ord"] = pd.Series(pd.PeriodIndex(tag.dropna(), freq="Q").astype("int64"),
                             index=tag.dropna().index).reindex(panel.index)
    panel = panel.merge(ttm_quarterly(), on=["ticker", "ord"], how="left", validate="many_to_one")
    panel["gross_margin_ttm_post"] = panel["gross_margin_ttm_post"]
    # The WACC of each end is valued at the end of the window its beta was
    # estimated over: the call date for `_pre`, and beta_post_252's last
    # trading day for `_post`.
    rf_ttm_pre = risk_free_at(rf_factors, panel["fecha"])
    rf_ttm_post = risk_free_at(rf_factors, panel["fecha"] + pd.to_timedelta(TTM_WINDOW_END_OFFSET * 7 / 5, unit="D"))
    tax_rate = panel["effective_tax_rate_pre"]
    cost_debt = panel["cost_of_debt_pre"]
    panel["roic_minus_wacc_ttm_pre"] = ttm_spread(
        rf_ttm_pre, panel["beta_pre"], erp, panel["operating_income_ttm_pre"],
        panel["invested_capital_ttm_pre"], panel["equity_ttm_pre"], panel["debt_ttm_pre"],
        tax_rate, cost_debt)
    panel["roic_minus_wacc_ttm_post"] = ttm_spread(
        rf_ttm_post, panel["beta_post_252"], erp, panel["operating_income_ttm_post"],
        panel["invested_capital_ttm_post"], panel["equity_ttm_post"], panel["debt_ttm_post"],
        tax_rate, cost_debt)

    panel = attach_leverage(panel)
    spine = L.GOLD_SPINE_COLUMNS["call"]
    L.write_gold("covariates", "call", "market", panel[spine + COV_MARKET], builder=BUILDER)
    L.write_gold("covariates", "call", "financials", panel[spine + COV_FINANCIALS + COV_FINANCIALS_TTM], builder=BUILDER)
    L.write_gold("targets", "call", "market", panel[spine + TGT_MARKET], builder=BUILDER)
    L.write_gold("targets", "call", "financials", panel[spine + TGT_FINANCIALS], builder=BUILDER)
    print(f"ERP geométrico usado: {erp:.4f}")


if __name__ == "__main__":
    main()
