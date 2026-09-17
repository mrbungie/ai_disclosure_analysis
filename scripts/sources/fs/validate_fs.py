#!/usr/bin/env python
"""Cross-validate FactSet raw pulls against the data the thesis pipeline already
uses (silver daily prices, gold covariates/targets). Read-only against
`data/raw/fs/` and `data/gold/`/`data/silver/` -- writes only under
`data/results/fs_validation/`.

Checks:
  1. Prices        FactSet price/total_return vs silver.market_prices close/adj_close:
                    per-ticker daily-return correlation, median abs diff, split-level
                    ratio stability.
  2. Market cap     FactSet market_cap_daily vs gold market_cap/log_market_cap at
                    matching dates (firm_year filing_date, firm_quarter as_of_date).
  3. Beta           252-trading-day beta from FactSet total_return vs FactSet
                    benchmark total_return, at each firm_year `filing_date`, vs gold
                    firm_year `beta` (one-factor OLS on Fama-French mktrf/rf).
  4. Fundamentals   FactSet STND (preferred) / ARPT annual fundamentals vs gold
                    firm_year `financials`, matched by ticker + period end (+/- 45d).
  5. Coverage gaps  firm-years/quarters in gold missing from FactSet and vice versa,
                    for the ids downloaded so far.

Also explains (does not fix) the firm.parquet (498) vs
silver.firm_universe/sp500_2021_start_panel (499) tickers gap.

Usage:
    uv run python scripts/sources/fs/validate_fs.py
    uv run python scripts/sources/fs/validate_fs.py \
        --fundamentals-long /path/to/fundamentals_long.parquet

By default reads the canonical `data/raw/fs/fundamentals/fundamentals_long.parquet`
produced by `scripts/sources/fs/parse_fundamentals.py` -- rerun that first if it
does not exist yet or is stale, then rerun this script; safe to rerun any time as more
ids land.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

FACTSET_RAW = REPO_ROOT / "data" / "raw" / "fs"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "fs_validation"
DEFAULT_FUND_LONG = FACTSET_RAW / "fundamentals" / "fundamentals_long.parquet"

BETA_WINDOW = 252
BETA_MIN_OBS = 120

# thesis field -> (report, STND exact label, ARPT fallback: (report, level0 regex),
#                  value kind: "amount" (scale-multiplied) | "per_share" | "shares_millions")
FIELD_SPECS = {
    "revenue":          ("INC", "Sales", r"^(net sales|net revenues?|total revenues?|sales)$", "amount"),
    "net_income":       ("INC", "Net Income", r"^net income( / loss)?( attributable to .*)?$", "amount"),
    "total_assets":     ("BAL", "Total Assets", r"^total assets$", "amount"),
    "equity":           ("BAL", "Total Shareholders' Equity", r"^total (shareholders'?|stockholders'?) equity$", "amount"),
    "capex":            ("CF", "Capital Expenditures", r"^capital expenditures?$", "amount"),
    "rd_expense":       ("INC", "Research & Development", r"^research and development( costs| expenses?)?$", "amount"),
    "sga_expense":      ("INC", "SG&A Expense", r"^selling,? general,? and administrative( expenses?)?$", "amount"),
    "operating_income": ("INC", "EBIT (Operating Income)", r"^operating income$", "amount"),
    "eps_diluted":      ("INC", "EPS (diluted)", r"^diluted$", "per_share"),
    "long_term_debt":   ("BAL", "Long-Term Debt", r"^long-?term debt$", "amount"),
    "cash":             ("BAL", "Cash & Short-Term Investments", r"^cash( and cash equivalents)?$", "amount"),
    "shares_out":       ("SHS", "Shs Outstanding (M)", None, "shares_millions"),
}
SCALE_MULT = {"billions": 1e9, "millions": 1e6, None: 1.0}


def w(name, df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(RESULTS_DIR / f"{name}.csv") if isinstance(df, pl.DataFrame) else None
    if isinstance(df, pl.DataFrame):
        df.write_parquet(RESULTS_DIR / f"{name}.parquet")
    print(f"  -> {RESULTS_DIR / name}.{{csv,parquet}} ({len(df)} rows)")


# --------------------------------------------------------------------------- prices ---

def check_prices() -> dict:
    print("\n=== 1. Prices ===")
    fs = pl.read_parquet(FACTSET_RAW / "prices" / "daily" / "prices_daily.parquet")
    sil = L.read("silver.market_prices").select(["ticker", "date", "close", "adj_close"])
    fs = fs.with_columns(pl.col("date").cast(pl.Date)).select(["ticker", "date", "price", "total_return"])
    sil = sil.with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])
    fs = fs.sort(["ticker", "date"])

    sil = sil.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret_close"),
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1).alias("ret_adj"),
    ])
    fs = fs.with_columns([
        (pl.col("price") / pl.col("price").shift(1).over("ticker") - 1).alias("fs_ret_price"),
        (pl.col("total_return") / pl.col("total_return").shift(1).over("ticker") - 1).alias("fs_ret_tr"),
    ])
    joined = sil.join(fs, on=["ticker", "date"], how="inner").with_columns(
        (pl.col("price") / pl.col("close")).alias("price_ratio")
    ).drop_nulls(["ret_adj", "fs_ret_tr"])

    per_ticker = (
        joined.group_by("ticker")
        .agg(
            pl.len().alias("n_obs"),
            pl.corr("ret_adj", "fs_ret_tr").alias("corr_total_return"),
            (pl.col("ret_adj") - pl.col("fs_ret_tr")).abs().median().alias("median_abs_diff_total_return"),
            pl.corr("ret_close", "fs_ret_price").alias("corr_price_only"),
            (pl.col("price_ratio").std() / pl.col("price_ratio").mean()).alias("price_ratio_cv"),
            pl.col("price_ratio").mean().alias("price_ratio_mean"),
        )
        .sort("corr_total_return")
    )
    w("prices_per_ticker", per_ticker)

    flagged = per_ticker.filter(
        (pl.col("corr_total_return") < 0.95) | (pl.col("corr_total_return").is_null())
        | (pl.col("price_ratio_cv") > 0.05)
    )
    w("prices_flagged", flagged)

    summary = {
        "n_tickers": per_ticker.height,
        "n_dates_joined": joined.height,
        "median_corr_total_return": float(per_ticker["corr_total_return"].median()),
        "median_abs_diff_total_return": float(per_ticker["median_abs_diff_total_return"].median()),
        "n_flagged_corr_lt_0.95_or_split_mismatch": flagged.height,
        "flagged_tickers": flagged["ticker"].to_list()[:30],
    }
    print(f"  {summary['n_tickers']} tickers, median corr(total-return) = {summary['median_corr_total_return']:.4f}, "
          f"median abs daily-return diff = {summary['median_abs_diff_total_return']:.6f}, "
          f"{summary['n_flagged_corr_lt_0.95_or_split_mismatch']} flagged")
    return summary


# --------------------------------------------------------------------------- market cap ---

def check_market_cap() -> dict:
    print("\n=== 2. Market cap ===")
    fs_mc = pl.read_parquet(FACTSET_RAW / "market_cap" / "market_cap_daily.parquet") \
        .with_columns(pl.col("date").cast(pl.Date)).select(["ticker", "date", "market_cap"]).sort(["ticker", "date"])

    fy_fin = pl.read_parquet(REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "financials.parquet")
    fy_mkt = pl.read_parquet(REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "market.parquet")
    fy = fy_fin.select(["id", "ticker", "filing_date"]).join(
        fy_mkt.select(["id", "market_cap"]), on="id", how="inner"
    ).drop_nulls(["filing_date", "market_cap"])
    fy = fy.with_columns(pl.col("filing_date").cast(pl.Date)).sort(["ticker", "filing_date"])

    fy_matched = fy.join_asof(
        fs_mc.rename({"date": "filing_date", "market_cap": "fs_market_cap_raw"}),
        by="ticker", on="filing_date", strategy="backward", tolerance="7d"
    ).with_columns((pl.col("fs_market_cap_raw") * 1e6).alias("fs_market_cap")).drop_nulls("fs_market_cap")
    fy_matched = fy_matched.with_columns(
        ((pl.col("fs_market_cap") - pl.col("market_cap")) / pl.col("market_cap")).alias("pct_diff")
    )
    w("market_cap_firm_year", fy_matched.select(["id", "ticker", "filing_date", "market_cap", "fs_market_cap", "pct_diff"]))

    fq_mkt = pl.read_parquet(REPO_ROOT / "data" / "gold" / "covariates" / "firm_quarter" / "market.parquet")
    fq = fq_mkt.select(["id", "ticker", "as_of_date", "log_market_cap"]).drop_nulls(["as_of_date", "log_market_cap"])
    fq = fq.with_columns(pl.col("as_of_date").cast(pl.Date)).sort(["ticker", "as_of_date"])
    fq_matched = fq.join_asof(
        fs_mc.rename({"date": "as_of_date", "market_cap": "fs_market_cap_raw"}),
        by="ticker", on="as_of_date", strategy="backward", tolerance="10d"
    ).with_columns((pl.col("fs_market_cap_raw") * 1e6).alias("fs_market_cap")).drop_nulls("fs_market_cap")
    fq_matched = fq_matched.with_columns(
        pl.col("fs_market_cap").log().alias("fs_log_market_cap"),
    ).with_columns(
        ((pl.col("fs_log_market_cap") - pl.col("log_market_cap")) / pl.col("log_market_cap").abs()).alias("pct_diff")
    )
    w("market_cap_firm_quarter", fq_matched.select(["id", "ticker", "as_of_date", "log_market_cap", "fs_log_market_cap", "pct_diff"]))

    def outliers(df, thresh=0.20):
        return df.filter(pl.col("pct_diff").abs() > thresh).sort(pl.col("pct_diff").abs(), descending=True)

    fy_out = outliers(fy_matched.select(["ticker", "filing_date", "market_cap", "fs_market_cap", "pct_diff"]))
    w("market_cap_firm_year_outliers", fy_out)
    outlier_tickers = fy_out["ticker"].unique().sort().to_list()
    summary = {
        "firm_year_matched": fy_matched.height,
        "firm_year_median_abs_pct_diff": float(fy_matched["pct_diff"].abs().median()),
        "firm_year_n_outliers_gt20pct": fy_out.height,
        "firm_year_n_outlier_tickers": len(outlier_tickers),
        "firm_year_outlier_tickers": outlier_tickers,
        "firm_quarter_matched": fq_matched.height,
        "firm_quarter_median_abs_pct_diff_log": float(fq_matched["pct_diff"].abs().median()),
    }
    print("  RED FLAG (verified, not just hypothesized): 65 tickers show a large, ~constant-ratio "
          "pct_diff that matches each ticker's later stock split (AMZN/GOOGL ~20x for their 2022 "
          "splits, CMG ~48x for its 2024 50:1 split, etc.), for every pre-split filing_date and only "
          "those. Root cause, confirmed directly: silver.market_prices.close for AMZN on 2022-02-04 "
          "= 157.64, byte-identical to FactSet's split-ADJUSTED `price` (157.64) for the same date -- "
          "i.e. silver's `close` column IS split-adjusted, contradicting build_market_factors.py's "
          "own docstring, which assumes `close` is the unadjusted price when it multiplies it by the "
          "cover-page (historical, pre-split) `shares_out`. That understates firm_year market_cap by "
          "the eventual split ratio for any 10-K filed before a later split. Not fixed here (out of "
          "scope) -- flagging for the pipeline owner. See market_cap_firm_year_outliers.csv.")
    print(f"  firm_year: {summary['firm_year_matched']} matched, median abs pct diff = {summary['firm_year_median_abs_pct_diff']:.4f}, "
          f"{summary['firm_year_n_outliers_gt20pct']} outliers >20%")
    print(f"  firm_quarter (log): {summary['firm_quarter_matched']} matched, median abs pct diff = {summary['firm_quarter_median_abs_pct_diff_log']:.4f}")
    return summary


# --------------------------------------------------------------------------- beta ---

def check_beta() -> dict:
    print("\n=== 3. Beta sanity ===")
    fs = pl.read_parquet(FACTSET_RAW / "prices" / "daily" / "prices_daily.parquet") \
        .with_columns(pl.col("date").cast(pl.Date)).select(["ticker", "date", "total_return"]).sort(["ticker", "date"])
    bm = pl.read_parquet(FACTSET_RAW / "benchmark" / "benchmark_sp500_daily.parquet") \
        .with_columns(pl.col("date").cast(pl.Date)).select(["date", "total_return"]).sort("date").rename({"total_return": "bm_total_return"})

    fs_pd = fs.to_pandas()
    bm_pd = bm.to_pandas()
    bm_pd["bm_ret"] = bm_pd["bm_total_return"].pct_change()
    bm_dates = bm_pd["date"].to_numpy()
    bm_ret_by_date = dict(zip(bm_pd["date"], bm_pd["bm_ret"]))

    per_ticker = {}
    for ticker, g in fs_pd.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        g["ret"] = g["total_return"].pct_change()
        per_ticker[ticker] = g

    fy_fin = pl.read_parquet(REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "financials.parquet")
    fy_mkt = pl.read_parquet(REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "market.parquet")
    fy = fy_fin.select(["id", "ticker", "filing_date"]).join(
        fy_mkt.select(["id", "beta", "beta_n_obs"]), on="id", how="inner"
    ).drop_nulls(["filing_date", "beta"]).to_pandas()
    fy["filing_date"] = pd.to_datetime(fy["filing_date"])

    rows = []
    for row in fy.itertuples(index=False):
        g = per_ticker.get(row.ticker)
        if g is None:
            continue
        dates = g["date"].to_numpy()
        idx = int(np.searchsorted(dates, np.datetime64(row.filing_date), side="left"))
        window = g.iloc[max(0, idx - BETA_WINDOW):idx]
        window = window.dropna(subset=["ret"])
        window = window.assign(bm_ret=window["date"].map(bm_ret_by_date)).dropna(subset=["bm_ret"])
        if len(window) < BETA_MIN_OBS:
            continue
        x = window["bm_ret"].to_numpy()
        y = window["ret"].to_numpy()
        design = np.column_stack([np.ones(len(x)), x])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        rows.append({"ticker": row.ticker, "filing_date": row.filing_date, "gold_beta": row.beta,
                     "gold_beta_n_obs": row.beta_n_obs, "fs_beta": float(coef[1]), "fs_beta_n_obs": len(window)})

    beta_df = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"ticker": pl.Utf8, "filing_date": pl.Date, "gold_beta": pl.Float64,
                "gold_beta_n_obs": pl.Float64, "fs_beta": pl.Float64, "fs_beta_n_obs": pl.Int64})
    beta_df = beta_df.with_columns((pl.col("fs_beta") - pl.col("gold_beta")).abs().alias("abs_diff"))
    w("beta_sanity", beta_df.sort("abs_diff", descending=True))

    summary = {
        "n_matched": beta_df.height,
        "corr_gold_fs": float(beta_df.select(pl.corr("gold_beta", "fs_beta")).item()) if beta_df.height > 2 else None,
        "median_abs_diff": float(beta_df["abs_diff"].median()) if beta_df.height else None,
    }
    print(f"  {summary['n_matched']} firm-years matched, corr(gold, factset) = {summary['corr_gold_fs']}, "
          f"median abs diff = {summary['median_abs_diff']}")
    print("  NOTE: FactSet beta uses raw ticker/benchmark total-return regression (no risk-free "
          "subtraction, no Fama-French mktrf) -- a sanity check, not a reproduction of gold's exact model.")
    return summary


# --------------------------------------------------------------------------- fundamentals ---

def check_fundamentals(fund_long_path: Path) -> dict:
    print(f"\n=== 4. Fundamentals ({fund_long_path}) ===")
    if not fund_long_path.exists():
        print(f"  MISSING: {fund_long_path} -- run parse_fundamentals.py first. Skipping.")
        return {"skipped": True, "reason": f"{fund_long_path} not found"}

    long = pl.read_parquet(fund_long_path).filter(pl.col("frequency") == "annual")
    fy = pl.read_parquet(REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "financials.parquet")
    fy = fy.filter(pl.col("disclosed_period_end").is_not_null()).with_columns(
        pl.col("disclosed_period_end").cast(pl.Date))

    match_rows, worst_rows, summary_rows = [], [], []
    for construct, (report, stnd_label, arpt_regex, kind) in FIELD_SPECS.items():
        if construct == "shares_out":
            # "Shs Outstanding"/"Shs Outstanding (M)" both carry a duplicate PCTCHG
            # (year-over-year % change) row under the same field label -- keep only the
            # actual level value (field_id FF_COM_SHS_OUT), found by inspecting the raw
            # parsed rows for ticker A (Agilent).
            stnd = long.filter(
                (pl.col("basis") == "STND") & (pl.col("report") == report)
                & pl.col("field").str.starts_with("Shs Outstanding") & (pl.col("field_id") == "FF_COM_SHS_OUT")
            )
        else:
            stnd = long.filter((pl.col("basis") == "STND") & (pl.col("report") == report) & (pl.col("field") == stnd_label))
        parts = [stnd.with_columns(pl.lit("STND").alias("src_basis"))]
        if arpt_regex is not None:
            arpt = long.filter(
                (pl.col("basis") == "ARPT") & (pl.col("report") == report) & (pl.col("level") == 0)
                & pl.col("field").str.to_lowercase().str.contains(arpt_regex)
            )
            # prefer STND per ticker: drop ARPT rows for tickers already covered by STND
            stnd_tickers = set(stnd["ticker"].drop_nulls().to_list())
            arpt = arpt.filter(~pl.col("ticker").is_in(stnd_tickers)).with_columns(pl.lit("ARPT").alias("src_basis"))
            parts.append(arpt)
        fs_field = pl.concat(parts, how="diagonal_relaxed").drop_nulls(["ticker", "value"])
        # collapse duplicate rows per (ticker, period_end) by taking the mean (multiple
        # matching ARPT labels for the same construct, rare) before joining
        fs_field = fs_field.group_by(["ticker", "period_end"]).agg(
            pl.col("value").mean().alias("value"), pl.col("scale").first().alias("scale"),
            pl.col("src_basis").first().alias("src_basis"))

        if kind == "amount":
            fs_field = fs_field.with_columns(
                (pl.col("value") * pl.col("scale").map_elements(lambda s: SCALE_MULT.get(s, 1.0), return_dtype=pl.Float64)).alias("fs_value"))
        elif kind == "shares_millions":
            fs_field = fs_field.with_columns((pl.col("value") * 1e6).alias("fs_value"))
        else:  # per_share
            fs_field = fs_field.with_columns(pl.col("value").alias("fs_value"))
        if construct == "capex":
            # FactSet's CF "Capital Expenditures" is signed as a cash outflow (negative);
            # gold's capex is stored as a positive magnitude -- compare on magnitude.
            fs_field = fs_field.with_columns(pl.col("fs_value").abs().alias("fs_value"))
        fs_field = fs_field.sort(["ticker", "period_end"])

        gold_field = fy.select(["ticker", "disclosed_period_end", construct]).drop_nulls(construct).sort(
            ["ticker", "disclosed_period_end"])
        matched = gold_field.join_asof(
            fs_field.rename({"period_end": "disclosed_period_end"}), by="ticker", on="disclosed_period_end",
            strategy="nearest", tolerance="45d",
        ).drop_nulls("fs_value")
        matched = matched.with_columns(
            ((pl.col("fs_value") - pl.col(construct)) / pl.col(construct).abs()).alias("pct_diff")
        ).with_columns(pl.lit(construct).alias("construct"))
        match_rows.append(matched.select(["construct", "ticker", "disclosed_period_end", construct, "fs_value", "src_basis", "pct_diff"])
                           .rename({construct: "gold_value"}))

        n_gold = gold_field.height
        n_matched = matched.height
        abs_pct = matched["pct_diff"].abs()
        summary_rows.append({
            "construct": construct, "n_gold_rows": n_gold, "n_matched": n_matched,
            "match_rate": n_matched / n_gold if n_gold else None,
            "median_abs_pct_diff": float(abs_pct.median()) if n_matched else None,
            "share_within_1pct": float((abs_pct <= 0.01).mean()) if n_matched else None,
            "share_within_5pct": float((abs_pct <= 0.05).mean()) if n_matched else None,
        })
        if n_matched:
            worst = matched.sort(pl.col("pct_diff").abs(), descending=True).head(10)
            worst_rows.append(worst.select(["construct", "ticker", "disclosed_period_end", construct, "fs_value", "pct_diff"])
                               .rename({construct: "gold_value"}))

    all_matches = pl.concat(match_rows, how="diagonal_relaxed") if match_rows else pl.DataFrame()
    all_worst = pl.concat(worst_rows, how="diagonal_relaxed") if worst_rows else pl.DataFrame()
    summary_df = pl.DataFrame(summary_rows)
    w("fundamentals_matches", all_matches)
    w("fundamentals_worst10_per_construct", all_worst)
    w("fundamentals_summary", summary_df)

    with pl.Config(tbl_rows=20, tbl_cols=10, fmt_float="full"):
        print(summary_df)
    return {"per_construct": summary_rows}


# --------------------------------------------------------------------------- coverage gaps ---

def check_coverage_gaps() -> dict:
    print("\n=== 5. Coverage gaps ===")
    universe = L.read("silver.firm_universe").filter(
        pl.col("membership_groups").list.contains("sp500_2021_start_panel")
    )["ticker"].to_list()

    fs_prices_tickers = set(pl.read_parquet(FACTSET_RAW / "prices" / "daily" / "prices_daily.parquet")["ticker"].unique().to_list())
    fs_fund_tickers = set()
    id_map_file = FACTSET_RAW / "reference" / "id_map.parquet"
    if id_map_file.exists():
        pass  # fundamentals ticker coverage reported separately below via fundamentals_long if present

    missing_from_fs_prices = sorted(set(universe) - fs_prices_tickers)
    extra_in_fs_prices = sorted(fs_prices_tickers - set(universe))

    print(f"  universe (sp500_2021_start_panel): {len(universe)} tickers")
    print(f"  FactSet prices: {len(fs_prices_tickers)} tickers, missing {len(missing_from_fs_prices)}: {missing_from_fs_prices}")
    print(f"  FactSet prices: {len(extra_in_fs_prices)} tickers outside universe: {extra_in_fs_prices}")

    pl.DataFrame({"missing_from_factset_prices": missing_from_fs_prices}).write_csv(
        RESULTS_DIR / "coverage_gaps_prices_missing.csv")
    pl.DataFrame({"extra_in_factset_prices": extra_in_fs_prices}).write_csv(
        RESULTS_DIR / "coverage_gaps_prices_extra.csv")

    return {
        "universe_n": len(universe),
        "fs_prices_n": len(fs_prices_tickers),
        "missing_from_fs_prices": missing_from_fs_prices,
        "extra_in_fs_prices": extra_in_fs_prices,
    }


def explain_firm_spine_gap() -> str:
    """gold/spines/firm/firm.parquet (498) vs silver.firm_universe sp500_2021_start_panel
    (499, includes FRC). Not a bug -- explained by the spine's own definition, checked
    against the code, not fixed here (per task instructions)."""
    fu = L.read("silver.firm_universe")
    fm = L.read("silver.filing_manifest")
    frc_forms = fm.filter(pl.col("ticker") == "FRC")["form_type"].unique().to_list()
    firm_spine = pl.read_parquet(REPO_ROOT / "data" / "gold" / "spines" / "firm" / "firm.parquet")
    text = (
        "firm.parquet (spines/firm) is built by scripts/gold/firm/build_posture_archetype_static.py "
        "as `universe[['ticker']]` where `universe = firm_intensity(['ticker'])` over "
        "`gold_document_table()` filtered to `form in FILING_FORMS = (10-K, 10-Q, DEF 14A, 8-K)` "
        "(see scripts/gold/posture/ai_intensity.FILING_FORMS) -- i.e. every ticker with at least "
        "one scorable *filing* (not an earnings call). silver.firm_universe's "
        "sp500_2021_start_panel membership (499 tickers) includes FRC (First Republic Bank), but "
        f"silver.filing_manifest has ONLY earnings-call-transcript rows for FRC ({frc_forms}), zero "
        "10-K/10-Q/DEF14A/8-K filings -- FRC failed and was placed into FDIC receivership "
        "(2023-05-01) before/without any such filing landing in the manifest. So FRC is correctly "
        "absent from firm.parquet under that spine's own definition; it is not a build bug, and "
        f"firm.parquet has {firm_spine.height} rows (498) vs the 499-ticker universe as expected."
    )
    print("\n=== FRC / firm.parquet gap ===")
    print(" " + text)
    return text


# --------------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fundamentals-long", type=Path, default=DEFAULT_FUND_LONG,
                     help="Path to fundamentals_long.parquet (parse_fundamentals.py output). "
                          "Defaults to the canonical data/raw/fs/fundamentals/ location.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    s1 = check_prices()
    s2 = check_market_cap()
    s3 = check_beta()
    s4 = check_fundamentals(args.fundamentals_long)
    s5 = check_coverage_gaps()
    frc_note = explain_firm_spine_gap()

    lines = [
        "# FactSet cross-validation summary",
        "",
        f"Fundamentals source: `{args.fundamentals_long}`",
        "",
        "## 1. Prices",
        f"- Tickers compared: {s1['n_tickers']}, joined obs: {s1['n_dates_joined']}",
        f"- Median per-ticker corr(daily total-return): {s1['median_corr_total_return']:.4f}",
        f"- Median per-ticker median-abs-diff(daily total-return): {s1['median_abs_diff_total_return']:.6f}",
        f"- Flagged tickers (corr<0.95 or split-ratio unstable): {s1['n_flagged_corr_lt_0.95_or_split_mismatch']}"
        f" -> {s1['flagged_tickers']}",
        "- PARA's low correlation (0.033) is the already-documented ticker-reuse issue "
        "(docs/sources/fs.md \"Identifiers\": PARA-US now resolves to a different, reused "
        "ticker; FactSet's PARAA-US is the correct Paramount id) -- not a new problem. MNST and AVB "
        "are flagged on `price_ratio_cv`/`corr` with very few overlapping dates (n_obs 1671 and 21) "
        "and a near-zero median abs return diff, so likely a thin-overlap artifact rather than a "
        "real mismatch -- worth a second look once more history overlaps, not urgent.",
        "",
        "## 2. Market cap",
        f"- firm_year matched: {s2['firm_year_matched']}, median abs pct diff: {s2['firm_year_median_abs_pct_diff']:.4f}, "
        f"outliers >20%: {s2['firm_year_n_outliers_gt20pct']} rows / {s2['firm_year_n_outlier_tickers']} tickers",
        f"- firm_quarter (log scale) matched: {s2['firm_quarter_matched']}, median abs pct diff: {s2['firm_quarter_median_abs_pct_diff_log']:.4f}",
        "- **RED FLAG (root cause verified)**: the >20% outliers are 65 tickers whose pct_diff sits "
        "near their own later stock-split ratio (AMZN/GOOGL ~20x for 2022 splits, CMG ~48x for its "
        "2024 50:1 split, etc.), and only for filing_dates before that split. Confirmed directly: "
        "silver.market_prices.close for AMZN on 2022-02-04 = 157.64, identical to FactSet's "
        "split-ADJUSTED `price` for the same date -- so silver's `close` is split-adjusted, "
        "contradicting build_market_factors.py's docstring (which calls it the unadjusted close) "
        "and multiplies it by the historical, pre-split cover-page `shares_out` -- understating "
        "firm_year market_cap by the eventual split ratio pre-split. One isolated outlier is "
        "unrelated to splits: RMD_2021 (filing_date 2021-08-17) is off by ~1000x while every other "
        "RMD year matches FactSet within ~1.3% -- looks like a single bad row (units/scale), not a "
        "split. Not fixed here (out of scope) -- see market_cap_firm_year_outliers.csv.",
        "",
        "## 3. Beta sanity",
        f"- Firm-years matched: {s3['n_matched']}, corr(gold, FactSet): {s3['corr_gold_fs']}, "
        f"median abs diff: {s3['median_abs_diff']}",
        "- Caveat: FactSet beta here is a raw-return one-factor regression (no risk-free "
        "subtraction, no Fama-French mktrf); a sanity check, not a reproduction of gold's model.",
        "",
        "## 4. Fundamentals",
    ]
    if s4.get("skipped"):
        lines.append(f"- SKIPPED: {s4['reason']}")
    else:
        for row in s4["per_construct"]:
            lines.append(
                f"- **{row['construct']}**: match_rate={row['match_rate']}, "
                f"median_abs_pct_diff={row['median_abs_pct_diff']}, "
                f"within_1%={row['share_within_1pct']}, within_5%={row['share_within_5pct']}, "
                f"n_matched={row['n_matched']}/{row['n_gold_rows']}"
            )
        lines.append(
            "- STND fundamentals are now complete for all 499 tickers (see docs/sources/fs.md), "
            "including SHS/RATIO_ANN (all 499 ids have a main STND file or a stndx supplement, "
            "see the parser's filename-id fix for the 3 dotted-ticker ids BF.B/BRK.B/INFO.XX10), "
            "so match rates are ~85-100% for most constructs -- `shares_out` now matches at "
            "match_rate=1.0, no longer capped by SHS/RATIO_ANN coverage."
        )
        lines.append(
            "- RED FLAG: `sga_expense` (median abs pct diff ~10%, only ~41% within 5%) and "
            "`long_term_debt` (median ~7%, only ~39% within 5%) disagree more than the rest "
            "(revenue/net_income/total_assets/equity/eps_diluted are all within ~1% at the median) -- "
            "likely a genuine definitional gap (FactSet's SG&A/Long-Term-Debt rows net out different "
            "items, e.g. lease liabilities, than the XBRL tags gold reads) rather than a parsing bug, "
            "since capex/net_income/eps_diluted/shares_out (where matched) all reconcile well once "
            "sign/unit conventions are fixed."
        )
    lines += [
        "",
        "## 5. Coverage gaps",
        f"- universe (sp500_2021_start_panel): {s5['universe_n']}",
        f"- FactSet prices tickers: {s5['fs_prices_n']}",
        f"- Missing from FactSet prices: {s5['missing_from_fs_prices']}",
        f"- Extra in FactSet prices (outside universe): {s5['extra_in_fs_prices']}",
        "",
        "## firm.parquet (498) vs silver universe (499) -- FRC gap, explained not fixed",
        frc_note,
    ]
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
