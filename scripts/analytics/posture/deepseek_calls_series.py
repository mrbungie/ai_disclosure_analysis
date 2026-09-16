"""
scripts/analytics/posture/deepseek_calls_series.py -- open-weight vocabulary in
earnings calls around DeepSeek.

One specification, chosen before looking at its p-value and stated here.
Monthly series of open-weight vocabulary per 1,000 words over earnings calls
only; months carrying fewer than MIN_CALLS calls are dropped, because a month
with thirty calls has a noisy rate that pulls the estimated break toward
whichever light month sits next to the event; the break is January 2025, the
first reporting-active month after DeepSeek-R1 (20 January 2025); seasonality
enters as month-of-year dummies, reporting composition as the log number of
firms, and standard errors are Newey-West.

    Y_t = alpha + beta*t + tau*Post_t + delta*(t - T0)*Post_t + season + e_t

Calls rather than the pooled corpus. A call can respond within weeks, while a
10-K published in a given month was written earlier and describes an earlier
period, so pooling mixes documents whose timing relative to an event differs.
That the pooled series shows nothing is consistent with dilution but does not
by itself establish it.

What the estimate does and does not settle: the break is dated to a month, and
January is not entirely after the 20th, so the design locates a change around
the turn of the year rather than on a day. What connects it to DeepSeek is the
text -- analysts name the model in those calls (`quotes` below) -- not the
timing alone.

Output: results/posture/deepseek_calls_series.csv (the monthly series),
        results/posture/deepseek_calls_its.json (the fit and the quotes)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

START = "2021-01"
MIN_CALLS = 100
EVENT = "2025-01"
HAC_LAGS = 6
QUOTE_PATTERN = r"(?i)(^|[^a-z0-9])(deepseek)([^a-z0-9]|$)"


def series() -> pd.DataFrame:
    docs = L.read_gold("document", ("covariates", "disclosure_volume"))
    calls = docs[(docs["form"] == "Earnings call") & docs["fecha"].notna()].copy()
    calls["month"] = pd.PeriodIndex(pd.to_datetime(calls["fecha"]), freq="M")
    agg = calls.groupby("month").agg(open_source=("n_open_source", "sum"),
                                     words=("n_words", "sum"),
                                     calls=("accession_number", "size"),
                                     firms=("ticker", "nunique"))
    agg["rate_per_1k"] = 1000.0 * agg["open_source"] / agg["words"]
    return agg[(agg.index >= pd.Period(START, freq="M")) & (agg["calls"] >= MIN_CALLS)].reset_index()


def fit(s: pd.DataFrame):
    ev = pd.Period(EVENT, freq="M")
    d = s.copy()
    d["t"] = (d["month"] - d["month"].min()).apply(lambda p: p.n)
    d["since"] = (d["month"] - ev).apply(lambda p: p.n)
    d["post"] = (d["since"] >= 0).astype(float)
    d["slope"] = d["since"] * d["post"]
    X = sm.add_constant(pd.concat(
        [d[["t", "post", "slope"]],
         pd.get_dummies(d["month"].dt.month, prefix="M", drop_first=True, dtype=float),
         np.log(d[["firms"]]).rename(columns={"firms": "log_firms"})], axis=1))
    return sm.OLS(d["rate_per_1k"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}), d


def quotes(limit: int = 3) -> list[dict]:
    """Dated call paragraphs naming the model, which is what ties the series to
    the event rather than to the calendar."""
    up = pl.scan_parquet(L.BRONZE / "unique_paragraphs.parquet")
    hits = (up.filter((pl.col("form") == "Earnings call")
                      & pl.col("paragraph_text").str.contains(QUOTE_PATTERN))
            .select("accession_number", "paragraph_text").collect().to_pandas())
    docs = L.read_gold("document", ("covariates", "disclosure_volume"))[["accession_number", "ticker", "fecha"]]
    hits = hits.merge(docs, on="accession_number", how="left").sort_values("fecha")
    hits["length"] = hits["paragraph_text"].str.len()
    pick = hits[(hits["length"] > 180) & (hits["length"] < 700)].head(limit)
    return [{"ticker": r.ticker, "date": str(pd.to_datetime(r.fecha).date()),
             "text": r.paragraph_text.strip()} for r in pick.itertuples()]


def main() -> None:
    warnings.filterwarnings("ignore")
    s = series()
    r, d = fit(s)
    ci = r.conf_int().loc["post"]
    out = {"event": EVENT, "months": len(d), "min_calls": MIN_CALLS,
           "step": float(r.params["post"]), "step_ci_low": float(ci[0]), "step_ci_high": float(ci[1]),
           "p_step": float(r.pvalues["post"]),
           "slope_change": float(r.params["slope"]), "p_slope_change": float(r.pvalues["slope"]),
           "pre_trend": float(r.params["t"]), "p_pre_trend": float(r.pvalues["t"]),
           "mean_before": float(s.loc[s["month"] < pd.Period(EVENT, freq="M"), "rate_per_1k"].mean()),
           "mean_after": float(s.loc[s["month"] >= pd.Period(EVENT, freq="M"), "rate_per_1k"].mean()),
           "r2": float(r.rsquared), "quotes": quotes()}
    d.assign(fitted=r.fittedvalues).to_csv(L.results_path("posture", "deepseek_calls_series.csv"), index=False)
    L.results_path("posture", "deepseek_calls_its.json").write_text(json.dumps(out, indent=2))
    print(f"meses {out['months']} | salto {out['step']:+.5f} "
          f"[{out['step_ci_low']:+.5f}, {out['step_ci_high']:+.5f}] p={out['p_step']:.3f}")
    print(f"pendiente posterior {out['slope_change']:+.5f} (p={out['p_slope_change']:.3f}) | "
          f"media antes {out['mean_before']:.5f} despues {out['mean_after']:.5f}")
    for q in out["quotes"]:
        print(f"\n[{q['ticker']} {q['date']}] {q['text'][:200]}")


if __name__ == "__main__":
    main()
