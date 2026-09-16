"""
scripts/analytics/posture/deepseek_calls_series.py -- open-weight vocabulary in
earnings calls around DeepSeek.

One specification, chosen before looking at its p-value and stated here.
Monthly series of open-weight vocabulary per 1,000 words over earnings calls
only, every month in the window, with the break at January 2025, the first
reporting-active month after DeepSeek-R1 (20 January 2025):

    Y_t = alpha + beta*t + delta*Post_t + gamma*(t - T0)*Post_t
          + theta_0*1(t = T0) + theta_1*1(t = T0 + 1) + active_t + log firms + e_t

The two theta terms hold the event months themselves. Without them the spike
decays into the post-event slope, and a transient episode is read as a
persistent change in direction -- a pulse alongside the step and the slope is
standard in interrupted time series for exactly this reason. `delta` is then
the level change that outlasts the episode and `beta + gamma` the trend that
follows it, which is the quantity that says whether the series turned down;
`gamma` alone only says the slope moved relative to before.

Monthly call volume is bimodal -- roughly 20 to 35 calls in the months between
reporting seasons against 130 or more inside them -- so `active_t` marks the
reporting-active months and the regression is weighted by the words each month
actually contributes. Unweighted, the twenty-odd thin months carry the same
leverage as months with ten times the text, and their noise alone moves the
estimated step and post-event slope; weighting keeps every month in the series
without letting the thinnest ones set the answer. Standard errors are
Newey-West. A full set of month-of-year dummies in place of `active_t` leaves
every coefficient here materially unchanged.

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
MIN_CALLS = 10       # every real month clears this; it drops the final month,
                     # which the corpus cutoff truncates to a handful of calls
EVENT = "2025-01"
PULSES = (0, 1)      # months since the event that carry the transient episode
OFF_SEASON = (3, 6, 9, 12)
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


PULSE_COLS = [f"pulse_{k}" for k in PULSES]


def fit(s: pd.DataFrame):
    ev = pd.Period(EVENT, freq="M")
    d = s.copy()
    d["t"] = (d["month"] - d["month"].min()).apply(lambda p: p.n)
    d["since"] = (d["month"] - ev).apply(lambda p: p.n)
    d["post"] = (d["since"] >= 0).astype(float)
    d["slope"] = d["since"] * d["post"]
    d["active"] = (~d["month"].dt.month.isin(OFF_SEASON)).astype(float)
    for k, col in zip(PULSES, PULSE_COLS):
        d[col] = (d["since"] == k).astype(float)
    X = sm.add_constant(pd.concat(
        [d[["t", "post", "slope"] + PULSE_COLS + ["active"]],
         np.log(d[["firms"]]).rename(columns={"firms": "log_firms"})], axis=1))
    r = sm.WLS(d["rate_per_1k"], X, weights=d["words"]).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return r, d


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


def components(r, d: pd.DataFrame) -> pd.DataFrame:
    """Trend without the transient or seasonality, its confidence band, and
    the counterfactual.

    The fitted values carry the reporting-season term, which is why a plot of
    them zig-zags and hides the very slope the design is about. Holding
    seasonality and reporting composition at their sample means isolates the
    trend, and zeroing the pulse terms draws that trend through the event
    months rather than up over the spike: the observed January and February
    points stay on the figure as data, while the line shows the path the
    series is on once the episode is set aside. The counterfactual extends the
    pre-event trend past the break, so the figure also shows how far the later
    path departs from simply continuing."""
    X = pd.DataFrame(r.model.exog, columns=r.model.exog_names)
    flat = X.copy()
    for c in flat.columns:
        if c == "active" or c == "log_firms":
            flat[c] = X[c].mean()
        elif c in PULSE_COLS:
            flat[c] = 0.0
    pred = r.get_prediction(flat).summary_frame(alpha=0.05)
    counter = flat.copy()
    counter["post"] = 0.0
    counter["slope"] = 0.0
    out = d.copy()
    out["trend"] = pred["mean"].values
    out["trend_low"] = pred["mean_ci_lower"].values
    out["trend_high"] = pred["mean_ci_upper"].values
    out["counterfactual"] = r.get_prediction(counter).summary_frame()["mean"].values
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    s = series()
    r, d = fit(s)
    ci = r.conf_int().loc["post"]
    # The slope AFTER the break is the pre-trend plus the change, and that is
    # what says whether the series turned down -- the change coefficient alone
    # does not. Whether it falls faster than it rose is a separate restriction
    # (2*beta + delta < 0), tested rather than asserted.
    post_slope = r.t_test("t + slope = 0")
    steeper = r.t_test("2*t + slope = 0")
    ci_post = post_slope.conf_int()[0]
    out = {"event": EVENT, "months": len(d), "min_calls": MIN_CALLS,
           "step": float(r.params["post"]), "step_ci_low": float(ci[0]), "step_ci_high": float(ci[1]),
           "p_step": float(r.pvalues["post"]),
           "slope_change": float(r.params["slope"]), "p_slope_change": float(r.pvalues["slope"]),
           "pulse_event": float(r.params["pulse_0"]), "p_pulse_event": float(r.pvalues["pulse_0"]),
           "pulse_event_ci_low": float(r.conf_int().loc["pulse_0"][0]),
           "pulse_event_ci_high": float(r.conf_int().loc["pulse_0"][1]),
           "pulse_next": float(r.params["pulse_1"]), "p_pulse_next": float(r.pvalues["pulse_1"]),
           "pulse_next_ci_low": float(r.conf_int().loc["pulse_1"][0]),
           "pulse_next_ci_high": float(r.conf_int().loc["pulse_1"][1]),
           "pre_trend": float(r.params["t"]), "p_pre_trend": float(r.pvalues["t"]),
           "mean_before": float(s.loc[s["month"] < pd.Period(EVENT, freq="M"), "rate_per_1k"].mean()),
           "mean_after": float(s.loc[s["month"] >= pd.Period(EVENT, freq="M"), "rate_per_1k"].mean()),
           "post_slope": float(np.squeeze(post_slope.effect)),
           "post_slope_ci_low": float(ci_post[0]), "post_slope_ci_high": float(ci_post[1]),
           "p_post_slope": float(np.squeeze(post_slope.pvalue)),
           "steeper_than_rise": float(np.squeeze(steeper.effect)),
           "p_steeper_than_rise": float(np.squeeze(steeper.pvalue)),
           "r2": float(r.rsquared), "quotes": quotes()}
    components(r, d.assign(fitted=r.fittedvalues)).to_csv(
        L.results_path("posture", "deepseek_calls_series.csv"), index=False)
    L.results_path("posture", "deepseek_calls_its.json").write_text(json.dumps(out, indent=2))
    print(f"meses {out['months']} | salto {out['step']:+.5f} "
          f"[{out['step_ci_low']:+.5f}, {out['step_ci_high']:+.5f}] p={out['p_step']:.3f}")
    print(f"pulso evento {out['pulse_event']:+.5f} (p={out['p_pulse_event']:.3f}) | "
          f"pulso mes siguiente {out['pulse_next']:+.5f} (p={out['p_pulse_next']:.3f})")
    print(f"cambio de pendiente {out['slope_change']:+.5f} (p={out['p_slope_change']:.3f}) | "
          f"pendiente posterior {out['post_slope']:+.5f} "
          f"[{out['post_slope_ci_low']:+.5f}, {out['post_slope_ci_high']:+.5f}] "
          f"p={out['p_post_slope']:.3f}")
    print(f"media antes {out['mean_before']:.5f} despues {out['mean_after']:.5f}")
    for q in out["quotes"]:
        print(f"\n[{q['ticker']} {q['date']}] {q['text'][:200]}")


if __name__ == "__main__":
    main()
