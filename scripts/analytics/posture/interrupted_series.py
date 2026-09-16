"""
scripts/analytics/posture/interrupted_series.py -- did market-wide AI discourse
break at the SEC's 2024 warning or at DeepSeek's release?

The question is about a break in the AGGREGATE series, not a difference between
groups, so the design is an interrupted time series rather than a
difference-in-differences:

    Y_t = alpha + beta*t + tau*Post_t + delta*(t - T0)*Post_t + season + e_t

`beta` is the trend already running, `tau` the immediate step at the event, and
`delta` the change in slope afterwards. Date fixed effects are deliberately
absent: they would absorb the common movement the design exists to measure. A
differential design answers a different question and can miss this one, since a
shock that moves every firm alike leaves no differential to detect.

Series are MONTHLY and expressed per 1,000 words, so a step cannot come from
a quarter simply carrying more documents -- the raw count of open-source
paragraphs rose from 29 to 99 between 2024Q4 and 2025Q1, and part of any such
rise is filings season. Reporting seasonality enters as quarter-of-year dummies
and the standard errors are Newey-West, since consecutive quarters of a
disclosure series are not independent.

What it cannot do: a break at one date cannot be separated from anything else
happening at that date. It discounts trend and seasonality; it does not
identify a cause.

Output: results/posture/interrupted_series.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
import layers as L  # noqa: E402

EVENTS = {"SEC warning": "2023-12", "DeepSeek-R1": "2025-01"}
RATES = {"open_source": "Open-weight vocabulary", "promo": "Promotional frames",
         "risk": "Risk frames", "gov": "Governance frames", "frames": "AI frames"}
HAC_LAGS = 6          # roughly two reporting quarters
START = "2021-01"     # the corpus before this is the panel ramping up -- 2020
                      # months carry 3 to 30 documents against a median of 833
MIN_DOCS = 50         # and the final month is truncated by the cutoff


def monthly() -> pd.DataFrame:
    """One row per month: frames of each kind per 1,000 words, over every
    document published that month, with the reporting mix that produced it.
    Monthly rather than quarterly because the series is what carries the
    identification here: 68 usable months against 26 quarters."""
    docs = L.read_gold("document", ("covariates", "disclosure_volume"))
    docs = docs[docs["fecha"].notna()].copy()
    docs["q"] = pd.PeriodIndex(pd.to_datetime(docs["fecha"]), freq="M")
    cols = {f"n_{k}": "sum" for k in RATES}
    agg = docs.groupby("q").agg(**{k: (k, "sum") for k in cols},
                                n_words=("n_words", "sum"),
                                n_docs=("accession_number", "size"),
                                n_firms=("ticker", "nunique"))
    for k in RATES:
        agg[f"{k}_per_1k"] = 1000.0 * agg[f"n_{k}"] / agg["n_words"]
    agg = agg[(agg.index >= pd.Period(START, freq="M")) & (agg["n_docs"] >= MIN_DOCS)]
    return agg.reset_index()


def fit(series: pd.DataFrame, rate: str, event: str) -> dict:
    s = series.dropna(subset=[f"{rate}_per_1k"]).reset_index(drop=True).copy()
    ev = pd.Period(event, freq="M")
    s["t"] = (s["q"] - s["q"].min()).apply(lambda p: p.n)
    s["since"] = (s["q"] - ev).apply(lambda p: p.n)
    s["post"] = (s["since"] >= 0).astype(float)
    s["slope"] = s["since"] * s["post"]
    season = pd.get_dummies(s["q"].dt.month, prefix="M", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([s[["t", "post", "slope"]], season,
                                   np.log(s[["n_firms"]]).rename(columns={"n_firms": "log_firms"})], axis=1))
    r = sm.OLS(s[f"{rate}_per_1k"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return {"event": event, "outcome": RATES[rate], "quarters": len(s),
            "pre_trend": float(r.params["t"]), "p_pre_trend": float(r.pvalues["t"]),
            "step": float(r.params["post"]), "p_step": float(r.pvalues["post"]),
            "slope_change": float(r.params["slope"]), "p_slope_change": float(r.pvalues["slope"]),
            "r2": float(r.rsquared)}


def main() -> None:
    warnings.filterwarnings("ignore")
    s = monthly()
    print(f"serie: {len(s)} meses, {s['q'].min()} .. {s['q'].max()}\n")
    rows = [fit(s, rate, event) for event in EVENTS.values() for rate in RATES]
    out = pd.DataFrame(rows)
    path = L.results_path("posture", "interrupted_series.csv")
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200):
        print(out.round(4).to_string(index=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
