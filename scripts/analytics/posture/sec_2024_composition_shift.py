"""
scripts/analytics/posture/sec_2024_composition_shift.py -- did posture
composition move around 2024Q1?

Design: firm-quarter panel, continuous pre-event exposure (mean posture
intensity over 2022, standardized) interacted with the post-2024Q1 indicator,
firm fixed effects, and time absorbed two ways -- quarter, then SECTOR x
QUARTER. The second absorbs anything common to an industry in a quarter, so
what identifies the coefficient is exposure varying WITHIN a sector-quarter.

What that does and does not remove. It removes the sectoral component of the
generative-AI boom. It does not remove the boom: firms that were already more
exposed may respond more to it than their own sector's average, and that
differential response is observationally identical to a response to scrutiny.
The estimate is a differential recomposition after 2024Q1, not an effect of the
SEC.

The three weights sum to one, so their coefficients sum to zero: this is ONE
compositional movement reported three ways, not three findings.

Inference note: with firm and sector-quarter dummies the design reaches ~920
parameters against 458 firm clusters. The model is estimable (rank 921, 2,533
residual degrees of freedom) but the cluster-robust covariance has rank at most
G-1 = 457, so it is singular and standard errors come out undefined. Absorbing
the fixed effects by alternating projections estimates the same coefficient
with one parameter, where clustering is well behaved.

Pre-trends are reported as a test that fails to reject, which is not proof of
parallel trends [@roth2022].

Output: results/posture/sec_2024_composition_shift.csv
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
import layers as L  # noqa: E402

EVENT = "2024Q1"
PLACEBOS = ("2023Q1", "2022Q3")
EXPOSURE_YEAR = ("2022Q1", "2022Q4")
WINDOW = 5
WEIGHTS = {"posture_ttm_w_voc": "Vocal", "posture_ttm_w_gov": "Governance",
           "posture_ttm_w_def": "Defensive"}


def panel() -> pd.DataFrame:
    weights = L.read_gold("firm_quarter", ("covariates", "posture_archetype"))
    volume = L.read_gold("firm_quarter", ("covariates", "disclosure_volume"))[
        ["ticker", "quarter", "posture_ttm_frames_per_1k"]]
    d = weights.merge(volume, on=["ticker", "quarter"], how="left")
    d["q"] = pd.PeriodIndex(d["quarter"], freq="Q")
    lo, hi = (pd.Period(p, freq="Q") for p in EXPOSURE_YEAR)
    exposure = (d[(d["q"] >= lo) & (d["q"] <= hi)]
                .groupby("ticker")["posture_ttm_frames_per_1k"].mean().rename("exposure"))
    d = d.merge(exposure, on="ticker", how="left").dropna(subset=["exposure"])
    d["exposure_z"] = (d["exposure"] - d["exposure"].mean()) / d["exposure"].std()
    return d


def absorb(frame: pd.DataFrame, columns: list[str], groups: list[str], iters: int = 40) -> pd.DataFrame:
    """Alternating projections: subtract each group's mean until they stop moving."""
    out = frame[columns].astype(float).copy()
    for _ in range(iters):
        for g in groups:
            out = out - out.groupby(frame[g]).transform("mean")
    return out


def estimate(d: pd.DataFrame, event: str) -> list[dict]:
    ev = pd.Period(event, freq="Q")
    x = d.assign(k=(d["q"] - ev).apply(lambda t: t.n))
    x = x[(x["k"] >= -WINDOW) & (x["k"] <= WINDOW)].copy()
    x["did"] = x["exposure_z"] * (x["k"] >= 0).astype(float)
    x["sector_quarter"] = x["sic2"].astype(str) + "_" + x["k"].astype(str)
    rows = []
    for col, label in WEIGHTS.items():
        s = x.dropna(subset=[col, "sic2"]).copy()
        row = {"event": event, "weight": label, "n": len(s), "firms": s["ticker"].nunique()}
        for name, groups in [("firm+quarter", ["ticker", "k"]),
                             ("firm+sector-quarter", ["ticker", "sector_quarter"])]:
            z = absorb(s, [col, "did"], groups)
            r = sm.OLS(z[col], z[["did"]]).fit(cov_type="cluster", cov_kwds={"groups": s["ticker"]})
            row[f"beta_{name}"] = float(r.params["did"])
            row[f"p_{name}"] = float(r.pvalues["did"])
        pre = s[s["k"] < 0].copy()
        pre["trend"] = pre["exposure_z"] * pre["k"]
        zp = absorb(pre, [col, "trend"], ["ticker", "k"])
        rp = sm.OLS(zp[col], zp[["trend"]]).fit(cov_type="cluster", cov_kwds={"groups": pre["ticker"]})
        row["beta_pretrend"], row["p_pretrend"] = float(rp.params["trend"]), float(rp.pvalues["trend"])
        rows.append(row)
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    d = panel()
    rows = [r for event in (EVENT, *PLACEBOS) for r in estimate(d, event)]
    out = pd.DataFrame(rows)
    path = L.results_path("posture", "sec_2024_composition_shift.csv")
    out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
