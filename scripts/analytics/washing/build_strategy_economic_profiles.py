"""Economic profiles of the posture archetypes (RQ3): the static posture
archetype of each firm-year (covariates/firm_year/posture_archetype_static,
model fit by scripts/gold/firm/build_posture_archetype_static.py).

Panels:
  A  crude median by archetype: characterization.
  B  sector x year residualized: X[i,t] = a[sector x t] + g_k*Archetype[i,k] + e,
     firm-clustered SE, reference = No AI.

Output: `strategy_economic_profiles.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
import layers as L  # noqa: E402
from fills import roic_wacc_debt_as_zero  # noqa: E402

YEARS = (2021, 2022, 2023, 2024, 2025)
VARS = {"log_market_cap": "log market cap", "rd_intensity": "R&D/sales", "gross_margin": "gross margin",
        "operating_margin": "operating margin", "next_revenue_yoy": "revenue growth (t+1)", "beta": "beta",
        "vol_pre_60d": "pre-filing volatility", "ps_ratio": "P/S", "roic_minus_wacc": "ROIC - WACC"}
ORDER = ["No AI", "Defensive Disclosers", "Governance-Led Disclosers", "Vocal Substantives"]
REF = "No AI"
TIMING = ("descriptive, contemporaneous: the static full-sample archetype of the firm-year against outcomes of the "
          "same firm-year (including next_revenue_yoy); not a point-in-time prediction")


def winsor(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def main() -> None:
    m = L.read_dataset(
        "firm_year",
        ("covariates", "financials", ["rd_intensity", "gross_margin", "operating_margin", "sic2", "nopat", "equity",
                                      "long_term_debt", "cost_of_equity", "cost_of_debt", "effective_tax_rate"]),
        ("covariates", "market", ["market_cap", "beta", "vol_pre_60d", "ps_ratio"]),
        ("targets", "financials", ["next_revenue_yoy"]),
        ("covariates", "posture_archetype_static", ["archetype"]),
    )
    m = m[m["year"].isin(YEARS)].copy()
    # Unreported long-term debt read as zero debt (gold leaves ROIC - WACC null there).
    m["roic_minus_wacc"] = roic_wacc_debt_as_zero(m)["roic_minus_wacc"]
    m["archetype"] = m["archetype"].fillna("No AI")
    m["log_market_cap"] = np.log(m["market_cap"])
    for v in VARS:
        m[v] = winsor(m[v])
    m["sector_year"] = m["sic2"].astype(str) + "_" + m["year"].astype(str)
    print(f"panel: {len(m):,} firm-years, {m.ticker.nunique()} firms | by archetype: {m.archetype.value_counts().to_dict()}\n")

    print("PANEL A -- crude median by archetype")
    a = m.groupby("archetype")[list(VARS)].median().reindex(ORDER)
    a["n"] = m.groupby("archetype").size().reindex(ORDER)
    print(a.round(3).T.to_string())

    print("\nPANEL B -- sector x year residualized: gamma of each archetype vs. 'No AI' (firm-clustered SE)")
    rows = {}
    for v, label in VARS.items():
        d = m.dropna(subset=[v, "sector_year", "archetype"])
        d = d[d.groupby("sector_year")[v].transform("size") >= 2]
        X = pd.DataFrame({f"seg_{s}": (d["archetype"] == s).astype(float) for s in ORDER if s != REF})
        g = d["sector_year"].to_numpy()
        Xd = X - X.groupby(g).transform("mean")
        yd = d[v] - d[v].groupby(g).transform("mean")
        res = sm.OLS(yd.to_numpy(dtype=float), Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
        rows[v] = {s: {"gamma": float(res.params[f"seg_{s}"]), "se": float(res.bse[f"seg_{s}"]), "p": float(res.pvalues[f"seg_{s}"])}
                   for s in ORDER if s != REF}
        rows[v]["n"] = int(len(d))
        rows[v]["sd"] = float(d[v].std())
        print(f"  {label:24s} n={len(d):5d} | " + " | ".join(
            f"{s}: {rows[v][s]['gamma']:+.3f} (p={rows[v][s]['p']:.3f})" for s in ORDER if s != REF))
    payload = {"timing": TIMING, "years": YEARS, "panel_a": json.loads(a.to_json(orient="index")), "panel_b": rows,
               "n_by_archetype": {k: int(v) for k, v in m.archetype.value_counts().items()}}
    destination = L.results_path("washing", "strategy_economic_profiles.json")
    destination.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
