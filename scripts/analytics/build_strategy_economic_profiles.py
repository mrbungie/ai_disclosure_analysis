"""Economic profiles of the strategy archetypes (RQ3), on the new
factor-analytic segmentation from `build_strategy_dimensions.py`.

Mirrors `economic_profiles.py` exactly (same variables, same sector-year
residualization, same winsorization), swapping the old 6-feature K-means
segment (`firm_year_segments.segmento`) for the new factor-analytic
archetype (`firm_year_strategy_dimensions.archetype`). Kept as a separate
script rather than editing `economic_profiles.py` in place, since the old
segmentation is still reported in Appendix C as a superseded robustness
check and needs its own economic profile to remain reproducible.

Panels:
  A  crude median by archetype: characterization.
  B  sector x year residualized: X[i,t] = a[sector x t] + g_k*Archetype[i,k] + e,
     firm-clustered SE, reference = No AI.

Output: `strategy_economic_profiles.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
YEARS = (2021, 2022, 2023, 2024, 2025)
VARS = {"log_market_cap": "log market cap", "rd_intensity": "R&D/sales", "gross_margin": "gross margin",
        "operating_margin": "operating margin", "next_revenue_yoy": "revenue growth (t+1)", "beta": "beta",
        "vol_pre_60d": "pre-filing volatility", "ps_ratio": "P/S", "roic_minus_wacc": "ROIC - WACC"}
ORDER = ["No AI", "Defensive Disclosers", "Governance-Led Disclosers", "Vocal Substantives"]
REF = "No AI"


def winsor(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def main() -> None:
    m = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")
    m = m[m["year"].isin(YEARS)].copy()
    rw = pd.read_parquet(OUT_DIR / "firm_year_roic_wacc.parquet")[["ticker", "year", "roic_minus_wacc"]]
    # `firm_year_master_v2` already carries an `archetype`/`archetype_dist` pair
    # from the OLD voice-archetype pipeline (`build_firm_clusters.py`) -- drop
    # those before merging so the new archetype doesn't get an `_y` suffix.
    m = m.drop(columns=[c for c in ("archetype", "archetype_dist") if c in m.columns])
    seg = pd.read_parquet(OUT_DIR / "firm_year_strategy_dimensions.parquet")[["ticker", "year", "archetype"]]
    m = m.merge(rw, on=["ticker", "year"], how="left").merge(seg, on=["ticker", "year"], how="left")
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
    payload = {"years": YEARS, "panel_a": json.loads(a.to_json(orient="index")), "panel_b": rows,
               "n_by_archetype": {k: int(v) for k, v in m.archetype.value_counts().items()}}
    (OUT_DIR / "strategy_economic_profiles.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR / 'strategy_economic_profiles.json'}")


if __name__ == "__main__":
    main()
