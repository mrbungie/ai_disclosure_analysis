"""Posture archetype weights across the full outcome family (post-call beta,
six firm fundamentals, NCSKEW/DUVOL), Benjamini-Hochberg over the archetype
tests, and two robustness checks on the crash-risk archetype weights (sector
exclusion, posture switchers: calls at which the firm's dominant archetype has
already changed, known at the call).

Design, panel and estimation: scripts/analytics/call_regression.py. Controls:
the outcome's pre-call level, log market cap, 60-day return and ROA (the
control that is itself the outcome is dropped); SIC2 x year fixed effects;
errors clustered by firm.

Archetype test family: the three weight coefficients (each against firms
without posture) for each of the nine outcomes (27 tests).

Outputs (data/results/crash_archetypes/):
  call_archetype_full_battery_targets.csv     coefficients per outcome
  call_archetype_full_battery_bh.csv          BH table over the 27 weight coefficients
  call_archetype_full_battery_robustness.csv  sector exclusion / switchers, NCSKEW and DUVOL

Usage:
  .venv/bin/python scripts/analytics/crash_archetypes/call_archetype_full_battery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
import call_regression as C  # noqa: E402

L = C.L
OUT_DIR = L.results_path("crash_archetypes", "call_archetype_full_battery_targets.csv").parent
TARGETS = {
    "beta_post_63": ("beta_pre", "Market Beta (post)"),
    "log_market_cap_post": ("log_market_cap_pre", "Log Market Cap"),
    "rd_intensity_post": ("rd_intensity_pre", "R&D / Sales"),
    "gross_margin_post": ("gross_margin_pre", "Gross Margin"),
    "ps_ratio_post": ("ps_ratio_pre", "Price-to-Sales"),
    "next_revenue_yoy_post": ("revenue_yoy_pre", "Revenue Growth (t+1)"),
    "roic_minus_wacc_post": ("roic_minus_wacc_pre", "ROIC - WACC"),
    "ncskew_post": ("ncskew_pre", "NCSKEW"),
    "duvol_post": ("duvol_pre", "DUVOL"),
}
EXCLUDED_SECTORS = ["Financial Services", "Utilities", "Healthcare & Pharma"]


def sector_map() -> dict[str, str]:
    fu = (L.scan("silver.firm_universe").filter(pl.col("ticker").is_not_null())
          .select("ticker", "sic").collect().to_pandas())
    fu["sic2"] = fu["sic"].astype(str).str.zfill(4).str[:2]

    def _map(sic2):
        try:
            s = int(sic2)
        except ValueError:
            return "Other / Diversified"
        if s in (28, 38, 80):
            return "Healthcare & Pharma"
        if 60 <= s <= 67:
            return "Financial Services"
        if s == 49:
            return "Utilities"
        return "Other / Diversified"

    fu["sector"] = fu["sic2"].apply(_map)
    return fu.set_index("ticker")["sector"].to_dict()


def switched_by_call(panel: pd.DataFrame) -> pd.Series:
    """True for a call once its firm has shown two different dominant
    archetypes (argmax of the as-of weights) at that call or earlier ones: a
    call with posture whose archetype differs from the archetype at some
    earlier call, and the firm's later calls, including calls without posture
    (the weights' comparison group)."""
    order = panel.sort_values(["ticker", "fecha"]).index
    has_posture = panel[C.WEIGHTS].sum(axis=1) > 0
    arch = pd.Series(panel[C.WEIGHTS].values.argmax(axis=1), index=panel.index).where(has_posture)
    seen = pd.concat([(arch == j).loc[order].groupby(panel.loc[order, "ticker"]).cummax()
                      for j in range(len(C.WEIGHTS))], axis=1)
    return (seen.sum(axis=1) >= 2).reindex(panel.index)


def fit_target(panel: pd.DataFrame, y_col: str):
    return C.fit(panel, y_col, C.AI_VARS + [TARGETS[y_col][0]] + C.controls_for(y_col))


def main() -> None:
    panel = C.attach_fundamentals(C.attach_crash_risk(C.load_call_panel()))
    panel["sector"] = panel["ticker"].map(sector_map()).fillna("Other / Diversified")

    tables = []
    for y_col, (pre_col, label) in TARGETS.items():
        res, d = fit_target(panel, y_col)
        regressors = C.AI_VARS + [pre_col] + C.controls_for(y_col)
        pr2 = C.partial_r2(panel, y_col, regressors, C.AI_VARS)
        tables.append(C.coef_table(res, C.AI_VARS + C.WEIGHTS, target=y_col, label_outcome=label, n_calls=len(d),
                                   n_firms=d.ticker.nunique(), r2=res.rsquared, partial_r2_ai_arch=pr2))
        print(f"{label:22s} n={len(d):5d} firms={d.ticker.nunique():4d} r2={res.rsquared:.3f} pR2={pr2:.4f} | "
              + " | ".join(f"{v} {res.params[v]:+.3f} (p={res.pvalues[v]:.3f})" for v in C.WEIGHTS))
    targets = pd.concat(tables, ignore_index=True)
    targets.to_csv(OUT_DIR / "call_archetype_full_battery_targets.csv", index=False)

    fam = targets[targets["variable"].isin(C.WEIGHTS)][["target", "variable", "beta_std", "p"]]
    fam = fam.sort_values("p").reset_index(drop=True)
    m = len(fam)
    fam["rank"] = np.arange(1, m + 1)
    fam["bh_crit"] = fam["rank"] / m * 0.05
    # step-up: every test ranked at or below the largest rank with p <= its critical value survives
    passing = np.where(fam["p"].values <= fam["bh_crit"].values)[0]
    fam["survives"] = fam["rank"] <= (passing.max() + 1 if len(passing) else 0)
    fam.to_csv(OUT_DIR / "call_archetype_full_battery_bh.csv", index=False)
    print(f"\n=== Benjamini-Hochberg across the {m}-test archetype family ===\n{fam.head(6).round(4).to_string(index=False)}")

    samples = {
        "sector_exclusion": panel[~panel["sector"].isin(EXCLUDED_SECTORS)],
        "posture_switchers": panel[switched_by_call(panel)],
    }
    robust = []
    for check, sample in samples.items():
        for y_col in ("ncskew_post", "duvol_post"):
            res, d = fit_target(sample, y_col)
            robust.append(C.coef_table(res, C.AI_VARS + C.WEIGHTS, check=check, target=y_col,
                                       n_calls=len(d), n_firms=d.ticker.nunique(), r2=res.rsquared))
            print(f"{check:18s} {y_col:12s} n={len(d)} firms={d.ticker.nunique()} | "
                  + " | ".join(f"{v} {res.params[v]:+.3f} (p={res.pvalues[v]:.3f})" for v in C.WEIGHTS))
    pd.concat(robust, ignore_index=True).to_csv(OUT_DIR / "call_archetype_full_battery_robustness.csv", index=False)


if __name__ == "__main__":
    main()
