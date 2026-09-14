"""Archetype-dummy regressions across the full outcome family (post-call beta,
six firm fundamentals, and NCSKEW/DUVOL), plus the two robustness checks on
the Governance-Led / crash-risk association (sector exclusion, posture
switchers).

Reconstructs, with the roa control and the `dum_noai` inclusion adopted in
`build_call_beta_panel.py` / `build_call_crash_and_archetypes.py`, the
archetype-dummy multi-target regression that used to live only in an
orphaned CSV (`call_multi_target_archetypes_results.csv`, no longer built by
any script in this repo). Reuses `panel_expanding_archetypes.parquet`
(built by `build_call_crash_and_archetypes.py`) and
`call_fundamentals_panel.parquet`.

Usage:
  source .venv/bin/activate
  .venv/bin/python scripts/analytics/build_call_crash_and_archetypes.py
  .venv/bin/python scripts/analytics/call_archetype_full_battery.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = ROOT / "data/processed/clusters"
PANEL_PATH = CLUSTERS / "call_beta_main_panel_10k10q_asof.parquet"
CRASH_PATH = CLUSTERS / "call_crash_risk_panel.parquet"
FUNDAMENTALS_PATH = CLUSTERS / "call_fundamentals_panel.parquet"
ARCH_PATH = CLUSTERS / "panel_expanding_archetypes.parquet"
DB_PATH = ROOT / "duckdb/thesis.duckdb"

OUT_TARGETS = CLUSTERS / "call_archetype_full_battery_targets.csv"
OUT_ROBUST = CLUSTERS / "call_archetype_full_battery_robustness.csv"

AI_VARS = ["hist_disclosure", "hist_substance", "surprise_disclosure", "surprise_substance"]
DUMMIES = ["dum_gov", "dum_voc"]
BASE_CTRLS = ["log_market_cap", "return60", "roa"]

TARGETS = {
    "beta_post_63": ("beta_pre", "Market Beta (post)"),
    "log_market_cap_post": ("log_market_cap_pre", "Log Market Cap"),
    "rd_intensity_post": ("rd_intensity_pre", "R&D / Sales"),
    "gross_margin_post": ("gross_margin_pre", "Gross Margin"),
    "ps_ratio_post": ("ps_ratio_pre", "Price-to-Sales"),
    "next_revenue_yoy_post": ("next_revenue_yoy_pre", "Revenue Growth (t+1)"),
    "roic_minus_wacc_post": ("roic_minus_wacc_pre", "ROIC - WACC"),
}


def sector_map() -> dict[str, str]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    fu = con.execute(
        "SELECT ticker, sic FROM firm_universe WHERE country_code = 'us' AND ticker IS NOT NULL"
    ).df()
    con.close()
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


def load_active_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    crash = pd.read_parquet(CRASH_PATH)
    fund = pd.read_parquet(FUNDAMENTALS_PATH)
    arch = pd.read_parquet(ARCH_PATH)

    panel = panel.merge(crash, on=["ticker", "fecha"], how="left")
    panel["year"] = pd.to_datetime(panel["fecha"]).dt.year
    panel = panel.merge(arch, on=["ticker", "year"], how="left")

    fund_cols = [c for c in fund.columns if c not in ("ticker", "fecha")]
    panel = panel.merge(fund[["ticker", "fecha"] + fund_cols], on=["ticker", "fecha"], how="left")

    active = panel[panel["arch_exp"].isin(
        ["Defensive Disclosers", "Governance-Led Disclosers", "Vocal Substantives"])].copy()
    active["dum_gov"] = (active["arch_exp"] == "Governance-Led Disclosers").astype(float)
    active["dum_voc"] = (active["arch_exp"] == "Vocal Substantives").astype(float)
    return active


def fit(df: pd.DataFrame, y_col: str, pre_col: str, extra_vars: list[str] | None = None,
        drop_ctrl: str | None = None, dummies: list[str] = DUMMIES
        ) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, list[str]]:
    extra_vars = extra_vars or []
    ctrls = [c for c in BASE_CTRLS if c != drop_ctrl]
    vars_all = AI_VARS + dummies + [pre_col] + ctrls + extra_vars
    d = df.dropna(subset=[y_col, "fe", *vars_all]).copy()
    d = d[d.groupby("fe")["ticker"].transform("size") >= 2].copy()
    if len(d) < 30:
        return None, d, vars_all
    y = (d[y_col] - d[y_col].mean()) / d[y_col].std(ddof=0)
    std_vars = [v for v in vars_all if v not in dummies]
    std_df = d[std_vars].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    design_vars = pd.concat([std_df, d[dummies]], axis=1)
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([design_vars, fe], axis=1))
    res = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return res, d, vars_all


def partial_r2_ai_arch(df: pd.DataFrame, y_col: str, pre_col: str, drop_ctrl: str | None) -> float:
    ctrls = [c for c in BASE_CTRLS if c != drop_ctrl]
    full_vars = AI_VARS + DUMMIES + [pre_col] + ctrls
    d = df.dropna(subset=[y_col, "fe", *full_vars]).copy()
    d = d[d.groupby("fe")["ticker"].transform("size") >= 2].copy()
    if len(d) < 30:
        return float("nan")
    y = (d[y_col] - d[y_col].mean()) / d[y_col].std(ddof=0)
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)

    def _r2(vars_std, vars_dum):
        std_df = d[vars_std].apply(lambda col: (col - col.mean()) / col.std(ddof=0)) if vars_std else pd.DataFrame(index=d.index)
        parts = [std_df]
        if vars_dum:
            parts.append(d[vars_dum])
        parts.append(fe)
        design = sm.add_constant(pd.concat(parts, axis=1))
        return sm.OLS(y, design).fit().rsquared

    r2_m0 = _r2([pre_col] + ctrls, [])
    r2_m1 = _r2(AI_VARS + [pre_col] + ctrls, DUMMIES)
    return r2_m1 - r2_m0


def main() -> None:
    active = load_active_panel()
    sectors = sector_map()
    active["sector"] = active["ticker"].map(sectors).fillna("Other / Diversified")

    rows = []
    for y_col, (pre_col, label) in TARGETS.items():
        target_base = y_col.replace("_post", "").replace("_post_63", "")
        drop_ctrl = target_base if target_base in BASE_CTRLS else None
        res, d, _ = fit(active, y_col, pre_col, drop_ctrl=drop_ctrl)
        if res is None:
            print(f"{label}: insufficient N, skipped")
            continue
        pr2 = partial_r2_ai_arch(active, y_col, pre_col, drop_ctrl)
        n_calls, n_firms = len(d), d.ticker.nunique()
        r2 = res.rsquared
        print(f"{label:24s} n={n_calls:5d} firms={n_firms:4d} r2={r2:.3f} partial_r2={pr2:.4f} | "
              + " | ".join(f"{v}: {res.params[v]:+.4f} (p={res.pvalues[v]:.4f})" for v in AI_VARS + DUMMIES))
        for v in AI_VARS + DUMMIES:
            rows.append({"target": y_col, "label": label, "variable": v, "beta_std": res.params[v],
                         "se": res.bse[v], "p": res.pvalues[v], "n_calls": n_calls, "n_firms": n_firms,
                         "r2": r2, "partial_r2_ai_arch": pr2})
    pd.DataFrame(rows).to_csv(OUT_TARGETS, index=False)
    print(f"\n-> {OUT_TARGETS}")

    # --- NCSKEW: sector-exclusion and posture-switcher checks, matching the
    # headline model's sample (Defensive/Governance-Led/Vocal/No AI, with
    # dum_noai) rather than the 3-archetype "active" sample used above for
    # the fundamentals battery. ---
    robust_rows = []
    DUMMIES_4 = DUMMIES + ["dum_noai"]

    panel_all = pd.read_parquet(PANEL_PATH)
    crash = pd.read_parquet(CRASH_PATH)
    arch = pd.read_parquet(ARCH_PATH)
    panel_all = panel_all.merge(crash, on=["ticker", "fecha"], how="left")
    panel_all["year"] = pd.to_datetime(panel_all["fecha"]).dt.year
    panel_all = panel_all.merge(arch, on=["ticker", "year"], how="left")
    active4 = panel_all[panel_all["arch_exp"].isin(
        ["Defensive Disclosers", "Governance-Led Disclosers", "Vocal Substantives", "No AI"])].copy()
    active4["dum_gov"] = (active4["arch_exp"] == "Governance-Led Disclosers").astype(float)
    active4["dum_voc"] = (active4["arch_exp"] == "Vocal Substantives").astype(float)
    active4["dum_noai"] = (active4["arch_exp"] == "No AI").astype(float)
    active4["sector"] = active4["ticker"].map(sectors).fillna("Other / Diversified")

    excl_mask = ~active4["sector"].isin(["Financial Services", "Utilities", "Healthcare & Pharma"])
    res_excl, d_excl, _ = fit(active4[excl_mask], "ncskew_post", "ncskew_pre", dummies=DUMMIES_4)
    print(f"\nSector-exclusion (ex Fin/Util/Healthcare): n={len(d_excl)}, firms={d_excl.ticker.nunique()}, "
          f"dum_gov={res_excl.params['dum_gov']:+.4f} (p={res_excl.pvalues['dum_gov']:.4f})")
    robust_rows.append({"check": "sector_exclusion", "n_calls": len(d_excl), "n_firms": d_excl.ticker.nunique(),
                        "dum_gov_beta": res_excl.params["dum_gov"], "dum_gov_p": res_excl.pvalues["dum_gov"]})

    switchers = active4.groupby("ticker")["arch_exp"].nunique()
    switcher_tickers = switchers[switchers > 1].index
    sw_mask = active4["ticker"].isin(switcher_tickers)
    res_sw, d_sw, _ = fit(active4[sw_mask], "ncskew_post", "ncskew_pre", dummies=DUMMIES_4)
    print(f"Posture switchers: n={len(d_sw)}, firms={d_sw.ticker.nunique()}, "
          f"dum_gov={res_sw.params['dum_gov']:+.4f} (p={res_sw.pvalues['dum_gov']:.4f})")
    robust_rows.append({"check": "posture_switchers", "n_calls": len(d_sw), "n_firms": d_sw.ticker.nunique(),
                        "dum_gov_beta": res_sw.params["dum_gov"], "dum_gov_p": res_sw.pvalues["dum_gov"]})

    pd.DataFrame(robust_rows).to_csv(OUT_ROBUST, index=False)
    print(f"-> {OUT_ROBUST}")

    # --- DUVOL, active sample with dummies (parity with the NCSKEW headline
    # model in build_call_crash_and_archetypes.py) ---
    res_du, d_du, _ = fit(active4, "duvol_post", "duvol_pre", dummies=DUMMIES_4)
    print(f"\nDUVOL Active Sample with Dummies: n={len(d_du)}, firms={d_du.ticker.nunique()}, r2={res_du.rsquared:.4f}")
    for v in AI_VARS + DUMMIES_4:
        print(f"  {v:22s}: beta={res_du.params[v]:+.4f}, se={res_du.bse[v]:.4f}, p={res_du.pvalues[v]:.4f}")

    # --- Benjamini-Hochberg across the full 18-test archetype family:
    # 2 dummies x 7 fundamentals (this script's targets battery) + 2 dummies
    # x {NCSKEW, DUVOL} (the headline crash-risk models, entered here by
    # hand from the companion script's printed output). ---
    fam = pd.read_csv(OUT_TARGETS)[["target", "variable", "p"]].copy()
    fam = fam[fam["variable"].isin(DUMMIES)]
    extra = pd.DataFrame([
        {"target": "ncskew_post", "variable": "dum_gov", "p": 0.0460},
        {"target": "ncskew_post", "variable": "dum_voc", "p": 0.9390},
        {"target": "duvol_post", "variable": "dum_gov", "p": res_du.pvalues["dum_gov"]},
        {"target": "duvol_post", "variable": "dum_voc", "p": res_du.pvalues["dum_voc"]},
    ])
    fam = pd.concat([fam, extra], ignore_index=True)
    fam = fam.sort_values("p").reset_index(drop=True)
    m = len(fam)
    fam["rank"] = np.arange(1, m + 1)
    fam["bh_crit"] = fam["rank"] / m * 0.05
    fam["survives"] = fam["p"] <= fam["bh_crit"]
    print(f"\n=== Benjamini-Hochberg across the {m}-test archetype family ===")
    print(fam.to_string(index=False))
    fam.to_csv(CLUSTERS / "call_archetype_full_battery_bh.csv", index=False)


if __name__ == "__main__":
    main()
