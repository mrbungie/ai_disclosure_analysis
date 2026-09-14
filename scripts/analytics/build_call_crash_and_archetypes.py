"""Builds expanding archetypes and runs NCSKEW/DUVOL tail risk regressions.
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from archetypes import AA as _AA_exp

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = REPO_ROOT / "data/processed/clusters"
PANEL_PATH = CLUSTERS / "call_beta_main_panel_10k10q_asof.parquet"
FUNDAMENTALS_PATH = CLUSTERS / "call_fundamentals_panel.parquet"
CRASH_PATH = CLUSTERS / "call_crash_risk_panel.parquet"
ARCH_PATH = CLUSTERS / "panel_expanding_archetypes.parquet"
DB_PATH = REPO_ROOT / "duckdb/thesis.duckdb"

print("Loading expanding archetypes from DuckDB...")
conn = duckdb.connect(str(DB_PATH), read_only=True)
conn.execute(f'SET file_search_path = "{REPO_ROOT}";')

_POSTURE_EXP = [
    "promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
    "temporal_posture", "ai_positioning", "specificity"
]
_CROSS_FEATS_EXP = _POSTURE_EXP + ["disclosure_intensity"]
_exp_names = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]

_frames_exp = conn.execute("""
    WITH frame_domains AS (
        SELECT text_hash, frame_id, bool_or(domain = 'customer_facing') AS is_customer_facing
        FROM gold_ai_activities WHERE has_activity GROUP BY text_hash, frame_id
    )
    SELECT fm.ticker, extract(year from fm.filing_date)::INT AS year,
           f.concepts, f.temporal, f.specificity, f.rhetoric,
           COALESCE(fd.is_customer_facing, false) AS is_customer_facing
    FROM gold_ai_frames f
    LEFT JOIN frame_domains fd ON fd.text_hash = f.text_hash AND fd.frame_id = f.frame_id
    JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code = 'us' AND f.has_frame AND fm.ticker IS NOT NULL
""").fetchdf()
conn.close()

def _posture_frame_rates(df):
    df = df.copy()
    concepts = df["concepts"].apply(lambda c: set(c) if c is not None else set())
    df["promotional_posture"] = df["rhetoric"].apply(
        lambda r: np.mean([x in list(r) if r is not None else False for x in ("promotional", "strategic")]))
    df["hedging_posture"] = df["rhetoric"].apply(lambda r: float("hedged" in list(r)) if r is not None else 0.0)
    df["risk_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("risk_") for c in s)))
    df["governance_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("gov_") for c in s)))
    df["temporal_posture"] = (df["temporal"] == "realized").astype(float)
    df["ai_positioning"] = df["is_customer_facing"].astype(float)
    df["specificity"] = df["specificity"].apply(lambda s: len(s) / 5.0 if s is not None else 0.0)
    return df

_df_exp = _posture_frame_rates(_frames_exp)
_m = pd.read_parquet(CLUSTERS / "firm_year_master_v2.parquet")
_dp_exp = pd.read_parquet(CLUSTERS / "document_panel.parquet")
_dp_exp["year"] = _dp_exp["fecha"].dt.year
_years_comp = sorted(_df_exp["year"].unique())

def _shrink_to_prior_exp(rates, counts):
    out = {}
    for col in rates.columns:
        p = rates[col]
        mean, var = float(p.mean()), float(p.var(ddof=1))
        if var <= 0 or not 0 < mean < 1:
            out[col] = p
            continue
        strength = max(mean * (1 - mean) / var - 1, 1e-6)
        alpha, beta = mean * strength, (1 - mean) * strength
        out[col] = (p * counts + alpha) / (counts + alpha + beta)
    return pd.DataFrame(out, index=rates.index)

_exp_rows = []
for _cutoff in _years_comp:
    _sub = _df_exp[_df_exp["year"] <= _cutoff]
    _counts = _sub.groupby("ticker").size()
    _active_tickers = _counts[_counts >= 5].index
    _rates = _sub.groupby("ticker")[_POSTURE_EXP].mean().loc[_active_tickers]
    _shrunk = _shrink_to_prior_exp(_rates, _counts.loc[_active_tickers])
    _words = _dp_exp[_dp_exp["year"] <= _cutoff].groupby("ticker").agg(
        w=("n_words", "sum"), fr=("n_frames", "sum")).reindex(_active_tickers)
    _fr1k = 1000 * _words["fr"] / _words["w"].replace(0, np.nan)
    _shrunk["disclosure_intensity"] = _fr1k.rank(pct=True)
    _train_df = _shrunk.dropna()
    _mu_e, _sd_e = _train_df.mean(), _train_df.std(ddof=0)
    _X_train = ((_train_df - _mu_e) / _sd_e).values
    _aa_e = _AA_exp(n_archetypes=3, random_state=42, max_iter=500)
    _aa_e.fit_transform(_X_train)
    _A_e = _aa_e.archetypes_
    _def_col = int(np.argmax(_A_e[:, _POSTURE_EXP.index("risk_orientation")]))
    _rem = [i for i in range(3) if i != _def_col]
    _gov_col = max(_rem, key=lambda i: _A_e[i, _POSTURE_EXP.index("governance_orientation")])
    _voc_col = [i for i in range(3) if i not in (_def_col, _gov_col)][0]
    _order_e = [_voc_col, _gov_col, _def_col]

    _cross = _m[_m["year"] == _cutoff][["ticker", "year", "archetype"] + _CROSS_FEATS_EXP].copy()
    _active = _cross["archetype"] != "No AI"
    _X_cross = ((_cross.loc[_active, _CROSS_FEATS_EXP] - _mu_e) / _sd_e).values
    _W_cross = _aa_e.transform(_X_cross)
    _cross.loc[_active, "arch_exp"] = np.array(_exp_names)[np.argmax(_W_cross[:, _order_e], axis=1)]
    _cross.loc[~_active, "arch_exp"] = "No AI"
    _exp_rows.append(_cross[["ticker", "year", "arch_exp"]])

_panel_exp = pd.concat(_exp_rows, ignore_index=True)
_panel_exp.to_parquet(ARCH_PATH, index=False)
print(f"Saved expanding archetypes to {ARCH_PATH}")

# Now load panels
panel = pd.read_parquet(PANEL_PATH)
crash = pd.read_parquet(CRASH_PATH)
fund = pd.read_parquet(FUNDAMENTALS_PATH)

panel = panel.merge(crash, on=["ticker", "fecha"], how="left")
panel["year"] = pd.to_datetime(panel["fecha"]).dt.year
panel = panel.merge(_panel_exp, on=["ticker", "year"], how="left")

# Call-level decoupling W:
# W = pctrank(disclosure) - pctrank(substance) within year
panel["w_call"] = panel.groupby("year")["disclosure"].rank(pct=True) - panel.groupby("year")["substance"].rank(pct=True)

# Merge fundamentals
post_cols = [c for c in fund.columns if c.endswith("_post") and c not in panel.columns]
pre_cols = [c for c in fund.columns if c.endswith("_pre") and c not in panel.columns]
panel = panel.merge(fund[["ticker", "fecha"] + post_cols + pre_cols], on=["ticker", "fecha"], how="left")

AI_VARS = ["hist_disclosure", "hist_substance", "surprise_disclosure", "surprise_substance"]
DUMMIES = ["dum_gov", "dum_voc", "dum_noai"]
BASE_CTRLS = ["log_market_cap", "return60", "roa"]

active = panel[panel["arch_exp"].isin(
    ["Defensive Disclosers", "Governance-Led Disclosers", "Vocal Substantives", "No AI"])].copy()
active["dum_gov"] = (active["arch_exp"] == "Governance-Led Disclosers").astype(float)
active["dum_voc"] = (active["arch_exp"] == "Vocal Substantives").astype(float)
active["dum_noai"] = (active["arch_exp"] == "No AI").astype(float)

def fit_outcome(df, y_col, pre_col, add_dummies=True, add_w=False):
    extra = DUMMIES if add_dummies else []
    if add_w:
        extra = extra + ["w_call"]
    ctrls = [pre_col] + [c for c in BASE_CTRLS if c != y_col.replace("_post", "")]
    vars_all = AI_VARS + extra + ctrls
    d = df.dropna(subset=[y_col, "fe", *vars_all]).copy()
    d = d[d.groupby("fe")["ticker"].transform("size") >= 2].copy()
    y = (d[y_col] - d[y_col].mean()) / d[y_col].std(ddof=0)
    std_vars = [v for v in vars_all if v not in DUMMIES]
    std_df = d[std_vars].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    if add_dummies:
        design_vars = pd.concat([std_df, d[DUMMIES]], axis=1)
    else:
        design_vars = std_df
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([design_vars, fe], axis=1))
    res = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return res, d

print("\n=== RESULTS: NCSKEW (Crash Risk) Full Sample ===")
r_nc_f, d_nc_f = fit_outcome(panel, "ncskew_post", "ncskew_pre", add_dummies=False)
print(f"N calls: {len(d_nc_f)}, Firms: {d_nc_f.ticker.nunique()}, R2: {r_nc_f.rsquared:.4f}")
for v in AI_VARS + ["ncskew_pre"]:
    print(f"  {v:22s}: beta={r_nc_f.params[v]:+.4f}, se={r_nc_f.bse[v]:.4f}, p={r_nc_f.pvalues[v]:.4f}")

print("\n=== RESULTS: NCSKEW (Crash Risk) with Decoupling W ===")
r_nc_w, d_nc_w = fit_outcome(panel, "ncskew_post", "ncskew_pre", add_dummies=False, add_w=True)
print(f"N calls: {len(d_nc_w)}, Firms: {d_nc_w.ticker.nunique()}, R2: {r_nc_w.rsquared:.4f}")
for v in ["w_call", "hist_disclosure", "hist_substance", "ncskew_pre"]:
    print(f"  {v:22s}: beta={r_nc_w.params[v]:+.4f}, se={r_nc_w.bse[v]:.4f}, p={r_nc_w.pvalues[v]:.4f}")

print("\n=== RESULTS: NCSKEW (Crash Risk) Active Sample with Dummies ===")
r_nc_a, d_nc_a = fit_outcome(active, "ncskew_post", "ncskew_pre", add_dummies=True)
print(f"N calls: {len(d_nc_a)}, Firms: {d_nc_a.ticker.nunique()}, R2: {r_nc_a.rsquared:.4f}")
for v in AI_VARS + DUMMIES + ["ncskew_pre"]:
    print(f"  {v:22s}: beta={r_nc_a.params[v]:+.4f}, se={r_nc_a.bse[v]:.4f}, p={r_nc_a.pvalues[v]:.4f}")

print("\n=== RESULTS: DUVOL (Down-to-Up Volatility) Full Sample ===")
r_du_f, d_du_f = fit_outcome(panel, "duvol_post", "duvol_pre", add_dummies=False)
print(f"N calls: {len(d_du_f)}, Firms: {d_du_f.ticker.nunique()}, R2: {r_du_f.rsquared:.4f}")
for v in AI_VARS + ["duvol_pre"]:
    print(f"  {v:22s}: beta={r_du_f.params[v]:+.4f}, se={r_du_f.bse[v]:.4f}, p={r_du_f.pvalues[v]:.4f}")

print("\n=== RESULTS: DUVOL with Decoupling W ===")
r_du_w, d_du_w = fit_outcome(panel, "duvol_post", "duvol_pre", add_dummies=False, add_w=True)
for v in ["w_call", "hist_disclosure", "hist_substance", "duvol_pre"]:
    print(f"  {v:22s}: beta={r_du_w.params[v]:+.4f}, se={r_du_w.bse[v]:.4f}, p={r_du_w.pvalues[v]:.4f}")

print("\n=== RESULTS: DUVOL (Down-to-Up Volatility) Active Sample with Dummies ===")
r_du_a, d_du_a = fit_outcome(active, "duvol_post", "duvol_pre", add_dummies=True)
print(f"N calls: {len(d_du_a)}, Firms: {d_du_a.ticker.nunique()}, R2: {r_du_a.rsquared:.4f}")
for v in AI_VARS + DUMMIES + ["duvol_pre"]:
    print(f"  {v:22s}: beta={r_du_a.params[v]:+.4f}, se={r_du_a.bse[v]:.4f}, p={r_du_a.pvalues[v]:.4f}")

# --- Export coefficient tables (with 95% CIs) for the crash-risk figure and
# tables in the thesis text, so those numbers are read from a file rather
# than retyped by hand. ---
def _coef_table(res, varlist, target):
    rows = []
    for v in varlist:
        b, se = res.params[v], res.bse[v]
        rows.append({"target": target, "variable": v, "beta_std": b, "se": se,
                     "ci95_low": b - 1.96 * se, "ci95_high": b + 1.96 * se, "p": res.pvalues[v]})
    return pd.DataFrame(rows)

_export = pd.concat([
    _coef_table(r_nc_f, AI_VARS + ["ncskew_pre"], "ncskew_full"),
    _coef_table(r_nc_w, ["w_call", "hist_disclosure", "hist_substance", "ncskew_pre"], "ncskew_w"),
    _coef_table(r_nc_a, AI_VARS + DUMMIES + ["ncskew_pre"], "ncskew_active_dummies"),
    _coef_table(r_du_f, AI_VARS + ["duvol_pre"], "duvol_full"),
    _coef_table(r_du_w, ["w_call", "hist_disclosure", "hist_substance", "duvol_pre"], "duvol_w"),
    _coef_table(r_du_a, AI_VARS + DUMMIES + ["duvol_pre"], "duvol_active_dummies"),
], ignore_index=True)
_samples = pd.DataFrame([
    {"target": "ncskew_full", "n_calls": len(d_nc_f), "n_firms": d_nc_f.ticker.nunique(), "r2": r_nc_f.rsquared},
    {"target": "ncskew_w", "n_calls": len(d_nc_w), "n_firms": d_nc_w.ticker.nunique(), "r2": r_nc_w.rsquared},
    {"target": "ncskew_active_dummies", "n_calls": len(d_nc_a), "n_firms": d_nc_a.ticker.nunique(), "r2": r_nc_a.rsquared},
    {"target": "duvol_full", "n_calls": len(d_du_f), "n_firms": d_du_f.ticker.nunique(), "r2": r_du_f.rsquared},
    {"target": "duvol_w", "n_calls": len(d_du_w), "n_firms": d_du_w.ticker.nunique(), "r2": r_du_w.rsquared},
    {"target": "duvol_active_dummies", "n_calls": len(d_du_a), "n_firms": d_du_a.ticker.nunique(), "r2": r_du_a.rsquared},
])
_export.to_csv(CLUSTERS / "call_crash_archetype_regressions.csv", index=False)
_samples.to_csv(CLUSTERS / "call_crash_archetype_regression_samples.csv", index=False)
print(f"\n-> {CLUSTERS / 'call_crash_archetype_regressions.csv'}")
print(f"-> {CLUSTERS / 'call_crash_archetype_regression_samples.csv'}")
