"""Configuration + decoupling specification, on the fully-cleaned panels
(see docs/analytics/12_config_decoupling_asof.md for the full writeup):

Y = g1*w_voc(as-of) + g2*w_gov(as-of) + g3*HistD(t-1, filled 0)
    + th1*HistW(as-of, expanding) + th2*SurpriseW(t)
    + log_market_cap + return60 + roa + Y(t-1)

- w_voc/w_gov: continuous expanding-window AA mixture weights, matched
  as-of the call's own fiscal quarter (derived from call_accession_number,
  NOT the buggy `fecha`), backward, strictly prior quarter. Firms/calls with
  no archetype classification ("No AI", insufficient history) get w_voc=0,
  w_gov=0 (no measured posture) instead of being dropped.
- hist_disclosure_filled: the firm's own expanding disclosure intensity
  (hist_disclosure, already expanding().shift(1) per ticker in the main
  panel), 0 filled where there is no prior AI disclosure at all -- replaces
  an earlier dum_noai flag with a continuous measure of the same state.
- HistW/SurpriseW: call-level decoupling (z(disclosure)-z(substance)),
  z-scored with an EXPANDING cross-sectional mean/std over calendar time
  (not the full-sample constant, which would leak the later, much-higher-
  disclosure years into an early call's standardization), then
  expanding().shift(1) per ticker on the panel sorted by (ticker, fecha,
  call_accession_number) for correct chronological order even where
  `fecha` collides across two real different calls.
- Controls: log_market_cap, return60, roa (the well-covered ones), plus the
  outcome's own pre-call level (beta_pre / ncskew_pre / ncskew_pre_63).
- Run for beta_post_63 (headline), ncskew_post (126d, via the deduplicated
  call_crash_risk_panel.parquet), and ncskew_post_63 (via ncskew_63.parquet,
  built by build_ncskew_63.py).
"""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = REPO_ROOT / "data/processed/clusters"

panel = pd.read_parquet(CLUSTERS / "call_beta_main_panel_10k10q_asof.parquet")
panel["fecha"] = pd.to_datetime(panel["fecha"])
print(f"panel: {len(panel)} rows (should be 10,507, clean)")

# --- Call's own quarter from call_accession_number, not fecha ---
extracted = panel["call_accession_number"].str.extract(r"_(\d{4})Q([1-4])$")
panel["call_year"] = extracted[0].astype(int)
panel["call_q"] = extracted[1].astype(int)
panel["call_period"] = pd.PeriodIndex.from_fields(year=panel["call_year"], quarter=panel["call_q"], freq="Q")

# --- Archetype weights, matched by call_period (as-of, strictly prior) ---
wq = pd.read_parquet(CLUSTERS / "panel_expanding_archetype_weights_quarterly.parquet")
wq["cutoff_quarter"] = wq["cutoff_quarter"].astype("period[Q]")

rows = []
for tkr, g in panel.groupby("ticker", sort=False):
    wqt = wq.loc[wq.ticker == tkr, ["cutoff_quarter", "w_voc", "w_gov"]].sort_values("cutoff_quarter")
    if wqt.empty:
        m = g[["ticker", "fecha", "call_accession_number"]].copy()
        m["w_voc"] = float("nan"); m["w_gov"] = float("nan")
        rows.append(m)
        continue
    g2 = g.sort_values("call_period")
    m = pd.merge_asof(g2, wqt, left_on="call_period", right_on="cutoff_quarter",
                       direction="backward", allow_exact_matches=False)
    rows.append(m[["ticker", "fecha", "call_accession_number", "w_voc", "w_gov"]])
weights = pd.concat(rows, ignore_index=True)
panel = panel.merge(weights, on=["ticker", "fecha", "call_accession_number"], how="left")
print(f"after weight merge: {len(panel)} rows (should still be same as panel)")

# No archetype classification available (insufficient AI-disclosure history
# to date) -> not missing data, a real state: w_voc=w_gov=0 (no measured
# posture) instead of dropping the row. Replaces the dum_noai flag with the
# firm's own expanding disclosure intensity (hist_disclosure, already
# expanding().shift(1) per ticker, 0 for a firm's first call / no prior AI
# disclosure at all) -- a continuous "how AI-active has this firm been"
# measure instead of a binary no-archetype flag.
panel["w_voc"] = panel["w_voc"].fillna(0.0)
panel["w_gov"] = panel["w_gov"].fillna(0.0)
panel["hist_disclosure_filled"] = panel["hist_disclosure"].fillna(0.0)

# --- HistW / SurpriseW, correctly ordered ---
# z_disclosure/z_substance must NOT use the full-sample mean/std (that
# constant is computed over ALL years, so it "knows" about the later,
# much-higher-disclosure years -- exactly the secular trend this thesis
# documents -- when standardizing an early call). Use an EXPANDING
# cross-sectional mean/std instead: for each call, only calls that
# happened strictly before it (across ALL tickers, not just its own)
# contribute to the standardization constant, so no call's z-score is
# influenced by data from after its own date.
panel = panel.sort_values(["fecha", "ticker", "call_accession_number"]).reset_index(drop=True)
n = pd.Series(range(1, len(panel) + 1), index=panel.index)
exp_mean_d = panel["disclosure"].expanding().mean().shift(1)
exp_var_d = panel["disclosure"].expanding().var(ddof=0).shift(1)
exp_mean_s = panel["substance"].expanding().mean().shift(1)
exp_var_s = panel["substance"].expanding().var(ddof=0).shift(1)
panel["z_disclosure"] = (panel["disclosure"] - exp_mean_d) / exp_var_d.pow(0.5)
panel["z_substance"] = (panel["substance"] - exp_mean_s) / exp_var_s.pow(0.5)
panel["w_raw"] = panel["z_disclosure"] - panel["z_substance"]

panel = panel.sort_values(["ticker", "fecha", "call_accession_number"]).reset_index(drop=True)
panel["HistW"] = panel.groupby("ticker")["w_raw"].transform(lambda s: s.expanding().mean().shift(1))
panel["SurpriseW"] = panel["w_raw"] - panel["HistW"]

VARS = ["w_voc", "w_gov", "hist_disclosure_filled", "HistW", "SurpriseW"]
CTRLS = ["log_market_cap", "return60", "roa"]

def fit(df, y_col, pre_col):
    vars_all = VARS + CTRLS + [pre_col]
    d = df.dropna(subset=[y_col, "fe", *vars_all]).copy()
    d = d[d.groupby("fe")["ticker"].transform("size") >= 2].copy()
    y = (d[y_col] - d[y_col].mean()) / d[y_col].std(ddof=0)
    std_df = d[vars_all].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    design_vars = std_df
    fe = pd.get_dummies(d["fe"], prefix="fe", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([design_vars, fe], axis=1))
    res = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]})
    return res, d

print(f"\n=== beta_post_63 (headline, trimestral), with beta_pre (Y t-1) ===")
res, d = fit(panel, "beta_post_63", "beta_pre")
print(f"N={len(d)}, firms={d.ticker.nunique()}, R2={res.rsquared:.4f}")
for v in VARS + CTRLS + ["beta_pre"]:
    print(f"  {v:12s}: beta={res.params[v]:+.4f}, p={res.pvalues[v]:.4f}")

crash = pd.read_parquet(CLUSTERS / "call_crash_risk_panel.parquet")
panel_c = panel.merge(crash, on=["ticker", "fecha"], how="left")
print(f"\n=== ncskew_post (126d), with ncskew_pre (Y t-1) ===")
res, d = fit(panel_c, "ncskew_post", "ncskew_pre")
print(f"N={len(d)}, firms={d.ticker.nunique()}, R2={res.rsquared:.4f}")
for v in VARS + CTRLS + ["ncskew_pre"]:
    print(f"  {v:12s}: beta={res.params[v]:+.4f}, p={res.pvalues[v]:.4f}")

ncskew63 = pd.read_parquet(CLUSTERS / "ncskew_63.parquet")
panel_n63 = panel.merge(ncskew63, on=["ticker", "fecha", "call_accession_number"], how="left")
print(f"\n=== ncskew_post_63 (63d, matching beta window), with ncskew_pre_63 (Y t-1) ===")
res, d = fit(panel_n63, "ncskew_post_63", "ncskew_pre_63")
print(f"N={len(d)}, firms={d.ticker.nunique()}, R2={res.rsquared:.4f}")
for v in VARS + CTRLS + ["ncskew_pre_63"]:
    print(f"  {v:12s}: beta={res.params[v]:+.4f}, p={res.pvalues[v]:.4f}")
