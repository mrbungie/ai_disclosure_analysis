#!/usr/bin/env python3
"""
SEC Scrutiny Experiment: Patent-Backed Narrative Excess (ExternalGap) Exposure
Comprehensive testing across event dates, resolutions, and gap formulations.
"""

import socket
_orig_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(*args, **kwargs):
    responses = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET] or responses
socket.getaddrinfo = _patched_getaddrinfo

import re
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
from ai_intensity import document_table, aggregate, FILING_FORMS

ENFORCEMENT_DATE = pd.Timestamp("2024-03-18")
WARNING_DATE = pd.Timestamp("2023-12-05")

def load_data():
    con = duckdb.connect(str(REPO_ROOT / "duckdb" / "thesis.duckdb"), read_only=True)
    docs = document_table(con)
    docs = docs[docs["form"].isin(FILING_FORMS)].copy()
    
    patents = pd.read_parquet(REPO_ROOT / "data" / "processed" / "sp500_firm_preshock_patent_capacity.parquet")
    
    fu = con.execute("SELECT ticker, sic FROM firm_universe WHERE country_code = 'us'").df()
    fu["sic2"] = fu["sic"].astype(str).str[:2].replace({"No": "99", "no": "99", "None": "99", "": "99", "na": "99"})
    
    mf = pd.read_parquet(REPO_ROOT / "data" / "processed" / "clusters" / "firm_year_market_factors.parquet")
    size_df = mf[mf["year"] == 2023].groupby("ticker")["market_cap"].mean().reset_index()
    size_df["log_size"] = np.log(size_df["market_cap"].replace(0, np.nan))
    
    return docs, patents, fu, size_df

def compute_external_gap(docs, patents, fu, size_df, shock_date=ENFORCEMENT_DATE, patent_col_prefix="pre_enforce", pre_2023_only=False):
    if pre_2023_only:
        pre_docs = docs[docs["fecha"] < pd.Timestamp("2023-01-01")].copy()
    else:
        pre_docs = docs[docs["fecha"] < shock_date].copy()
        
    pre_agg = pre_docs.groupby("ticker").agg(
        n_frames=("n_frames", "sum"),
        n_words=("n_words", "sum")
    ).reset_index()
    pre_agg["disc_rate"] = 1000.0 * pre_agg["n_frames"] / pre_agg["n_words"]
    pre_agg["log_disc"] = np.log1p(pre_agg["disc_rate"])
    
    df = pre_agg.merge(patents, on="ticker", how="inner")
    df = df.merge(fu[["ticker", "sic2"]], on="ticker", how="left").fillna({"sic2": "99"})
    df = df.merge(size_df[["ticker", "log_size"]], on="ticker", how="left")
    df["log_size"] = df["log_size"].fillna(df["log_size"].median())
    
    ai_col = f"ai_families_{patent_col_prefix}"
    non_ai_col = f"non_ai_families_{patent_col_prefix}"
    
    df["log_ai_patents"] = np.log1p(df[ai_col].astype(float))
    df["log_non_ai_patents"] = np.log1p(df[non_ai_col].astype(float))
    
    # Robust SIC2
    sic_counts = df["sic2"].value_counts()
    valid_sic = sic_counts[sic_counts > 1].index
    df["sic2_clean"] = df["sic2"].where(df["sic2"].isin(valid_sic), "99")
    
    formula = "log_disc ~ log_ai_patents + log_non_ai_patents + log_size + C(sic2_clean)"
    first_stage = smf.ols(formula, data=df).fit()
    
    df["external_gap"] = first_stage.resid
    df["external_gap_z"] = (df["external_gap"] - df["external_gap"].mean()) / df["external_gap"].std()
    
    # Simple rank difference robustness: Rank(Disclosure) - Rank(AI Patents)
    df["rank_disc"] = df["log_disc"].rank(pct=True)
    df["rank_ai_pat"] = df["log_ai_patents"].rank(pct=True)
    df["rank_gap"] = df["rank_disc"] - df["rank_ai_pat"]
    df["rank_gap_z"] = (df["rank_gap"] - df["rank_gap"].mean()) / df["rank_gap"].std()
    
    return df, first_stage

def fe_ols(data: pd.DataFrame, outcome: str, rhs: str):
    import patsy
    X = patsy.dmatrix(rhs, data, return_type="dataframe")
    X = X.drop(columns=[c for c in X.columns if c == "Intercept"], errors="ignore")
    groups = data["ticker"].to_numpy()
    y = data[outcome] - data.groupby("ticker")[outcome].transform("mean")
    Xd = X - X.groupby(groups).transform("mean")
    
    # Drop zero variance or collinear columns
    keep_cols = []
    for col in Xd.columns:
        if Xd[col].std() > 1e-8:
            keep_cols.append(col)
    Xd = Xd[keep_cols]
    
    return sm.OLS(y.to_numpy(dtype=float), Xd).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]}
    )

def run_panel_analysis(docs, treatment_df, event_type="enforcement", treat_var="external_gap_z", label=""):
    print(f"\n{'=' * 80}\nESTIMATION: {label} | Event: {event_type.upper()} | Variable: {treat_var}\n{'=' * 80}")
    
    panel = aggregate(docs, ["ticker", "quarter"])
    mix = (docs.assign(is_proxy=(docs.form == "DEF 14A") * docs.n_words,
                       is_10k=(docs.form == "10-K") * docs.n_words)
           .groupby(["ticker", "quarter"]).agg(p=("is_proxy", "sum"), k=("is_10k", "sum"), n=("n_words", "sum")))
    panel = panel.merge((mix.p / mix.n).rename("mix_proxy").reset_index(), on=["ticker", "quarter"])
    panel = panel.merge((mix.k / mix.n).rename("mix_10k").reset_index(), on=["ticker", "quarter"])
    
    panel = panel.merge(treatment_df[["ticker", treat_var]], on="ticker", how="inner")
    
    # Event definition
    if event_type == "enforcement":
        event_q = pd.Period("2024Q1", freq="Q")
    else:
        event_q = pd.Period("2023Q4", freq="Q")
        
    ref_offset = -1
    panel["event_time"] = (panel["quarter"].astype("period[Q]") - event_q).apply(lambda x: x.n)
    
    # Filter window [-4, +4]
    window_data = panel[panel["event_time"].abs() <= 4].copy()
    
    sides = window_data.groupby("ticker")["event_time"].agg(
        pre=lambda s: (s < 0).sum(), post=lambda s: (s >= 0).sum()
    )
    balanced = sides[(sides["pre"] >= 2) & (sides["post"] >= 2)].index
    balanced_data = window_data[window_data["ticker"].isin(balanced)].copy()
    
    balanced_data["ev"] = pd.Categorical(balanced_data["event_time"], categories=sorted(balanced_data["event_time"].unique()))
    balanced_data["q_str"] = balanced_data["quarter"].astype(str)
    
    outcomes = ["promo_per_1k", "spec_per_1k", "quant_per_1k", "risk_per_1k", "frames_per_1k"]
    controls = "mix_proxy + mix_10k + np.log(n_words)"
    
    print(f"Sample: {len(balanced_data)} firm-quarters across {balanced_data['ticker'].nunique()} firms.")
    
    # 1. DYNAMIC DID
    print("\n--- DYNAMIC EVENT-STUDY COEFFICIENTS ---")
    for out in outcomes:
        rhs = f"C(ev, Treatment(reference={ref_offset})):{treat_var} + C(ev) + {controls}"
        model = fe_ols(balanced_data, out, rhs)
        
        terms = []
        for name, val in model.params.items():
            if f":{treat_var}" in name or f"{treat_var}:" in name:
                m = re.search(r"\[T?\.?(-?\d+)\]", name)
                if m:
                    ev_t = int(m.group(1))
                    if ev_t != ref_offset:
                        terms.append({
                            "event_time": ev_t, "coef": float(val),
                            "se": float(model.bse[name]), "p": float(model.pvalues[name]),
                            "term": name
                        })
        
        tdf = pd.DataFrame(terms).sort_values("event_time")
        pre_terms = tdf[tdf["event_time"] < 0]["term"].tolist()
        post_terms = tdf[tdf["event_time"] >= 0]["term"].tolist()
        
        pre_f, pre_p = np.nan, np.nan
        if pre_terms:
            try:
                res_pre = model.f_test(" = 0, ".join(pre_terms) + " = 0")
                pre_f, pre_p = float(np.squeeze(res_pre.fvalue)), float(np.squeeze(res_pre.pvalue))
            except Exception:
                pass
                
        post_f, post_p = np.nan, np.nan
        if post_terms:
            try:
                res_post = model.f_test(" = 0, ".join(post_terms) + " = 0")
                post_f, post_p = float(np.squeeze(res_post.fvalue)), float(np.squeeze(res_post.pvalue))
            except Exception:
                pass
                
        post_mean = tdf[tdf["event_time"] >= 0]["coef"].mean()
        verdict = "PASS (flat)" if pre_p > 0.10 else ("BORDERLINE" if pre_p > 0.05 else "FAIL (trend)")
        
        print(f"\n{out.upper()}:")
        print("  " + "  ".join(f"t{r.event_time:+d}={r.coef:+.3f}{'*' if r.p < 0.05 else ''}" for r in tdf.itertuples()))
        print(f"  Pre-trends: F={pre_f:.2f}, p={pre_p:.4f} -> {verdict} | Post mean: {post_mean:+.4f}, Post joint p={post_p:.4f}")

    # 2. STATIC PRE/POST DID
    print("\n--- STATIC PRE/POST DID COEFFICIENT (Post x Treatment) ---")
    balanced_data["post"] = (balanced_data["event_time"] >= 0).astype(float)
    balanced_data["post_x_treat"] = balanced_data["post"] * balanced_data[treat_var]
    
    for out in outcomes:
        rhs_static = f"post_x_treat + C(q_str) + {controls}"
        m_static = fe_ols(balanced_data, out, rhs_static)
        coef = m_static.params["post_x_treat"]
        se = m_static.bse["post_x_treat"]
        pval = m_static.pvalues["post_x_treat"]
        tval = coef / se if se > 0 else 0
        star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ("+" if pval < 0.10 else "")))
        print(f"  {out:15s}: beta = {coef:+.4f} (SE = {se:.4f}, t = {tval:+.2f}, p = {pval:.4f}) {star}")

def run_monthly_analysis(docs, treatment_df, treat_var="external_gap_z", label=""):
    print(f"\n{'=' * 80}\nMONTHLY HIGH-RESOLUTION DID: {label} (March 2024 Enforcement Shock)\n{'=' * 80}")
    
    docs["year_month"] = docs["fecha"].dt.to_period("M")
    panel_m = aggregate(docs, ["ticker", "year_month"])
    mix = (docs.assign(is_proxy=(docs.form == "DEF 14A") * docs.n_words,
                       is_10k=(docs.form == "10-K") * docs.n_words)
           .groupby(["ticker", "year_month"]).agg(p=("is_proxy", "sum"), k=("is_10k", "sum"), n=("n_words", "sum")))
    panel_m = panel_m.merge((mix.p / mix.n).rename("mix_proxy").reset_index(), on=["ticker", "year_month"])
    panel_m = panel_m.merge((mix.k / mix.n).rename("mix_10k").reset_index(), on=["ticker", "year_month"])
    panel_m = panel_m.merge(treatment_df[["ticker", treat_var]], on="ticker", how="inner")
    
    event_m = pd.Period("2024-03", freq="M")
    panel_m["m_time"] = (panel_m["year_month"].astype("period[M]") - event_m).apply(lambda x: x.n)
    
    # Window [-6, +6] months
    m_win = panel_m[panel_m["m_time"].abs() <= 6].copy()
    sides = m_win.groupby("ticker")["m_time"].agg(pre=lambda s: (s < 0).sum(), post=lambda s: (s >= 0).sum())
    balanced = sides[(sides["pre"] >= 3) & (sides["post"] >= 3)].index
    b_data = m_win[m_win["ticker"].isin(balanced)].copy()
    b_data["ev_m"] = pd.Categorical(b_data["m_time"], categories=sorted(b_data["m_time"].unique()))
    b_data["m_str"] = b_data["year_month"].astype(str)
    
    controls = "mix_proxy + mix_10k + np.log(n_words)"
    outcomes = ["promo_per_1k", "spec_per_1k", "quant_per_1k", "risk_per_1k"]
    
    print(f"Sample: {len(b_data)} firm-months across {b_data['ticker'].nunique()} firms.")
    
    for out in outcomes:
        rhs = f"C(ev_m, Treatment(reference=-1)):{treat_var} + C(ev_m) + {controls}"
        model = fe_ols(b_data, out, rhs)
        
        terms = []
        for name, val in model.params.items():
            if f":{treat_var}" in name or f"{treat_var}:" in name:
                m = re.search(r"\[T?\.?(-?\d+)\]", name)
                if m:
                    ev_t = int(m.group(1))
                    if ev_t != -1:
                        terms.append({"m_time": ev_t, "coef": float(val), "se": float(model.bse[name]), "p": float(model.pvalues[name]), "term": name})
        
        tdf = pd.DataFrame(terms).sort_values("m_time")
        pre_terms = tdf[tdf["m_time"] < 0]["term"].tolist()
        post_terms = tdf[tdf["m_time"] >= 0]["term"].tolist()
        
        pre_p, post_p = np.nan, np.nan
        if pre_terms:
            try:
                res_pre = model.f_test(" = 0, ".join(pre_terms) + " = 0")
                pre_p = float(np.squeeze(res_pre.pvalue))
            except Exception:
                pass
        if post_terms:
            try:
                res_post = model.f_test(" = 0, ".join(post_terms) + " = 0")
                post_p = float(np.squeeze(res_post.pvalue))
            except Exception:
                pass
                
        print(f"\n{out.upper()}:")
        print("  " + "  ".join(f"m{r.m_time:+d}={r.coef:+.3f}{'*' if r.p < 0.05 else ''}" for r in tdf.itertuples()))
        print(f"  Pre-trends p={pre_p:.4f}  |  Post joint p={post_p:.4f}")

def main():
    docs, patents, fu, size_df = load_data()
    
    # 1. Main Specification: Pre-Enforcement ExternalGap (< 2024-03-18)
    gap_enforce, fs1 = compute_external_gap(docs, patents, fu, size_df, ENFORCEMENT_DATE, "pre_enforce", pre_2023_only=False)
    run_panel_analysis(docs, gap_enforce, event_type="enforcement", treat_var="external_gap_z", label="Main ExternalGap (Pre-Enforce < 2024-03-18)")
    
    # Rank robustness
    run_panel_analysis(docs, gap_enforce, event_type="enforcement", treat_var="rank_gap_z", label="Robustness: Rank(Disc) - Rank(AI Patents)")
    
    # Monthly resolution
    run_monthly_analysis(docs, gap_enforce, treat_var="external_gap_z", label="Main ExternalGap")
    
    # 2. Alternative Specification: Pre-2023 ExternalGap (Clean Pre-GenAI baseline)
    gap_pre2023, fs2 = compute_external_gap(docs, patents, fu, size_df, ENFORCEMENT_DATE, "pre_enforce", pre_2023_only=True)
    run_panel_analysis(docs, gap_pre2023, event_type="enforcement", treat_var="external_gap_z", label="Pre-2023 ExternalGap (Exogenous Pre-Boom Exposure)")
    
    # 3. Regulatory Warning Shock (Dec 5, 2023)
    gap_warn, fs3 = compute_external_gap(docs, patents, fu, size_df, WARNING_DATE, "pre_warning", pre_2023_only=False)
    run_panel_analysis(docs, gap_warn, event_type="warning", treat_var="external_gap_z", label="Warning Shock ExternalGap (Pre-Warning < 2023-12-05)")

if __name__ == "__main__":
    main()
