#!/usr/bin/env python3
"""
Empirical Document Channel Validation for Archetypal Analysis (k=3).

Tests the hypothesis that disclosure archetypes reflect strategic regulatory channel
decisions (where the firm chooses to speak: DEF 14A vs. 10-K Item 1A vs. 10-K Items 1/7)
rather than purely textual style differences across identical document mixes.

Outputs:
- Correlation matrix between archetypal weights (w_Vocal, w_Gov, w_Def) and document fractions.
- Concept rates across filing types and 10-K sections.
- docs/analytics/archetype_document_channels.json summary.
"""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
from archetypes import AA

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "duckdb" / "thesis.duckdb"
STRAT_PATH = REPO_ROOT / "data" / "processed" / "clusters" / "firm_strategy_dimensions.parquet"
OUT_JSON = REPO_ROOT / "docs" / "analytics" / "archetype_document_channels.json"

# 1. Fit Archetypal Analysis (k=3)
df = pd.read_parquet(STRAT_PATH)
FEATS = ["promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
         "temporal_posture", "ai_positioning", "specificity", "disclosure_intensity"]
fit_pop = df[df["archetype"] != "No AI"].copy()
X = fit_pop[FEATS].values
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
aa = AA(n_archetypes=3, random_state=42, max_iter=500)
W = aa.fit_transform(X_std)
A = aa.archetypes_

def_col = int(np.argmax(A[:, FEATS.index("risk_orientation")]))
rem_cols = [i for i in range(3) if i != def_col]
gov_col = max(rem_cols, key=lambda i: A[i, FEATS.index("governance_orientation")])
voc_col = [i for i in range(3) if i not in (def_col, gov_col)][0]

fit_pop["w_Vocal"] = W[:, voc_col]
fit_pop["w_Gov"] = W[:, gov_col]
fit_pop["w_Def"] = W[:, def_col]

# 2. Extract Document Channel Sourcing from duckdb
con = duckdb.connect(str(DB_PATH), read_only=True)
frames_df = con.execute("""
    SELECT fm.ticker,
           fm.form_type,
           f.item_key,
           count(*) as n
    FROM gold_ai_frames f
    JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code = 'us' AND f.has_frame AND fm.ticker IS NOT NULL
    GROUP BY fm.ticker, fm.form_type, f.item_key
""").df()

# Compute rates by form and section
# v2 concept enum groups risk/governance more coarsely than v1 did (see
# docs/migration_v1_to_v2_analytics.md §1.4) -- rather than enumerate every
# risk_*/gov_* leaf, match the whole family by prefix, same pattern used in
# ai_intensity.py's `is_gov`.
RISK_RATE_SQL = "round(avg(list_bool_or(list_transform(f.concepts, c -> c LIKE 'risk_%'))::int), 3) as risk_rate"
GOV_RATE_SQL = "round(avg(list_bool_or(list_transform(f.concepts, c -> c LIKE 'gov_%'))::int), 3) as gov_rate"
PROMO_RATE_SQL = "round(avg(list_contains(f.rhetoric, 'promotional')::int), 3) as promo_rate"
SPEC_RATE_SQL = ("round(avg((list_contains(f.specificity, 'process')::int "
                  "+ list_contains(f.specificity, 'product')::int) / 2.0), 3) as spec_rate")

rates_form = con.execute(f"""
    SELECT
        fm.form_type,
        count(*) as total_frames,
        {RISK_RATE_SQL},
        {GOV_RATE_SQL},
        {PROMO_RATE_SQL},
        {SPEC_RATE_SQL}
    FROM gold_ai_frames f
    JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code = 'us' AND f.has_frame
    GROUP BY fm.form_type
    ORDER BY total_frames DESC
""").df()

rates_10k_section = con.execute(f"""
    SELECT
        f.item_key,
        count(*) as total_frames,
        {RISK_RATE_SQL},
        {GOV_RATE_SQL},
        {PROMO_RATE_SQL},
        {SPEC_RATE_SQL}
    FROM gold_ai_frames f
    JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code = 'us' AND f.has_frame AND fm.form_type = '10-K'
    GROUP BY f.item_key
    ORDER BY total_frames DESC
""").df()
con.close()

# 3. Firm-level Channel Aggregation
piv = frames_df.copy()
piv["is_def14a"] = piv["form_type"].str.contains("14A", na=False)
piv["is_10k_1a"] = (piv["form_type"] == "10-K") & (piv["item_key"].isin(["1A", "item_1a"]))
piv["is_10k_biz_mda"] = (piv["form_type"] == "10-K") & (piv["item_key"].isin(["1", "item_1", "7", "item_7"]))

firm_totals = piv.groupby("ticker")["n"].sum().rename("tot_frames")
firm_def14a = piv[piv["is_def14a"]].groupby("ticker")["n"].sum().rename("n_def14a")
firm_10k_1a = piv[piv["is_10k_1a"]].groupby("ticker")["n"].sum().rename("n_10k_1a")
firm_10k_biz = piv[piv["is_10k_biz_mda"]].groupby("ticker")["n"].sum().rename("n_10k_biz_mda")

chk = pd.DataFrame({"ticker": fit_pop["ticker"]}).merge(firm_totals, on="ticker", how="left")
chk = chk.merge(firm_def14a, on="ticker", how="left").fillna(0)
chk = chk.merge(firm_10k_1a, on="ticker", how="left").fillna(0)
chk = chk.merge(firm_10k_biz, on="ticker", how="left").fillna(0)

chk["frac_def14a"] = chk["n_def14a"] / chk["tot_frames"]
chk["frac_10k_1a"] = chk["n_10k_1a"] / chk["tot_frames"]
chk["frac_10k_biz_mda"] = chk["n_10k_biz_mda"] / chk["tot_frames"]

merged = fit_pop.merge(chk, on="ticker")
cols_corr = ["w_Vocal", "w_Gov", "w_Def", "frac_def14a", "frac_10k_1a", "frac_10k_biz_mda"]
corr_mat = merged[cols_corr].corr().round(3)

print("=" * 70)
print("EMPIRICAL CHANNEL VALIDATION: CORRELATION MATRIX (N=441 firms)")
print("=" * 70)
print(corr_mat.to_string())

print("\n" + "=" * 70)
print("CONCEPT RATES BY FILING TYPE")
print("=" * 70)
print(rates_form.to_string(index=False))

print("\n" + "=" * 70)
print("CONCEPT RATES WITHIN 10-K BY SECTION")
print("=" * 70)
print(rates_10k_section.to_string(index=False))

results = {
    "correlation_matrix": corr_mat.to_dict(),
    "rates_by_form": rates_form.to_dict(orient="records"),
    "rates_10k_by_section": rates_10k_section.to_dict(orient="records")
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved channel validation results to: {OUT_JSON}")
