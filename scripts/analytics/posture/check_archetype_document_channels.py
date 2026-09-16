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
import os

# Pin BLAS to one thread before numpy/archetypes load: multi-threaded BLAS
# reduction order isn't deterministic run-to-run, which can flip a seeded
# AA.fit() to a different local optimum (see posture_features.py).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
from posture_features import CLUSTER_FEATURES, fit_aa  # noqa: E402
import layers as L  # noqa: E402

OUT_JSON = L.results_path("posture", "archetype_document_channels.json")

# 1. Fit Archetypal Analysis (k=3)
df = L.read_gold("firm", ("covariates", "posture_archetype_static"))
# the features the archetypes are fitted on, from the one place that defines
# them: intensity is a covariate, not a posture, and is not among them
FEATS = list(CLUSTER_FEATURES)
fit_pop = df[df["archetype"] != "No AI"].copy()
X = fit_pop[FEATS].values
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
aa, W = fit_aa(X_std, 3)
A = aa.archetypes_

def_col = int(np.argmax(A[:, FEATS.index("risk_orientation")]))
rem_cols = [i for i in range(3) if i != def_col]
gov_col = max(rem_cols, key=lambda i: A[i, FEATS.index("governance_orientation")])
voc_col = [i for i in range(3) if i not in (def_col, gov_col)][0]

fit_pop["w_Vocal"] = W[:, voc_col]
fit_pop["w_Gov"] = W[:, gov_col]
fit_pop["w_Def"] = W[:, def_col]

# 2. Extract Document Channel Sourcing from the frame and manifest tables
frames = (L.scan("silver.ai_frames").filter(pl.col("has_frame"))
          .join(L.scan("silver.filing_manifest").select("country_code", "accession_number", "ticker", "form_type"),
                on=["country_code", "accession_number"], how="inner"))
frames_df = (frames.filter(pl.col("ticker").is_not_null())
             .group_by("ticker", "form_type", "item_key")
             .agg(pl.len().cast(pl.Int64).alias("n"))
             .collect().to_pandas())

# Compute rates by form and section
# v2 concept enum groups risk/governance more coarsely than v1 did (see
# docs/migration_v1_to_v2_analytics.md §1.4) -- rather than enumerate every
# risk_*/gov_* leaf, match the whole family by prefix, same pattern used in
# ai_intensity.py's `n_risk`/`n_gov`. A frame whose concept list is empty (or
# null) has no value for the risk/gov indicator and drops out of that mean.
def _any_concept_like(prefix: str) -> pl.Expr:
    concepts = pl.col("concepts").list.drop_nulls()
    hit = concepts.list.eval(pl.element().str.contains(f"(?s)^{prefix}.")).list.any()
    return pl.when(concepts.list.len() > 0).then(hit).cast(pl.Int32)


def _rate(expr: pl.Expr, name: str) -> pl.Expr:
    return expr.mean().round(3, mode="half_away_from_zero").alias(name)


RATES = [
    pl.len().cast(pl.Int64).alias("total_frames"),
    _rate(_any_concept_like("risk"), "risk_rate"),
    _rate(_any_concept_like("gov"), "gov_rate"),
    _rate(pl.col("rhetoric").list.contains("promotional").cast(pl.Int32), "promo_rate"),
    _rate((pl.col("specificity").list.contains("process").cast(pl.Int32)
           + pl.col("specificity").list.contains("product").cast(pl.Int32)) / 2.0, "spec_rate"),
]

# ties on total_frames are broken by the group key so the record order is fixed
rates_form = (frames.group_by("form_type").agg(RATES)
              .sort(["total_frames", "form_type"], descending=[True, False], nulls_last=True)
              .collect().to_pandas())

rates_10k_section = (frames.filter(pl.col("form_type") == "10-K")
                     .group_by("item_key").agg(RATES)
                     .sort(["total_frames", "item_key"], descending=[True, False], nulls_last=True)
                     .collect().to_pandas())

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

# 4. DEF 14A coverage across disclosing firms (thesis.qmd
# "Institutional channels: where vs. how firms disclose", ~1477-1485): how
# many disclosing (non-"No AI") firms ever filed a DEF 14A at all, and of
# those, how many actually mention AI in it versus omit it -- a low
# w_Gov weight only signals active omission if the proxy statement exists.
disclosing_tickers = set(fit_pop["ticker"])
has_def14a = set(
    L.scan("silver.filing_manifest").filter(pl.col("form_type") == "DEF 14A")
    .select("ticker").drop_nulls().unique().collect().to_pandas()["ticker"]
)
coverage = disclosing_tickers & has_def14a
mentions_def14a = set(
    L.scan("silver.ai_frames").filter(pl.col("has_frame"))
    .join(L.scan("silver.filing_manifest").filter(pl.col("form_type") == "DEF 14A")
          .select("country_code", "accession_number", "ticker"),
          on=["country_code", "accession_number"], how="inner")
    .select("ticker").drop_nulls().unique().collect().to_pandas()["ticker"]
)
n_coverage_mention = len(coverage & mentions_def14a)
def14a_coverage = {
    "n_disclosing": len(disclosing_tickers),
    "n_has_def14a": len(coverage),
    "coverage_pct": round(100 * len(coverage) / len(disclosing_tickers)),
    "n_mention": n_coverage_mention,
    "n_omit": len(coverage) - n_coverage_mention,
}
print("\n" + "=" * 70)
print("DEF 14A COVERAGE ACROSS DISCLOSING FIRMS")
print("=" * 70)
print(def14a_coverage)

results = {
    "correlation_matrix": corr_mat.to_dict(),
    "rates_by_form": rates_form.to_dict(orient="records"),
    "rates_10k_by_section": rates_10k_section.to_dict(orient="records"),
    "def14a_coverage": def14a_coverage,
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved channel validation results to: {OUT_JSON}")
