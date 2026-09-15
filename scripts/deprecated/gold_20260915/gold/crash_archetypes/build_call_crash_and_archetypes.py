"""Builds the yearly-cutoff expanding posture archetype (one Archetypal
Analysis fit per fiscal year, point-in-time safe) used as a predictor in
the crash-risk regressions. The NCSKEW/DUVOL regressions that used to live
in this script moved to
scripts/analytics/crash_archetypes/call_crash_regressions.py -- this file
is data/model construction only (spine -> archetype fit -> prediction),
per the gold-stage rule that gold holds no regressions/p-values.

Writes data/gold/predictions/firm_year/posture_archetype_expanding.parquet
directly (id `TICKER_YYYY`) -- no separate consolidate/predictions wrapper,
since the fit and the final prediction are produced in the same walk-forward
loop (one AA fit per cutoff year, immediately applied to that cutoff's
cross-section). Reads the pooled archetype label from
data/gold/predictions/firm_year/posture_archetype_static.parquet (only used
to flag "No AI" ticker-years, not as a point-in-time predictor) joined with
its posture features from data/gold/covariates/firm_year/posture.parquet,
and document-level word/frame totals from
data/gold/covariates/document/document_panel.parquet.
"""
import os

# Pin BLAS to one thread before numpy/archetypes load: multi-threaded BLAS
# reduction order isn't deterministic run-to-run, which can flip a seeded
# AA.fit() to a different local optimum (see build_strategy_dimensions.py).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
from posture_features import fit_aa, name_vertices  # noqa: E402

GOLD_ARCHETYPE_STATIC = REPO_ROOT / "data" / "gold" / "predictions" / "firm_year" / "posture_archetype_static.parquet"
GOLD_POSTURE = REPO_ROOT / "data" / "gold" / "covariates" / "firm_year" / "posture.parquet"
GOLD_DOCUMENT_PANEL = REPO_ROOT / "data" / "gold" / "covariates" / "document" / "document_panel.parquet"
ARCH_PATH = REPO_ROOT / "data" / "gold" / "predictions" / "firm_year" / "posture_archetype_expanding.parquet"

print("Loading frames for the expanding archetypes...")

_POSTURE_EXP = [
    "promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
    "temporal_posture", "ai_positioning", "specificity"
]
_CROSS_FEATS_EXP = _POSTURE_EXP + ["disclosure_intensity"]

_frame_domains = (L.scan("silver.ai_activities").filter(pl.col("has_activity"))
                  .group_by("text_hash", "frame_id")
                  .agg((pl.col("domain") == "customer_facing").any().alias("is_customer_facing")))
_frames_exp = (L.scan("silver.ai_frames").filter(pl.col("has_frame"))
               .join(_frame_domains, on=["text_hash", "frame_id"], how="left")
               .join(L.scan("silver.filing_manifest").select("country_code", "accession_number", "ticker", "filing_date"),
                     on=["country_code", "accession_number"], how="inner")
               .filter(pl.col("ticker").is_not_null())
               .sort(["ticker", "accession_number", "text_hash", "frame_id", "item_key", "paragraph_index"], nulls_last=True)
               .select("ticker", pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"),
                       "concepts", "temporal", "specificity", "rhetoric",
                       pl.col("is_customer_facing").fill_null(False))
               .collect()
               .to_pandas())

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
_m = (pd.read_parquet(GOLD_ARCHETYPE_STATIC)[["ticker", "year", "archetype"]]
      .merge(pd.read_parquet(GOLD_POSTURE)[["ticker", "year"] + _CROSS_FEATS_EXP], on=["ticker", "year"], how="inner"))
_dp_exp = pd.read_parquet(GOLD_DOCUMENT_PANEL)
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
    _aa_e, _ = fit_aa(_X_train, 3)
    _cross = _m[_m["year"] == _cutoff][["ticker", "year", "archetype"] + _CROSS_FEATS_EXP].copy()
    _active = _cross["archetype"] != "No AI"
    _X_cross = ((_cross.loc[_active, _CROSS_FEATS_EXP] - _mu_e) / _sd_e).values
    _W_cross = _aa_e.transform(_X_cross)
    # Vertices are named on the global posture PCA axes (posture_features.name_vertices).
    _vertex_names, _vertex_corr = name_vertices(_W_cross, _cross.loc[_active])
    _cross.loc[_active, "arch_exp"] = [_vertex_names[j] for j in np.argmax(_W_cross, axis=1)]
    _cross.loc[~_active, "arch_exp"] = "No AI"
    _exp_rows.append(_cross[["ticker", "year", "arch_exp"]])

    # Persist THIS cutoff's fitted model (a fresh AA fit every fiscal year,
    # by design -- no-lookahead, distinct from posture_archetype_expanding's
    # quarterly-cutoff sibling above and from posture_archetype_static's
    # single full-panel fit). Predictable path from cutoff alone.
    _model_dir = REPO_ROOT / "models" / "posture_archetype_expanding_yearly" / f"cutoff={_cutoff}"
    _model_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        "model": _aa_e, "mean": _mu_e, "std": _sd_e, "vertex_names": _vertex_names,
        "vertex_pc_corr": _vertex_corr, "feature_names": _CROSS_FEATS_EXP, "cutoff_year": int(_cutoff), "seed": 42,
    }, _model_dir / "model.pkl")

_panel_exp = pd.concat(_exp_rows, ignore_index=True)
_panel_exp.insert(0, "id", _panel_exp["ticker"] + "_" + _panel_exp["year"].astype(int).astype(str))
ARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
_panel_exp.to_parquet(ARCH_PATH, index=False)
print(f"Saved expanding archetypes to {ARCH_PATH}")
