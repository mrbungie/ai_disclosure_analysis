"""Rebuilds expanding archetype weights at QUARTERLY cutoffs (not annual),
so each call can be matched to the most recent cutoff strictly before its
own call date -- true "as of" weights, no leakage, without discarding
close-to-a-year of legitimate recent information the way a full calendar-
year lag would.
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
import layers as L  # noqa: E402
from posture_features import UNDEFINED_ARCHETYPE, fit_aa, name_vertices  # noqa: E402

DOCUMENT_PANEL = L.gold_path("covariates", "document", "document_panel")
OUT = L.gold_path("covariates", "firm_quarter", "archetype_weights_expanding")

_POSTURE_EXP = ["promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
                "temporal_posture", "ai_positioning", "specificity"]
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
               .select("ticker", "filing_date", "concepts", "temporal", "specificity", "rhetoric",
                       pl.col("is_customer_facing").fill_null(False))
               .collect()
               .to_pandas())

_frames_exp["filing_date"] = pd.to_datetime(_frames_exp["filing_date"])
_frames_exp["quarter"] = _frames_exp["filing_date"].dt.to_period("Q")

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
_dp_exp = pd.read_parquet(DOCUMENT_PANEL)
_dp_exp["quarter"] = pd.to_datetime(_dp_exp["fecha"]).dt.to_period("Q")

_quarters = sorted(_df_exp["quarter"].unique())
print(f"{len(_quarters)} quarterly cutoffs: {_quarters[0]} .. {_quarters[-1]}")

_exp_rows = []
for _qi, _cutoff in enumerate(_quarters):
    if _qi < 3:
        continue  # need enough history to fit a stable geometry
    _sub = _df_exp[_df_exp["quarter"] <= _cutoff]
    _counts = _sub.groupby("ticker").size()
    _active_tickers = _counts[_counts >= 5].index
    if len(_active_tickers) < 20:
        continue
    _rates = _sub.groupby("ticker")[_POSTURE_EXP].mean().loc[_active_tickers]
    # shrink using expanding().shift(1)-style: use counts as of this cutoff (already excludes anything after cutoff)
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
    _shrunk = _shrink_to_prior_exp(_rates, _counts.loc[_active_tickers])
    _words = _dp_exp[_dp_exp["quarter"] <= _cutoff].groupby("ticker").agg(
        w=("n_words", "sum"), fr=("n_frames", "sum")).reindex(_active_tickers)
    _fr1k = 1000 * _words["fr"] / _words["w"].replace(0, np.nan)
    _shrunk["disclosure_intensity"] = _fr1k.rank(pct=True)
    _train_df = _shrunk.dropna()
    if len(_train_df) < 20:
        continue
    _mu_e, _sd_e = _train_df.mean(), _train_df.std(ddof=0)
    _sd_e = _sd_e.replace(0, 1.0)
    _X_train = ((_train_df - _mu_e) / _sd_e).values
    _aa_e, _ = fit_aa(_X_train, 3)
    _tickers_cross = _active_tickers.intersection(_train_df.index)
    _X_cross = ((_train_df.loc[_tickers_cross] - _mu_e) / _sd_e).values
    _W_cross = _aa_e.transform(_X_cross)
    # Vertices are named on the global posture PCA axes; an archetype with no
    # matching vertex at this cutoff gets a missing weight.
    _vertex_names, _vertex_corr = name_vertices(_W_cross, _train_df.loc[_tickers_cross])
    _col = {name: j for j, name in _vertex_names.items() if name != UNDEFINED_ARCHETYPE}
    def _weight(name):
        return _W_cross[:, _col[name]] if name in _col else np.full(len(_tickers_cross), np.nan)
    _exp_rows.append(pd.DataFrame({
        "ticker": _tickers_cross, "cutoff_quarter": _cutoff,
        "w_voc": _weight("Vocal Substantives"), "w_gov": _weight("Governance-Led Disclosers"),
        "w_def": _weight("Defensive Disclosers"),
    }))

    # Persist THIS cutoff's fitted model (a fresh AA fit every cutoff, by
    # design -- no-lookahead). Predictable path from cutoff_quarter alone,
    # no manifest needed to find it. Lets predictions be re-derived (model
    # + a dataset) as a separate step instead of reusing this in-process fit.
    _model_dir = REPO_ROOT / "models" / "posture_archetype_expanding" / f"cutoff={_cutoff}"
    _model_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        "model": _aa_e, "mean": _mu_e, "std": _sd_e, "vertex_names": _vertex_names, "vertex_pc_corr": _vertex_corr,
        "feature_names": _CROSS_FEATS_EXP, "cutoff_quarter": str(_cutoff), "seed": 42,
    }, _model_dir / "model.pkl")
    if _qi % 4 == 0:
        print(f"  {_cutoff}: {len(_tickers_cross)} firms")

_panel_q = pd.concat(_exp_rows, ignore_index=True)
_panel_q.to_parquet(OUT, index=False)
print(f"\nSaved {len(_panel_q)} ticker-quarter soft weights to {OUT}")
