"""Robustness check: refit the k=3 posture Archetypal Analysis on Form 10-K
frames alone (thesis.qmd "Institutional channels: where vs. how firms
disclose" / Appendix E, ~1487-1567 and ~4550-4565).

The full-corpus fit pools every form (10-K, DEF 14A, 8-K, ...); this script
asks whether the same three archetypes emerge from 10-K text on its own, or
whether one of them (Governance-Led, expected to load on DEF 14A board
language) is an artifact of pooling across statutory channels. It:

1. Loads the full-corpus posture archetype model bundle
   (models/posture_archetype_static/model.pkl) and applies its fitted
   scaler + AA `.transform()` (never refits) to the same firm population
   (covariates/firm/posture_archetype_static.parquet) to get each firm's
   continuous full-corpus archetype weights (w_Vocal, w_Gov, w_Def).
2. Rebuilds the same 7 frame-level posture rates plus disclosure intensity
   from Form 10-K frames only (silver.ai_frames joined to
   silver.filing_manifest, restricted to form_type == '10-K'), applies the
   same empirical-Bayes shrinkage as the main construction
   (scripts/gold/firm/build_posture_archetype_static.py), and genuinely
   refits a fresh AA(k=3) on this 10-K-only feature matrix -- this is the
   one legitimate new fit in this file, since the whole point of the check
   is whether an independent fit on a text subset recovers the same
   archetypes.
3. Correlates the two firms' weight sets on the tickers common to both
   fits, and reports vertex-level standardized coordinates (SD from the
   fit's own pooled mean) for the governance and risk dimensions.

Output: data/results/posture/archetype_10k_refit.json
    {"correlations": {"vocal": r, "governance": r, "defensive": r},
     "vertices": {"governance_10k_on_governance": z, "governance_10k_on_risk": z,
                  "defensive_full_on_risk": z, "defensive_10k_on_risk": z}}

Usage:
    uv run python scripts/analytics/posture/archetype_10k_refit.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import joblib
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
from posture_features import fit_aa  # noqa: E402
import layers as L  # noqa: E402

FEATS = ["promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
         "temporal_posture", "ai_positioning", "specificity", "disclosure_intensity"]
POSTURE_COLS_10K = FEATS[:-1]  # everything but disclosure_intensity


def _posture_10k(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    concepts = d["concepts"].apply(lambda c: set(c) if c is not None else set())
    d["promotional_posture"] = d["rhetoric"].apply(
        lambda r: np.mean([x in list(r) if r is not None else False for x in ("promotional", "strategic")]))
    d["hedging_posture"] = d["rhetoric"].apply(lambda r: float("hedged" in list(r)) if r is not None else 0.0)
    d["risk_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("risk_") for c in s)))
    d["governance_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("gov_") for c in s)))
    d["temporal_posture"] = (d["temporal"] == "realized").astype(float)
    d["ai_positioning"] = d["is_customer_facing"].astype(float)
    d["specificity"] = d["specificity"].apply(lambda s: len(s) / 5.0 if s is not None else 0.0)
    out = d.groupby("ticker")[POSTURE_COLS_10K].mean()
    out["n_frames_10k"] = d.groupby("ticker").size()
    return out.reset_index()


def _shrink_10k(rates: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
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


def main() -> None:
    # 1. Full-corpus continuous weights: apply (not refit) the frozen bundle.
    bundle = joblib.load(REPO_ROOT / "models" / "posture_archetype_static" / "model.pkl")
    model, scaler, feature_names, cluster_names = (bundle["model"], bundle["scaler"],
                                                    bundle["feature_names"], bundle["cluster_names"])
    df_full = L.read_gold("firm", ("covariates", "posture_archetype_static"))
    fit_pop = df_full[df_full["archetype"] != "No AI"].copy()
    W_full = model.transform(scaler.transform(fit_pop[feature_names].values))
    name_to_col = {name: cid for cid, name in cluster_names.items()}
    W_named = pd.DataFrame({
        "Vocal": W_full[:, name_to_col["Vocal Substantives"]],
        "Governance": W_full[:, name_to_col["Governance-Led Disclosers"]],
        "Defensive": W_full[:, name_to_col["Defensive Disclosers"]],
    }, index=fit_pop["ticker"])

    # Full-corpus vertex (matrix A) standardized coordinate on risk, for the
    # Defensive archetype -- read from the same bundle, no fit needed.
    A_full = model.archetypes_
    def_col_full = name_to_col["Defensive Disclosers"]
    z_def_full_on_risk = float(A_full[def_col_full, feature_names.index("risk_orientation")])

    # 2. Rebuild posture rates from 10-K frames only.
    frame_domains = (L.scan("silver.ai_activities").filter(pl.col("has_activity"))
                     .group_by("text_hash", "frame_id")
                     .agg((pl.col("domain") == "customer_facing").any().alias("is_customer_facing")))
    frames_10k = (
        L.scan("silver.ai_frames").filter(pl.col("has_frame"))
        .join(L.scan("silver.filing_manifest").filter(pl.col("form_type") == "10-K")
              .select("country_code", "accession_number", "ticker"),
              on=["country_code", "accession_number"], how="inner")
        .filter(pl.col("ticker").is_not_null())
        .join(frame_domains, on=["text_hash", "frame_id"], how="left")
        .with_columns(pl.col("is_customer_facing").fill_null(False))
        .select("ticker", "concepts", "temporal", "specificity", "rhetoric", "is_customer_facing")
        .collect().to_pandas()
    )

    p10k = _posture_10k(frames_10k)
    p10k = p10k[p10k["n_frames_10k"] >= 5].reset_index(drop=True)
    shrunk10k = _shrink_10k(p10k[POSTURE_COLS_10K], p10k["n_frames_10k"])
    shrunk10k["disclosure_intensity"] = p10k["n_frames_10k"].rank(pct=True).values
    X10k_raw = shrunk10k[FEATS].values
    X10k = (X10k_raw - X10k_raw.mean(axis=0)) / X10k_raw.std(axis=0, ddof=0)

    aa10k, W10k = fit_aa(X10k, 3)
    A10k = aa10k.archetypes_
    def_col10k = int(np.argmax(A10k[:, FEATS.index("risk_orientation")]))
    rem10k = [i for i in range(3) if i != def_col10k]
    gov_col10k = max(rem10k, key=lambda i: A10k[i, FEATS.index("governance_orientation")])
    voc_col10k = [i for i in range(3) if i not in (def_col10k, gov_col10k)][0]

    w10k_named = pd.DataFrame({
        "Vocal": W10k[:, voc_col10k], "Governance": W10k[:, gov_col10k], "Defensive": W10k[:, def_col10k],
    }, index=p10k["ticker"])

    common = W_named.index.intersection(w10k_named.index)
    r_vocal = float(W_named.loc[common, "Vocal"].corr(w10k_named.loc[common, "Vocal"]))
    r_gov = float(W_named.loc[common, "Governance"].corr(w10k_named.loc[common, "Governance"]))
    r_def = float(W_named.loc[common, "Defensive"].corr(w10k_named.loc[common, "Defensive"]))

    out = {
        "n_firms_10k_fit": int(len(p10k)),
        "n_common_tickers": int(len(common)),
        "correlations": {"vocal": r_vocal, "governance": r_gov, "defensive": r_def},
        "vertices": {
            "governance_10k_on_governance": float(A10k[gov_col10k, FEATS.index("governance_orientation")]),
            "governance_10k_on_risk": float(A10k[gov_col10k, FEATS.index("risk_orientation")]),
            "defensive_full_on_risk": z_def_full_on_risk,
            "defensive_10k_on_risk": float(A10k[def_col10k, FEATS.index("risk_orientation")]),
        },
    }
    out_path = L.results_path("posture", "archetype_10k_refit.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
