"""Diagnostics and cross-pipeline regressions for the k=3 posture archetype
(G-M1): PCA/correlation/bootstrap-by-k robustness checks, year-over-year
archetype persistence, and the two regressions validating disclosure
posture against the INDEPENDENT disclosed-activity extraction
(promotional_posture ~ log(1 + activities); log(1 + activities) by
archetype, raw vs. volume-adjusted). None of this is the fit itself or its
predictions -- `scripts/gold/firm/build_posture_archetype_static.py` (the
fit) and `scripts/gold/firm_year/build_posture.py` (model + firm-year
posture -> firm-year labels) own those. This script recomputes
the diagnostics from the same `posture_features` functions the gold fit
uses (so the numbers can't silently drift apart) plus the gold outputs
already on disk, and writes ONLY a report.

Output: `data/results/posture/strategy_dimensions_diagnostics.json`,
replacing `data/processed/clusters/strategy_dimensions_manifest.json`
(thesis.qmd lines ~211-218, ~2182, ~4342). Same top-level keys as the old
manifest: `min_frames_pooled`, `cluster_features`, `correlation_matrix`,
`pca_diagnostic`, `k`, `stability_by_k`, `cluster_sizes`,
`persistence_year_over_year`, `promotional_excess_regression`,
`activity_volume_regression`.

`n_activities` (the regressor) is NOT new gold data: it is the activity
spine `data/gold/spines/activity/activity.parquet` (already gold, built by
`build_activity.py`) grouped by ticker -- computed here on the fly,
not persisted as its own artifact. `promotional_excess` (the OLS residual)
IS a statistic, not a covariate: it is written alongside this JSON as
`data/results/posture/promotional_excess.parquet` (ticker,
promotional_excess), not as a gold covariate.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from posture_features import (  # noqa: E402  (pins BLAS threads before numpy/archetypes import)
    ARCHETYPE_PRIORITY, CLUSTER_FEATURES, INTENSITY, MIN_FRAMES, OUTPUT_FEATURES, POSTURE, bootstrap_stability,
    build_posture, load_frames, pca_diagnostic, shrink_to_prior,
)

from sklearn.preprocessing import StandardScaler  # noqa: E402

import layers as L  # noqa: E402
from ai_intensity import firm_intensity  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "posture_archetype_static" / "model.pkl"


def main() -> None:
    frames = load_frames()
    universe = firm_intensity(["ticker"])
    posture = build_posture(frames, ["ticker"])
    merged = universe.merge(posture, on="ticker", how="left")
    merged[POSTURE] = merged[POSTURE].fillna(0.0)
    merged["n_frames"] = merged["n_frames"].fillna(0).astype(int)
    active = merged[merged["n_frames"] >= MIN_FRAMES].reset_index(drop=True)

    shrunk = shrink_to_prior(active[POSTURE], active["n_frames"])
    shrunk[INTENSITY] = active["frames_per_1k"].rank(pct=True).values
    # Descriptive, so it keeps intensity: how volume relates to each posture
    # dimension is worth seeing precisely BECAUSE intensity no longer enters the
    # fit. The PCA below stays on the fitted features.
    corr = shrunk[OUTPUT_FEATURES].corr()

    X = StandardScaler().fit_transform(shrunk[CLUSTER_FEATURES].values)
    pca = pca_diagnostic(X, CLUSTER_FEATURES)

    print("=== choosing k by bootstrap stability (25 replicates, floor 0.60) -- reproduced for the report ===")
    stabilities = {}
    for k in (2, 3, 4, 5):
        s = bootstrap_stability(frames, universe, k)
        stabilities[k] = s
        print(f"k={k}: " + " ".join(f"{v:.2f}" for v in s) + f" | min {s.min():.2f}")

    # k itself is a FITTING decision (already made, persisted in the model
    # bundle) -- read it back instead of re-deriving a floor rule here, so
    # this report can never silently disagree with what gold actually fit.
    bundle = joblib.load(MODEL_PATH)
    k, cluster_names = bundle["k"], bundle["cluster_names"]

    pooled = L.read_gold("firm", ("covariates", "posture_archetype_static"))
    cluster_sizes = pooled.loc[pooled["archetype"] != "No AI", "archetype"].value_counts().to_dict()

    # Year-over-year persistence, from the firm-year projection of this same
    # pooled model (data/gold/covariates/firm_year/posture_archetype_static.parquet),
    # over the firm-years the model labels.
    panel = L.read_gold("firm_year", ("covariates", "posture_archetype_static", ["archetype"]))
    panel = panel[panel["archetype"].notna()].sort_values(["ticker", "year"])
    panel["prev"] = panel.groupby("ticker")["archetype"].shift()
    transitions = panel.dropna(subset=["prev"])
    persistence = float((transitions["archetype"] == transitions["prev"]).mean())

    # Cross-pipeline validation against the INDEPENDENT disclosed-activity
    # extraction: n_activities per ticker, pooled across every channel and
    # year, from the gold activity spine (not from any processed panel).
    acts = L.read_gold("activity", spine_columns=["ticker"])
    n_activities = acts.groupby("ticker").size().rename("n_activities")
    pooled = pooled.merge(n_activities, on="ticker", how="left")
    pooled["n_activities"] = pooled["n_activities"].fillna(0).astype(int)
    pooled["log_activities"] = np.log1p(pooled["n_activities"])
    reg = pooled[pooled["archetype"] != "No AI"].copy()
    reg["log_frames"] = np.log1p(reg["n_frames"])

    ols = sm.OLS(reg["promotional_posture"], sm.add_constant(reg["log_activities"])).fit()
    ols_ctrl = sm.OLS(reg["promotional_posture"], sm.add_constant(reg[["log_activities", "log_frames"]])).fit()
    promotional_excess = pd.Series(np.nan, index=pooled.index)
    promotional_excess.loc[reg.index] = ols.resid.values
    print(f"\n=== promotional_posture ~ log(1 + disclosed activities) ===\n"
          f"{ols.summary().tables[1]}\nR^2 = {ols.rsquared:.3f}, n = {len(reg)}")
    promo_reg = {"const": float(ols.params["const"]), "beta_log_activities": float(ols.params["log_activities"]),
                "se_log_activities": float(ols.bse["log_activities"]), "p_log_activities": float(ols.pvalues["log_activities"]),
                "r2": float(ols.rsquared), "n": int(len(reg)),
                "controlled": {"beta_log_activities": float(ols_ctrl.params["log_activities"]),
                              "p_log_activities": float(ols_ctrl.pvalues["log_activities"]),
                              "beta_log_frames": float(ols_ctrl.params["log_frames"]),
                              "p_log_frames": float(ols_ctrl.pvalues["log_frames"])}}

    ref = "Defensive Disclosers"
    label_priority = [label for _, label in ARCHETYPE_PRIORITY] + ["Enterprise Communicators"]
    present = set(cluster_names.values())
    others = [label for label in label_priority if label != ref and label in present]
    dummies = pd.get_dummies(reg["archetype"]).astype(float)[others]
    raw_m = sm.OLS(reg["log_activities"], sm.add_constant(dummies)).fit(cov_type="HC1")
    adj_m = sm.OLS(reg["log_activities"], sm.add_constant(dummies.assign(log_frames=reg["log_frames"].values))).fit(cov_type="HC1")
    raw_ci = raw_m.conf_int(alpha=0.05)
    adj_ci = adj_m.conf_int(alpha=0.05)
    activity_reg = {
        "reference": ref,
        "raw": {a: {"ratio": float(np.exp(raw_m.params[a])), "p": float(raw_m.pvalues[a]),
                    "ratio_ci95": [float(np.exp(raw_ci.loc[a, 0])), float(np.exp(raw_ci.loc[a, 1]))]} for a in others},
        "volume_adjusted": {a: {"ratio": float(np.exp(adj_m.params[a])), "p": float(adj_m.pvalues[a]),
                                "ratio_ci95": [float(np.exp(adj_ci.loc[a, 0])), float(np.exp(adj_ci.loc[a, 1]))]} for a in others},
        "log_frames_coef": float(adj_m.params["log_frames"]), "log_frames_p": float(adj_m.pvalues["log_frames"]),
        "r2_adjusted": float(adj_m.rsquared), "n": int(len(reg)),
    }
    print(f"\n=== log(1+activities) by archetype vs. {ref}, raw and volume-adjusted ===\n"
          f"{json.dumps(activity_reg, indent=2)}")

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "min_frames_pooled": MIN_FRAMES,
        "cluster_features": CLUSTER_FEATURES,
        "correlation_matrix": json.loads(corr.round(3).to_json() or "{}"),
        "pca_diagnostic": pca,
        "k": int(k), "stability_by_k": {str(kk): list(v) for kk, v in stabilities.items()},
        "cluster_sizes": cluster_sizes,
        "persistence_year_over_year": persistence,
        "promotional_excess_regression": promo_reg,
        "activity_volume_regression": activity_reg,
    }
    out = L.results_path("posture", "strategy_dimensions_diagnostics.json")
    out.write_text(json.dumps(manifest, indent=2, default=float))

    excess_out = L.results_path("posture", "promotional_excess.parquet")
    pd.DataFrame({"ticker": pooled["ticker"], "promotional_excess": promotional_excess.values}).to_parquet(excess_out, index=False)

    print(f"\n-> {out}")
    print(f"-> {excess_out}")


if __name__ == "__main__":
    main()
