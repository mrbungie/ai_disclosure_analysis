"""One disclosure-posture taxonomy, validated against an independent
activity inventory, instead of two Chapter 4 instruments that never had to
agree with each other.

The old Chapter 4 ran two analyses in parallel: a K-means over 6 raw
features (three of which -- deployment, capability, adoption stage -- are
declared ACTIONS, not communication style) producing three archetypes, and a
separate tercile cut of "voice" against "behaviour" producing a 3x3 grid.
Two problems with that: (1) nothing forced the two instruments to agree, so
a firm could be a Risk Lister with high voice and there was no way to tell
signal from noise; (2) mixing declared-action concepts (deployed, invested
in infrastructure, own vs. third-party capability) into the SAME clustering
space as pure communication-style signals (promotional register, risk
framing) blurred "how a firm talks about AI" with "what it says it does" --
exactly the two things this thesis extracts with two SEPARATE LLM passes
(semantic frames, then disclosed activities) precisely so they can validate
each other instead of being collapsed together.

This script fixes both:

1. **Clustering uses ONLY disclosure posture** -- seven frame-level signals
   about HOW a firm communicates, none of them a declared action: promotional
   register, hedging register, risk framing, governance framing, temporal
   stance (realized vs. hypothetical), customer-facing vs. internal
   positioning, and specificity. Plus disclosure intensity (how much a firm
   says, at all). Nothing about deployment, capability, or third-party
   sourcing enters this step -- those live in the disclosed-activity
   extraction and are reserved for validation, never for defining the
   archetypes.
2. Continuous Archetypal Analysis (@cutler1994) on these 8 standardized
   posture dimensions gives the archetypes: every firm is a convex
   combination of k extreme vertices, and its DOMINANT vertex (argmax of
   its weights) is the discrete `archetype` label stored below. k is
   chosen by resampling each firm's own frames, refitting the whole
   pipeline, and keeping the largest k whose worst-recovered dominant-
   vertex partition still clears 0.60 mean Jaccard overlap. K-means never
   appears in this pipeline -- it exists only as an Appendix C robustness
   benchmark against this fit, computed independently inside thesis.qmd.

This script fits the model and persists it plus its direct output (labels,
posture features). It does NOT persist the PCA diagnostic, the correlation
matrix, the bootstrap-stability-by-k table, or the promotional-excess /
activity-volume regressions against the independent activity inventory --
those are reports about the fit, not the fit or its predictions, and live in
`scripts/analytics/posture/strategy_dimensions_diagnostics.py`
(`data/results/posture/strategy_dimensions_diagnostics.json`), reading gold
(this script's own outputs) and recomputing the same diagnostics from
`posture_features` so the numbers match exactly.

Outputs:
  - `models/posture_archetype_static/model.pkl`: the fitted AA model,
    scaler, feature names and cluster-name mapping.
  - `spines/firm/firm`: one row per firm with at least one scorable filing
    (`delisted`, `delisting_date` from silver.firm_universe).
  - `covariates/firm/posture_archetype_static`: the 8 posture dimensions
    (pooled over the whole panel), dominant archetype and its bootstrap
    stability. A single fit over every year: descriptive, not point in time
    (the firm-year projection of this model is
    `scripts/gold/firm_year/build_posture.py`).

Usage:
    uv run python scripts/gold/firm/build_posture_archetype_static.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "document"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from posture_features import (  # noqa: E402  (pins BLAS threads before numpy/archetypes import)
    CLUSTER_FEATURES, INTENSITY, MIN_FRAMES, POSTURE, SEED, bootstrap_stability, build_posture,
    fit_aa, load_frames, name_archetypes, shrink_to_prior,
)

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import layers as L  # noqa: E402
from ai_intensity import FILING_FORMS, aggregate  # noqa: E402
from build_document import gold_document_table  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "posture_archetype_static" / "model.pkl"
BUILDER = "scripts/gold/firm/build_posture_archetype_static.py"


def firm_intensity(keys: list[str]) -> pd.DataFrame:
    """`ai_intensity.firm_intensity` over the gold document tables: filing
    frames and words per `keys` for every firm-year or firm with filings."""
    docs = gold_document_table()
    docs = docs[docs["form"].isin(FILING_FORMS)].assign(year=lambda d: d["fecha"].dt.year)
    return aggregate(docs, list(keys))[list(keys) + ["n_docs", "n_paragraphs", "n_words", "n_frames", "frames_per_1k", "any_ai"]]


def main() -> None:
    frames = load_frames()
    universe = firm_intensity(["ticker"])
    print(f"{len(frames):,} frames | {universe['ticker'].nunique():,} firms with filings")

    posture = build_posture(frames, ["ticker"])
    merged = universe.merge(posture, on="ticker", how="left")
    # Fit input only: a firm without posture-form frames enters the fit
    # population with zero rates. Nothing filled here is written as a covariate.
    merged[POSTURE] = merged[POSTURE].fillna(0.0)
    merged["n_frames"] = merged["n_frames"].fillna(0).astype(int)
    has_frames = merged["n_frames"] >= MIN_FRAMES
    active = merged[has_frames].reset_index(drop=True)
    print(f"active population (>= {MIN_FRAMES} frames): {len(active):,} firms | "
          f"{int((~has_frames).sum())} firms below the threshold go to 'No AI'")

    shrunk = shrink_to_prior(active[POSTURE], active["n_frames"])
    shrunk[INTENSITY] = active["frames_per_1k"].rank(pct=True).values
    X = StandardScaler().fit_transform(shrunk[CLUSTER_FEATURES].values)

    print("\n=== choosing k by bootstrap stability (25 replicates, floor 0.60) ===")
    stabilities, chosen = {}, None
    for k in (2, 3, 4, 5):
        s = bootstrap_stability(frames, universe, k)
        stabilities[k] = s
        ok = "yes" if s.min() >= 0.60 else "NO"
        print(f"k={k}: " + " ".join(f"{v:.2f}" for v in s) + f" | min {s.min():.2f} -> usable: {ok}")
        if s.min() >= 0.60:
            chosen = k
    k = chosen or 2
    print(f"\nk chosen: {k}")

    aa, W = fit_aa(X, k)
    active = active.reset_index(drop=True)
    for col in CLUSTER_FEATURES:
        active[col] = shrunk[col].values
    active["cluster"] = W.argmax(axis=1)
    z_profile = pd.DataFrame(aa.archetypes_, columns=CLUSTER_FEATURES)
    cluster_names = name_archetypes(z_profile)
    active["archetype"] = active["cluster"].map(cluster_names)
    active["archetype_stability"] = active["cluster"].map(dict(enumerate(stabilities[k])))

    print("\n=== archetype profile (mean posture rate) ===")
    display = active.groupby("archetype")[CLUSTER_FEATURES].mean().round(3)
    display.insert(0, "firms", active.groupby("archetype").size())
    display.insert(1, "stability", active.groupby("archetype")["archetype_stability"].first().round(2))
    print(display.to_string())

    no_ai = merged[~has_frames].copy()
    no_ai[CLUSTER_FEATURES] = 0.0
    no_ai["cluster"], no_ai["archetype"], no_ai["archetype_stability"] = -1, "No AI", 1.0
    pooled = pd.concat([active, no_ai], ignore_index=True, sort=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": aa, "scaler": StandardScaler().fit(active[CLUSTER_FEATURES].values),
        "feature_names": CLUSTER_FEATURES, "cluster_names": cluster_names, "k": k, "seed": SEED,
    }, MODEL_PATH)
    print(f"-> {MODEL_PATH}")

    spine = universe[["ticker"]].assign(id=universe["ticker"]).merge(L.firm_delistings(), on="ticker", how="left")
    L.write_gold("spines", "firm", "firm", spine, builder=BUILDER,
                 extra={"grain": "firm (ticker with at least one scorable filing)"})
    pooled_out = pooled[["ticker", "n_frames"] + CLUSTER_FEATURES + ["cluster", "archetype", "archetype_stability"]]
    L.write_gold("covariates", "firm", "posture_archetype_static", pooled_out.assign(id=pooled_out["ticker"]),
                 builder=BUILDER, inputs=[MODEL_PATH],
                 extra={"point_in_time": False, "use": "descriptive: one Archetypal Analysis fit over the whole panel"})


if __name__ == "__main__":
    main()
