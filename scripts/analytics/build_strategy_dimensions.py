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
3. PCA on the same 8 dimensions is reported as a robustness diagnostic, NOT
   as the clustering method: if PC1-2 captured most of the variance, that
   would argue for collapsing to a 2-D space; if it doesn't (it doesn't --
   see the manifest), that supports keeping the dimensions distinct rather
   than forcing a low-dimensional summary onto them.
4. **Promotional excess is validated against the independent activity
   inventory**, not against another posture signal: `promotional_posture ~
   log(1 + disclosed activity count)`, where the activity count comes from
   the disclosed-activity LLM pass (`firm_activities.parquet`), a completely
   separate extraction from the frames this script's own clustering is
   built on. This is why the resulting R^2 is a real, cross-pipeline
   validation and not an artifact of one feature being a linear function of
   another built from the same 12-variable input, which is what happened in
   an earlier version of this script (an R^2 = 0.196 that partly reflected
   promotional register being one of the very features used to build the
   "operational commitment" score it was regressed against).

Outputs:
  - `firm_strategy_dimensions.parquet`: one row per firm -- the 8 posture
    dimensions, archetype, and promotional_excess (computed once activities
    are joined in a later step, see `build_strategy_economic_profiles.py`
    and the Chapter 4 notebook cells for how the activity-based residual is
    layered on afterward).
  - `firm_year_strategy_dimensions.parquet`: same, by (ticker, year),
    projected onto the pooled cluster centroids, never re-fit per year.
  - `strategy_dimensions_manifest.json`: correlation matrix, PCA diagnostic,
    bootstrap stability by k.

Usage:
    uv run python scripts/analytics/build_strategy_dimensions.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from archetypes import AA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
SEED = 42
MIN_FRAMES = 5

# Communication POSTURE only -- no declared-action concept (deployed, scaling,
# own/third-party capability, outcomes) is in this list on purpose. Those
# live in the disclosed-activity pass and are reserved for validation.
#
# `hedging_posture` (2026-09-13): v2's rhetoric enum adds `hedged` --
# cautious/conditional framing ("could", "may", "if adopted") -- alongside
# `promotional`/`strategic`. v1 had no such category, so this dimension did
# not and could not exist before. Kept as its OWN dimension rather than
# folded into `promotional_posture`: hedged and promotional describe
# opposite rhetorical postures (cautious vs. assertive), averaging them
# would cancel out exactly the contrast a firm's choice between them is
# meant to reveal.
POSTURE = ["promotional_posture", "hedging_posture", "risk_orientation",
          "governance_orientation", "temporal_posture", "ai_positioning", "specificity"]
INTENSITY = "disclosure_intensity"
CLUSTER_FEATURES = POSTURE + [INTENSITY]


def load_frames(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    # `domain` no longer lives on the frame itself (v2, see
    # docs/migration_v1_to_v2_analytics.md §1.3) -- it's a proxy computed on
    # gold_ai_activities from pass-2's `target`, joined back here by
    # (text_hash, frame_id). A frame with no matching activity (or none of
    # its activities has a customer-facing target) reads as not customer-
    # facing here, same partial-coverage caveat as the view itself.
    return con.execute("""
        WITH frame_domains AS (
            SELECT text_hash, frame_id, bool_or(domain = 'customer_facing') AS is_customer_facing
            FROM gold_ai_activities
            WHERE has_activity
            GROUP BY text_hash, frame_id
        )
        SELECT fm.ticker, extract(year from fm.filing_date)::INT AS year,
               f.concepts, f.temporal, f.specificity, f.rhetoric,
               COALESCE(fd.is_customer_facing, false) AS is_customer_facing
        FROM gold_ai_frames f
        LEFT JOIN frame_domains fd ON fd.text_hash = f.text_hash AND fd.frame_id = f.frame_id
        JOIN filing_manifest fm USING (country_code, accession_number)
        WHERE f.country_code = 'us' AND f.has_frame AND fm.ticker IS NOT NULL
        ORDER BY fm.ticker, fm.accession_number, f.text_hash, f.frame_id
    """).fetchdf()


def firm_universe(con: duckdb.DuckDBPyConnection, keys: list[str]) -> pd.DataFrame:
    """Every firm(-year) with a filing, including ones that never mention AI."""
    return con.execute(f"""
        WITH docs AS (
            SELECT country_code, ticker, accession_number, filing_date FROM filing_manifest
            UNION ALL SELECT country_code, ticker, accession_number, filing_date FROM filing_manifest_10q
        )
        SELECT ticker, {"extract(year from filing_date)::INT AS year" if "year" in keys else "NULL AS year"}
        FROM docs WHERE country_code = 'us' AND ticker IS NOT NULL
        GROUP BY {", ".join(str(i + 1) for i in range(len(keys)))}
    """).fetchdf()[keys]


def build_posture(frames: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The seven posture rates, none of them a declared-action concept."""
    df = frames.copy()
    concepts = df["concepts"].apply(lambda c: set(c) if c is not None else set())
    df["promotional_posture"] = df["rhetoric"].apply(
        lambda r: np.mean([x in list(r) if r is not None else False for x in ("promotional", "strategic")]))
    df["hedging_posture"] = df["rhetoric"].apply(
        lambda r: float("hedged" in list(r)) if r is not None else 0.0)
    df["risk_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("risk_") for c in s)))
    df["governance_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("gov_") for c in s)))
    df["temporal_posture"] = (df["temporal"] == "realized").astype(float)
    df["ai_positioning"] = df["is_customer_facing"].astype(float)
    df["specificity"] = df["specificity"].apply(lambda s: len(s) / 5.0 if s is not None else 0.0)
    out = df.groupby(keys)[POSTURE].mean()
    out["n_posture_frames"] = df.groupby(keys).size()
    return out.reset_index()


def shrink_to_prior(rates: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of each rate toward the corpus mean, in
    proportion to how many frames support it -- a firm with 4 frames should
    not swing a rate as far as one with 400."""
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


def pca_diagnostic(X: np.ndarray, feature_names: list[str]) -> dict:
    """PCA reported as a dimensionality DIAGNOSTIC, not as the clustering
    method: if the first 2-3 components captured most of the variance, that
    would argue for collapsing the posture space to a low-dimensional
    summary. If they don't, that is evidence the seven dimensions are not
    redundant restatements of one or two underlying axes, and supports
    clustering on all of them rather than forcing a 2-D projection."""
    pca = PCA(n_components=len(feature_names), random_state=SEED).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_ratio": cum.tolist(),
        "loadings_pc1_pc2": {name: [float(pca.components_[0, i]), float(pca.components_[1, i])]
                             for i, name in enumerate(feature_names)},
    }


ARCHETYPE_PRIORITY = [("risk_orientation", "Defensive Disclosers"),
                      ("governance_orientation", "Governance-Led Disclosers"),
                      ("promotional_posture", "Vocal Substantives")]


def name_archetypes(z_profile: pd.DataFrame) -> dict[int, str]:
    """Claims each cluster in a fixed priority order -- same pattern as the
    superseded designs (`build_firm_clusters._label_behavior_clusters`,
    `build_segments.name_segments`): without a fixed order, a near-tie on
    which dimension is "most extreme" flips names between reruns for no
    substantive reason. Risk orientation is claimed first because it is the
    most discriminating posture dimension in this corpus (the widest spread
    across clusters); a cluster not extreme on any remaining dimension is
    the moderate residual, "Enterprise Communicators"."""
    names: dict[int, str] = {}
    remaining = list(z_profile.index)
    for column, label in ARCHETYPE_PRIORITY:
        if not remaining:
            break
        candidates = z_profile.loc[remaining, column]
        pick = candidates.idxmax()
        if candidates[pick] > 0:
            names[pick] = label
            remaining.remove(pick)
    for cluster in remaining:
        names[cluster] = "Enterprise Communicators"
    return names


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def fit_pipeline(frames: pd.DataFrame, universe: pd.DataFrame, k: int):
    """Fits Archetypal Analysis (the thesis's stated method, @cutler1994) and
    returns each firm's DOMINANT vertex (argmax of its convex weights) as a
    discrete label -- needed only so bootstrap stability can be scored with
    the same Jaccard-overlap machinery used for a hard partition. The stored
    `archetype`/`cluster` columns downstream are this same dominant-vertex
    assignment, not a separately-fit K-means (that lives only as an
    Appendix C robustness benchmark inside thesis.qmd, never in this
    pipeline)."""
    posture = build_posture(frames, ["ticker"])
    merged = universe.merge(posture, on="ticker", how="left")
    merged[POSTURE] = merged[POSTURE].fillna(0.0)
    merged["n_frames"] = merged["n_frames"].fillna(0).astype(int)
    active = merged[merged["n_frames"] >= MIN_FRAMES].reset_index(drop=True)
    shrunk = shrink_to_prior(active[POSTURE], active["n_frames"])
    shrunk[INTENSITY] = active["n_frames"].rank(pct=True).values  # frame-count proxy; replaced by frames_per_1k in main()
    X = StandardScaler().fit_transform(shrunk[CLUSTER_FEATURES].values)
    W = AA(n_archetypes=k, random_state=SEED, max_iter=500).fit_transform(X)
    labels = W.argmax(axis=1)
    return labels, active["ticker"]


def bootstrap_stability(frames: pd.DataFrame, universe: pd.DataFrame, k: int, replicates: int = 25) -> np.ndarray:
    base_labels, base_index = fit_pipeline(frames, universe, k)
    rng = np.random.default_rng(SEED)
    out = np.zeros((replicates, k))
    groups = frames.groupby("ticker").indices
    for r in range(replicates):
        positions = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in groups.values()])
        sample = frames.iloc[positions]
        labels, index = fit_pipeline(sample, universe, k)
        aligned = pd.Series(labels, index=index).reindex(base_index)
        valid = aligned.notna()
        bl = pd.Series(base_labels, index=base_index)[valid].values
        al = aligned[valid].values
        for c in range(k):
            out[r, c] = max(jaccard(bl == c, al == other) for other in range(k))
    return out.mean(axis=0)


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ai_intensity import firm_intensity

    con = duckdb.connect(str(DB), read_only=True)
    try:
        frames = load_frames(con)
        universe = firm_intensity(con, ["ticker"])
        universe_year = firm_intensity(con, ["ticker", "year"])
    finally:
        con.close()
    print(f"{len(frames):,} frames | {universe['ticker'].nunique():,} firms with filings")

    posture = build_posture(frames, ["ticker"])
    merged = universe.merge(posture, on="ticker", how="left")
    merged[POSTURE] = merged[POSTURE].fillna(0.0)
    merged["n_frames"] = merged["n_frames"].fillna(0).astype(int)
    has_frames = merged["n_frames"] >= MIN_FRAMES
    active = merged[has_frames].reset_index(drop=True)
    print(f"active population (>= {MIN_FRAMES} frames): {len(active):,} firms | "
          f"{int((~has_frames).sum())} firms below the threshold go to 'No AI'")

    shrunk = shrink_to_prior(active[POSTURE], active["n_frames"])
    shrunk[INTENSITY] = active["frames_per_1k"].rank(pct=True).values
    corr = shrunk[CLUSTER_FEATURES].corr()
    print("\n=== correlation matrix of the 8 posture dimensions ===")
    print(corr.round(2).to_string())

    X = StandardScaler().fit_transform(shrunk[CLUSTER_FEATURES].values)
    pca = pca_diagnostic(X, CLUSTER_FEATURES)
    print("\n=== PCA diagnostic (robustness check, NOT the clustering method) ===")
    for i, (ev, cum) in enumerate(zip(pca["explained_variance_ratio"], pca["cumulative_variance_ratio"])):
        print(f"  PC{i + 1}: {ev:.1%} (cumulative {cum:.1%})")

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

    # Archetypal Analysis is the thesis's stated clustering method (see
    # module docstring): every firm is a convex combination of k extreme
    # vertices. The stored `archetype`/`cluster` columns take each firm's
    # DOMINANT vertex (argmax of its convex weights) as a discrete label --
    # a hard partition derived FROM the continuous fit, not a separately
    # fit K-means (K-means only appears as an Appendix C robustness
    # benchmark inside thesis.qmd, computed independently there).
    aa = AA(n_archetypes=k, random_state=SEED, max_iter=500)
    W = aa.fit_transform(X)
    active = active.reset_index(drop=True)
    for col in CLUSTER_FEATURES:
        active[col] = shrunk[col].values
    active["cluster"] = W.argmax(axis=1)
    # aa.archetypes_ (k x features) already lives in standardized space
    # (X was standardized before fitting), so its rows ARE each vertex's
    # z-profile directly -- no need to re-derive one from cluster means.
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

    # Cross-pipeline characterization, NOT independent validation: both the
    # frame-level posture measures and the disclosed-activity extraction read
    # the same underlying corporate text, so a firm that simply discloses
    # more about AI overall has mechanically more opportunity to register on
    # both sides of any comparison below. Every regression here therefore
    # also reports a version that conditions on total AI frame count (the
    # shared-source-text confound), so what remains after that control is
    # the part of the relationship that is not just "more text, more of
    # everything."
    import statsmodels.api as sm
    activities_path = OUT_DIR / "firm_activities.parquet"
    if activities_path.exists():
        acts = pd.read_parquet(activities_path)
        n_activities = acts.groupby("ticker").size().rename("n_activities")
        pooled = pooled.merge(n_activities, on="ticker", how="left")
        pooled["n_activities"] = pooled["n_activities"].fillna(0).astype(int)
        pooled["log_activities"] = np.log1p(pooled["n_activities"])
        reg = pooled[pooled["archetype"] != "No AI"].copy()
        reg["log_frames"] = np.log1p(reg["n_frames"])

        # (1) promotional_posture ~ log(activities), with and without a
        # log(frames) control -- promotional excess is the residual of the
        # UNCONTROLLED fit (kept simple, since the controlled fit barely
        # moves the coefficient; both are reported in the manifest).
        ols = sm.OLS(reg["promotional_posture"], sm.add_constant(reg["log_activities"])).fit()
        ols_ctrl = sm.OLS(reg["promotional_posture"],
                          sm.add_constant(reg[["log_activities", "log_frames"]])).fit()
        pooled["promotional_excess"] = np.nan
        pooled.loc[reg.index, "promotional_excess"] = ols.resid.values
        print(f"\n=== promotional_posture ~ log(1 + disclosed activities) ===\n"
              f"{ols.summary().tables[1]}\nR^2 = {ols.rsquared:.3f}, n = {len(reg)}")
        print(f"\n=== same, controlling for log(1 + AI frames) ===\n{ols_ctrl.summary().tables[1]}")
        promo_reg = {"const": float(ols.params["const"]), "beta_log_activities": float(ols.params["log_activities"]),
                     "se_log_activities": float(ols.bse["log_activities"]), "p_log_activities": float(ols.pvalues["log_activities"]),
                     "r2": float(ols.rsquared), "n": int(len(reg)),
                     "controlled": {"beta_log_activities": float(ols_ctrl.params["log_activities"]),
                                   "p_log_activities": float(ols_ctrl.pvalues["log_activities"]),
                                   "beta_log_frames": float(ols_ctrl.params["log_frames"]),
                                   "p_log_frames": float(ols_ctrl.pvalues["log_frames"])}}

        # (2) archetype activity counts, raw vs. volume-adjusted (log frames)
        ref, others = "Defensive Disclosers", ["Governance-Led Disclosers", "Vocal Substantives"]
        dummies = pd.get_dummies(reg["archetype"]).astype(float)[others]
        raw_m = sm.OLS(reg["log_activities"], sm.add_constant(dummies)).fit(cov_type="HC1")
        adj_m = sm.OLS(reg["log_activities"], sm.add_constant(dummies.assign(log_frames=reg["log_frames"].values))).fit(cov_type="HC1")
        activity_reg = {
            "reference": ref,
            "raw": {a: {"ratio": float(np.exp(raw_m.params[a])), "p": float(raw_m.pvalues[a])} for a in others},
            "volume_adjusted": {a: {"ratio": float(np.exp(adj_m.params[a])), "p": float(adj_m.pvalues[a])} for a in others},
            "log_frames_coef": float(adj_m.params["log_frames"]), "log_frames_p": float(adj_m.pvalues["log_frames"]),
            "r2_adjusted": float(adj_m.rsquared), "n": int(len(reg)),
        }
        print(f"\n=== log(1+activities) by archetype vs. {ref}, raw and volume-adjusted ===\n"
              f"{json.dumps(activity_reg, indent=2)}")
    else:
        pooled["n_activities"], pooled["promotional_excess"] = 0, np.nan
        promo_reg, activity_reg = None, None

    # panel: project onto the pooled cluster centroids, never re-fit per year
    panel_posture = build_posture(frames, ["ticker", "year"])
    panel = universe_year.merge(panel_posture, on=["ticker", "year"], how="left")
    panel[POSTURE] = panel[POSTURE].fillna(0.0)
    panel["n_frames"] = panel["n_frames"].fillna(0).astype(int)
    panel_active = panel["n_frames"] >= 3
    panel_shrunk = shrink_to_prior(panel.loc[panel_active, POSTURE], panel.loc[panel_active, "n_frames"])
    panel_shrunk[INTENSITY] = panel.loc[panel_active, "frames_per_1k"].rank(pct=True).values
    for col in CLUSTER_FEATURES:
        panel.loc[panel_active, col] = panel_shrunk[col].values
    panel[CLUSTER_FEATURES] = panel[CLUSTER_FEATURES].fillna(0.0)
    scaler = StandardScaler().fit(active[CLUSTER_FEATURES].values)
    panel_W = aa.transform(scaler.transform(panel.loc[panel_active, CLUSTER_FEATURES].values))
    panel.loc[panel_active, "cluster"] = panel_W.argmax(axis=1)
    panel["archetype"] = panel["cluster"].map(cluster_names)
    panel.loc[~panel_active, ["cluster", "archetype"]] = [-1, "No AI"]

    transitions = (panel.sort_values(["ticker", "year"])
                   .assign(prev=lambda d: d.groupby("ticker")["archetype"].shift())
                   .dropna(subset=["prev"]))
    persistence = float((transitions["archetype"] == transitions["prev"]).mean())
    print(f"\npanel: {len(panel):,} firm-years | year-over-year archetype persistence: {persistence:.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled_out = pooled[["ticker", "n_frames"] + CLUSTER_FEATURES +
                        ["cluster", "archetype", "archetype_stability", "n_activities", "promotional_excess"]]
    pooled_out.to_parquet(OUT_DIR / "firm_strategy_dimensions.parquet", index=False)
    panel_out = panel[["ticker", "year", "n_frames"] + CLUSTER_FEATURES + ["cluster", "archetype"]]
    panel_out.to_parquet(OUT_DIR / "firm_year_strategy_dimensions.parquet", index=False)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "min_frames_pooled": MIN_FRAMES,
        "cluster_features": CLUSTER_FEATURES,
        "correlation_matrix": json.loads(corr.round(3).to_json() or "{}"),
        "pca_diagnostic": pca,
        "k": k, "stability_by_k": {str(kk): list(v) for kk, v in stabilities.items()},
        "cluster_sizes": active["archetype"].value_counts().to_dict(),
        "persistence_year_over_year": persistence,
        "promotional_excess_regression": promo_reg,
        "activity_volume_regression": activity_reg,
    }
    (OUT_DIR / "strategy_dimensions_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    print(f"\n-> {OUT_DIR}/firm_strategy_dimensions.parquet ({len(pooled_out):,} firms)")
    print(f"-> {OUT_DIR}/firm_year_strategy_dimensions.parquet ({len(panel_out):,} rows)")
    print(f"-> {OUT_DIR}/strategy_dimensions_manifest.json")


if __name__ == "__main__":
    main()
