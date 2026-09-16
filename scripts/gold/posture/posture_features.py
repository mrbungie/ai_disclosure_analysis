"""Shared posture-feature machinery used by both the gold archetype fit
(`scripts/gold/firm/build_posture_archetype_static.py`) and its analytics diagnostics
(`scripts/analytics/posture/strategy_dimensions_diagnostics.py`).

Factored out so the diagnostic report (PCA, bootstrap stability across all
k, correlation matrix -- none of them the fitted model itself) can be
recomputed independently in `scripts/analytics/` from the same silver
frames, without the analytics script re-importing a gold builder's `main()`
or the gold builder persisting stats/regressions it shouldn't (see
docs/tasks/stage_violations_audit.md G-M1). The archetype fit itself (which
of these get called, with which k/seed/features) lives only in
build_posture_archetype_static.py -- this module holds no fitting DECISIONS.
"""
from __future__ import annotations

import os

# Pin BLAS to one thread BEFORE numpy/archetypes ever touch it. Apple's
# Accelerate (and OpenBLAS/MKL) parallelize matrix reduction with a
# non-deterministic summation order, so bootstrap_stability()'s Jaccard
# scores (and AA.fit's internal float64 matmuls) are not reproducible
# run-to-run without this, independent of the fixed SEED below. Every
# importer of this module (the gold fit and its analytics diagnostics)
# gets this for free by importing posture_features before numpy/archetypes
# themselves get imported anywhere in the process.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import polars as pl
from archetypes import AA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402


_BOOT: dict = {}
SEED = 42
MIN_FRAMES = 5
# Archetypal Analysis is non-convex: a single uniform start lands in whichever
# local optimum it reaches first. Every fit uses FurthestSum seeding with
# AA_N_INIT restarts and keeps the lowest-objective solution, which makes the
# archetypes independent of the random seed.
AA_INIT = "furthest_sum"
AA_N_INIT = 25
AA_MAX_ITER = 500

# Communication POSTURE only -- no declared-action concept (deployed, scaling,
# own/third-party capability, outcomes) is in this list on purpose. Those
# live in the disclosed-activity pass and are reserved for validation.
POSTURE = ["promotional_posture", "hedging_posture", "risk_orientation",
          "governance_orientation", "temporal_posture", "ai_positioning", "specificity"]
INTENSITY = "disclosure_intensity"
# Posture only, and intensity is NOT one of them. It used to be, and it made
# volume the dominant axis: the correlation between a firm's frame count and
# its Vocal weight was 0.66, the firms at the Vocal vertex held a median of 339
# frames against 30 at the Defensive one, and the mean weights fell monotonically
# across quartiles of frame count. An archetype is meant to describe HOW a firm
# writes about AI; how much it writes is a separate measure that enters the
# analysis on its own. Removing it takes that correlation to 0.51.
CLUSTER_FEATURES = list(POSTURE)
# What the gold tables carry: the clustering features plus intensity, which is
# still a covariate the analysis uses -- it just no longer shapes the vertices.
OUTPUT_FEATURES = CLUSTER_FEATURES + [INTENSITY]

ARCHETYPE_PRIORITY = [("risk_orientation", "Defensive Disclosers"),
                      ("governance_orientation", "Governance-Led Disclosers"),
                      ("promotional_posture", "Vocal Substantives")]


# Posture is measured on the annual and event filings. Earnings calls and 10-Q
# discuss different content (calls: governance 0.02, risk 0.06 of frames;
# DEF 14A: governance 0.38) and outnumber filings, so frame-pooled rates would
# measure a firm's channel mix instead of its posture.
POSTURE_FORMS = ("10-K", "8-K", "DEF 14A")


def load_frames(forms: tuple[str, ...] | None = POSTURE_FORMS) -> pd.DataFrame:
    """One row per AI frame of the analysis universe from the `forms` channels
    (None = every channel: 10-K, 10-Q, 8-K, DEF 14A, earnings calls), with the
    date the text became public (`available_date`: filing date or call date)
    and its calendar year.

    `is_customer_facing` comes from silver.ai_activities (a frame none of whose
    activities targets customers/developers reads as not customer-facing)."""
    frame_domains = (L.scan("silver.ai_activities")
                     .filter(pl.col("has_activity"))
                     .group_by("text_hash", "frame_id")
                     .agg((pl.col("domain") == "customer_facing").any().alias("is_customer_facing")))
    manifest = L.scan("silver.filing_manifest")
    documents = pl.concat([
        manifest.filter(pl.col("accession_number").is_not_null())
        .select("accession_number", "ticker", "filing_date"),
        manifest.filter(pl.col("accession_number").is_null())
        .select(pl.col("document_id").alias("accession_number"), "ticker", "filing_date"),
        L.scan("silver.filing_manifest_10q").select("accession_number", "ticker", "filing_date"),
    ]).unique(subset=["accession_number", "ticker"])
    return (L.scan("silver.ai_frames")
            .filter(pl.col("has_frame"))
            .join(frame_domains, on=["text_hash", "frame_id"], how="left")
            .join(documents, on="accession_number", how="inner")
            .filter(pl.col("ticker").is_not_null() & pl.col("filing_date").is_not_null())
            .filter(pl.col("form").is_in(list(forms)) if forms else pl.lit(True))
            .sort(["ticker", "accession_number", "text_hash", "frame_id", "item_key", "paragraph_index"],
                  nulls_last=True)
            .select("ticker", "form", pl.col("filing_date").alias("available_date"),
                    pl.col("filing_date").dt.year().cast(pl.Int32).alias("year"),
                    "concepts", "temporal", "specificity", "rhetoric",
                    pl.col("is_customer_facing").fill_null(False))
            .collect()
            .to_pandas())


def frame_indicators(frames: pd.DataFrame) -> pd.DataFrame:
    """The seven posture indicators per frame, none of them a declared-action concept."""
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
    return df


def build_posture(frames: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The seven posture rates per `keys` group (`frames` may already carry
    the indicators, as the bootstrap passes them to avoid recomputing)."""
    df = frames if set(POSTURE) <= set(frames.columns) else frame_indicators(frames)
    out = df.groupby(keys)[POSTURE].mean()
    out["n_posture_frames"] = df.groupby(keys).size()
    return out.reset_index()


def shrink_to_prior(rates: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of each rate toward the corpus mean, in
    proportion to how many frames support it."""
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
    method (see docstring in strategy_dimensions_diagnostics.py)."""
    pca = PCA(n_components=len(feature_names), random_state=SEED).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_ratio": cum.tolist(),
        "loadings_pc1_pc2": {name: [float(pca.components_[0, i]), float(pca.components_[1, i])]
                             for i, name in enumerate(feature_names)},
    }


def name_archetypes(z_profile: pd.DataFrame) -> dict[int, str]:
    """Claims each cluster in a fixed priority order so a near-tie doesn't
    flip names between reruns. A cluster not extreme on any remaining
    dimension is the moderate residual, "Enterprise Communicators"."""
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


def fit_aa(X: np.ndarray, k: int, seed: int = SEED, n_init: int = AA_N_INIT) -> tuple[AA, np.ndarray]:
    """Archetypal Analysis with the project-wide fitting settings; returns the
    fitted model and the (n x k) convex weights."""
    model = AA(n_archetypes=k, init=AA_INIT, n_init=n_init, max_iter=AA_MAX_ITER, random_state=seed)
    return model, model.fit_transform(X)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def fit_pipeline(frames: pd.DataFrame, universe: pd.DataFrame, k: int, n_init: int = AA_N_INIT):
    """Fits Archetypal Analysis and returns each firm's DOMINANT vertex
    (argmax of its convex weights) as a discrete label, needed only so
    bootstrap stability can be scored with Jaccard overlap."""
    posture = build_posture(frames, ["ticker"])
    merged = universe.merge(posture, on="ticker", how="left")
    # Fit input only (never written): same zero-rate fill as the gold fit.
    merged[POSTURE] = merged[POSTURE].fillna(0.0)
    merged["n_frames"] = merged["n_frames"].fillna(0).astype(int)
    active = merged[merged["n_frames"] >= MIN_FRAMES].reset_index(drop=True)
    shrunk = shrink_to_prior(active[POSTURE], active["n_frames"])
    shrunk[INTENSITY] = active["n_frames"].rank(pct=True).values
    X = StandardScaler().fit_transform(shrunk[CLUSTER_FEATURES].values)
    _, W = fit_aa(X, k, n_init=n_init)
    labels = W.argmax(axis=1)
    return labels, active["ticker"]


BOOT_WORKERS = 8
BOOT_N_INIT = 5   # restarts per bootstrap refit; the reported fit keeps AA_N_INIT


def _boot_state(frames: pd.DataFrame, universe: pd.DataFrame, k: int) -> None:
    _BOOT["frames"], _BOOT["universe"], _BOOT["k"] = frames, universe, k
    _BOOT["groups"] = list(frames.groupby("ticker").indices.values())
    _BOOT["base"] = fit_pipeline(frames, universe, k)


def _boot_replicate(r: int) -> np.ndarray:
    """Best-matching Jaccard per base cluster when each firm's frames are resampled."""
    frames, universe, k = _BOOT["frames"], _BOOT["universe"], _BOOT["k"]
    base_labels, base_index = _BOOT["base"]
    rng = np.random.default_rng([SEED, k, r])
    positions = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in _BOOT["groups"]])
    labels, index = fit_pipeline(frames.iloc[positions], universe, k, n_init=BOOT_N_INIT)
    aligned = pd.Series(labels, index=index).reindex(base_index)
    valid = aligned.notna().values
    bl, al = np.asarray(base_labels)[valid], aligned[valid].values
    return np.array([max(jaccard(bl == c, al == other) for other in range(k)) for c in range(k)])


def bootstrap_stability(frames: pd.DataFrame, universe: pd.DataFrame, k: int, replicates: int = 25) -> np.ndarray:
    """Mean Jaccard reproducibility per cluster over `replicates` frame-level
    bootstrap refits, run in parallel with a generator per replicate (so the
    result does not depend on the number of workers). `frames` carries the
    posture indicators, computed once."""
    frames = frames if set(POSTURE) <= set(frames.columns) else frame_indicators(frames)
    with ProcessPoolExecutor(BOOT_WORKERS, initializer=_boot_state, initargs=(frames, universe, k)) as pool:
        rows = list(pool.map(_boot_replicate, range(replicates), chunksize=2))
    return np.vstack(rows).mean(axis=0)
