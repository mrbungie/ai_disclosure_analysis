"""
Script 13: Exploratory Factor Analysis on BoW features.

Empirically validates the hand-constructed D1-D9 dimensional constructs
used in script 11 by identifying latent dimensions in the BoW feature space.

Usage:
    uv run python scripts/13_factor_analysis.py [--n-factors N]
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep boolean/numeric columns with 1% < prevalence < 99%."""
    # Drop identifier / text columns
    drop_patterns = ["chunk_id", "_text", "_id", "ticker", "filing_date",
                     "section_name", "cik", "accession", "year"]
    cols_to_drop = [c for c in df.columns
                    if any(p in c for p in drop_patterns)]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Cast all remaining columns to float
    df = df.astype(float)

    # Drop near-zero-variance: keep only 1% < mean < 99%
    means = df.mean()
    mask = (means > 0.01) & (means < 0.99)
    kept = df.loc[:, mask]

    n_dropped = len(df.columns) - len(kept.columns)
    pipeline_logger.log_event(
        pipeline_step="factor_analysis",
        level="INFO",
        message=(
            f"Feature selection: kept {len(kept.columns)} / {len(df.columns)} "
            f"columns ({n_dropped} dropped by prevalence filter)."
        ),
    )
    print(f"  Features kept after prevalence filter: {len(kept.columns)}")
    return kept.dropna(axis=1)


# ---------------------------------------------------------------------------
# Suitability tests
# ---------------------------------------------------------------------------

def run_suitability_tests(X: np.ndarray) -> None:
    print("\n--- Suitability Tests ---")
    chi2, p = calculate_bartlett_sphericity(X)
    print(f"  Bartlett's test: chi2={chi2:.2f}, p={p:.4e}")

    _, kmo_model = calculate_kmo(X)
    print(f"  KMO (overall): {kmo_model:.4f}")
    pipeline_logger.log_event(
        pipeline_step="factor_analysis",
        level="INFO",
        message=f"Bartlett chi2={chi2:.2f} p={p:.2e}; KMO={kmo_model:.4f}",
    )


# ---------------------------------------------------------------------------
# Parallel analysis (Horn's method)
# ---------------------------------------------------------------------------

def parallel_analysis(X: np.ndarray, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (actual_eigenvalues, mean_random_eigenvalues)."""
    n, p = X.shape
    # Actual eigenvalues from correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    actual_eigs = np.linalg.eigvalsh(corr)[::-1]

    # Random eigenvalues
    rand_eigs = []
    for _ in range(n_iter):
        rand_data = np.random.normal(size=(n, p))
        rand_corr = np.corrcoef(rand_data, rowvar=False)
        eigs = np.linalg.eigvalsh(rand_corr)[::-1]
        rand_eigs.append(eigs)

    mean_rand_eigs = np.mean(rand_eigs, axis=0)
    return actual_eigs, mean_rand_eigs


def determine_n_factors(
    actual_eigs: np.ndarray,
    mean_rand_eigs: np.ndarray,
    cli_override: int | None,
) -> int:
    if cli_override is not None:
        n = int(cli_override)
        print(f"\n  --n-factors override: using n_factors={n}")
        return n

    # Count factors where actual > random (parallel analysis criterion)
    n = int(np.sum(actual_eigs > mean_rand_eigs))
    n = max(2, min(12, n))
    print(f"\n  Parallel analysis suggests n_factors={n} (clamped to [2, 12])")
    pipeline_logger.log_event(
        pipeline_step="factor_analysis",
        level="INFO",
        message=f"Parallel analysis: n_factors={n}",
    )
    return n


# ---------------------------------------------------------------------------
# Factor analysis
# ---------------------------------------------------------------------------

def fit_factor_model(X: np.ndarray, n_factors: int) -> FactorAnalyzer:
    print(f"\n--- Fitting FactorAnalyzer (n_factors={n_factors}, varimax, ml) ---")
    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="ml")
    fa.fit(X)
    return fa


def print_variance_table(fa: FactorAnalyzer, n_factors: int) -> None:
    ev, cv, prop = fa.get_factor_variance()
    print("\n  Variance Explained:")
    header = f"  {'Factor':<10} {'SS Loadings':>12} {'Prop Var':>10} {'Cum Var':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(n_factors):
        print(f"  F{i+1:<9} {ev[i]:>12.4f} {prop[i]:>10.4f} {cv[i]:>10.4f}")


def print_top_features(
    loadings: np.ndarray,
    feature_names: list,
    n_factors: int,
    threshold: float = 0.35,
) -> None:
    print(f"\n  Top features per factor (|loading| > {threshold}):")
    for j in range(n_factors):
        col = loadings[:, j]
        idx = np.where(np.abs(col) > threshold)[0]
        idx_sorted = idx[np.argsort(np.abs(col[idx]))[::-1]]
        if len(idx_sorted) == 0:
            print(f"  F{j+1}: (no features above threshold)")
            continue
        items = ", ".join(
            f"{feature_names[i]}({col[i]:+.2f})" for i in idx_sorted
        )
        print(f"  F{j+1}: {items}")


# ---------------------------------------------------------------------------
# Outputs: loadings CSV
# ---------------------------------------------------------------------------

def save_loadings(
    loadings: np.ndarray,
    feature_names: list,
    n_factors: int,
    out_path: Path,
) -> pd.DataFrame:
    factor_cols = [f"F{i+1}" for i in range(n_factors)]
    df = pd.DataFrame(loadings, index=feature_names, columns=factor_cols)
    # Communalities = sum of squared loadings per row
    df["communality"] = (df[factor_cols] ** 2).sum(axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"\n  Loadings saved → {out_path}")
    return df


# ---------------------------------------------------------------------------
# Outputs: chunk & firm-year factor scores
# ---------------------------------------------------------------------------

def save_factor_scores(
    fa: FactorAnalyzer,
    X: np.ndarray,
    meta_df: pd.DataFrame,
    n_factors: int,
    chunk_out: Path,
    firm_year_out: Path,
) -> None:
    scores = fa.transform(X)
    factor_cols = [f"F{i+1}" for i in range(n_factors)]
    scores_df = pd.DataFrame(scores, columns=factor_cols, index=meta_df.index)

    # Attach metadata
    for col in ["ticker", "year", "section_name"]:
        if col in meta_df.columns:
            scores_df[col] = meta_df[col].values

    chunk_out.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(chunk_out, index=False)
    print(f"  Chunk factor scores saved → {chunk_out}")

    # Firm-year aggregation
    group_cols = [c for c in ["ticker", "year"] if c in scores_df.columns]
    if group_cols:
        firm_year_df = scores_df.groupby(group_cols)[factor_cols].mean().reset_index()
        firm_year_out.parent.mkdir(parents=True, exist_ok=True)
        firm_year_df.to_parquet(firm_year_out, index=False)
        print(f"  Firm-year factor scores saved → {firm_year_out}")
    else:
        print("  Warning: could not aggregate to firm-year (missing ticker/year columns).")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scree(
    actual_eigs: np.ndarray,
    mean_rand_eigs: np.ndarray,
    n_factors: int,
    out_path: Path,
) -> None:
    n_show = min(20, len(actual_eigs))
    x = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, actual_eigs[:n_show], "o-", label="Actual eigenvalues", color="#2171b5")
    ax.plot(
        x, mean_rand_eigs[:n_show], "s--",
        label="Parallel analysis (mean random)", color="#e6550d", alpha=0.8,
    )
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.2, label="Kaiser criterion (λ=1)")
    ax.axvline(x=n_factors, color="#238b45", linestyle="--", linewidth=1.2,
               label=f"Selected n_factors={n_factors}")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot with Parallel Analysis")
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Scree plot saved → {out_path}")


def plot_loadings_heatmap(
    loadings_df: pd.DataFrame,
    n_factors: int,
    out_path: Path,
    threshold: float = 0.3,
) -> None:
    factor_cols = [f"F{i+1}" for i in range(n_factors)]
    # Keep rows with at least one absolute loading > threshold
    mask = (loadings_df[factor_cols].abs() > threshold).any(axis=1)
    plot_df = loadings_df.loc[mask, factor_cols]

    if plot_df.empty:
        print("  Warning: no features above heatmap threshold; skipping heatmap.")
        return

    height = max(6, len(plot_df) * 0.28)
    width = max(5, n_factors * 0.9)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        plot_df,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        annot=(len(plot_df) <= 40),
        fmt=".2f",
        annot_kws={"size": 7},
    )
    ax.set_title("Factor Loadings Heatmap (|loading| > 0.3)")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loadings heatmap saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory Factor Analysis on BoW disclosure features."
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of factors (default: determined via parallel analysis).",
    )
    args = parser.parse_args()

    config = load_config()
    candidate_chunks_dir = Path(config["paths"]["candidate_chunks"])
    bow_path = candidate_chunks_dir / "ai_disclosure_bow_features.parquet"
    scored_path = candidate_chunks_dir / "ai_scored_chunks.parquet"

    factors_dir = Path("data/processed/factors")
    reports_dir = Path("reports")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pipeline_logger.log_event(
        pipeline_step="factor_analysis",
        level="INFO",
        message="Loading BoW features and scored chunks metadata...",
    )
    print("Loading BoW features...")

    if not bow_path.exists():
        print(f"ERROR: BoW features file not found: {bow_path}")
        pipeline_logger.log_event(
            pipeline_step="factor_analysis",
            level="ERROR",
            message=f"BoW features file not found: {bow_path}",
        )
        return

    bow_df = pd.read_parquet(bow_path)
    print(f"  BoW features shape: {bow_df.shape}")

    # Load metadata for factor score enrichment
    meta_cols = ["ticker", "filing_date", "section_name"]
    if scored_path.exists():
        scored_df = pd.read_parquet(scored_path, columns=meta_cols)
        # Align on index
        scored_df = scored_df.reindex(bow_df.index)
        if "filing_date" in scored_df.columns:
            scored_df["year"] = pd.to_datetime(
                scored_df["filing_date"], errors="coerce"
            ).dt.year
    else:
        print(f"  Warning: scored chunks not found at {scored_path}. Metadata will be empty.")
        scored_df = pd.DataFrame(index=bow_df.index)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    print("\nSelecting features...")
    feat_df = select_features(bow_df)
    feature_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    print(f"  Final feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # Suitability tests
    # ------------------------------------------------------------------
    run_suitability_tests(X)

    # ------------------------------------------------------------------
    # Parallel analysis → n_factors
    # ------------------------------------------------------------------
    print("\nRunning parallel analysis (100 iterations)...")
    actual_eigs, mean_rand_eigs = parallel_analysis(X, n_iter=100)
    n_factors = determine_n_factors(actual_eigs, mean_rand_eigs, args.n_factors)

    # ------------------------------------------------------------------
    # Fit factor model
    # ------------------------------------------------------------------
    fa = fit_factor_model(X, n_factors)
    print_variance_table(fa, n_factors)
    loadings = fa.loadings_
    assert loadings is not None, "FactorAnalyzer.fit() did not produce loadings"
    print_top_features(loadings, feature_names, n_factors)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    pipeline_logger.log_event(
        pipeline_step="factor_analysis",
        level="INFO",
        message=f"Saving factor outputs (n_factors={n_factors})...",
    )

    loadings_df = save_loadings(
        loadings, feature_names, n_factors,
        factors_dir / "factor_loadings.csv",
    )

    save_factor_scores(
        fa, X, scored_df, n_factors,
        factors_dir / "chunk_factor_scores.parquet",
        factors_dir / "firm_year_factor_scores.parquet",
    )

    plot_scree(
        actual_eigs, mean_rand_eigs, n_factors,
        reports_dir / "factor_scree.png",
    )

    plot_loadings_heatmap(
        loadings_df, n_factors,
        reports_dir / "factor_loadings_heatmap.png",
    )

    pipeline_logger.log_event(
        pipeline_step="factor_analysis",
        level="SUCCESS",
        message=(
            f"Factor analysis complete. n_factors={n_factors}. "
            f"Outputs written to {factors_dir} and {reports_dir}."
        ),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
