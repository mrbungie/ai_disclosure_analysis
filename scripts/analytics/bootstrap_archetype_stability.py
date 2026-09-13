"""200-replicate dual-bootstrap stability of the k=3 posture archetype
(@fig-archetype-stability, @tbl-a3-kmeans-stability in thesis.qmd).

This did not exist as a script before 2026-09-13: the JSON it now produces
(`bootstrap_jaccard_200_results.json`) was generated once by ad-hoc session
code and only the OUTPUT survived on disk, uncommitted anywhere as code --
exactly the pattern this project's own memory says to stop doing (see
`build_firm_clusters.py`'s original docstring for the same lesson learned
the first time). Any change to the posture feature set (e.g. adding
`hedging_posture`) silently made that cached file stale with no way to
tell short of noticing the numbers look wrong.

Two resampling regimes, both against the same base k-way partition from
`build_strategy_dimensions.fit_pipeline`:

  frame_level  Resample each firm's OWN frames with replacement (same
               count), refit, compare to the base partition. Answers:
               how much does measurement noise in which frames a firm
               happens to have move its assigned archetype?
  firm_level   Resample the cross-section of FIRMS with replacement
               (bootstrap sample of rows, duplicates allowed), refit,
               compare each draw's bootstrap label to that firm's base
               label. Answers: how much does which firms happen to be in
               the sample move the partition?

For each k in (2, 3, 4, 5) and each regime: per-cluster mean and standard
error of the best-matching Jaccard overlap across replicates, the minimum
across clusters (the stability floor check), and the overall mean.

Usage:
    uv run python scripts/analytics/bootstrap_archetype_stability.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_strategy_dimensions import (  # noqa: E402
    DB, SEED, fit_pipeline, jaccard, load_frames,
)
from ai_intensity import firm_intensity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
DOCS_DIR = REPO_ROOT / "docs" / "analytics"
REPLICATES = 200
KS = (2, 3, 4, 5)


def bootstrap_frame_level(frames: pd.DataFrame, universe: pd.DataFrame, k: int,
                          replicates: int, seed: int) -> np.ndarray:
    """(replicates, k) Jaccard matrix: resample each firm's own frames."""
    base_labels, base_index = fit_pipeline(frames, universe, k)
    rng = np.random.default_rng(seed)
    out = np.zeros((replicates, k))
    groups = frames.groupby("ticker").indices
    base_series = pd.Series(base_labels, index=base_index)
    for r in range(replicates):
        positions = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in groups.values()])
        sample = frames.iloc[positions]
        labels, index = fit_pipeline(sample, universe, k)
        aligned = pd.Series(labels, index=index).reindex(base_index)
        valid = aligned.notna()
        bl = base_series[valid].values
        al = aligned[valid].values
        for c in range(k):
            out[r, c] = max(jaccard(bl == c, al == other) for other in range(k))
    return out


def bootstrap_firm_level(frames: pd.DataFrame, universe: pd.DataFrame, k: int,
                         replicates: int, seed: int) -> np.ndarray:
    """(replicates, k) Jaccard matrix: resample which firms are in the sample.

    Each bootstrap draw gets a synthetic per-draw ticker (`AAPL__3`) so a
    firm pulled twice counts as two independent rows in the refit, the way
    a genuine cross-sectional bootstrap should -- `fit_pipeline` groups by
    `ticker`, so without this a repeated firm would silently collapse back
    into one row instead of being resampled."""
    base_labels, base_index = fit_pipeline(frames, universe, k)
    base_map = dict(zip(base_index, base_labels))
    by_ticker = {t: df for t, df in frames.groupby("ticker")}
    n_frames_map = universe.set_index("ticker")["n_frames"].to_dict()
    tickers = np.array(list(base_index))
    rng = np.random.default_rng(seed)
    out = np.zeros((replicates, k))
    for r in range(replicates):
        draw = rng.choice(tickers, size=len(tickers), replace=True)
        parts = []
        synth_tickers = []
        synth_n_frames = []
        for i, t in enumerate(draw):
            synth = f"{t}__{i}"
            sub = by_ticker[t].copy()
            sub["ticker"] = synth
            parts.append(sub)
            synth_tickers.append(synth)
            synth_n_frames.append(n_frames_map.get(t, 0))
        sample_frames = pd.concat(parts, ignore_index=True)
        sample_universe = pd.DataFrame({"ticker": synth_tickers, "n_frames": synth_n_frames})
        labels, index = fit_pipeline(sample_frames, sample_universe, k)
        true_tickers = [ix.rsplit("__", 1)[0] for ix in index]
        bl = np.array([base_map[t] for t in true_tickers])
        al = np.array(labels)
        for c in range(k):
            out[r, c] = max(jaccard(bl == c, al == other) for other in range(k))
    return out


def summarize(mat: np.ndarray) -> dict:
    mean_by_cluster = mat.mean(axis=0)
    se_by_cluster = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    return {
        "mean_by_cluster": mean_by_cluster.tolist(),
        "se_by_cluster": se_by_cluster.tolist(),
        "min": float(mean_by_cluster.min()),
        "overall_mean": float(mean_by_cluster.mean()),
    }


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        frames = load_frames(con)
        universe = firm_intensity(con, ["ticker"])
    finally:
        con.close()
    print(f"{len(frames):,} frames | {universe['ticker'].nunique():,} firms | {REPLICATES} replicates/regime/k")

    results = {"frame_level": {}, "firm_level": {}}
    for k in KS:
        print(f"k={k}: frame-level bootstrap...", flush=True)
        fl = bootstrap_frame_level(frames, universe, k, REPLICATES, SEED)
        results["frame_level"][str(k)] = summarize(fl)
        print(f"  frame-level min={results['frame_level'][str(k)]['min']:.3f} "
              f"overall_mean={results['frame_level'][str(k)]['overall_mean']:.3f}")

        print(f"k={k}: firm-level bootstrap...", flush=True)
        fm = bootstrap_firm_level(frames, universe, k, REPLICATES, SEED)
        results["firm_level"][str(k)] = summarize(fm)
        print(f"  firm-level  min={results['firm_level'][str(k)]['min']:.3f} "
              f"overall_mean={results['firm_level'][str(k)]['overall_mean']:.3f}")

    results["built_at"] = datetime.now(timezone.utc).isoformat()
    results["replicates"] = REPLICATES

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2)
    (OUT_DIR / "bootstrap_jaccard_200_results.json").write_text(payload)
    (DOCS_DIR / "bootstrap_jaccard_200_results.json").write_text(payload)
    print(f"\n-> {OUT_DIR}/bootstrap_jaccard_200_results.json")
    print(f"-> {DOCS_DIR}/bootstrap_jaccard_200_results.json")


if __name__ == "__main__":
    main()
