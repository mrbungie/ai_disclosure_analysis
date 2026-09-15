"""Dual-bootstrap stability of the k=3 posture archetype
(@fig-archetype-stability, @tbl-a3-kmeans-stability in thesis.qmd).

This did not exist as a script before 2026-09-13: the JSON it now produces
(then `bootstrap_jaccard_200_results.json`, now `bootstrap_jaccard_results.json`) was generated once by ad-hoc session
code and only the OUTPUT survived on disk, uncommitted anywhere as code --
exactly the pattern this project's own memory says to stop doing (see
`build_firm_clusters.py`'s original docstring for the same lesson learned
the first time). Any change to the posture feature set (e.g. adding
`hedging_posture`) silently made that cached file stale with no way to
tell short of noticing the numbers look wrong.

Two resampling regimes, both against the same base k-way partition from
`posture_features.fit_pipeline`:

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
error of the best-matching Jaccard overlap across REPLICATES replicates
(BOOT_N_INIT restarts per refit), the
minimum across clusters (the stability floor check), and the overall mean.

Replicates run in parallel (WORKERS processes). Each replicate draws from its
own generator seeded by (SEED, regime, k, replicate), so results do not
depend on the number of workers or the order replicates finish. Frame
indicators are computed once and resampled as rows.

Usage:
    uv run python scripts/analytics/posture/bootstrap_archetype_stability.py
"""
from __future__ import annotations

import os

# Pin BLAS to one thread BEFORE numpy loads: multi-threaded BLAS reduction
# order isn't deterministic run-to-run, which can flip a seeded AA.fit() to
# a different local optimum (see posture_features.py, which this
# script calls into repeatedly).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gold" / "posture"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "common"))
from posture_features import POSTURE, SEED, fit_pipeline, frame_indicators, jaccard, load_frames  # noqa: E402
from ai_intensity import firm_intensity  # noqa: E402
import layers as L  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
REPLICATES = 50
WORKERS = 8
# Restarts per bootstrap refit. The base partition uses the project-wide
# posture_features.AA_N_INIT; replicates only need to land near the same
# optimum, which FurthestSum seeding reaches with a few restarts.
BOOT_N_INIT = 5
REGIMES = ("frame_level", "firm_level")
KS = (2, 3, 4, 5)


_STATE: dict = {}


def _init(frames: pd.DataFrame, universe: pd.DataFrame) -> None:
    _STATE["frames"] = frames
    _STATE["universe"] = universe
    _STATE["groups"] = frames.groupby("ticker").indices
    _STATE["by_ticker"] = {t: df for t, df in frames.groupby("ticker")}
    _STATE["n_frames"] = universe.set_index("ticker")["n_frames"].to_dict()
    _STATE["base"] = {}


def _base(k: int):
    if k not in _STATE["base"]:
        _STATE["base"][k] = fit_pipeline(_STATE["frames"], _STATE["universe"], k)
    return _STATE["base"][k]


def _best_match(bl: np.ndarray, al: np.ndarray, k: int) -> np.ndarray:
    return np.array([max(jaccard(bl == c, al == other) for other in range(k)) for c in range(k)])


def replicate(regime: str, k: int, r: int) -> np.ndarray:
    """Best-matching Jaccard per base cluster for one bootstrap draw."""
    frames, universe = _STATE["frames"], _STATE["universe"]
    base_labels, base_index = _base(k)
    rng = np.random.default_rng([SEED, REGIMES.index(regime), k, r])
    if regime == "frame_level":
        # resample each firm's own frames
        positions = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in _STATE["groups"].values()])
        labels, index = fit_pipeline(frames.iloc[positions], universe, k, n_init=BOOT_N_INIT)
        aligned = pd.Series(labels, index=index).reindex(base_index)
        valid = aligned.notna().values
        return _best_match(np.asarray(base_labels)[valid], aligned[valid].values, k)
    # resample firms; a firm drawn twice becomes two synthetic tickers (`AAPL__3`)
    # so the refit, which groups by ticker, counts it twice
    base_map = dict(zip(base_index, base_labels))
    draw = rng.choice(np.array(list(base_index)), size=len(base_index), replace=True)
    parts, synth = [], []
    for i, t in enumerate(draw):
        sub = _STATE["by_ticker"][t].copy()
        sub["ticker"] = f"{t}__{i}"
        parts.append(sub)
        synth.append((f"{t}__{i}", _STATE["n_frames"].get(t, 0)))
    sample_universe = pd.DataFrame(synth, columns=["ticker", "n_frames"])
    labels, index = fit_pipeline(pd.concat(parts, ignore_index=True), sample_universe, k, n_init=BOOT_N_INIT)
    bl = np.array([base_map[ix.rsplit("__", 1)[0]] for ix in index])
    return _best_match(bl, np.asarray(labels), k)


def _run(task: tuple[str, int, int]) -> tuple[str, int, int, np.ndarray]:
    return (*task, replicate(*task))


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
    frames = frame_indicators(load_frames())[["ticker", *POSTURE]]
    universe = firm_intensity(["ticker"])
    print(f"{len(frames):,} frames | {universe['ticker'].nunique():,} firms | "
          f"{REPLICATES} replicates/regime/k | {WORKERS} workers", flush=True)

    tasks = [(regime, k, r) for k in KS for regime in REGIMES for r in range(REPLICATES)]
    mats = {(regime, k): np.zeros((REPLICATES, k)) for k in KS for regime in REGIMES}
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(frames, universe)) as pool:
        for i, (regime, k, r, row) in enumerate(pool.map(_run, tasks, chunksize=4), start=1):
            mats[(regime, k)][r] = row
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} replicates", flush=True)

    results = {regime: {str(k): summarize(mats[(regime, k)]) for k in KS} for regime in REGIMES}
    for k in KS:
        print(f"k={k}: " + " | ".join(f"{regime} min={results[regime][str(k)]['min']:.3f} "
                                       f"mean={results[regime][str(k)]['overall_mean']:.3f}" for regime in REGIMES))
    results["built_at"] = datetime.now(timezone.utc).isoformat()
    results["replicates"] = REPLICATES
    results["restarts_per_refit"] = BOOT_N_INIT

    out_path = L.results_path("posture", "bootstrap_jaccard_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
