"""Out-of-sample temporal validation of the k=3 posture archetype
(thesis.qmd, Appendix C, "Out-of-sample temporal validation", ~4260-4322):
fit k-means (k=3) on a training window of firm-years, assign a held-out
year's firm-years to the FROZEN training centroids, and compare against an
unconstrained k-means refit on the held-out year alone (ARI + per-archetype
Jaccard recovery). Answers whether the taxonomy is an artifact of which
years happened to be in the fitting sample, not a re-derivation of the
archetype itself (that is Archetypal Analysis; this is a k-means
cross-check, exactly as the qmd chunk does it).

Reproduces the qmd chunk's own `_kmeans`/`_ari`/`_oos` verbatim (a
self-contained NumPy k-means, not sklearn's, and a from-scratch pairwise
Adjusted Rand Index) so the numbers can be reproduced from the exact same
algorithm, not merely an equivalent one. Reads the gold firm-year posture
covariate (`covariates/firm_year/posture.parquet`) joined with the static
archetype label (`covariates/firm_year/posture_archetype_static.parquet`).

Output: data/results/posture/oos_validation.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
import layers as L  # noqa: E402
from posture_features import CLUSTER_FEATURES  # noqa: E402

FE = list(CLUSTER_FEATURES)


def _kmeans(X, k=3, seed=42, n_init=10, iters=300):
    best = None
    for s in range(n_init):
        rng = np.random.default_rng(seed + s)
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(C, C2):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


def _ari(a, b):
    ct = pd.crosstab(a, b).values
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    n = ct.sum(); sij = comb(ct).sum(); si = comb(ct.sum(1)).sum(); sj = comb(ct.sum(0)).sum()
    e = si * sj / comb(n)
    return (sij - e) / ((si + sj) / 2 - e)


def _oos(fy: pd.DataFrame, train_years: list[int], test_year: int = 2025):
    tr = fy[fy["year"].isin(train_years)]
    te = fy[fy["year"] == test_year]
    mu, sd = tr[FE].mean(), tr[FE].std()
    Xtr = ((tr[FE] - mu) / sd).values
    Xte = ((te[FE] - mu) / sd).values
    C, _ = _kmeans(Xtr)
    frozen = np.argmin(((Xte[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    _, refit = _kmeans(Xte)
    lab_tr = np.argmin(((Xtr[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    names = {}
    for j in range(3):
        names[j] = tr["archetype"].values[lab_tr == j].tolist()
        names[j] = max(set(names[j]), key=names[j].count) if names[j] else f"cluster {j}"
    jac = {}
    for j in range(3):
        a = frozen == j
        jac[names[j]] = max(((a & (refit == r)).sum() / max((a | (refit == r)).sum(), 1)) for r in range(3)) * 100
    return _ari(frozen, refit), jac, len(te)


def main() -> None:
    fy = L.read_dataset("firm_year", ("covariates", "posture", FE), ("covariates", "posture_archetype_static", ["archetype"]))
    fy = fy[fy["archetype"].notna() & (fy["archetype"] != "No AI") & fy[FE].notna().all(axis=1)]
    # _kmeans seeds its centroids by row position: pin the row order to the firm-year key.
    fy = fy.sort_values("id", ignore_index=True)

    oos_ari_2124, oos_jac_2124, oos_n_2025 = _oos(fy, [2021, 2022, 2023, 2024], 2025)
    oos_ari_2123, oos_jac_2123, _ = _oos(fy, [2021, 2022, 2023], 2025)
    oos_ari_2125, _, oos_n_2026 = _oos(fy, [2021, 2022, 2023, 2024, 2025], 2026)
    oos_jac_min, oos_jac_max = min(oos_jac_2124.values()), max(oos_jac_2124.values())
    oos_vs_2123 = oos_jac_2123.get("Vocal Substantives", float("nan"))

    result = {
        "oos_ari_2124": float(oos_ari_2124), "oos_jac_2124": {k: float(v) for k, v in oos_jac_2124.items()},
        "oos_n_2025": int(oos_n_2025),
        "oos_ari_2123": float(oos_ari_2123), "oos_jac_2123": {k: float(v) for k, v in oos_jac_2123.items()},
        "oos_ari_2125": float(oos_ari_2125), "oos_n_2026": int(oos_n_2026),
        "oos_jac_min": float(oos_jac_min), "oos_jac_max": float(oos_jac_max),
        "oos_vs_2123": float(oos_vs_2123),
    }
    out = L.results_path("posture", "oos_validation.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
