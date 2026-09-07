"""Robustez de la segmentación de Cap. 3 excluyendo la dimensión de intensidad.

El comité de tesis planteó la pregunta obvia: si `intensidad_ia` es una de las
features del K-means, ¿no es "Product Deployer" simplemente "empresa que habla
mucho de IA"? Este script re-corre exactamente el mismo pipeline de
`build_segments.py` (mismas features de comportamiento, mismo encogimiento
empírico-Bayes, mismo k) pero SIN la columna de intensidad en la matriz de
clustering, y compara la partición resultante contra `firm_segments.parquet`
(la que sí incluye intensidad) con un Adjusted Rand Index y una tabla de
contingencia.

No modifica `build_segments.py` ni sus salidas — es un chequeo de robustez de
una sola vez, determinístico, sin LLM.

Salida: `data/processed/clusters/segments_no_intensity_robustness.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import DB, OUT_DIR, SEED, load_frames  # noqa: E402
from build_segments import (ALL_FEATURES, INTENSITY, firm_features,  # noqa: E402
                             name_segments, shrunk_matrix, with_universe)
from ai_intensity import firm_intensity  # noqa: E402


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        frames = load_frames(con)
        universe = firm_intensity(con, ["ticker"])
    finally:
        con.close()

    pooled = with_universe(firm_features(frames, ["ticker"]), universe, ["ticker"])
    has_frames = (pooled["n_frames"] > 0).to_numpy()

    current = pd.read_parquet(OUT_DIR / "firm_segments.parquet")[["ticker", "segmento", "cluster"]]
    k = int(current.loc[current["cluster"] >= 0, "cluster"].nunique())
    print(f"Current segmentation: k={k}, segments={sorted(current['segmento'].unique())}")

    shrunk = shrunk_matrix(pooled, ["ticker"])
    X_with = StandardScaler().fit_transform(shrunk[ALL_FEATURES + [INTENSITY]].values)
    X_without = StandardScaler().fit_transform(shrunk[ALL_FEATURES].values)

    model_without = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X_without[has_frames])
    pooled["cluster_no_intensity"] = model_without.predict(X_without)
    profile_no_intensity = pooled.loc[has_frames].groupby("cluster_no_intensity")[ALL_FEATURES].mean()
    labels_no_intensity = name_segments(profile_no_intensity.assign(**{INTENSITY: 0.0}))
    pooled["segmento_no_intensity"] = pooled["cluster_no_intensity"].map(labels_no_intensity)

    # sanity re-fit WITH intensity, same code path, to isolate the effect of
    # dropping the column rather than any incidental refactor difference
    model_with = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X_with[has_frames])
    pooled["cluster_with_intensity_refit"] = model_with.predict(X_with)
    ari_refit_vs_original = adjusted_rand_score(
        current.loc[current["cluster"] >= 0, "cluster"],
        pooled.loc[current["cluster"] >= 0, "cluster_with_intensity_refit"])
    print(f"Sanity: refit WITH intensity vs. original firm_segments.parquet -> ARI = {ari_refit_vs_original:.3f} "
          f"(should be ~1.0; anything less is refactor drift, not the intensity effect)")

    merged = pooled.merge(current, on="ticker", how="inner", suffixes=("", "_orig"))
    with_frames = merged[merged["n_frames"] > 0]
    ari = adjusted_rand_score(with_frames["cluster"], with_frames["cluster_no_intensity"])
    print(f"\nAdjusted Rand Index (with-intensity segmento vs. without-intensity segmento, "
          f"firms with >=1 AI frame, n={len(with_frames)}): {ari:.3f}")

    print("\nContingency table (rows = original segment, cols = no-intensity segment), row %:")
    ct = pd.crosstab(with_frames["segmento"], with_frames["segmento_no_intensity"], normalize="index") * 100
    print(ct.round(1).to_string())

    print("\nNo-intensity segment profile (mean %, includes frames_per_1k for reference only, NOT in the fit):")
    prof = with_frames.groupby("segmento_no_intensity").agg(
        n=("ticker", "size"), frames_per_1k_median=("frames_per_1k", "median"),
        pct_despliegue=("pct_despliegue", "mean"), pct_producto=("pct_producto", "mean"),
        pct_riesgo=("pct_riesgo", "mean"), pct_gobernanza=("pct_gobernanza", "mean"),
    )
    prof[["pct_despliegue", "pct_producto", "pct_riesgo", "pct_gobernanza"]] *= 100
    print(prof.round(2).to_string())

    # Does the firm set closest to "Product Deployers" survive?
    pd_orig = set(with_frames.loc[with_frames["segmento"] == "desplegadores_de_producto", "ticker"])
    if not pd_orig:
        pd_orig = set(with_frames.loc[with_frames["segmento"] == "desplegadores", "ticker"])
    best_overlap_seg, best_overlap = None, -1.0
    for seg in with_frames["segmento_no_intensity"].unique():
        cand = set(with_frames.loc[with_frames["segmento_no_intensity"] == seg, "ticker"])
        jac = len(pd_orig & cand) / len(pd_orig | cand) if (pd_orig | cand) else 0.0
        if jac > best_overlap:
            best_overlap, best_overlap_seg = jac, seg
    print(f"\nProduct-Deployer-like original segment (n={len(pd_orig)}) vs. best-matching "
          f"no-intensity segment '{best_overlap_seg}': Jaccard overlap = {best_overlap:.3f}")

    result = {
        "k": k,
        "ari_refit_with_intensity_vs_original": float(ari_refit_vs_original),
        "ari_with_vs_without_intensity": float(ari),
        "n_firms_with_frames": int(len(with_frames)),
        "contingency_table_row_pct": ct.round(1).to_dict(),
        "no_intensity_segment_profile": prof.round(2).to_dict(),
        "product_deployer_jaccard_vs_best_no_intensity_segment": {
            "best_match_segment": best_overlap_seg, "jaccard": float(best_overlap),
            "n_original": len(pd_orig),
        },
    }
    out_path = OUT_DIR / "segments_no_intensity_robustness.json"
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
