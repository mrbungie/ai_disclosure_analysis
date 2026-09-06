"""¿Los arquetipos de voz son estructura o son ruido? Y si son estructura,
¿k-means es la herramienta?

`06_...md` ya reportaba silhouette 0,150-0,162 para todo k probado y aun así
`01_`, `07_` y `08_` tratan los 4 grupos como categorías. Este script mide tres
cosas que faltaban antes de decidir eso:

  1. CONFIABILIDAD de cada feature. Las 9 métricas de voz son TASAS: la media
     por empresa de una bandera booleana a nivel frame. Una tasa estimada con 5
     frames y otra con 500 entran a k-means con el mismo peso, y la primera es
     casi toda ruido de muestreo. Para cada feature se compara la varianza
     ENTRE empresas contra la varianza de muestreo esperada — lo que queda es
     la señal real, y el cociente es una confiabilidad al estilo de un ICC:
     0,9 significa "las diferencias entre empresas son reales", 0,3 significa
     "estás agrupando denominadores".

  2. DIMENSIONALIDAD. Análisis factorial y PCA sobre las mismas 9 tasas: si dos
     factores explican casi todo, los "4 arquetipos" son cortes arbitrarios
     sobre un plano continuo, y reportar dos scores es más honesto (y más útil)
     que reportar cuatro etiquetas.

  3. ESTABILIDAD de la partición. Bootstrap que remuestrea los FRAMES de cada
     empresa (no las empresas): así el remuestreo propaga exactamente la
     incertidumbre que el punto 1 mide. Para cada cluster se reporta el Jaccard
     medio contra su mejor pareja en cada réplica — la convención de
     `clusterboot`: por debajo de 0,6 el cluster no es reproducible, por encima
     de 0,75 se lo puede tratar como real.

Y compara cuatro métodos sobre los mismos datos (k-means, Ward, mezcla de
gaussianas por BIC, y k-means sobre tasas con encogimiento empírico-Bayes hacia
la media global), porque "k-means da grupos feos" y "no hay grupos" son
conclusiones distintas y hay que saber cuál es.

Uso:
    uv run python scripts/analytics/cluster_diagnostics.py
    uv run python scripts/analytics/cluster_diagnostics.py --bootstrap 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import (DB, OUT_DIR, POOLED_MIN_FRAMES, SEED, VOICE_FEATURES,
                                 load_frames, voice_metrics)

K_RANGE = (2, 3, 4, 5, 6)


def reliability(rates: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """Cuánta de la varianza observada entre empresas es señal y no muestreo.

    Para una tasa binomial, la varianza de muestreo de la empresa i es
    p(1-p)/n_i. Si el promedio de esas varianzas se acerca a la varianza
    observada entre empresas, entonces las diferencias que k-means está usando
    para separar grupos son, en su mayoría, denominadores distintos."""
    rows = []
    for column in rates.columns:
        observed = float(rates[column].var(ddof=1))
        grand = float(rates[column].mean())
        sampling = float((grand * (1 - grand) / counts).mean())
        signal = max(observed - sampling, 0.0)
        rows.append({"feature": column, "media": grand, "var_observada": observed,
                     "var_muestreo": sampling,
                     "confiabilidad": signal / observed if observed > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("confiabilidad")


def shrink(rates: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """Encogimiento empírico-Bayes: cada tasa se corre hacia la media global en
    proporción a lo poco que se sabe de esa empresa.

    Es la corrección mínima que hace comparables una tasa de 5 frames con una
    de 500: la primera queda casi en la media global (no se sabe nada), la
    segunda casi intacta. Prior beta por momentos sobre las tasas observadas."""
    out = {}
    for column in rates.columns:
        p = rates[column]
        mean, var = float(p.mean()), float(p.var(ddof=1))
        if var <= 0 or not 0 < mean < 1:
            out[column] = p
            continue
        strength = max(mean * (1 - mean) / var - 1, 1e-6)
        alpha, beta = mean * strength, (1 - mean) * strength
        out[column] = (p * counts + alpha) / (counts + alpha + beta)
    return pd.DataFrame(out, index=rates.index)


def fit_partition(X: np.ndarray, method: str, k: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
    if method == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if method == "gmm":
        return GaussianMixture(n_components=k, covariance_type="full",
                               random_state=SEED, n_init=5).fit_predict(X)
    raise ValueError(method)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def bootstrap_stability(frames: pd.DataFrame, method: str, k: int, replicates: int,
                        shrunk: bool, seed: int = SEED) -> np.ndarray:
    """Remuestrea los FRAMES de cada empresa y vuelve a agrupar.

    Remuestrear empresas mediría otra cosa (sensibilidad a la composición de la
    muestra); lo que acá importa es si la etiqueta de una empresa sobrevive a
    que le hubieran tocado otros frames suyos — que es la incertidumbre real de
    una tasa estimada con 5 observaciones."""
    reference_rates, reference_counts = firm_rates(frames)
    reference = fit_partition(prepare(reference_rates, reference_counts, shrunk), method, k)
    rng = np.random.default_rng(seed)
    scores = np.zeros((replicates, k))
    for replicate in range(replicates):
        # Remuestreo con reemplazo DENTRO de cada empresa, por índices: pandas
        # ya no deja aplicar una función que necesita las columnas de grupo.
        positions = np.concatenate([
            rng.choice(index_array, size=len(index_array), replace=True)
            for index_array in frames.groupby("ticker").indices.values()])
        sampled = frames.iloc[positions]
        rates, counts = firm_rates(sampled)
        rates = rates.reindex(reference_rates.index).fillna(reference_rates.mean())
        counts = counts.reindex(reference_rates.index).fillna(1)
        labels = fit_partition(prepare(rates, counts, shrunk), method, k)
        for cluster in range(k):
            mask = reference == cluster
            scores[replicate, cluster] = max(
                jaccard(mask, labels == other) for other in range(k))
    return scores.mean(axis=0)


def firm_rates(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    metrics = voice_metrics(frames, ["ticker"]).set_index("ticker")
    metrics = metrics[metrics["n_frames"] >= POOLED_MIN_FRAMES]
    return metrics[VOICE_FEATURES], metrics["n_frames"]


def prepare(rates: pd.DataFrame, counts: pd.Series, shrunk: bool) -> np.ndarray:
    values = shrink(rates, counts) if shrunk else rates
    return StandardScaler().fit_transform(values.values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "cluster_diagnostics.json")
    parser.add_argument("--bootstrap", type=int, default=25)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load_frames(con)
    finally:
        con.close()
    rates, counts = firm_rates(frames)
    print(f"{len(frames):,} frames | {len(rates):,} empresas con >= {POOLED_MIN_FRAMES} frames "
          f"| frames por empresa: mediana {counts.median():.0f}, mín {counts.min()}, "
          f"máx {counts.max()}\n")

    report = {}
    print("=" * 78)
    print("1. CONFIABILIDAD DE CADA FEATURE (cuánta varianza NO es muestreo)")
    print("=" * 78)
    table = reliability(rates, counts)
    print(table.round(4).to_string(index=False))
    report["reliability"] = table.to_dict("records")
    print(f"\nconfiabilidad mediana: {table['confiabilidad'].median():.3f}")

    print("\n" + "=" * 78)
    print("2. DIMENSIONALIDAD (PCA y análisis factorial sobre las 9 tasas)")
    print("=" * 78)
    X = prepare(rates, counts, shrunk=False)
    pca = PCA().fit(X)
    variance = pca.explained_variance_ratio_
    print("varianza explicada acumulada: " +
          " ".join(f"PC{i+1}={v:.2f}" for i, v in enumerate(np.cumsum(variance)[:5])))
    factors = FactorAnalysis(n_components=2, random_state=SEED).fit(X)
    loadings = pd.DataFrame(factors.components_.T, index=VOICE_FEATURES,
                            columns=["factor_1", "factor_2"])
    print("\ncargas factoriales (2 factores):")
    print(loadings.round(2).to_string())
    report["pca_cumulative"] = list(np.cumsum(variance))
    report["loadings"] = loadings.round(3).to_dict()

    print("\n" + "=" * 78)
    print("3. MÉTODOS Y k (silhouette; BIC sólo para la mezcla de gaussianas)")
    print("=" * 78)
    Xs = prepare(rates, counts, shrunk=True)
    print(f"{'método':22s} " + " ".join(f"k={k:<6d}" for k in K_RANGE))
    grid = {}
    for method, data, label in (("kmeans", X, "k-means (tasas crudas)"),
                                ("kmeans", Xs, "k-means (encogidas)"),
                                ("ward", X, "Ward"),
                                ("gmm", X, "mezcla gaussiana")):
        scores = []
        for k in K_RANGE:
            labels = fit_partition(data, method, k)
            scores.append(silhouette_score(data, labels))
        grid[label] = scores
        print(f"{label:22s} " + " ".join(f"{s:<8.3f}" for s in scores))
    report["silhouette"] = grid
    bics = [GaussianMixture(n_components=k, covariance_type="full", random_state=SEED,
                            n_init=5).fit(X).bic(X) for k in K_RANGE]
    print(f"{'BIC (gaussiana)':22s} " + " ".join(f"{b:<8.0f}" for b in bics))
    print(f"  -> BIC mínimo en k={K_RANGE[int(np.argmin(bics))]}")
    report["gmm_bic"] = bics

    print("\n" + "=" * 78)
    print(f"4. ESTABILIDAD BOOTSTRAP ({args.bootstrap} réplicas remuestreando frames)")
    print("=" * 78)
    print("Jaccard medio por cluster — <0,6 no reproducible, >0,75 sólido")
    for label, method, shrunk in (("k-means k=4 (crudas)", "kmeans", False),
                                  ("k-means k=4 (encogidas)", "kmeans", True),
                                  ("k-means k=2 (crudas)", "kmeans", False)):
        k = 2 if "k=2" in label else 4
        scores = bootstrap_stability(frames, method, k, args.bootstrap, shrunk)
        print(f"{label:26s} " + " ".join(f"c{i}={s:.2f}" for i, s in enumerate(scores))
              + f"  | media {scores.mean():.2f}")
        report[f"stability_{label}"] = list(scores)

    args.output.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
