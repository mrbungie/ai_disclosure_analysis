"""Segmentación de empresas por CÓMO divulgan IA — la pregunta central de
`docs/thesis_proposal.md`.

    "How do public firms differ in their AI disclosure behaviors regarding AI
     adoption, capabilities, risks, and governance, and what distinct disclosure
     archetypes emerge?"

Reemplaza los arquetipos A/B/C/D, que no eran reproducibles (Jaccard bootstrap
0,53; ver `cluster_diagnostics.py`). Tres decisiones, todas para que el
resultado sea usable río abajo:

1. **Las features son las dimensiones de la pregunta**, no lo que había
   quedado disponible. Adopción, capacidades, riesgos, gobernanza y registro
   discursivo, cada una en porcentaje de las afirmaciones de IA de la empresa —
   una unidad que se lee sin traducción ("38% de lo que dice sobre IA describe
   despliegue").

2. **Los conceptos raros se agrupan en su dimensión, no se usan sueltos.**
   `ai_investment`, `ai_infrastructure`, `ai_talent` y `proprietary_ai`
   aparecen cada uno en 2-3% de los frames, y por separado una tasa así es casi
   toda ruido de muestreo (confiabilidad 0,35-0,52, medido en
   `behavior_block_eval.py`). Juntos son "capacidades" y llegan a 10%, que sí
   se mide. Agrupar es sustantivo: la pregunta es por capacidades, no por
   `ai_talent` en particular.

3. **k se elige por ESTABILIDAD, no por silhouette.** Se remuestrean los frames
   de cada empresa y se vuelve a segmentar: si a una empresa le hubieran tocado
   otros de sus propios frames, ¿queda en el mismo grupo? Se toma el k más
   grande donde todos los segmentos superan 0,6 de Jaccard medio, que es la
   convención de `clusterboot`. Un segmento que no sobrevive a eso no se puede
   usar para nada río abajo.

Salidas:
  - `firm_segments.parquet`: un segmento por empresa, con sus features en
    porcentaje y la estabilidad de su grupo.
  - `firm_year_segments.parquet`: la misma asignación por empresa-año,
    proyectando sobre los centroides ya entrenados (no se re-segmenta por año,
    o las etiquetas no serían comparables entre años).

Uso:
    uv run python scripts/analytics/build_segments.py
    uv run python scripts/analytics/build_segments.py --bootstrap 40
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import DB, OUT_DIR, SEED, load_frames, shrink_rates

# Todas las empresas con filings entran. La que no habla de IA tiene sus tasas
# en el prior (encogimiento con n=0) y su intensidad en cero: es una empresa
# más, no una excluida. Nada condiciona a hablar de IA.
from ai_intensity import firm_intensity
K_RANGE = (2, 3, 4, 5)
STABILITY_FLOOR = 0.60

# Cada feature es el % de las afirmaciones de IA de la empresa que cae en esa
# categoría. Los grupos son las dimensiones de la pregunta de investigación.
FEATURES = {
    # --- ADOPCIÓN: ¿en qué etapa está y dónde la aplica? ---
    "pct_despliegue": ["deployed"],
    "pct_escalamiento": ["expansion_or_scaling"],
    "pct_etapa_temprana": ["pilot_or_testing", "exploring"],
    "pct_resultados": ["productivity_outcome", "revenue_outcome",
                       "cost_outcome", "customer_outcome"],
    # --- CAPACIDADES: ¿construye o compra? ---
    "pct_capacidad_propia": ["ai_investment", "ai_infrastructure",
                             "ai_talent", "proprietary_ai"],
    "pct_terceros": ["third_party_ai"],
}
# Dimensiones que no salen de `concepts` sino de otras columnas del frame.
DERIVED = ("pct_riesgo", "pct_gobernanza", "pct_promocional", "pct_cuantificado",
           "pct_hipotetico", "pct_producto")
ALL_FEATURES = list(FEATURES) + list(DERIVED)
# Cuánto del filing se dedica a IA (frames por 1.000 párrafos, en log). Es la
# dimensión que separa a quien no habla de quien habla; las tasas de arriba
# sólo dicen cómo habla quien habla.
INTENSITY = "intensidad_ia"
MATRIX_FEATURES = ALL_FEATURES + [INTENSITY]


def firm_features(frames: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Un porcentaje por dimensión y por unidad (empresa, o empresa-año)."""
    df = frames.copy()
    concepts = df["concepts"].apply(lambda c: set(c) if c is not None else set())
    for name, members in FEATURES.items():
        df[name] = concepts.apply(lambda s, m=set(members): float(bool(s & m)))
    df["pct_riesgo"] = concepts.apply(
        lambda s: float(any(str(c).startswith("risk_") for c in s)))
    df["pct_gobernanza"] = concepts.apply(
        lambda s: float(any(str(c).startswith("gov_") for c in s)))
    df["pct_promocional"] = df["rhetoric_promotional"].astype(float)
    df["pct_cuantificado"] = df["specificity_quantified_metric"].astype(float)
    df["pct_hipotetico"] = (df["temporal"] == "hypothetical").astype(float)
    df["pct_producto"] = (df["domain"] == "customer_facing").astype(float)
    out = df.groupby(keys)[ALL_FEATURES].mean()
    out["n_frames"] = df.groupby(keys).size()
    return out.reset_index()


def with_universe(features: pd.DataFrame, universe: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Reindexa las tasas a TODAS las unidades con filings. Sin frames: tasas 0
    con n_frames 0 (el encogimiento las lleva exactamente al prior) e
    intensidad = log(1 + frames por 1.000 párrafos)."""
    out = universe[keys + ["frames_per_1k"]].merge(features, on=keys, how="left")
    out["n_frames"] = out["n_frames"].fillna(0).astype(int)
    out[ALL_FEATURES] = out[ALL_FEATURES].fillna(0.0)
    out[INTENSITY] = np.log1p(out["frames_per_1k"])
    return out


def shrunk_matrix(features: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = features.set_index(keys)
    shrunk = shrink_rates(idx[ALL_FEATURES], idx["n_frames"])
    shrunk[INTENSITY] = idx[INTENSITY]
    return shrunk


def matrix(features: pd.DataFrame) -> np.ndarray:
    """Tasas encogidas hacia el promedio, más la intensidad, estandarizadas.

    El encogimiento es lo que impide que una empresa con 8 frames pese igual
    que una con 500: su tasa se corre hacia el promedio del corpus en
    proporción a lo poco que se sabe de ella; con 0 frames queda en el prior."""
    return StandardScaler().fit_transform(shrunk_matrix(features, ["ticker"]).values)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def stability(frames: pd.DataFrame, universe: pd.DataFrame, k: int, replicates: int,
              seed: int = SEED) -> np.ndarray:
    """Jaccard medio por segmento remuestreando los frames de cada empresa."""
    base = with_universe(firm_features(frames, ["ticker"]), universe, ["ticker"])
    reference = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(matrix(base))
    rng = np.random.default_rng(seed)
    scores = np.zeros((replicates, k))
    for replicate in range(replicates):
        positions = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True)
            for idx in frames.groupby("ticker").indices.values()])
        sample = with_universe(firm_features(frames.iloc[positions], ["ticker"]), universe, ["ticker"])
        sample = sample.set_index("ticker").reindex(base["ticker"]).reset_index()
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(matrix(sample))
        for cluster in range(k):
            scores[replicate, cluster] = max(
                jaccard(reference == cluster, labels == other) for other in range(k))
    return scores.mean(axis=0)


FEATURE_NAMES = {
    "pct_despliegue": "desplegadores",
    "pct_resultados": "desplegadores",
    "pct_producto": "desplegadores_de_producto",
    "pct_riesgo": "listadores_de_riesgo",
    "pct_hipotetico": "listadores_de_riesgo",
    "pct_gobernanza": "adoptantes_con_gobernanza",
    "pct_capacidad_propia": "constructores_de_capacidad",
    "pct_escalamiento": "escaladores",
    "pct_etapa_temprana": "exploradores",
    "pct_promocional": "promocionales",
    "pct_cuantificado": "cuantificadores",
    "pct_terceros": "integradores_de_terceros",
    INTENSITY: "intensivos_en_ia",
}


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Nombre = la dimensión donde el segmento MÁS se despega del promedio.

    No por el id de sklearn, que se renumera en cada re-ajuste y renombraría en
    silencio a todas las empresas. Y no por un orden de reclamo fijo: eso le
    ponía "constructores de capacidad" a un segmento cuya capacidad (8,5%) es
    MENOR que la de los desplegadores (11,2%) — el nombre salía de quién
    quedaba libre, no del perfil. Con z-scores el nombre lo decide el dato: si
    mañana el segmento se define por otra dimensión, se llama distinto."""
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
    names, used = {}, set()
    for cluster in z.index:
        if INTENSITY in z.columns and z.loc[cluster, INTENSITY] <= -1.0 and "silentes" not in used:
            names[cluster] = "silentes"; used.add("silentes"); continue   # casi no habla de IA
        ordered = z.loc[cluster].drop(labels=[INTENSITY], errors="ignore").sort_values(ascending=False)
        for feature in ordered.index:
            candidate = FEATURE_NAMES.get(feature)
            if candidate and candidate not in used:
                names[cluster] = candidate
                used.add(candidate)
                break
        names.setdefault(cluster, f"segmento_{cluster}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=25)
    parser.add_argument("--k", type=int, default=0, help="0 = elegir por estabilidad")
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load_frames(con)
        universe = firm_intensity(con, ["ticker"])
        universe_year = firm_intensity(con, ["ticker", "year"])
    finally:
        con.close()
    pooled = with_universe(firm_features(frames, ["ticker"]), universe, ["ticker"])
    print(f"{len(frames):,} frames | {len(pooled):,} empresas con filings, "
          f"{int((pooled['n_frames'] == 0).sum())} sin ningún frame de IA\n")

    print("=" * 74)
    print(f"ELECCIÓN DE k POR ESTABILIDAD ({args.bootstrap} réplicas, piso {STABILITY_FLOOR})")
    print("=" * 74)
    chosen, stabilities = None, {}
    for k in K_RANGE:
        scores = stability(frames, universe, k, args.bootstrap)
        stabilities[k] = scores
        ok = "sí" if scores.min() >= STABILITY_FLOOR else "NO"
        print(f"k={k}: " + " ".join(f"{s:.2f}" for s in scores) +
              f"  | mínimo {scores.min():.2f} -> usable: {ok}")
        if scores.min() >= STABILITY_FLOOR:
            chosen = k
    k = args.k or chosen or 2
    print(f"\nk elegido: {k}" + ("" if args.k else " (el mayor que supera el piso)"))

    X = matrix(pooled)
    model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
    pooled["cluster"] = model.labels_
    profile = pooled.groupby("cluster")[MATRIX_FEATURES].mean()
    labels = name_segments(profile)
    pooled["segmento"] = pooled["cluster"].map(labels)
    pooled["estabilidad_segmento"] = pooled["cluster"].map(
        dict(enumerate(stabilities.get(k, np.full(k, np.nan)))))

    print("\n" + "=" * 74)
    print("PERFIL DE CADA SEGMENTO (% de las afirmaciones de IA de la empresa)")
    print("=" * 74)
    display = (pooled.groupby("segmento")[ALL_FEATURES].mean() * 100).round(1)
    display.insert(0, "empresas", pooled.groupby("segmento").size())
    display.insert(1, "frames_por_1k", pooled.groupby("segmento")["frames_per_1k"].median().round(2))
    display.insert(1, "frames_medianos", pooled.groupby("segmento")["n_frames"].median())
    display.insert(2, "estabilidad", pooled.groupby("segmento")["estabilidad_segmento"]
                   .first().round(2))
    print(display.to_string())

    print("\nempresas típicas de cada segmento (las más cercanas al centroide):")
    distances = model.transform(X)
    for cluster, name in labels.items():
        rows = pooled.index[pooled["cluster"] == cluster]
        closest = pooled.loc[rows].assign(d=distances[rows, cluster]).nsmallest(8, "d")
        print(f"  {name:28s} {', '.join(closest['ticker'])}")

    panel = with_universe(firm_features(frames, ["ticker", "year"]), universe_year, ["ticker", "year"])
    panel_shrunk = shrunk_matrix(panel, ["ticker", "year"])
    scaler = StandardScaler().fit(shrunk_matrix(pooled, ["ticker"]).values)
    panel["cluster"] = model.predict(scaler.transform(panel_shrunk.values))
    panel["segmento"] = panel["cluster"].map(labels)
    transitions = (panel.sort_values(["ticker", "year"])
                   .assign(anterior=lambda d: d.groupby("ticker")["segmento"].shift())
                   .dropna(subset=["anterior"]))
    persistence = float((transitions["segmento"] == transitions["anterior"]).mean())
    print(f"\npanel empresa-año: {len(panel):,} filas | persistencia año a año: "
          f"{persistence:.1%} sobre {len(transitions):,} pares")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(args.output_dir / "firm_segments.parquet", index=False)
    panel.to_parquet(args.output_dir / "firm_year_segments.parquet", index=False)
    (args.output_dir / "firm_segments_manifest.json").write_text(json.dumps({
        "k": k, "stability_by_k": {str(kk): list(v) for kk, v in stabilities.items()},
        "features": MATRIX_FEATURES, "min_frames": 0,
        "persistence_year_over_year": persistence,
        "built_at": datetime.now(timezone.utc).isoformat()}, indent=2, default=float))
    print(f"\n-> {args.output_dir}/firm_segments.parquet ({len(pooled):,} empresas)")
    print(f"-> {args.output_dir}/firm_year_segments.parquet ({len(panel):,} filas)")


if __name__ == "__main__":
    main()
