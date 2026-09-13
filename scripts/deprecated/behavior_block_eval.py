"""¿Cómo hay que medir la conducta declarada para que sea usable a nivel empresa?

El bloque de conducta se venía agregando como TASAS (fracción de frames de la
empresa con cada concepto). Medido, 11 de 16 de esas tasas tienen confiabilidad
baja — pero la causa no es que el corpus no hable de conducta: `deployed`
aparece en 25% de los frames y los dominios en 23%/59%. La causa es la
agregación: una *tasa* es una forma pésima de resumir un concepto raro. Una
empresa con 3 menciones de inversión en IA sobre 300 frames y otra con 1 sobre
100 tienen la misma tasa (1%) y evidencia muy distinta.

Este script compara cuatro representaciones del MISMO dato, con confiabilidad
split-half (partir los frames de cada empresa en dos mitades al azar, calcular
la medida en cada una y correlacionarlas entre empresas, con corrección de
Spearman-Brown). Sirve para cualquier medida, no sólo para tasas binomiales:

  tasa          fracción de frames con el concepto (lo que se usaba)
  presencia     ¿la empresa lo menciona alguna vez? (binaria)
  log-conteo    log(1 + menciones) — el volumen absoluto, no la proporción
  familia       tasa sobre grupos de conceptos afines, que suben la tasa base:
                construcción de capacidad, etapa temprana, resultados,
                despliegue/escala

Y después vuelve a correr la correlación canónica voz×conducta con el bloque
ganador, para ver si con una medición mejor los dos bloques siguen siendo casi
el mismo eje (canónica 0,93 con tasas crudas) o se separan.

Uso:
    uv run python scripts/analytics/behavior_block_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import (BEHAVIOR_CONCEPTS, DB, OUT_DIR, POOLED_MIN_FRAMES, SEED,
                                 VOICE_FEATURES, load_frames, shrink_rates, voice_metrics)

# Familias: conceptos afines que por separado tienen tasa base de 1-3% y juntos
# llegan a algo medible. El agrupamiento es sustantivo, no estadístico — cada
# familia responde a una pregunta distinta sobre qué hace la empresa.
FAMILIES = {
    "capability_building": ["ai_investment", "ai_infrastructure", "ai_talent", "proprietary_ai"],
    "early_stage": ["pilot_or_testing", "exploring"],
    "outcomes": ["productivity_outcome", "revenue_outcome", "cost_outcome", "customer_outcome"],
    "deployment_scale": ["deployed", "expansion_or_scaling"],
    "third_party": ["third_party_ai"],
}


def concept_matrix(frames: pd.DataFrame) -> pd.DataFrame:
    """Una columna 0/1 por concepto, a nivel frame."""
    out = pd.DataFrame({"ticker": frames["ticker"].values})
    concepts = frames["concepts"].apply(lambda c: set(c) if c is not None else set())
    for concept in BEHAVIOR_CONCEPTS:
        out[concept] = concepts.apply(lambda s, c=concept: int(c in s)).values
    for family, members in FAMILIES.items():
        out[f"fam_{family}"] = concepts.apply(
            lambda s, m=members: int(bool(s & set(m)))).values
    out["domain_customer_facing"] = (frames["domain"] == "customer_facing").astype(int).values
    out["domain_internal"] = (frames["domain"] == "internal").astype(int).values
    return out


def aggregate(matrix: pd.DataFrame, kind: str) -> pd.DataFrame:
    columns = [c for c in matrix.columns if c != "ticker"]
    grouped = matrix.groupby("ticker")[columns]
    if kind == "tasa":
        return grouped.mean()
    if kind == "presencia":
        return (grouped.sum() > 0).astype(float)
    if kind == "log_conteo":
        return np.log1p(grouped.sum())
    raise ValueError(kind)


def split_half_reliability(matrix: pd.DataFrame, kind: str, seed: int = SEED,
                           repeats: int = 5) -> pd.Series:
    """Confiabilidad split-half con corrección de Spearman-Brown, promediada
    sobre varias particiones.

    Es la única medida que se puede aplicar igual a una tasa, a un conteo y a
    una bandera de presencia, y por eso permite comparar representaciones."""
    rng = np.random.default_rng(seed)
    columns = [c for c in matrix.columns if c != "ticker"]
    accumulated = pd.Series(0.0, index=columns)
    for _ in range(repeats):
        half = rng.random(len(matrix)) < 0.5
        a = aggregate(matrix[half], kind)
        b = aggregate(matrix[~half], kind)
        common = a.index.intersection(b.index)
        for column in columns:
            x, y = a.loc[common, column], b.loc[common, column]
            if x.std() == 0 or y.std() == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            accumulated[column] += 2 * r / (1 + r) if r > -1 else 0.0
    return (accumulated / repeats).sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load_frames(con)
    finally:
        con.close()
    counts = frames.groupby("ticker").size()
    frames = frames[frames["ticker"].isin(counts[counts >= POOLED_MIN_FRAMES].index)]
    matrix = concept_matrix(frames)
    print(f"{len(frames):,} frames | {matrix['ticker'].nunique():,} empresas\n")

    print("=" * 78)
    print("1. CONFIABILIDAD SPLIT-HALF POR REPRESENTACIÓN (Spearman-Brown)")
    print("=" * 78)
    table = pd.DataFrame({kind: split_half_reliability(matrix, kind)
                          for kind in ("tasa", "presencia", "log_conteo")})
    table["tasa_base"] = matrix.drop(columns="ticker").mean()
    print(table.round(3).to_string())
    print(f"\nmediana por representación: " +
          ", ".join(f"{k}={table[k].median():.3f}" for k in ("tasa", "presencia", "log_conteo")))
    families = table.loc[[c for c in table.index if c.startswith("fam_")]]
    singles = table.loc[[c for c in table.index if c in BEHAVIOR_CONCEPTS]]
    print(f"conceptos sueltos (mediana log-conteo): {singles['log_conteo'].median():.3f} | "
          f"familias: {families['log_conteo'].median():.3f}")

    print("\n" + "=" * 78)
    print("2. CCA VOZ x CONDUCTA CON CADA BLOQUE")
    print("=" * 78)
    voice = voice_metrics(frames, ["ticker"]).set_index("ticker")
    firm_counts = voice["n_frames"]
    V = StandardScaler().fit_transform(
        shrink_rates(voice[VOICE_FEATURES], firm_counts).values)
    blocks = {
        "conceptos sueltos, tasa": aggregate(matrix, "tasa")[BEHAVIOR_CONCEPTS],
        "conceptos sueltos, log-conteo": aggregate(matrix, "log_conteo")[BEHAVIOR_CONCEPTS],
        "familias, log-conteo": aggregate(matrix, "log_conteo")[
            [f"fam_{f}" for f in FAMILIES] + ["domain_customer_facing", "domain_internal"]],
        "familias, tasa": aggregate(matrix, "tasa")[
            [f"fam_{f}" for f in FAMILIES] + ["domain_customer_facing", "domain_internal"]],
    }
    report = {}
    for name, block in blocks.items():
        block = block.loc[voice.index]
        B = StandardScaler().fit_transform(block.values)
        cca = CCA(n_components=1, max_iter=1000).fit(V, B)
        U, T = cca.transform(V, B)
        canonical = float(np.corrcoef(U[:, 0], T[:, 0])[0, 1])
        factors = FactorAnalysis(n_components=3, random_state=SEED).fit(B)
        explained = float(np.sum(factors.noise_variance_) / B.shape[1])
        print(f"{name:32s} canónica={canonical:.3f} | "
              f"varianza no explicada por 3 factores={explained:.2f} | "
              f"columnas={block.shape[1]}")
        report[name] = {"canonical": canonical, "unexplained": explained,
                        "columns": int(block.shape[1])}

    destination = args.output_dir / "behavior_block_eval.json"
    destination.write_text(json.dumps(
        {"reliability": table.round(4).to_dict(), "cca": report}, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
