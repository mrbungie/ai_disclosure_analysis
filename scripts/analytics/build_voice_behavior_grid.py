"""Grilla voz × comportamiento: dos ejes, nueve celdas.

La pregunta de `docs/thesis_proposal.md` es si una empresa habla de IA más de lo
que hace. Eso son DOS ejes, y conviene que sean dos ejes explícitos y no el
subproducto de un clustering:

  VOZ            % de las afirmaciones de IA de la empresa en registro
                 promocional o estratégico (`rhetoric_promotional`,
                 `rhetoric_strategic_importance`). Cuánto de lo que dice es
                 superlativo o "esto nos transforma".

  COMPORTAMIENTO % de sus afirmaciones que describen conducta concreta: etapa de
                 uso (despliegue, escalamiento, piloto) o resultado
                 (productividad, ingresos, costos, cliente) o capacidad
                 (inversión, infraestructura, talento, IA propia/de terceros).

Los dos son PORCENTAJES sobre los frames de la misma empresa, así que ninguno
premia a la que más habla — la diferencia entre una empresa con 500 frames y
otra con 20 no entra en los ejes, sólo en la confianza que merecen sus tasas,
que es lo que corrige el encogimiento empírico-Bayes.

Cada eje se corta en TERCILES y se cruzan: 9 celdas. Las cuatro esquinas son las
categorías que la tesis necesita nombrar:

    voz alta + conducta baja   -> candidatos a AI-washing
    voz baja + conducta alta   -> sustancia callada
    voz alta + conducta alta   -> vocales sustantivos
    voz baja + conducta baja   -> silenciosos

Terciles y no k-means a propósito: un corte por cuantil sobre un índice
encogido es reproducible por construcción, mientras que la partición de 4
arquetipos que esto reemplaza no sobrevivía al remuestreo (Jaccard 0,53). La
estabilidad se mide igual, remuestreando los frames de cada empresa.

Salidas: `firm_voice_behavior_grid.parquet` y `firm_year_voice_behavior_grid.parquet`.

Uso:
    uv run python scripts/analytics/build_voice_behavior_grid.py
    uv run python scripts/analytics/build_voice_behavior_grid.py --bins 2   # 2x2
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import BEHAVIOR_CONCEPTS, DB, OUT_DIR, SEED, load_frames, shrink_rates

MIN_FRAMES = 8
MIN_FRAMES_YEAR = 4
LEVELS = {3: ["baja", "media", "alta"], 2: ["baja", "alta"], 4: ["q1", "q2", "q3", "q4"]}
CORNERS = {
    ("alta", "baja"): "washing (voz alta, conducta baja)",
    ("baja", "alta"): "sustancia callada (voz baja, conducta alta)",
    ("alta", "alta"): "vocales sustantivos",
    ("baja", "baja"): "silenciosos",
}


def axes(frames: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Los dos ejes, en porcentaje de las afirmaciones de cada unidad."""
    df = frames.copy()
    concepts = df["concepts"].apply(lambda c: set(c) if c is not None else set())
    behavior_set = set(BEHAVIOR_CONCEPTS)
    df["voz"] = (df["rhetoric_promotional"].astype(bool)
                 | df["rhetoric_strategic_importance"].astype(bool)).astype(float)
    df["comportamiento"] = concepts.apply(lambda s: float(bool(s & behavior_set)))
    out = df.groupby(keys)[["voz", "comportamiento"]].mean()
    out["n_frames"] = df.groupby(keys).size()
    return out.reset_index()


def shrink_axes(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    indexed = table.set_index(keys)
    shrunk = shrink_rates(indexed[["voz", "comportamiento"]], indexed["n_frames"])
    return shrunk.rename(columns={"voz": "voz_shrunk",
                                  "comportamiento": "comportamiento_shrunk"}).reset_index()


def label(values: pd.Series, bins: int, edges: np.ndarray | None = None):
    """Corte por cuantil. `edges` permite aplicar a un panel los MISMOS cortes
    calculados sobre el pooled: sin eso, un año con más empresas vocales
    reetiquetaría a todas y las series de tiempo no serían comparables."""
    if edges is None:
        edges = np.quantile(values, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    return pd.cut(values, bins=edges, labels=LEVELS[bins], include_lowest=True), edges


def cell_name(voice: str, behavior: str) -> str:
    return CORNERS.get((voice, behavior), f"voz {voice} / conducta {behavior}")


def stability(frames: pd.DataFrame, reference: pd.DataFrame, bins: int,
              replicates: int, seed: int = SEED) -> tuple[dict, pd.Series]:
    """Remuestrea los frames de cada empresa y vuelve a asignar celdas.

    Devuelve dos cosas, y la segunda es la que sirve río abajo: la CONFIANZA de
    cada empresa, o sea en qué fracción de los remuestreos cae en su misma
    celda. Una empresa con 200 frames aterriza siempre en el mismo lado; una con
    9 baila. Publicar esa columna es mejor que publicar un promedio: quien use
    la grilla puede exigir el nivel de confianza que su análisis necesite en vez
    de creerle a todas por igual."""
    order = {level: index for index, level in enumerate(LEVELS[bins])}
    baseline = reference.set_index("ticker")
    rng = np.random.default_rng(seed)
    hits = pd.Series(0.0, index=baseline.index)
    trials = pd.Series(0.0, index=baseline.index)
    same, adjacent = [], []
    for _ in range(replicates):
        positions = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True)
            for idx in frames.groupby("ticker").indices.values()])
        sample = axes(frames.iloc[positions], ["ticker"])
        sample = sample[sample["ticker"].isin(baseline.index)]
        labels = assign(sample, ["ticker"], bins)[0].set_index("ticker")
        common = baseline.index.intersection(labels.index)
        matched = baseline.loc[common, "celda"].values == labels.loc[common, "celda"].values
        hits.loc[common] += matched
        trials.loc[common] += 1
        same.append(float(matched.mean()))
        voice_gap = np.abs(baseline.loc[common, "nivel_voz"].astype(str).map(order).values
                           - labels.loc[common, "nivel_voz"].astype(str).map(order).values)
        behavior_gap = np.abs(
            baseline.loc[common, "nivel_conducta"].astype(str).map(order).values
            - labels.loc[common, "nivel_conducta"].astype(str).map(order).values)
        adjacent.append(float(((voice_gap <= 1) & (behavior_gap <= 1)).mean()))
    confidence = (hits / trials.replace(0, np.nan)).rename("confianza_celda")
    return {"misma_celda": float(np.mean(same)),
            "igual_o_adyacente": float(np.mean(adjacent))}, confidence


def assign(table: pd.DataFrame, keys: list[str], bins: int,
           edges: tuple | None = None) -> tuple[pd.DataFrame, tuple]:
    shrunk = shrink_axes(table, keys)
    out = table.merge(shrunk, on=keys)
    voice_edges = edges[0] if edges else None
    behavior_edges = edges[1] if edges else None
    out["nivel_voz"], voice_edges = label(out["voz_shrunk"], bins, voice_edges)
    out["nivel_conducta"], behavior_edges = label(out["comportamiento_shrunk"], bins,
                                                  behavior_edges)
    out["celda"] = [cell_name(v, b) for v, b in zip(out["nivel_voz"], out["nivel_conducta"])]
    return out, (voice_edges, behavior_edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bins", type=int, default=3, choices=(2, 3, 4))
    parser.add_argument("--bootstrap", type=int, default=25)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load_frames(con)
    finally:
        con.close()
    pooled = axes(frames, ["ticker"])
    pooled = pooled[pooled["n_frames"] >= MIN_FRAMES].reset_index(drop=True)
    print(f"{len(frames):,} frames | {len(pooled):,} empresas con >= {MIN_FRAMES} frames")
    print(f"eje VOZ: media {pooled['voz'].mean()*100:.1f}% de las afirmaciones | "
          f"eje CONDUCTA: media {pooled['comportamiento'].mean()*100:.1f}%")
    correlation = pooled[["voz", "comportamiento"]].corr().iloc[0, 1]
    print(f"correlación entre ejes: {correlation:.3f} — "
          f"{'no son el mismo eje' if abs(correlation) < 0.7 else 'ojo: casi el mismo eje'}")

    assigned, edges = assign(pooled, ["ticker"], args.bins)
    grid = pd.crosstab(assigned["nivel_voz"], assigned["nivel_conducta"])
    print(f"\n=== GRILLA {args.bins}x{args.bins} (empresas) ===")
    print(grid.to_string())

    print("\n=== PERFIL DE CADA CELDA ===")
    profile = assigned.groupby("celda").agg(
        empresas=("ticker", "size"), frames=("n_frames", "median"),
        pct_voz=("voz", lambda s: round(s.mean() * 100, 1)),
        pct_conducta=("comportamiento", lambda s: round(s.mean() * 100, 1)))
    print(profile.sort_values("empresas", ascending=False).to_string())

    for corner in CORNERS.values():
        members = assigned[assigned["celda"] == corner].nlargest(10, "n_frames")
        if len(members):
            print(f"\n{corner} (top 10 por volumen): {', '.join(members['ticker'])}")

    print(f"\n=== ESTABILIDAD ({args.bootstrap} réplicas remuestreando frames) ===")
    agreement, confidence = stability(frames, assigned, args.bins, args.bootstrap)
    assigned = assigned.merge(confidence.reset_index(), on="ticker", how="left")
    print(f"empresas que caen en la MISMA celda: {agreement['misma_celda']:.1%} "
          f"(azar con {args.bins**2} celdas: ~{100/args.bins**2:.0f}%)")
    print(f"en la misma o una ADYACENTE: {agreement['igual_o_adyacente']:.1%} "
          f"— cuando se mueve, se mueve un paso, no cruza la grilla")
    print(f"empresas con confianza >= 0,80: "
          f"{int((assigned['confianza_celda'] >= 0.8).sum())} de {len(assigned)}")
    print("\nconfianza mediana por celda:")
    print(assigned.groupby("celda")["confianza_celda"].median().round(2).to_string())

    panel = axes(frames, ["ticker", "year"])
    panel = panel[panel["n_frames"] >= MIN_FRAMES_YEAR].reset_index(drop=True)
    # Mismos cortes que el pooled: las celdas tienen que significar lo mismo en
    # 2021 y en 2026 o la serie de tiempo no dice nada.
    panel_assigned, _ = assign(panel, ["ticker", "year"], args.bins, edges)
    consecutive = (panel_assigned.sort_values(["ticker", "year"])
                   .assign(anterior=lambda d: d.groupby("ticker")["celda"].shift())
                   .dropna(subset=["anterior"]))
    persistence = float((consecutive["celda"] == consecutive["anterior"]).mean())
    print(f"\npanel empresa-año: {len(panel_assigned):,} filas | "
          f"persistencia año a año: {persistence:.1%}")
    evolution = pd.crosstab(panel_assigned["year"], panel_assigned["nivel_voz"],
                            normalize="index").round(3) * 100
    print("\ncomposición del eje de VOZ por año (% de empresas-año):")
    print(evolution.to_string())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assigned.to_parquet(args.output_dir / "firm_voice_behavior_grid.parquet", index=False)
    panel_assigned.to_parquet(args.output_dir / "firm_year_voice_behavior_grid.parquet",
                              index=False)
    (args.output_dir / "voice_behavior_grid_manifest.json").write_text(json.dumps({
        "bins": args.bins, "axis_correlation": float(correlation),
        "cell_stability": agreement, "persistence_year_over_year": persistence,
        "firms_confidence_over_80": int((assigned["confianza_celda"] >= 0.8).sum()),
        "voice_edges": list(edges[0]), "behavior_edges": list(edges[1]),
        "built_at": datetime.now(timezone.utc).isoformat()}, indent=2, default=float))
    print(f"\n-> {args.output_dir}/firm_voice_behavior_grid.parquet ({len(assigned):,} empresas)")
    print(f"-> {args.output_dir}/firm_year_voice_behavior_grid.parquet "
          f"({len(panel_assigned):,} filas)")


if __name__ == "__main__":
    main()
