"""Validación del score de AI-washing: ¿mide algo estable, o mide la
especificación con que se lo calculó?

Corre cinco chequeos sobre `washing_score.py`. Ninguno necesita una etiqueta
externa de "esta empresa hace washing" — no existe — así que la estrategia es
la que se usa para validar un instrumento sin gold standard: consistencia
interna, estabilidad ante remuestreo, persistencia temporal, sensibilidad de
especificación y comportamiento bajo la hipótesis nula.

  1. PLACEBO       Permuta la etiqueta promocional DENTRO de cada formulario,
                   preservando la tasa de cada forma y la mezcla documental de
                   cada empresa. Bajo el nulo el estimador no debería marcar
                   casi nada; si marca, el test está mal calibrado.
  2. SPLIT-HALF    Parte los frames de cada empresa en dos mitades al azar y
                   compara. Si "hablar de más" es un rasgo de la empresa, las
                   dos mitades tienen que ordenarla parecido; si es ruido, no.
  3. PERSISTENCIA  Mismo ejercicio partiendo por época (filings <=2023 vs
                   >=2024) en vez de al azar. Es la prueba que hundió la
                   definición anterior de washing: sus candidatos no sostenían
                   la etiqueta año a año.
  4. ESPECIFICACIÓN  Corre la grilla unidad x controles x dispersión y mide
                   cuánto se superponen las colas. Es la sensibilidad que
                   motivó la reescritura: sin control de formulario la cola
                   está poblada por empresas con proxy extenso.
  5. CRITERIO EXTERNO  Ubica en el ranking los únicos casos con evidencia
                   independiente: Welltower, la única carta de comentario de la
                   SEC del corpus que cuestiona divulgación de IA, y los tres
                   falsos positivos léxicos de esa misma búsqueda.

Uso:
    uv run python scripts/analytics/validate_washing_score.py
    uv run python scripts/analytics/validate_washing_score.py --placebo-runs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from washing_score import DB, OUT_DIR, load, score

# Welltower: único caso del corpus con una carta de comentario de la SEC que
# pregunta explícitamente por sus afirmaciones de IA (abril 2025, ver
# docs/analytics/01_...md). ANET/HPE/NVDA salieron en la misma búsqueda léxica
# y al leerlas resultaron ser contabilidad de segmentos, no escrutinio de
# disclosure: sirven como controles negativos débiles.
EXTERNAL_CASES = {"WELL": "carta SEC sobre disclosure de IA (abril 2025)",
                  "ANET": "carta SEC, contabilidad de segmentos (falso positivo)",
                  "HPE": "carta SEC, contabilidad de segmentos (falso positivo)",
                  "NVDA": "carta SEC, reconocimiento de ingresos (falso positivo)"}
SPEC_GRID = [("unique", "behavior+form", "document"),
             ("unique", "behavior+form", "none"),
             ("unique", "behavior", "document"),
             ("unique", "behavior+form+sector", "document"),
             ("instance", "behavior", "none")]


def standardized_excess(table: pd.DataFrame) -> pd.Series:
    """Exceso en unidades de desvío binomial: comparable entre empresas de
    distinto volumen, que es justo lo que la versión de clusters no hacía."""
    variance = (table["n_frames"] * table["p_esperada"] * (1 - table["p_esperada"])).clip(lower=1e-9)
    return ((table["k_promo"] - table["k_esperado"]) / np.sqrt(variance)).rename("z")


def run(frames: pd.DataFrame, controls: str, dispersion: str, min_frames: int) -> pd.DataFrame:
    table, _ = score(frames, controls, dispersion, min_frames, verbose=False)
    table["z"] = standardized_excess(table)
    return table


def placebo(frames: pd.DataFrame, controls: str, dispersion: str, min_frames: int,
            runs: int, seed: int = 42) -> list[dict]:
    """Permuta `promotional` dentro de cada formulario: conserva la tasa
    promocional de cada forma y la mezcla documental de cada empresa, y destruye
    sólo la asociación empresa-retórica."""
    rng = np.random.default_rng(seed)
    results = []
    for i in range(runs):
        shuffled = frames.copy()
        shuffled["promotional"] = (
            shuffled.groupby("form_type")["promotional"]
            .transform(lambda values: rng.permutation(values.to_numpy())))
        table, diagnostics = score(shuffled, controls, dispersion, min_frames, verbose=False)
        results.append({"run": i, "washing": diagnostics["washing"],
                        "callada": diagnostics["callada"]})
    return results


def split_frames(frames: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(frames)) < 0.5
    return frames[mask], frames[~mask]


def agreement(left: pd.DataFrame, right: pd.DataFrame, min_frames: int = 10) -> dict:
    merged = left.merge(right, on="ticker", suffixes=("_a", "_b"))
    merged = merged[(merged["n_frames_a"] >= min_frames) & (merged["n_frames_b"] >= min_frames)]
    if len(merged) < 10:
        return {"n": len(merged)}
    rho, p = stats.spearmanr(merged["z_a"], merged["z_b"])
    top_a = set(merged.nlargest(max(5, len(merged) // 10), "z_a")["ticker"])
    top_b = set(merged.nlargest(max(5, len(merged) // 10), "z_b")["ticker"])
    overlap = len(top_a & top_b) / len(top_a)
    return {"n": int(len(merged)), "spearman": float(rho), "p": float(p),
            "top_decil_overlap": float(overlap)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--controls", default="behavior+form")
    parser.add_argument("--dispersion", default="document")
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--placebo-runs", type=int, default=5)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames_unique = load(con, "unique")
        frames_instance = load(con, "instance")
        sectors = con.execute("""
            SELECT ticker, LEFT(sic, 2) AS sic2 FROM firm_universe
            WHERE country_code = 'us' AND ticker IS NOT NULL AND sic IS NOT NULL
        """).fetchdf().drop_duplicates("ticker")
    finally:
        con.close()
    frames_unique = frames_unique.merge(sectors, on="ticker", how="left")
    frames_instance = frames_instance.merge(sectors, on="ticker", how="left")

    report: dict = {}
    baseline = run(frames_unique, args.controls, args.dispersion, args.min_frames)
    flagged = set(baseline.loc[baseline["washing"], "ticker"])
    quiet = set(baseline.loc[baseline["callada"], "ticker"])
    print(f"especificación de referencia [{args.controls}, {args.dispersion}, unique]: "
          f"{len(flagged)} washing, {len(quiet)} callada, {len(baseline)} empresas\n")

    print("=" * 74)
    print("1. PLACEBO — etiqueta promocional permutada dentro de cada formulario")
    print("=" * 74)
    runs = placebo(frames_unique, args.controls, args.dispersion, args.min_frames,
                   args.placebo_runs)
    washing_counts = [r["washing"] for r in runs]
    quiet_counts = [r["callada"] for r in runs]
    print(f"{args.placebo_runs} permutaciones: washing {washing_counts} | callada {quiet_counts}")
    print(f"promedio de falsos positivos: {np.mean(washing_counts):.1f} washing, "
          f"{np.mean(quiet_counts):.1f} callada (observado real: {len(flagged)} y {len(quiet)})")
    report["placebo"] = {"runs": runs, "observed_washing": len(flagged),
                         "observed_quiet": len(quiet)}

    print("\n" + "=" * 74)
    print("2. SPLIT-HALF — dos mitades aleatorias de los frames de cada empresa")
    print("=" * 74)
    half_a, half_b = split_frames(frames_unique)
    table_a = run(half_a, args.controls, args.dispersion, args.min_frames)
    table_b = run(half_b, args.controls, args.dispersion, args.min_frames)
    split = agreement(table_a, table_b)
    print(f"n={split['n']} empresas con >=10 frames en ambas mitades | "
          f"Spearman(z) = {split['spearman']:.3f} (p={split['p']:.1e}) | "
          f"solapamiento del decil superior = {split['top_decil_overlap']:.0%}")
    report["split_half"] = split

    print("\n" + "=" * 74)
    print("3. PERSISTENCIA — filings <=2023 contra >=2024")
    print("=" * 74)
    early = frames_unique[frames_unique["year"] <= 2023]
    late = frames_unique[frames_unique["year"] >= 2024]
    table_early = run(early, args.controls, args.dispersion, args.min_frames)
    table_late = run(late, args.controls, args.dispersion, args.min_frames)
    persistence = agreement(table_early, table_late)
    print(f"n={persistence['n']} empresas con >=10 frames en ambas épocas | "
          f"Spearman(z) = {persistence['spearman']:.3f} (p={persistence['p']:.1e}) | "
          f"solapamiento del decil superior = {persistence['top_decil_overlap']:.0%}")
    both = flagged & set(table_early.loc[table_early["washing"], "ticker"]) \
                   & set(table_late.loc[table_late["washing"], "ticker"])
    print(f"empresas marcadas en el pool Y en ambas épocas por separado: "
          f"{sorted(both) if both else 'ninguna'}")
    report["persistence"] = {**persistence, "flagged_in_both_eras": sorted(both)}

    print("\n" + "=" * 74)
    print("4. ESPECIFICACIÓN — cuánto depende la cola de cómo se calcula")
    print("=" * 74)
    spec_results = {}
    print(f"{'unidad':9s} {'controles':22s} {'dispersión':11s} {'washing':>8s} {'callada':>8s} "
          f"{'∩ referencia':>13s}")
    for unit, controls, dispersion in SPEC_GRID:
        source = frames_unique if unit == "unique" else frames_instance
        table = run(source, controls, dispersion, args.min_frames)
        tail = set(table.loc[table["washing"], "ticker"])
        quiet_tail = set(table.loc[table["callada"], "ticker"])
        shared = len(tail & flagged) / max(len(tail | flagged), 1)
        spec_results[f"{unit}|{controls}|{dispersion}"] = {
            "washing": sorted(tail), "callada": sorted(quiet_tail), "jaccard_vs_ref": shared}
        print(f"{unit:9s} {controls:22s} {dispersion:11s} {len(tail):8d} {len(quiet_tail):8d} "
              f"{shared:12.0%}")
    report["specifications"] = spec_results
    stable = set.intersection(*[set(v["washing"]) for v in spec_results.values()
                                if v["washing"]]) if spec_results else set()
    print(f"\nempresas en la cola de washing bajo TODAS las especificaciones con cola no vacía: "
          f"{sorted(stable) if stable else 'ninguna'}")
    report["stable_across_specs"] = sorted(stable)

    print("\n" + "=" * 74)
    print("5. CRITERIO EXTERNO — casos con evidencia independiente")
    print("=" * 74)
    ranked = baseline.sort_values("z", ascending=False).reset_index(drop=True)
    ranked["percentil"] = 100 * (1 - ranked.index / max(len(ranked) - 1, 1))
    external = []
    for ticker, note in EXTERNAL_CASES.items():
        row = ranked[ranked["ticker"] == ticker]
        if row.empty:
            print(f"{ticker:6s} sin frames suficientes en el panel — {note}")
            external.append({"ticker": ticker, "note": note, "in_panel": False})
            continue
        record = row.iloc[0]
        print(f"{ticker:6s} percentil {record['percentil']:5.1f} | z={record['z']:+.2f} | "
              f"{int(record['k_promo'])}/{int(record['n_frames'])} promocionales "
              f"(esperados {record['k_esperado']:.1f}) | marcado: "
              f"{'sí' if record['washing'] else 'no'} — {note}")
        external.append({"ticker": ticker, "note": note, "in_panel": True,
                         "percentile": float(record["percentil"]), "z": float(record["z"]),
                         "flagged": bool(record["washing"])})
    report["external_cases"] = external

    destination = args.output_dir / "washing_score_validation.json"
    destination.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
