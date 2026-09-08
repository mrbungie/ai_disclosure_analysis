"""Validación del índice de AI-washing (brecha divulgación-sustancia,
W_it = pctrank(Disclosure_it) - pctrank(Substance_it), sólo empresas-año con
divulgación de IA): ¿mide algo estable y económicamente coherente, o es ruido?

  1. CONSISTENCIA MECÁNICA  Spearman(W, Grounding), restringido a
                             empresas-año con al menos una actividad
                             declarada. No se reporta como validación
                             convergente independiente: Grounding es la
                             mitad del score por construcción.
  2. PERSISTENCIA            Panel empresa-año, correlación de W entre el
                             año t y t+1 dentro de la misma empresa.
  3. COHERENCIA ECONÓMICA    Spearman(W, intensidad de I+D), y frente a
                             tamaño, beta, volatilidad idiosincrática y
                             valoración (donde no se espera relación fuerte:
                             el score no debería ser un proxy de esas
                             variables).
  4. SPLIT-HALF              Los años de cada empresa se parten al azar en
                              dos mitades; W se recalcula en cada mitad
                              (mismo pipeline, mismas empresas) y se compara.

Uso:
    uv run python scripts/analytics/validate_washing_score.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from washing_score import OUT_DIR, build, validations


def split_half(panel: pd.DataFrame, seed: int = 42) -> dict:
    """Empresas con >=2 años en la muestra analítica: se parten sus años al
    azar en dos mitades, se recalcula el promedio de W en cada mitad, y se
    compara entre empresas."""
    rng = np.random.default_rng(seed)
    counts = panel.groupby("ticker").size()
    eligible = counts[counts >= 2].index
    rows = []
    for tk in eligible:
        years = panel.loc[panel["ticker"] == tk, ["year", "w"]].sort_values("year")
        mask = rng.random(len(years)) < 0.5
        if mask.all() or (~mask).all():
            mask[0] = True
            mask[-1] = False
        rows.append({"ticker": tk, "w_a": years.loc[mask, "w"].mean(), "w_b": years.loc[~mask, "w"].mean()})
    df = pd.DataFrame(rows)
    if len(df) < 10:
        return {"n": len(df)}
    rho, p = stats.spearmanr(df["w_a"], df["w_b"])
    return {"n": int(len(df)), "spearman": float(rho), "p": float(p)}


def main() -> None:
    panel = build(("ticker", "year"))
    report: dict = {"n_panel": int(len(panel)), "n_firms": int(panel["ticker"].nunique())}

    val = validations(panel)
    report.update(val)

    print("=" * 74)
    print("1. CONSISTENCIA MECÁNICA — Spearman(W, Grounding), n_activities > 0")
    print("=" * 74)
    v = val.get("mechanical_consistency_vs_grounding")
    if v:
        print(f"n={v['n']} | Spearman={v['spearman']:+.3f} (p={v['p']:.3g})")

    print("\n" + "=" * 74)
    print("2. PERSISTENCIA — panel empresa-año, W en el año t vs. t+1")
    print("=" * 74)
    v = val.get("persistence")
    if v:
        print(f"n={v['n_pairs']} pares empresa-año consecutivos | Spearman={v['spearman']:+.3f} (p={v['p']:.3g})")

    print("\n" + "=" * 74)
    print("3. COHERENCIA ECONÓMICA")
    print("=" * 74)
    for var, vv in val["economic_coherence"].items():
        print(f"  {var:16s} n={vv['n']:5d} Spearman={vv['spearman']:+.3f} (p={vv['p']:.3g})")

    print("\n" + "=" * 74)
    print("4. SPLIT-HALF — años de cada empresa partidos al azar en dos mitades")
    print("=" * 74)
    sh = split_half(panel)
    report["split_half"] = sh
    if sh.get("n", 0) >= 10:
        print(f"n={sh['n']} empresas con >=2 años | Spearman(W) = {sh['spearman']:+.3f} (p={sh['p']:.3g})")
    else:
        print(f"n insuficiente: {sh}")

    destination = OUT_DIR / "washing_score_validation.json"
    destination.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
