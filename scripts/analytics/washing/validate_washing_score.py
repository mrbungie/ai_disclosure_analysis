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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L


def validations(panel: pd.DataFrame) -> dict:
    """Consistencia mecánica, persistencia y coherencia económica del índice
    W (ver washing_score.py). Vivía en el gold builder (`washing_score.py`)
    porque ahí se calcula `panel`; movida acá porque es una validación, no
    parte de la construcción del índice -- gold sólo construye el dato."""
    out: dict = {}

    grounded = panel[panel["n_activities"] > 0]
    if grounded["grounding_index"].nunique() > 1:
        rho, p = stats.spearmanr(grounded["w"], grounded["grounding_index"])
        out["mechanical_consistency_vs_grounding"] = {"n": int(len(grounded)), "spearman": float(rho), "p": float(p)}

    p2 = panel[["ticker", "year", "w"]].copy()
    nxt = p2.assign(year=p2["year"] - 1).rename(columns={"w": "w_next"})
    pairs = p2.merge(nxt, on=["ticker", "year"])
    if len(pairs) > 10:
        rho, p = stats.spearmanr(pairs["w"], pairs["w_next"])
        out["persistence"] = {"n_pairs": int(len(pairs)), "spearman": float(rho), "p": float(p)}

    m = L.read_dataset("firm_year", ("covariates", "financials", ["rd_intensity"]),
                       ("covariates", "market", ["market_cap", "beta", "idio_vol_252d", "ps_ratio"]),
                       spine_columns=["id", "ticker", "year"]).drop(columns="id")
    econ = panel.merge(m, on=["ticker", "year"], how="left")
    econ["log_market_cap"] = np.log(econ["market_cap"])
    coherence = {}
    for var in ("rd_intensity", "log_market_cap", "beta", "idio_vol_252d", "ps_ratio"):
        sub = econ.dropna(subset=["w", var])
        if sub[var].nunique() > 1:
            rho, p = stats.spearmanr(sub["w"], sub[var])
            coherence[var] = {"n": int(len(sub)), "spearman": float(rho), "p": float(p)}
    out["economic_coherence"] = coherence
    return out


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
    panel = L.read_gold("firm_year", ("covariates", "washing_score"))
    panel = panel[panel["w"].notna()].reset_index(drop=True)
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

    destination = L.results_path("washing", "washing_score_validation.json")
    destination.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
