"""AI-washing como score continuo con incertidumbre, en vez de un cruce de clusters.

Reemplaza la definición de `06_voice_vs_behavior_clustering.md` ("voz D ×
comportamiento 1"), que resultó no ser medible: los 4 arquetipos de voz se
construyen sobre 9 TASAS cuyo denominador va de 5 a 500 frames según la
empresa, y K-means trata una tasa estimada con 5 frames como igual de
confiable que una estimada con 500. Consecuencias medidas sobre este
corpus:

  - La tasa promocional del corpus es 9,6%. Con 4 frames, la probabilidad
    de observar CERO promocionales por puro muestreo es 67%.
  - 84,6% de las empresas-año con 3-5 frames tienen tasa promocional
    exactamente 0, contra 1,1% de las que tienen 50+.
  - La mediana de frames totales sube monótonamente con el arquetipo
    (A 17, B 26, C 36, D 59): la escalera de arquetipos es también una
    escalera de volumen.
  - Cuatro empresas con CERO frames promocionales (CMS, FE, CHD, TFC,
    todas con 5-9 frames) quedaban etiquetadas "líderes vocales de IA",
    porque D también se define por `realized_share` alto y `risk_share`
    bajo, y con pocos frames eso sale por azar.

El enfoque de acá no estima una tasa por empresa y la compara contra una
frontera: modela los CONTEOS y pregunta si el exceso de lenguaje
promocional de una empresa es mayor que lo que explica el muestreo.

Modelo:

  1. Nivel frame: regresión logística de `rhetoric_promotional` sobre el
     índice de comportamiento de su empresa. Cada frame es una
     observación, así que una empresa con 500 frames pesa 100 veces más
     que una con 5 — que es exactamente el peso que corresponde.
  2. Nivel empresa: k = frames promocionales observados, n = frames
     totales, p̂ = probabilidad predicha por el modelo dado su
     comportamiento. Test binomial EXACTO de k contra Binomial(n, p̂).
     Con n chico el test simplemente no rechaza: no hace falta ningún
     umbral de volumen arbitrario, la potencia estadística se encarga.
  3. Corrección de Benjamini-Hochberg sobre todas las empresas, porque
     son ~450 tests simultáneos.

Salida: una fila por empresa con el exceso observado, su p-valor exacto y
si sobrevive al FDR — en las dos direcciones. Cola superior = dice más de
lo que su comportamiento declarado justifica (washing). Cola inferior =
dice menos (sustancia callada).

LIMITACIÓN QUE NO SE PUEDE ARREGLAR ACÁ: voz y comportamiento se miden
sobre los MISMOS frames, así que hay dependencia mecánica entre predictor
y respuesta. Un diseño limpio mediría el comportamiento contra una fuente
externa al texto (capex, I+D, contrataciones), que es justamente lo que
intentan `02_...md` y `04_...md`. Esto acota el problema, no lo elimina.

Uso:
    uv run python scripts/analytics/washing_score.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import load_frames, BEHAVIOR_CONCEPTS

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
FDR_Q = 0.05


def behavior_flag(concepts) -> bool:
    """Un frame 'describe comportamiento' si trae al menos un concepto de
    etapa de uso o resultado — no de riesgo ni de gobernanza, que son
    afirmaciones sobre el mundo y no sobre lo que la empresa hace."""
    if concepts is None:
        return False
    return any(c in BEHAVIOR_CONCEPTS for c in concepts)


def benjamini_hochberg(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """True donde el p-valor sobrevive BH al nivel q."""
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresholds = (np.arange(1, m + 1) / m) * q
    passing = ranked <= thresholds
    cutoff = np.max(np.flatnonzero(passing)) if passing.any() else -1
    keep = np.zeros(m, dtype=bool)
    if cutoff >= 0:
        keep[order[: cutoff + 1]] = True
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Sólo para no reportar filas sin ninguna potencia; el test "
                             "no necesita umbral (default: 5)")
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load_frames(con)
    finally:
        con.close()
    frames = frames.copy()
    frames["promotional"] = frames["rhetoric_promotional"].astype(int)
    frames["behavior"] = frames["concepts"].apply(behavior_flag).astype(int)

    # El índice de comportamiento se mide SÓLO sobre los frames NO
    # promocionales de la empresa. Voz y comportamiento salen del mismo
    # texto, y la dependencia a nivel frame es enorme: 92,6% de los frames
    # promocionales también describen comportamiento, contra 54,7% de los
    # no promocionales. Predecir "es promocional" con un índice que
    # incluye a los propios frames promocionales infla la relación por
    # construcción. Excluirlos rompe ese lazo sin sesgar el predictor:
    # sigue midiendo cuánto comportamiento describe la empresa, sólo que
    # en la parte de su texto que no está en discusión.
    per_firm = frames.groupby("ticker").agg(
        n_frames=("promotional", "size"),
        k_promo=("promotional", "sum"),
    ).reset_index()
    non_promotional = frames[frames["promotional"] == 0].groupby("ticker")["behavior"].mean()
    per_firm = per_firm.merge(non_promotional.rename("behavior_index").reset_index(), on="ticker")
    per_firm = per_firm[per_firm["n_frames"] >= args.min_frames].reset_index(drop=True)
    print(f"{len(frames):,} frames | {len(per_firm):,} empresas con >= {args.min_frames} frames")
    print(f"tasa promocional del corpus: {frames['promotional'].mean()*100:.1f}% | "
          f"frames que describen comportamiento: {frames['behavior'].mean()*100:.1f}%")

    # --- 1. relación voz ~ comportamiento, ajustada a nivel FRAME ---
    fitted = frames.merge(per_firm[["ticker", "behavior_index"]], on="ticker")
    model = LogisticRegression(max_iter=1000)
    model.fit(fitted[["behavior_index"]].values, fitted["promotional"].values)
    slope = float(model.coef_[0][0])
    print(f"\nlogit(P(promocional)) = {float(model.intercept_[0]):.3f} "
          f"+ {slope:.3f} * indice_comportamiento")
    print("  -> a más comportamiento descrito, "
          f"{'MÁS' if slope > 0 else 'MENOS'} lenguaje promocional")

    per_firm["p_esperada"] = model.predict_proba(per_firm[["behavior_index"]].values)[:, 1]
    per_firm["k_esperado"] = per_firm["p_esperada"] * per_firm["n_frames"]
    per_firm["exceso"] = per_firm["k_promo"] - per_firm["k_esperado"]
    per_firm["tasa_obs"] = per_firm["k_promo"] / per_firm["n_frames"]

    # --- 2. test binomial exacto, dos colas separadas ---
    per_firm["p_mas"] = [
        stats.binomtest(int(k), int(n), float(p), alternative="greater").pvalue
        for k, n, p in zip(per_firm.k_promo, per_firm.n_frames, per_firm.p_esperada)]
    per_firm["p_menos"] = [
        stats.binomtest(int(k), int(n), float(p), alternative="less").pvalue
        for k, n, p in zip(per_firm.k_promo, per_firm.n_frames, per_firm.p_esperada)]

    # --- 3. FDR sobre cada cola ---
    per_firm["washing"] = benjamini_hochberg(per_firm["p_mas"].values)
    per_firm["callada"] = benjamini_hochberg(per_firm["p_menos"].values)

    n_w, n_c = int(per_firm.washing.sum()), int(per_firm.callada.sum())
    print(f"\nCon FDR {FDR_Q:.0%} sobre {len(per_firm)} empresas:")
    print(f"  habla MÁS de lo que su comportamiento justifica: {n_w} empresas")
    print(f"  habla MENOS: {n_c} empresas")
    print(f"  indistinguibles del modelo: {len(per_firm) - n_w - n_c}")

    show = ["ticker", "n_frames", "k_promo", "k_esperado", "tasa_obs", "p_esperada", "p_mas"]
    if n_w:
        print("\n--- candidatos a washing (ordenados por exceso) ---")
        top = per_firm[per_firm.washing].nlargest(15, "exceso")[show]
        print(top.round(3).to_string(index=False))
    if n_c:
        print("\n--- sustancia callada (top 10 por déficit) ---")
        bot = per_firm[per_firm.callada].nsmallest(10, "exceso")[
            ["ticker", "n_frames", "k_promo", "k_esperado", "tasa_obs", "p_menos"]]
        print(bot.round(3).to_string(index=False))

    print("\n--- potencia: cuántas empresas PODRÍAN haber dado significativas ---")
    per_firm["bucket"] = pd.cut(per_firm.n_frames, [0, 10, 25, 50, 100, 10**6],
                                labels=["5-10", "11-25", "26-50", "51-100", "100+"])
    power = per_firm.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({
            "empresas": len(g),
            "washing": int(g.washing.sum()),
            "callada": int(g.callada.sum()),
        }), include_groups=False)
    print(power.to_string())
    print("  (con pocos frames el test no rechaza casi nunca — eso es correcto,")
    print("   no un defecto: no hay evidencia suficiente para afirmar nada)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "firm_washing_score.parquet"
    per_firm.drop(columns=["bucket"]).to_parquet(out, index=False)
    print(f"\n-> {out} ({len(per_firm):,} filas)")


if __name__ == "__main__":
    main()
