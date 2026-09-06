"""Voz × comportamiento como dos bloques de factores, cruzados con correlación
canónica — y el residuo como medida continua de AI-washing.

Por qué no alcanzan dos factores de voz solos: describen CÓMO habla una empresa
y nada más. La pregunta de la tesis es si eso se despega de lo que la empresa
DICE QUE HACE, y para responderla hay que cruzar los dos bloques, no reportar
uno.

Lo que hace este script, sobre tasas con encogimiento empírico-Bayes en ambos
bloques (imprescindible: `cluster_diagnostics.py` midió que varias tasas crudas
son mayoritariamente ruido de muestreo a nivel empresa):

  1. **Factores por bloque.** Análisis factorial sobre las 9 tasas de VOZ y
     sobre las 15 de COMPORTAMIENTO por separado, con sus cargas.
  2. **Correlación canónica (CCA) entre bloques.** Responde con un número la
     premisa de `06_...md`: ¿cuánto comparten "cómo habla" y "qué dice que
     hace"? Si la primera correlación canónica fuera ~1, los dos ejes son el
     mismo y la matriz voz×comportamiento no tiene contenido; si es intermedia,
     hay una parte compartida y un residuo, y el residuo es justamente lo
     interesante.
  3. **El residuo como score continuo.** Se regresa el eje de voz sobre TODO el
     bloque de comportamiento; lo que queda sin explicar es "habla más (o
     menos) de lo que su comportamiento declarado predice", ahora en una escala
     continua y corregida por confiabilidad, no como un cruce de dos etiquetas
     de cluster.
  4. **Estabilidad bootstrap** de la primera correlación canónica remuestreando
     los frames de cada empresa, para saber si el eje compartido es real o es
     el ruido de las tasas apareciendo dos veces.

Salida: `data/processed/clusters/firm_voice_behavior_factors.parquet`.

Uso:
    uv run python scripts/analytics/voice_behavior_factors.py
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import (BEHAVIOR_CONCEPTS, DB, OUT_DIR, POOLED_MIN_FRAMES, SEED,
                                 VOICE_FEATURES, concept_shares, domain_shares,
                                 load_frames, shrink_rates, voice_metrics)

BEHAVIOR_FEATURES = [f"behavior_share_{c}" for c in BEHAVIOR_CONCEPTS] + \
                    ["domain_share_customer_facing", "domain_share_internal"]


def firm_blocks(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Los dos bloques de tasas por empresa, más el conteo que las sostiene."""
    voice = voice_metrics(frames, ["ticker"]).set_index("ticker")
    counts = voice["n_frames"]
    behavior = concept_shares(frames, ["ticker"], BEHAVIOR_CONCEPTS)
    behavior.columns = [f"behavior_share_{c}" for c in behavior.columns]
    domains = domain_shares(frames, ["ticker"])
    domains.columns = [f"domain_share_{c}" for c in domains.columns]
    block = behavior.join(domains)
    keep = counts >= POOLED_MIN_FRAMES
    return (voice.loc[keep, VOICE_FEATURES],
            block.loc[keep, [c for c in BEHAVIOR_FEATURES if c in block.columns]],
            counts[keep])


def standardize(rates: pd.DataFrame, counts: pd.Series) -> np.ndarray:
    return StandardScaler().fit_transform(shrink_rates(rates, counts).values)


def orient(scores: np.ndarray, reference: pd.Series, column: int = 0) -> np.ndarray:
    """Fija el signo del factor contra una tasa de referencia: sin esto un
    re-ajuste invierte el eje y todas las lecturas quedan al revés."""
    if np.corrcoef(scores[:, column], reference.values)[0, 1] < 0:
        scores[:, column] *= -1
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=25)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load_frames(con)
    finally:
        con.close()
    voice_rates, behavior_rates, counts = firm_blocks(frames)
    print(f"{len(frames):,} frames | {len(voice_rates):,} empresas | "
          f"{len(VOICE_FEATURES)} tasas de voz x {behavior_rates.shape[1]} de comportamiento")

    V = standardize(voice_rates, counts)
    B = standardize(behavior_rates, counts)

    print("\n" + "=" * 74)
    print("1. FACTORES POR BLOQUE (cargas)")
    print("=" * 74)
    voice_fa = FactorAnalysis(n_components=2, random_state=SEED).fit(V)
    behavior_fa = FactorAnalysis(n_components=2, random_state=SEED).fit(B)
    voice_scores = orient(voice_fa.transform(V), voice_rates["risk_share"])
    behavior_scores = orient(behavior_fa.transform(B),
                             behavior_rates["behavior_share_deployed"])
    print("VOZ:")
    print(pd.DataFrame(voice_fa.components_.T, index=VOICE_FEATURES,
                       columns=["voz_1", "voz_2"]).round(2).to_string())
    print("\nCOMPORTAMIENTO:")
    print(pd.DataFrame(behavior_fa.components_.T, index=behavior_rates.columns,
                       columns=["comp_1", "comp_2"]).round(2).to_string())

    print("\n" + "=" * 74)
    print("2. CORRELACIÓN CANÓNICA ENTRE LOS DOS BLOQUES")
    print("=" * 74)
    cca = CCA(n_components=2, max_iter=1000).fit(V, B)
    U, T = cca.transform(V, B)
    canonical = [float(np.corrcoef(U[:, i], T[:, i])[0, 1]) for i in range(U.shape[1])]
    print(f"correlaciones canónicas: " + ", ".join(f"{c:.3f}" for c in canonical))
    print(f"varianza compartida por el primer par: {canonical[0]**2:.1%}")
    loadings = pd.DataFrame({
        "voz_con_U1": [np.corrcoef(V[:, i], U[:, 0])[0, 1] for i in range(V.shape[1])]},
        index=VOICE_FEATURES).round(2)
    behavior_loadings = pd.DataFrame({
        "comp_con_T1": [np.corrcoef(B[:, i], T[:, 0])[0, 1] for i in range(B.shape[1])]},
        index=behavior_rates.columns).round(2)
    print("\nqué es el eje compartido (correlación de cada tasa con su variable canónica):")
    print(loadings.sort_values("voz_con_U1").to_string())
    print(behavior_loadings.sort_values("comp_con_T1").to_string())

    print("\n" + "=" * 74)
    print(f"3. ESTABILIDAD DE LA PRIMERA CORRELACIÓN CANÓNICA ({args.bootstrap} réplicas)")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(args.bootstrap):
        positions = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True)
            for idx in frames.groupby("ticker").indices.values()])
        vr, br, cn = firm_blocks(frames.iloc[positions])
        common = vr.index.intersection(voice_rates.index)
        u, t_ = CCA(n_components=1, max_iter=500).fit_transform(
            standardize(vr.loc[common], cn.loc[common]),
            standardize(br.loc[common], cn.loc[common]))
        values.append(float(np.corrcoef(u[:, 0], t_[:, 0])[0, 1]))
    print(f"primera correlación canónica: observada {canonical[0]:.3f} | "
          f"bootstrap media {np.mean(values):.3f}, "
          f"IC 95% [{np.percentile(values, 2.5):.3f}, {np.percentile(values, 97.5):.3f}]")

    print("\n" + "=" * 74)
    print("4. RESIDUO: voz que el comportamiento NO explica")
    print("=" * 74)
    # El eje de voz que interesa es el 1 (riesgo hipotético <-> despliegue
    # afirmado). Se lo regresa sobre TODO el bloque de comportamiento y el
    # residuo es "habla distinto de lo que su comportamiento declarado predice".
    model = LinearRegression().fit(B, voice_scores[:, 0])
    predicted = model.predict(B)
    residual = voice_scores[:, 0] - predicted
    r2 = float(np.corrcoef(predicted, voice_scores[:, 0])[0, 1] ** 2)
    print(f"el comportamiento explica {r2:.1%} de la varianza del eje de voz; "
          f"el resto es el residuo")
    out = pd.DataFrame({
        "ticker": voice_rates.index, "n_frames": counts.values,
        "voice_factor_risk_vs_deployment": voice_scores[:, 0],
        "voice_factor_quantified_vs_governance": voice_scores[:, 1],
        "behavior_factor_deployment": behavior_scores[:, 0],
        "behavior_factor_2": behavior_scores[:, 1],
        "canonical_voice": U[:, 0], "canonical_behavior": T[:, 0],
        "voice_residual": residual,
    })
    # Cuadrantes con los dos ejes ya corregidos por confiabilidad. "Vocal" es el
    # extremo de despliegue afirmado del eje de voz (factor negativo).
    out["quadrant"] = np.where(
        out.voice_factor_risk_vs_deployment < out.voice_factor_risk_vs_deployment.median(),
        np.where(out.behavior_factor_deployment >= out.behavior_factor_deployment.median(),
                 "voz_y_conducta_altas", "voz_alta_conducta_baja"),
        np.where(out.behavior_factor_deployment >= out.behavior_factor_deployment.median(),
                 "conducta_alta_voz_baja", "voz_y_conducta_bajas"))
    print(out.groupby("quadrant").agg(empresas=("ticker", "size"),
                                      frames=("n_frames", "median")).to_string())
    print("\nresiduo más alto (habla más de lo que su conducta predice):")
    print(", ".join(out.nlargest(10, "voice_residual").ticker))
    print("residuo más bajo (conducta por delante del discurso):")
    print(", ".join(out.nsmallest(10, "voice_residual").ticker))

    destination = args.output_dir / "firm_voice_behavior_factors.parquet"
    out.to_parquet(destination, index=False)
    report = {"canonical_correlations": canonical, "bootstrap_first": values,
              "behavior_explains_voice_r2": r2,
              "quadrants": out["quadrant"].value_counts().to_dict()}
    (args.output_dir / "voice_behavior_factors.json").write_text(
        json.dumps(report, indent=2, default=float))
    print(f"\n-> {destination} ({len(out):,} filas)")


if __name__ == "__main__":
    main()
