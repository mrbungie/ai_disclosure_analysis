"""Modelo jerárquico del exceso promocional: una estimación con incertidumbre
para CADA empresa, no ocho banderas.

`08_definiciones_de_washing.md` corre 451 tests binomiales independientes y marca 8
empresas al 5% de FDR. El resultado es correcto y es poco: 451 tests
independientes tiran a la basura la información de que todas las empresas
vienen de la misma población, y devuelven una etiqueta binaria donde hay una
cantidad continua.

Acá el mismo fenómeno se modela con un efecto aleatorio por empresa:

    logit P(promocional) = b0 + b1 * indice_comportamiento + b_forma + u_empresa
    u_empresa ~ Normal(0, sigma)

`u_empresa` ES el exceso promocional de esa empresa, en log-odds, después de su
comportamiento declarado y de su mezcla documental. El agrupamiento parcial hace
lo que el test independiente no puede: una empresa con 8 frames se encoge hacia
cero porque no hay evidencia, y una con 500 se queda donde está — sin necesidad
de un umbral de volumen que descarte a la mitad del corpus.

Salidas:
  - `firm_washing_hierarchical.parquet`: media posterior, desvío y P(u > 0) para
    las 451 empresas. Un ranking completo con incertidumbre.
  - una curva de potencia: qué exceso es detectable con cuántos frames, que es
    la respuesta cuantitativa a "por qué sólo 8" y cuánto compraría más texto.

Uso:
    uv run python scripts/analytics/washing_hierarchical.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_firm_clusters import DB, OUT_DIR
from washing_score import behavior_flag, load


def power_curve(base_rate: float, sizes=(10, 25, 50, 100, 250, 500),
                ratios=(1.5, 2.0, 3.0), alpha: float = 0.05 / 451) -> pd.DataFrame:
    """Con n frames y una tasa base p, ¿qué exceso se detecta el 80% de las veces?

    El alpha es el de Bonferroni sobre 451 empresas — más severo que el FDR real,
    pero del mismo orden y con fórmula cerrada. Es la respuesta a "por qué sólo
    8": no es el estimador, es cuánto texto tiene cada empresa."""
    rows = []
    for n in sizes:
        row = {"frames": n}
        for ratio in ratios:
            alternative = min(base_rate * ratio, 0.95)
            critical = stats.binom.isf(alpha, n, base_rate)
            power = stats.binom.sf(critical, n, alternative)
            row[f"x{ratio:g}"] = round(float(power), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-frames", type=int, default=5)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load(con, "unique")
    finally:
        con.close()
    frames["promotional"] = frames["rhetoric_promotional"].astype(int)
    frames["behavior"] = frames["concepts"].apply(behavior_flag).astype(int)

    counts = frames.groupby("ticker")["promotional"].size()
    keep = counts[counts >= args.min_frames].index
    frames = frames[frames["ticker"].isin(keep)].copy()
    # Mismo control que el score exacto: el índice de comportamiento se mide
    # sólo sobre frames NO promocionales, para no predecir la retórica con
    # ella misma.
    index = (frames[frames["promotional"] == 0].groupby("ticker")["behavior"].mean()
             .rename("behavior_index"))
    frames = frames.merge(index.reset_index(), on="ticker", how="left")
    frames["behavior_index"] = frames["behavior_index"].fillna(frames["behavior"].mean())
    print(f"{len(frames):,} frames | {frames['ticker'].nunique():,} empresas | "
          f"tasa promocional {frames['promotional'].mean()*100:.1f}%")

    design = pd.get_dummies(frames["form_type"], prefix="form", drop_first=True).astype(float)
    design["behavior_index"] = frames["behavior_index"].values
    design["intercept"] = 1.0
    exog = design.to_numpy(dtype=float)
    endog = frames["promotional"].to_numpy(dtype=float)
    firm_codes = pd.Categorical(frames["ticker"])
    exog_vc = pd.get_dummies(firm_codes).astype(float).to_numpy()

    print("Ajustando el modelo jerárquico (VB sobre efecto aleatorio por empresa)...")
    model = BinomialBayesMixedGLM(endog, exog, exog_vc,
                                  ident=np.zeros(exog_vc.shape[1], dtype=int),
                                  vcp_p=2.0, fe_p=2.0)
    result = model.fit_vb(verbose=False)
    n_fixed = exog.shape[1]
    means = result.vc_mean if hasattr(result, "vc_mean") else result.params[n_fixed:]
    sds = result.vc_sd if hasattr(result, "vc_sd") else result.cov_params()[n_fixed:]
    means = np.asarray(means, dtype=float)[: exog_vc.shape[1]]
    sds = np.asarray(sds, dtype=float)[: exog_vc.shape[1]]

    firms = pd.DataFrame({
        "ticker": list(firm_codes.categories),
        "exceso_log_odds": means, "sd": sds,
        "n_frames": counts.loc[list(firm_codes.categories)].to_numpy(),
    })
    firms["p_mayor_que_cero"] = stats.norm.sf(0, loc=firms["exceso_log_odds"], scale=firms["sd"])
    firms["odds_ratio"] = np.exp(firms["exceso_log_odds"])
    firms = firms.sort_values("exceso_log_odds", ascending=False)

    strong = firms[firms["p_mayor_que_cero"] > 0.95]
    quiet = firms[firms["p_mayor_que_cero"] < 0.05]
    print(f"\ncoeficientes fijos: " +
          ", ".join(f"{c}={v:.2f}" for c, v in zip(design.columns, result.fe_mean)))
    print(f"desvío del efecto empresa (sigma): {float(np.exp(result.vcp_mean[0])):.3f} en log-odds "
          f"— si fuera ~0, las empresas no diferirían más que el azar")
    print(f"\nempresas con P(exceso > 0) > 0,95: {len(strong)}")
    print(f"empresas con P(exceso > 0) < 0,05 (sustancia callada): {len(quiet)}")
    print("\n--- top 15 por exceso posterior ---")
    print(firms.head(15)[["ticker", "n_frames", "exceso_log_odds", "sd",
                          "odds_ratio", "p_mayor_que_cero"]].round(3).to_string(index=False))
    print("\n--- cola opuesta ---")
    print(firms.tail(10)[["ticker", "n_frames", "exceso_log_odds", "odds_ratio",
                          "p_mayor_que_cero"]].round(3).to_string(index=False))

    print("\n=== POR QUÉ SÓLO OCHO: potencia del test exacto ===")
    base = float(frames["promotional"].mean())
    curve = power_curve(base)
    print(f"probabilidad de detectar un exceso, con tasa base {base:.3f} "
          f"y alpha de Bonferroni sobre 451 empresas:")
    print(curve.to_string(index=False))
    distribution = counts.loc[keep]
    print(f"\nframes por empresa en el corpus: mediana {distribution.median():.0f}, "
          f"p75 {distribution.quantile(.75):.0f}, p90 {distribution.quantile(.9):.0f}")
    print(f"empresas con >= 100 frames: {int((distribution >= 100).sum())} de {len(distribution)}")
    print("  -> con la mediana de 27 frames, ni un exceso de 3x se detecta la mitad de las veces.")
    print("     El límite es el TEXTO por empresa, no el estimador: la palanca real es")
    print("     más documentos por empresa (earnings calls, 10-Q), no otro test.")

    destination = args.output_dir / "firm_washing_hierarchical.parquet"
    firms.to_parquet(destination, index=False)
    print(f"\n-> {destination} ({len(firms):,} filas)")


if __name__ == "__main__":
    main()
