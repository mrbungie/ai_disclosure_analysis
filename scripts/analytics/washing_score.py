"""AI-washing como exceso de lenguaje promocional sobre lo que predice el
comportamiento declarado — con la mezcla documental controlada y la
dependencia entre frames contabilizada.

Reemplaza la definición de `06_voice_vs_behavior_clustering.md` ("voz D ×
comportamiento 1"), que no era medible: los 4 arquetipos de voz se construyen
sobre 9 TASAS cuyo denominador va de 5 a 500 frames, y K-means trata una tasa
estimada con 5 frames como igual de confiable que una estimada con 500. Cuatro
empresas con CERO frames promocionales quedaban etiquetadas "líderes vocales".

En vez de estimar una tasa por empresa y compararla contra una frontera, acá se
modelan los CONTEOS y se pregunta si el exceso promocional de una empresa supera
lo que explican (a) el muestreo, (b) su nivel de comportamiento declarado y
(c) la mezcla de formularios en la que habla.

## Qué cambió respecto de la primera versión, y por qué

1. **Unidad = frame único, no instancia de párrafo** (`--unit`). `gold_ai_frames`
   expande cada texto clasificado a una fila por INSTANCIA de párrafo, así que
   un boilerplate repetido en varios filings entra varias veces. Son 1.492
   textos repetidos que generan 3.844 instancias: contar instancias infla `n` y
   `k` sin agregar información independiente.

2. **Efectos fijos de formulario** (`--controls`). La DEF 14A tiene 16,2% de
   frames promocionales contra 7,2% del 10-K (medido sobre textos únicos), así
   que una empresa cuyo proxy aporta la mitad de sus frames parece promocional
   por COMPOSICIÓN DOCUMENTAL, no por su discurso. Sin este control la cola de
   washing tiene 41% de frames de proxy contra 23% del resto del corpus, y al
   agregarlo caen 9 de las 22 empresas detectadas (AAPL, ADBE, AMZN, IBM entre
   ellas) y aparecen otras 3. No es un detalle de especificación: cambia la
   lista.

3. **Sobredispersión por DOCUMENTO, no por empresa** (`--dispersion`). El test
   binomial supone frames independientes y no lo son: los frames de un mismo
   filing comparten párrafo, sección y decisiones de redacción. Se estima la
   correlación intra-documento (ANOVA de una vía sobre los residuos del modelo,
   agrupando por `accession_number`) y se convierte en un efecto de diseño
   `deff = 1 + (frames por filing - 1) * ICC` que infla la varianza del conteo.

   El nivel del clustering es una decisión sustantiva, no técnica. Estimar la
   dispersión a nivel EMPRESA (`--dispersion firm`) trata la heterogeneidad
   entre empresas como ruido — y esa heterogeneidad es exactamente el fenómeno
   que se quiere medir: con esa especificación el test no marca a NADIE, ni en
   la cola alta ni en la baja, porque "esta empresa es más promocional que las
   otras" queda absorbido en el parámetro de dispersión. A nivel DOCUMENTO se
   corrige lo que sí es dependencia mecánica (el mismo texto repitiéndose
   dentro de un filing) y se preserva como señal lo que varía entre documentos
   de la misma empresa.

4. **Poisson-binomial en vez de binomial.** Con controles, cada frame tiene su
   propia probabilidad predicha (según su formulario), así que el conteo nulo
   NO es binomial sino Poisson-binomial. Se calcula exacta por convolución.

## Lo que NO cambia

El índice de comportamiento se sigue midiendo SÓLO sobre los frames no
promocionales de cada empresa: voz y comportamiento salen del mismo texto y la
dependencia a nivel frame es grande (92,6% de los frames promocionales también
describen comportamiento, contra 54,7% de los no promocionales). Excluirlos
rompe ese lazo sin sesgar el predictor.

Y sigue en pie la limitación de fondo: el comportamiento se mide en el mismo
texto que la voz. Un diseño limpio lo mediría contra capex/I+D/contrataciones,
que es lo que intentan `02_...md` y `04_...md` con resultados débiles.

Uso:
    uv run python scripts/analytics/washing_score.py
    uv run python scripts/analytics/washing_score.py --controls behavior --dispersion none --unit instance  # v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
DISPERSION_TRIM = 0.10


def behavior_flag(concepts) -> bool:
    """Un frame 'describe comportamiento' si trae al menos un concepto de etapa
    de uso o resultado — no de riesgo ni de gobernanza, que son afirmaciones
    sobre el mundo y no sobre lo que la empresa hace."""
    if concepts is None:
        return False
    return any(c in BEHAVIOR_CONCEPTS for c in concepts)


def benjamini_hochberg(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """True donde el p-valor sobrevive BH al nivel q."""
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    passing = ranked <= (np.arange(1, m + 1) / m) * q
    keep = np.zeros(m, dtype=bool)
    if passing.any():
        keep[order[: int(np.max(np.flatnonzero(passing))) + 1]] = True
    return keep


def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    """Distribución exacta de la suma de Bernoullis con probabilidades
    DISTINTAS, por convolución sucesiva.

    Hace falta porque con efectos fijos de formulario cada frame tiene su
    propia probabilidad: un frame de proxy es a priori más promocional que uno
    de 10-K, así que el conteo nulo de una empresa depende de en qué documentos
    habló, no sólo de cuántas veces."""
    pmf = np.ones(1)
    for p in probabilities:
        pmf = np.convolve(pmf, [1 - p, p])
    return pmf


def tail_probabilities(probabilities: np.ndarray, k: int) -> tuple[float, float]:
    """P(K >= k) y P(K <= k) bajo la Poisson-binomial exacta."""
    pmf = poisson_binomial_pmf(probabilities)
    upper = float(pmf[k:].sum()) if k < len(pmf) else 0.0
    lower = float(pmf[: k + 1].sum())
    return min(1.0, upper), min(1.0, lower)


def intraclass_correlation(residuals: np.ndarray, groups: np.ndarray) -> float:
    """ICC de una vía (ANOVA) de los residuos, agrupados por documento.

    Mide cuánto se parecen entre sí los frames del MISMO filing una vez
    descontado lo que el modelo ya explica. Es la dependencia mecánica que
    invalida el supuesto binomial; lo que varía ENTRE documentos de una misma
    empresa se deja intacto, porque eso es el fenómeno bajo estudio."""
    frame = pd.DataFrame({"e": residuals, "g": groups})
    sizes = frame.groupby("g")["e"].size()
    k = len(sizes)
    n = len(frame)
    if k < 2 or n <= k:
        return 0.0
    grand = frame["e"].mean()
    means = frame.groupby("g")["e"].mean()
    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(((frame["e"] - frame["g"].map(means)) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    if ms_within <= 0:
        return 0.0
    # m0: tamaño de grupo "efectivo" del estimador ANOVA con grupos desiguales.
    m0 = (n - float((sizes ** 2).sum()) / n) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (m0 - 1) * ms_within)
    return float(np.clip(icc, 0.0, 0.95))


def design_effect(frames_per_filing: float, icc: float) -> float:
    """deff = 1 + (m - 1) * ICC — la varianza del conteo se infla por esto."""
    return float(max(1.0, 1 + (max(frames_per_filing, 1.0) - 1) * icc))


def firm_dispersion_rho(counts: np.ndarray, expected: np.ndarray, variance: np.ndarray,
                        sizes: np.ndarray, trim: float = DISPERSION_TRIM) -> float:
    """Alternativa (`--dispersion firm`): rho que iguala a 1 la media recortada
    de los residuos de Pearson ajustados, resolviendo por bisección.

    Se conserva porque es la especificación más conservadora imaginable y sirve
    de cota: si con esto algo sobrevive, sobrevive con cualquier cosa. En este
    corpus no sobrevive nada, y la sección del docstring explica por qué eso no
    significa "no hay washing" sino "este estimador absorbió el fenómeno"."""
    usable = variance > 0
    base = (counts[usable] - expected[usable]) ** 2 / variance[usable]
    n_i = sizes[usable]

    def trimmed_mean(rho: float) -> float:
        adjusted = base / (1 + (n_i - 1) * rho)
        return float(np.mean(np.sort(adjusted)[: max(1, int(len(adjusted) * (1 - trim)))]))

    if trimmed_mean(0.0) <= 1.0:
        return 0.0
    low, high = 0.0, 0.5
    for _ in range(60):
        mid = (low + high) / 2
        if trimmed_mean(mid) > 1.0:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)


def betabinom_tails(n: int, k: int, p: float, rho: float) -> tuple[float, float]:
    """Colas de una beta-binomial con media p y correlación intra-empresa rho."""
    if rho <= 0 or p <= 0 or p >= 1:
        return (float(stats.binom.sf(k - 1, n, p)), float(stats.binom.cdf(k, n, p)))
    theta = (1 - rho) / rho
    a, b = p * theta, (1 - p) * theta
    return (float(stats.betabinom.sf(k - 1, n, a, b)),
            float(stats.betabinom.cdf(k, n, a, b)))


def build_design(frames: pd.DataFrame, controls: str) -> tuple[np.ndarray, list[str]]:
    """Matriz de diseño del logit a nivel frame."""
    columns = [frames[["behavior_index"]].to_numpy(dtype=float)]
    names = ["behavior_index"]
    if "form" in controls:
        dummies = pd.get_dummies(frames["form_type"], prefix="form", drop_first=True)
        columns.append(dummies.to_numpy(dtype=float))
        names += list(dummies.columns)
    if "sector" in controls:
        dummies = pd.get_dummies(frames["sic2"].fillna("NA"), prefix="sic", drop_first=True)
        columns.append(dummies.to_numpy(dtype=float))
        names += list(dummies.columns)
    return np.hstack(columns), names


def score(frames: pd.DataFrame, controls: str, dispersion: str,
          min_frames: int, verbose: bool = True,
          keys: tuple[str, ...] = ("ticker",)) -> tuple[pd.DataFrame, dict]:
    """El estimador completo. Devuelve la tabla por unidad y el diagnóstico.

    `keys` es la unidad de análisis: `("ticker",)` da el score pooled por
    empresa; `("ticker", "year")` da el panel empresa-año, donde el mismo test
    se aplica a cada año por separado y el FDR corre sobre todas las
    empresas-año a la vez. El panel es lo que permite preguntar si el exceso
    promocional de una empresa cambia después de un evento — con la advertencia
    de que partir los frames por año deja denominadores mucho más chicos, así
    que la potencia cae y casi todo pasa a ser "sin evidencia"."""
    keys = list(keys)
    per_firm = frames.groupby(keys).agg(
        n_frames=("promotional", "size"), k_promo=("promotional", "sum")).reset_index()
    non_promotional = (frames[frames["promotional"] == 0]
                       .groupby(keys)["behavior"].mean().rename("behavior_index"))
    per_firm = per_firm.merge(non_promotional.reset_index(), on=keys, how="left")
    per_firm["behavior_index"] = per_firm["behavior_index"].fillna(
        frames["behavior"].mean())
    per_firm = per_firm[per_firm["n_frames"] >= min_frames].reset_index(drop=True)

    fitted = frames.merge(per_firm[keys + ["behavior_index"]], on=keys)
    design, names = build_design(fitted, controls)
    model = LogisticRegression(max_iter=2000).fit(design, fitted["promotional"].values)
    fitted["p_hat"] = model.predict_proba(design)[:, 1]
    coefficients = dict(zip(names, model.coef_[0].round(3)))

    grouped = fitted.groupby(keys)["p_hat"]
    per_firm = per_firm.merge(
        grouped.agg(p_esperada="mean", k_esperado="sum").reset_index(), on=keys)
    per_firm["exceso"] = per_firm["k_promo"] - per_firm["k_esperado"]
    per_firm["tasa_obs"] = per_firm["k_promo"] / per_firm["n_frames"]

    # pandas entrega la clave como tupla de 1 elemento cuando `keys` es una
    # lista de un solo nombre; se normaliza para que el índice de abajo sirva
    # tanto para el score pooled como para el panel empresa-año.
    probabilities = {(unit[0] if isinstance(unit, tuple) and len(keys) == 1 else unit):
                     group.to_numpy() for unit, group in grouped}
    unit_index = (per_firm[keys[0]] if len(keys) == 1
                  else pd.MultiIndex.from_frame(per_firm[keys]))
    variance = np.array([float((probabilities[u] * (1 - probabilities[u])).sum())
                         for u in unit_index])
    # Frames por filing de cada empresa: es el tamaño de cluster que entra en el
    # efecto de diseño. Una empresa que dice 40 cosas sobre IA repartidas en 6
    # filings tiene menos dependencia que otra que dice las mismas 40 en uno.
    per_filing = (fitted.groupby(keys + ["accession_number"]).size()
                  .groupby(keys).mean().rename("frames_por_filing"))
    per_firm = per_firm.merge(per_filing.reset_index(), on=keys, how="left")
    per_firm["frames_por_filing"] = per_firm["frames_por_filing"].fillna(1.0)

    icc = 0.0
    if dispersion == "document":
        icc = intraclass_correlation(
            (fitted["promotional"] - fitted["p_hat"]).to_numpy(float),
            fitted["accession_number"].to_numpy())
        if verbose:
            print(f"dependencia intra-documento: ICC={icc:.4f} | "
                  f"frames por filing (mediana) {per_firm['frames_por_filing'].median():.1f} "
                  f"-> deff mediano {design_effect(float(per_firm['frames_por_filing'].median()), icc):.2f}")
    firm_rho = 0.0
    if dispersion == "firm":
        firm_rho = firm_dispersion_rho(per_firm["k_promo"].to_numpy(float),
                                       per_firm["k_esperado"].to_numpy(float),
                                       variance, per_firm["n_frames"].to_numpy(float))
        if verbose:
            print(f"sobredispersión a nivel empresa: rho={firm_rho:.4f}")

    upper, lower, rhos = [], [], []
    for unit, row in zip(unit_index, per_firm.itertuples(index=False)):
        p_i = probabilities[unit]
        if dispersion == "document" and icc > 0:
            deff = design_effect(float(row.frames_por_filing), icc)
            # deff = 1 + (n-1) * rho_equivalente sobre el conteo de la empresa.
            rho_i = float(np.clip((deff - 1) / max(row.n_frames - 1, 1), 0, 0.95))
            u, l = betabinom_tails(int(row.n_frames), int(row.k_promo),
                                   float(p_i.mean()), rho_i)
        elif dispersion == "firm" and firm_rho > 0:
            rho_i = firm_rho
            u, l = betabinom_tails(int(row.n_frames), int(row.k_promo),
                                   float(p_i.mean()), rho_i)
        else:
            rho_i = 0.0
            u, l = tail_probabilities(p_i, int(row.k_promo))
        upper.append(u)
        lower.append(l)
        rhos.append(rho_i)
    per_firm["p_mas"] = np.clip(upper, 0, 1)
    per_firm["p_menos"] = np.clip(lower, 0, 1)
    per_firm["washing"] = benjamini_hochberg(per_firm["p_mas"].to_numpy())
    per_firm["callada"] = benjamini_hochberg(per_firm["p_menos"].to_numpy())
    per_firm["controls"] = controls
    per_firm["dispersion"] = dispersion
    per_firm["dispersion_rho"] = rhos

    diagnostics = {"controls": controls, "dispersion": dispersion,
                   "icc_document": icc, "rho_firm": firm_rho,
                   "intercept": float(model.intercept_[0]), "coefficients": coefficients,
                   "n_firms": int(len(per_firm)), "n_frames": int(len(frames)),
                   "washing": int(per_firm["washing"].sum()),
                   "callada": int(per_firm["callada"].sum())}
    return per_firm, diagnostics


def load(con, unit: str) -> pd.DataFrame:
    frames = load_frames(con)
    frames["promotional"] = frames["rhetoric_promotional"].astype(int)
    frames["behavior"] = frames["concepts"].apply(behavior_flag).astype(int)
    if unit == "unique":
        before = len(frames)
        frames = frames.drop_duplicates(["ticker", "text_hash", "frame_index"])
        print(f"unidad = frame único: {before:,} instancias -> {len(frames):,} frames "
              f"({before - len(frames):,} repeticiones del mismo texto descartadas)")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--unit", choices=("unique", "instance"), default="unique")
    parser.add_argument("--controls", default="behavior+form",
                        choices=("behavior", "behavior+form", "behavior+form+sector"))
    parser.add_argument("--dispersion", choices=("document", "firm", "none"),
                        default="document",
                        help="Nivel al que se corrige la dependencia entre frames. "
                             "'document' (default): frames del mismo filing. 'firm': "
                             "trata TODA la heterogeneidad entre empresas como ruido — "
                             "cota superior de conservadurismo, no marca a nadie. "
                             "'none': supone frames independientes (versión original).")
    parser.add_argument("--by-year", action="store_true",
                        help="Calcular el score por (empresa, año de filing) además del "
                             "pooled, y escribir firm_year_washing_score.parquet. Permite "
                             "preguntar si el exceso promocional de una empresa cambia tras "
                             "un evento; la contra es que partir por año achica el "
                             "denominador y la potencia cae mucho.")
    parser.add_argument("--min-frames", type=int, default=5,
                        help="Sólo para no reportar filas sin ninguna potencia; el test "
                             "no necesita umbral (default: 5)")
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load(con, args.unit)
        sectors = con.execute("""
            SELECT ticker, LEFT(sic, 2) AS sic2 FROM firm_universe
            WHERE country_code = 'us' AND ticker IS NOT NULL AND sic IS NOT NULL
        """).fetchdf().drop_duplicates("ticker")
    finally:
        con.close()
    frames = frames.merge(sectors, on="ticker", how="left")

    print(f"{len(frames):,} frames | tasa promocional {frames['promotional'].mean()*100:.1f}% | "
          f"describen comportamiento {frames['behavior'].mean()*100:.1f}%")
    by_form = frames.groupby("form_type")["promotional"].agg(["mean", "size"])
    print("tasa promocional por formulario:",
          {f: f"{row['mean']*100:.1f}% (n={int(row['size']):,})" for f, row in by_form.iterrows()})

    per_firm, diagnostics = score(frames, args.controls, args.dispersion, args.min_frames)
    print(f"\nlogit(P(promocional)) = {diagnostics['intercept']:.3f} + "
          + " + ".join(f"{v} * {k}" for k, v in diagnostics["coefficients"].items()))
    print(f"\nCon FDR {FDR_Q:.0%} sobre {diagnostics['n_firms']} empresas "
          f"[{args.controls}, {args.dispersion}, {args.unit}]:")
    print(f"  habla MÁS de lo que su comportamiento y su mezcla documental justifican: "
          f"{diagnostics['washing']}")
    print(f"  habla MENOS: {diagnostics['callada']}")
    print(f"  indistinguibles: {diagnostics['n_firms'] - diagnostics['washing'] - diagnostics['callada']}")

    show = ["ticker", "n_frames", "k_promo", "k_esperado", "tasa_obs", "p_esperada", "p_mas"]
    if diagnostics["washing"]:
        print("\n--- cola de washing (por exceso) ---")
        print(per_firm[per_firm.washing].nlargest(15, "exceso")[show].round(3).to_string(index=False))
    if diagnostics["callada"]:
        print("\n--- sustancia callada (por déficit) ---")
        print(per_firm[per_firm.callada].nsmallest(10, "exceso")[
            ["ticker", "n_frames", "k_promo", "k_esperado", "tasa_obs", "p_menos"]
        ].round(3).to_string(index=False))

    print("\n--- potencia por volumen de texto ---")
    per_firm["bucket"] = pd.cut(per_firm.n_frames, [0, 10, 25, 50, 100, 10**6],
                                labels=["5-10", "11-25", "26-50", "51-100", "100+"])
    power = per_firm.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({"empresas": len(g), "washing": int(g.washing.sum()),
                             "callada": int(g.callada.sum())}), include_groups=False)
    print(power.to_string())
    print("  (con pocos frames el test no rechaza casi nunca — eso es correcto:")
    print("   ausencia de evidencia queda registrada como ausencia de evidencia)")

    if args.by_year:
        print("\n" + "=" * 74)
        print("PANEL EMPRESA-AÑO — mismo test por año, FDR sobre todas las empresas-año")
        print("=" * 74)
        panel, panel_diagnostics = score(frames, args.controls, args.dispersion,
                                         args.min_frames, verbose=False,
                                         keys=("ticker", "year"))
        print(f"{len(panel):,} empresas-año con >= {args.min_frames} frames | "
              f"{panel_diagnostics['washing']} marcadas washing, "
              f"{panel_diagnostics['callada']} callada")
        flagged = panel[panel["washing"]]
        if len(flagged):
            counts = flagged.groupby("ticker").size().sort_values(ascending=False)
            print(f"\nempresas marcadas en más de un año: "
                  f"{counts[counts > 1].to_dict() or 'ninguna'}")
            print("\n" + flagged.sort_values(["ticker", "year"])[
                ["ticker", "year", "n_frames", "k_promo", "k_esperado", "tasa_obs"]
            ].round(2).to_string(index=False))
        # Exceso estandarizado antes vs. después del escrutinio SEC (marzo 2024).
        # Es descriptivo: los grupos no son aleatorios y no hay contrafactual.
        panel["z"] = ((panel["k_promo"] - panel["k_esperado"]) /
                      np.sqrt((panel["n_frames"] * panel["p_esperada"]
                               * (1 - panel["p_esperada"])).clip(lower=1e-9)))
        era = panel.assign(era=np.where(panel["year"] <= 2023, "<=2023", ">=2024"))
        print("\nexceso estandarizado medio por época (descriptivo, sin contrafactual):")
        print(era.groupby("era")["z"].agg(["mean", "median", "size"]).round(3).to_string())
        panel_out = args.output_dir / "firm_year_washing_score.parquet"
        panel.to_parquet(panel_out, index=False)
        print(f"\n-> {panel_out} ({len(panel):,} filas)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "firm_washing_score.parquet"
    per_firm.drop(columns=["bucket"]).to_parquet(out, index=False)
    manifest = args.output_dir / "firm_washing_score_manifest.json"
    manifest.write_text(json.dumps(
        {**diagnostics, "unit": args.unit, "min_frames": args.min_frames,
         "built_at": datetime.now(timezone.utc).isoformat()}, indent=2, default=float))
    print(f"\n-> {out} ({len(per_firm):,} filas)\n-> {manifest}")


if __name__ == "__main__":
    main()
