"""Señal incremental: ¿cuánto agrega cada capa de medición de la divulgación de
IA a lo que ya explican los fundamentals, sobre outcomes medidos después?

Panel point in time: datasets/firm_quarter/firm_quarter. Cada fila es una
empresa-trimestre usable desde `as_of_date` (día siguiente al cierre del
trimestre): los predictores son covariates publicadas antes de `as_of_date`
y los outcomes son targets medidos estrictamente después.

Modelos anidados, la misma muestra (efectos fijos sector SIC2 x trimestre por
demeaning, errores cluster por empresa):

    M0  Y = a[sector×trimestre] + b·fundamentals + e
    M1  M0 + log(1 + frames de IA por 1.000 palabras)                  (menciones)
    M2  M1 + postura: las siete tasas de postura de los frames y los
        pesos de arquetipo w_voc, w_gov (w_def es el complemento)      (postura)
    M3  M2 + actividades: log(1 + actividades por 1.000 palabras) de
        seis familias                                                    (actividad)
    M4  M3 + índice de decoupling W                                      (washing)

Texto: familia de canal `filings` (10-K, 10-Q, 8-K, DEF 14A) en ventana `ttm`
(documentos de los últimos cuatro trimestres); arquetipos en su única familia
`posture_ttm`; W = `filings_ttm_w` (covariates/firm_quarter/washing_score,
estimado dentro de cada trimestre). Sin frames de IA en la ventana, las tasas
de postura y los pesos quedan en cero (la ausencia la mide M1), igual que los
pesos cuando la familia de postura (10-K, 8-K, DEF 14A) no tiene frames; sin
actividades, cero. W sólo existe para empresas-trimestre con divulgación de
IA: M4 se compara con M3 sobre esa submuestra.

Fundamentals (covariates as of `as_of_date`): log(market cap) al cierre
previo (covariates/firm_quarter/market), margen bruto, margen operativo y
rotación de activos del último trimestre fiscal presentado
(covariates/firm_quarter/financials; cada razón sólo cuando sus componentes
son del mismo trimestre fiscal). Winsorización 1/99 de fundamentals y
outcomes.

Outcomes (targets/firm_quarter) y su versión anual anterior:
    beta                        beta_post_63       (beta de 252 días antes del 10-K)
    volatilidad_idiosincratica  idio_vol_post_63   (idio vol de 252 días antes del 10-K)
    price_to_sales              ps_ratio_post      (P/S antes del 10-K)
    rd_sobre_ventas             next_rd_intensity  (R&D/ventas del ejercicio del 10-K)
    crecimiento_ingresos_t1     next_revenue_yoy   (crecimiento del ejercicio siguiente)
    volatilidad_total_63d       vol_post_63        (robustez; vol total de 60 días antes)

Inferencia: ΔR² de cada capa con intervalo bootstrap por empresa, R²
ajustado, test de Wald conjunto de cada bloque (SE cluster por empresa).
Robustez: efectos fijos de empresa en vez de sector×trimestre.

Salida: data/results/shock/incremental_signal.json y
data/results/shock/incremental_signal_coefficients.csv (coeficientes
estandarizados de M3, postura y actividad, con IC95 y p). El M3 ajustado por
outcome se persiste en models/incremental_signal/<outcome>/model.pkl (gold no
lee ese directorio).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import layers as L  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTCOMES = {"beta": "beta_post_63", "volatilidad_idiosincratica": "idio_vol_post_63", "price_to_sales": "ps_ratio_post",
            "rd_sobre_ventas": "next_rd_intensity", "crecimiento_ingresos_t1": "next_revenue_yoy"}
EXTRA_OUTCOMES = {"volatilidad_total_63d": "vol_post_63"}
FUNDAMENTALS = ["log_market_cap", "gross_margin", "operating_margin", "asset_turnover"]
TEXT = "filings_ttm"
VOLUME = f"{TEXT}_frames_per_1k"
POSTURE_RATES = ["promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
                 "temporal_posture", "ai_positioning", "specificity"]
POSTURE = [f"{TEXT}_{c}" for c in POSTURE_RATES] + ["posture_ttm_w_voc", "posture_ttm_w_gov"]
ACTIVITY_FAMILIES = ["customer_facing_deployment", "internal_deployment", "proprietary_ai", "third_party_named_provider",
                     "infrastructure_investment", "quantified_outcome"]
ACTIVITY = [f"{TEXT}_{c}_per_1k" for c in ACTIVITY_FAMILIES]
WASHING = f"{TEXT}_w"
BLOCKS = {"M1": [VOLUME], "M2": POSTURE, "M3": ACTIVITY, "M4": [WASHING]}
LAYER_NAMES = {"M1": "menciones", "M2": "postura", "M3": "actividad", "M4": "washing"}
FAMILIES = [
    ("covariates", "market", ["log_market_cap"]),
    ("covariates", "financials", ["revenue", "revenue_quarter", "cogs", "cogs_quarter", "operating_income",
                                  "operating_income_quarter", "assets", "assets_quarter"]),
    ("covariates", "disclosure_volume", [VOLUME, f"{TEXT}_n_frames"]),
    ("covariates", "posture_rates", [f"{TEXT}_{c}" for c in POSTURE_RATES]),
    ("covariates", "posture_archetype", ["posture_ttm_w_voc", "posture_ttm_w_gov"]),
    ("covariates", "activities", ACTIVITY),
    ("covariates", "washing_score", [WASHING]),
    ("targets", "market", ["beta_post_63", "idio_vol_post_63", "vol_post_63", "ps_ratio_post"]),
    ("targets", "financials", ["next_revenue_yoy", "next_rd_intensity"]),
]
SEED = 42


def winsor(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def load() -> pd.DataFrame:
    m = L.read_dataset("firm_quarter", *FAMILIES, spine_columns=L.GOLD_SPINE_COLUMNS["firm_quarter"] + ["sic2"])
    same = lambda a, b: m[f"{a}_quarter"] == m[f"{b}_quarter"]  # noqa: E731
    revenue = m["revenue"].where(m["revenue"] > 0)
    m["gross_margin"] = ((m["revenue"] - m["cogs"]) / revenue).where(same("revenue", "cogs"))
    m["operating_margin"] = (m["operating_income"] / revenue).where(same("revenue", "operating_income"))
    m["asset_turnover"] = (m["revenue"] / m["assets"].where(m["assets"] > 0)).where(same("revenue", "assets"))
    for c in FUNDAMENTALS + list(OUTCOMES.values()) + list(EXTRA_OUTCOMES.values()):
        m[c] = winsor(m[c])
    no_frames = m[f"{TEXT}_n_frames"].fillna(0) == 0
    m[VOLUME] = m[VOLUME].fillna(0.0)
    for c in POSTURE:
        m[c] = m[c].where(~no_frames, 0.0)
    # the weights come from the posture channel family (10-K, 8-K, DEF 14A): null
    # when only 10-Q frames exist in the window, read as no posture weight
    m[["posture_ttm_w_voc", "posture_ttm_w_gov"]] = m[["posture_ttm_w_voc", "posture_ttm_w_gov"]].fillna(0.0)
    m[ACTIVITY] = m[ACTIVITY].fillna(0.0)
    m["sector_quarter"] = m["sic2"].astype(str) + "_" + m["quarter"].astype(str)
    return m


def design(d: pd.DataFrame, y: str, block: str, fe: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fundamentals plus every layer up to `block`, demeaned within `fe`."""
    X = d[FUNDAMENTALS].copy()
    for b in ("M1", "M2", "M3", "M4")[: int(block[1])]:
        for c in BLOCKS[b]:
            X[c] = np.log1p(d[c]) if b in ("M1", "M3") else d[c]
    g = d[fe].to_numpy()
    Xd = X - X.groupby(g).transform("mean")
    yd = d[y] - d[y].groupby(g).transform("mean")
    return Xd.to_numpy(dtype=float), yd.to_numpy(dtype=float), list(X.columns)


def r2(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    tss = float(np.sum(y ** 2))
    return 1 - float(np.sum((y - X @ beta) ** 2)) / tss if tss > 0 else np.nan


def adj_r2(X: np.ndarray, y: np.ndarray, n_groups: int) -> float:
    """R² ajustado contando los efectos fijos absorbidos (n_groups) como parámetros."""
    n, k = X.shape
    dof = n - k - n_groups
    return 1 - (1 - r2(X, y)) * (n - 1) / dof if dof > 0 else np.nan


def wald(d: pd.DataFrame, y: str, block: str, fe: str) -> dict:
    """H0: los coeficientes de la capa `block` son cero en el modelo `block`."""
    import statsmodels.api as sm
    X, yy, names = design(d, y, block, fe)
    res = sm.OLS(yy, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
    idx = [i for i, n in enumerate(names) if n in BLOCKS[block]]
    R = np.zeros((len(idx), len(names)))
    for row, i in enumerate(idx):
        R[row, i] = 1.0
    ft = res.wald_test(R, use_f=True, scalar=True)
    return {"F": float(ft.statistic), "p": float(ft.pvalue), "df": len(idx)}


def sample(m: pd.DataFrame, y: str, fe: str, washing: bool = False) -> pd.DataFrame:
    need = [y, fe, "ticker"] + FUNDAMENTALS + ([WASHING] if washing else [])
    d = m.dropna(subset=need)
    return d[d.groupby(fe)[y].transform("size") >= 2].reset_index(drop=True)


def nested(m: pd.DataFrame, y: str, fe: str = "sector_quarter") -> tuple[dict, pd.DataFrame]:
    d = sample(m, y, fe)
    out = {"n": int(len(d)), "n_firms": int(d["ticker"].nunique()), "n_quarters": int(d["quarter"].nunique())}
    n_groups = int(d[fe].nunique())
    for block in ("M0", "M1", "M2", "M3"):
        X, yy, _ = design(d, y, block, fe)
        out[f"r2_{block}"], out[f"adj_r2_{block}"] = r2(X, yy), adj_r2(X, yy, n_groups)
    for prev, block in (("M0", "M1"), ("M1", "M2"), ("M2", "M3")):
        out[f"delta_r2_{LAYER_NAMES[block]}"] = out[f"r2_{block}"] - out[f"r2_{prev}"]
        out[f"wald_{LAYER_NAMES[block]}"] = wald(d, y, block, fe)
    dw = sample(m, y, fe, washing=True)
    out["n_washing"], out["n_washing_firms"] = int(len(dw)), int(dw["ticker"].nunique())
    if len(dw) > 20:
        X3, yy, _ = design(dw, y, "M3", fe)
        X4, _, _ = design(dw, y, "M4", fe)
        out["r2_M3_washing_sample"], out["r2_M4"] = r2(X3, yy), r2(X4, yy)
        out["delta_r2_washing"] = out["r2_M4"] - out["r2_M3_washing_sample"]
        out["wald_washing"] = wald(dw, y, "M4", fe)
    return out, d


def bootstrap(d: pd.DataFrame, y: str, fe: str, reps: int, seed: int = SEED) -> dict:
    """Bootstrap por empresa (cluster) del ΔR² de menciones, postura y actividad."""
    rng = np.random.default_rng(seed)
    firms, groups = d["ticker"].unique(), d.groupby("ticker").indices
    draws = {LAYER_NAMES[b]: [] for b in ("M1", "M2", "M3")}
    for _ in range(reps):
        s = d.iloc[np.concatenate([groups[f] for f in rng.choice(firms, size=len(firms), replace=True)])]
        s = s.reset_index(drop=True)
        r = {b: r2(*design(s, y, b, fe)[:2]) for b in ("M0", "M1", "M2", "M3")}
        for prev, b in (("M0", "M1"), ("M1", "M2"), ("M2", "M3")):
            draws[LAYER_NAMES[b]].append(r[b] - r[prev])
    return {f"delta_r2_{k}_ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in draws.items()} | {"bootstrap_reps": reps}


def coefficients(d: pd.DataFrame, y: str, fe: str, model_name: str) -> dict:
    """Coeficientes estandarizados de M3 (menciones, postura, actividad), SE
    cluster por empresa; el ajuste se guarda en models/incremental_signal/."""
    import joblib
    import statsmodels.api as sm
    X, yy, names = design(d, y, "M3", fe)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    res = sm.OLS(yy / (yy.std() or 1), X / sd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
    model_dir = REPO_ROOT / "models" / "incremental_signal" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": res, "feature_names": names, "outcome": y, "fe": fe}, model_dir / "model.pkl")
    return {n: {"beta_std": float(b), "p": float(p), "ci95": [float(lo), float(hi)]}
            for n, b, p, (lo, hi) in zip(names, res.params, res.pvalues, res.conf_int(alpha=0.05))
            if n not in FUNDAMENTALS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    m = load()
    print(f"panel: {len(m):,} empresas-trimestre, {m.ticker.nunique()} empresas, "
          f"{m['quarter'].min()} .. {m['quarter'].max()}\n")
    report = {"unit": "firm_quarter", "text": f"{TEXT} (10-K, 10-Q, 8-K, DEF 14A; last four quarters)",
              "timing": "predictors published before as_of_date; outcomes measured after as_of_date",
              "outcome_columns": {**OUTCOMES, **EXTRA_OUTCOMES}, "outcomes": {}, "robustez": {}}
    print(f"{'outcome':26s} {'n':>6s} {'R²M0':>6s} {'ΔR² menc':>9s} {'ΔR² post':>9s} {'ΔR² act':>8s} {'ΔR² W':>8s} "
          f"{'p post':>7s} {'p act':>6s} {'p W':>6s}")
    coef_rows = []
    for label, y in OUTCOMES.items():
        res, d = nested(m, y)
        res.update(bootstrap(d, y, "sector_quarter", args.bootstrap))
        res["coeficientes_M3"] = coefficients(d, y, "sector_quarter", label)
        coef_rows += [{"outcome": label, "feature": f, **{k: v[k] for k in ("beta_std", "p")},
                       "ci95_low": v["ci95"][0], "ci95_high": v["ci95"][1]} for f, v in res["coeficientes_M3"].items()]
        report["outcomes"][label] = res
        print(f"{label:26s} {res['n']:6d} {res['r2_M0']:6.3f} {res['delta_r2_menciones']:+9.4f} {res['delta_r2_postura']:+9.4f} "
              f"{res['delta_r2_actividad']:+8.4f} {res.get('delta_r2_washing', np.nan):+8.4f} "
              f"{res['wald_postura']['p']:7.3f} {res['wald_actividad']['p']:6.3f} {res.get('wald_washing', {}).get('p', np.nan):6.3f}")
    for label, y in {**OUTCOMES, **EXTRA_OUTCOMES}.items():
        for vname, fe in (("sector_trimestre", "sector_quarter"), ("efectos_fijos_empresa", "ticker")):
            if vname == "sector_trimestre" and label in OUTCOMES:
                continue
            res, _ = nested(m, y, fe)
            report["robustez"].setdefault(vname, {})[label] = res
    print("\nROBUSTEZ — efectos fijos de empresa")
    for label, res in report["robustez"]["efectos_fijos_empresa"].items():
        print(f"{label:26s} {res['n']:6d} ΔR² menciones {res['delta_r2_menciones']:+.4f} postura {res['delta_r2_postura']:+.4f} "
              f"actividad {res['delta_r2_actividad']:+.4f} washing {res.get('delta_r2_washing', np.nan):+.4f}")
    out_json = L.results_path("shock", "incremental_signal.json")
    out_json.write_text(json.dumps(report, indent=2, default=float) + "\n")
    out_csv = L.results_path("shock", "incremental_signal_coefficients.csv")
    pd.DataFrame(coef_rows).to_csv(out_csv, index=False)
    print(f"\n-> {out_json}\n-> {out_csv}")


if __name__ == "__main__":
    main()
