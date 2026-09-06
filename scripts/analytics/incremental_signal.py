"""Señal incremental: ¿el contenido semántico de la divulgación de IA aporta
información más allá de los fundamentals y del volumen de IA?

Es el análisis central de la tesis (RQ4). Para cada outcome Y se ajustan
modelos anidados sobre el panel de TODAS las empresas-año con filings
(`firm_year_master_v2`, modo extensivo: intensidades con cero cuando la
empresa no habla de IA), ejercicios 2021-2025:

    M0  Y = a[sector×año] + b·fundamentals + e
    M1  M0 + θ·log(1 + frames de IA por 1.000 párrafos)           (volumen)
    M2  M1 + bloque semántico: log(1 + x por 1.000 párrafos) para
        realizado, despliegue, capacidad (inversión + infraestructura),
        riesgo, gobernanza, promocional, especificidad              (contenido)
    M3  M1 + dummies de segmento (`firm_year_segments`)            (arquetipos)

Lo que se reporta no son 80 coeficientes sino ΔR² = R²(M2) − R²(M1) y el R²
parcial del bloque semántico, (R²M2 − R²M1)/(1 − R²M1), con intervalo
bootstrap por empresa. Responde: ¿cuánto agregan las palabras una vez que se
observan sus fundamentals, sector, año y volumen de divulgación de IA?

Outcomes (cinco): beta, volatilidad idiosincrática (desv. est. de los residuos
del modelo de mercado sobre los 252 días previos al filing, anualizada), P/S,
R&D/ventas, crecimiento de ingresos t+1. La volatilidad total va como robustez.

Inferencia sobre el bloque semántico, además del ΔR²: R² ajustado de cada
modelo (siete regresores más nunca bajan el R² crudo), test conjunto de Wald
con SE cluster por empresa (H0: los siete coeficientes son cero) y test de
permutación del ΔR² permutando las siete features entre empresas DENTRO de
cada celda sector×año. Robustez de composición: el bloque semántico como
shares del total de frames (frames_k / frames de IA), encogidos hacia la media
del corpus con el mismo prior empírico-Bayes que los segmentos, para separar
"cómo se reparte" de "cuánto"; sin frames, los shares quedan en el prior. Fundamentals: log(market cap),
margen bruto, margen operativo, rotación de activos (R&D, capex y deuda/equity
quedan fuera por cobertura XBRL; R&D es outcome). Winsorización 1/99 de todo
lo financiero.

Robustez, la misma para todos los outcomes: sólo 10-K en el canal de texto,
excluyendo SIC 35/36/48/73 (IT y comunicaciones), efectos fijos de empresa en
vez de sector×año (qué parte es transversal y qué parte within-firm).

Salida: `incremental_signal.json` y una tabla por outcome en consola.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_intensity import document_table, aggregate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
YEARS = (2021, 2022, 2023, 2024, 2025)
OUTCOMES = {"beta": "beta", "volatilidad_idiosincratica": "idio_vol_252d", "price_to_sales": "ps_ratio",
            "rd_sobre_ventas": "rd_intensity", "crecimiento_ingresos_t1": "next_revenue_yoy"}
EXTRA_OUTCOMES = {"volatilidad_total_60d": "vol_pre_60d"}    # robustez, no cabecera
# Sin R&D ni capex entre los fundamentals: su cobertura XBRL (45% y 87%) dejaría
# el panel en un quinto; R&D es además uno de los outcomes.
FUNDAMENTALS = ["log_market_cap", "gross_margin", "operating_margin", "asset_turnover"]
SEMANTIC = {"realizado": "realized_per_1k", "despliegue": "deployed_per_1k", "capacidad": "capability_per_1k",
            "riesgo": "risk_per_1k", "gobernanza": "gov_per_1k", "promocional": "promo_per_1k", "especificidad": "spec_per_1k"}
IT_COMM_SIC2 = {"35", "36", "48", "73"}
SEED = 42


def winsor(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def load(con) -> pd.DataFrame:
    m = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")
    m = m[m["year"].isin(YEARS)].copy()
    m["capability_per_1k"] = m["ai_investment_per_1k"] + m["ai_infrastructure_per_1k"]
    seg = pd.read_parquet(OUT_DIR / "firm_year_segments.parquet")[["ticker", "year", "segmento"]]
    m = m.merge(seg, on=["ticker", "year"], how="left")
    m["segmento"] = m["segmento"].fillna("sin_ia")
    m["log_market_cap"] = np.log(m["market_cap"])
    for c in ["gross_margin", "operating_margin", "debt_to_equity", "asset_turnover", "capex_intensity", "rd_intensity",
              "beta", "idio_vol_252d", "vol_pre_60d", "ps_ratio", "next_revenue_yoy"]:
        m[c] = winsor(m[c])
    # shares semánticos encogidos (composición, independiente de la cantidad)
    counts = {"realizado": "n_realized", "despliegue": "n_deployed", "capacidad": None, "riesgo": "n_risk",
              "gobernanza": "n_gov", "promocional": "n_promo", "especificidad": "n_spec"}
    m["n_capacidad"] = m["n_ai_investment"] + m["n_ai_infrastructure"]
    for name, col in counts.items():
        col = col or "n_capacidad"
        raw = np.where(m["n_frames"] > 0, m[col] / m["n_frames"], np.nan)
        s = pd.Series(raw, index=m.index)
        mean, var = float(s.mean()), float(s.var(ddof=1))
        strength = max(mean * (1 - mean) / var - 1, 1e-6) if 0 < mean < 1 and var > 0 else 1.0
        alpha, beta_ = mean * strength, (1 - mean) * strength
        m[f"share_{name}"] = (s.fillna(0) * m["n_frames"] + alpha) / (m["n_frames"] + alpha + beta_)
    m["sector_year"] = m["sic2"].astype(str) + "_" + m["year"].astype(str)
    # texto sólo 10-K, para la robustez
    docs = document_table(con)
    k = aggregate(docs[docs["form"] == "10-K"].assign(year=lambda d: d["fecha"].dt.year), ["ticker", "year"])
    k["capability_per_1k"] = k["ai_investment_per_1k"] + k["ai_infrastructure_per_1k"]
    cols = ["frames_per_1k"] + list(SEMANTIC.values())
    m = m.merge(k[["ticker", "year"] + cols].rename(columns={c: f"{c}__10k" for c in cols}), on=["ticker", "year"], how="left")
    return m


def design(d: pd.DataFrame, y: str, block: str, fe: str, suffix: str = "") -> tuple[np.ndarray, np.ndarray, list[str]]:
    fund = [c for c in FUNDAMENTALS if c != y]
    X = d[fund].copy()
    if block in ("M1", "M2", "M3"):
        X["volumen"] = np.log1p(d["frames_per_1k" if suffix == "__share" else f"frames_per_1k{suffix}"])
    if block == "M2":
        for name, col in SEMANTIC.items():
            X[name] = d[f"share_{name}"] if suffix == "__share" else np.log1p(d[f"{col}{suffix}"])
    if block == "M3":
        for s in sorted(d["segmento"].unique()):
            if s != "sin_ia":
                X[f"seg_{s}"] = (d["segmento"] == s).astype(float)
    # efectos fijos por demeaning dentro del grupo
    g = d[fe].to_numpy()
    Xd = X - X.groupby(g).transform("mean")
    yd = d[y] - d[y].groupby(g).transform("mean")
    return Xd.to_numpy(dtype=float), yd.to_numpy(dtype=float), list(X.columns)


def r2(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    tss = float(np.sum(y ** 2))
    return 1 - float(np.sum(resid ** 2)) / tss if tss > 0 else np.nan


def adj_r2(X: np.ndarray, y: np.ndarray, n_groups: int) -> float:
    """R² ajustado contando los efectos fijos absorbidos (n_groups) como parámetros."""
    n, k = X.shape
    r = r2(X, y)
    dof = n - k - n_groups
    return 1 - (1 - r) * (n - 1) / dof if dof > 0 else np.nan


def wald_semantic(d: pd.DataFrame, y: str, fe: str, suffix: str) -> dict:
    """H0: los siete coeficientes del bloque semántico son cero, en M2, con
    covarianza cluster por empresa."""
    import statsmodels.api as sm
    X, yy, names = design(d, y, "M2", fe, suffix)
    res = sm.OLS(yy, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
    idx = [i for i, n in enumerate(names) if n in SEMANTIC]
    R = np.zeros((len(idx), len(names)))
    for r_, i in enumerate(idx): R[r_, i] = 1.0
    ft = res.wald_test(R, use_f=True, scalar=True)
    return {"F": float(ft.statistic), "p": float(ft.pvalue), "df": len(idx)}


def permutation_semantic(d: pd.DataFrame, y: str, fe: str, suffix: str, observed: float, reps: int, seed: int = SEED) -> dict:
    """ΔR² nulo permutando las siete features semánticas entre empresas-año
    DENTRO de cada celda sector×año (o empresa, si fe=ticker): preserva la
    estructura de sector, año y volumen; destruye sólo la asociación entre
    contenido y outcome."""
    rng = np.random.default_rng(seed)
    cols = [f"share_{n}" for n in SEMANTIC] if suffix == "__share" else [f"{c}{suffix}" for c in SEMANTIC.values()]
    groups = d.groupby(fe).indices
    null = []
    for _ in range(reps):
        s = d.copy()
        block = s[cols].to_numpy().copy()
        for idx in groups.values():
            if len(idx) > 1:
                block[idx] = block[rng.permutation(idx)]
        s[cols] = block
        X1, yy, _ = design(s, y, "M1", fe, suffix); X2, _, _ = design(s, y, "M2", fe, suffix)
        null.append(r2(X2, yy) - r2(X1, yy))
    null = np.array(null)
    return {"p": float((np.sum(null >= observed) + 1) / (len(null) + 1)), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "reps": int(len(null))}


def nested(d: pd.DataFrame, y: str, fe: str = "sector_year", suffix: str = "") -> dict:
    if suffix == "__share":
        need = [y] + [c for c in FUNDAMENTALS if c != y] + ["frames_per_1k"] + [f"share_{n}" for n in SEMANTIC] + [fe, "ticker", "segmento"]
    else:
        need = [y] + [c for c in FUNDAMENTALS if c != y] + [f"frames_per_1k{suffix}"] + [f"{c}{suffix}" for c in SEMANTIC.values()] + [fe, "ticker", "segmento"]
    d = d.dropna(subset=need)
    d = d[d.groupby(fe)[y].transform("size") >= 2]
    out = {"n": int(len(d)), "n_firms": int(d["ticker"].nunique())}
    n_groups = int(d[fe].nunique())
    for block in ("M0", "M1", "M2", "M3"):
        X, yy, _ = design(d, y, block, fe, suffix)
        out[f"r2_{block}"] = r2(X, yy)
        out[f"adj_r2_{block}"] = adj_r2(X, yy, n_groups)
    out["delta_adj_r2_semantica"] = out["adj_r2_M2"] - out["adj_r2_M1"]
    out["delta_r2_semantica"] = out["r2_M2"] - out["r2_M1"]
    out["partial_r2_semantica"] = out["delta_r2_semantica"] / (1 - out["r2_M1"]) if out["r2_M1"] < 1 else np.nan
    out["delta_r2_volumen"] = out["r2_M1"] - out["r2_M0"]
    out["delta_r2_segmentos"] = out["r2_M3"] - out["r2_M1"]
    return out, d


def bootstrap(d: pd.DataFrame, y: str, fe: str, suffix: str, reps: int, seed: int = SEED) -> dict:
    """Bootstrap por empresa (cluster) de ΔR² y R² parcial del bloque semántico."""
    rng = np.random.default_rng(seed)
    firms = d["ticker"].unique(); groups = d.groupby("ticker").indices
    deltas, partials = [], []
    for _ in range(reps):
        pick = rng.choice(firms, size=len(firms), replace=True)
        idx = np.concatenate([groups[f] for f in pick])
        s = d.iloc[idx].reset_index(drop=True)
        try:
            X1, yy, _ = design(s, y, "M1", fe, suffix); X2, _, _ = design(s, y, "M2", fe, suffix)
            r1, r2_ = r2(X1, yy), r2(X2, yy)
            deltas.append(r2_ - r1); partials.append((r2_ - r1) / (1 - r1))
        except Exception:
            continue
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"delta_r2_ci95": q(deltas), "partial_r2_ci95": q(partials), "reps": len(deltas)}


def coefficients(d: pd.DataFrame, y: str, fe: str, suffix: str) -> dict:
    """Coeficientes estandarizados del bloque semántico en M2, SE cluster por
    empresa — para saber QUÉ dimensión aporta, no sólo cuánto."""
    import statsmodels.api as sm
    X, yy, names = design(d, y, "M2", fe, suffix)
    sd = X.std(axis=0); sd[sd == 0] = 1
    Xs = X / sd; ys = yy / (yy.std() or 1)
    res = sm.OLS(ys, Xs).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
    return {n: {"beta_std": float(b), "p": float(p)} for n, b, p in zip(names, res.params, res.pvalues) if n in SEMANTIC or n == "volumen"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    con = duckdb.connect(str(args.database), read_only=True)
    try:
        m = load(con)
    finally:
        con.close()
    print(f"panel: {len(m):,} empresas-año {YEARS[0]}-{YEARS[-1]}, {m.ticker.nunique()} empresas, "
          f"{m.any_ai.mean()*100:.0f}% con algún frame de IA\n")
    report = {"years": YEARS, "outcomes": {}}
    hdr = (f"{'outcome':26s} {'n':>5s} {'R²M0':>6s} {'R²M1':>6s} {'R²M2':>6s} {'R²M3':>6s} {'ΔR² sem':>8s} {'IC95':>16s} "
           f"{'ΔR²aj sem':>9s} {'R²parc':>7s} {'Wald F':>7s} {'p':>6s} {'p perm':>7s}")
    print("PRINCIPAL — sector×año, todos los filings"); print(hdr)
    for label, y in OUTCOMES.items():
        res, d = nested(m, y)
        boot = bootstrap(d, y, "sector_year", "", args.bootstrap)
        coefs = coefficients(d, y, "sector_year", "")
        wald = wald_semantic(d, y, "sector_year", "")
        perm = permutation_semantic(d, y, "sector_year", "", res["delta_r2_semantica"], args.permutations)
        res.update(boot); res["coeficientes_M2"] = coefs; res["wald_semantica"] = wald; res["permutacion_semantica"] = perm
        ci = boot["delta_r2_ci95"]
        print(f"{label:26s} {res['n']:5d} {res['r2_M0']:6.3f} {res['r2_M1']:6.3f} {res['r2_M2']:6.3f} {res['r2_M3']:6.3f} "
              f"{res['delta_r2_semantica']:+8.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {res['delta_adj_r2_semantica']:+9.4f} "
              f"{res['partial_r2_semantica']:7.4f} {wald['F']:7.2f} {wald['p']:6.3f} {perm['p']:7.3f}")
        print(f"    R² ajustado M0/M1/M2/M3: {res['adj_r2_M0']:.3f} / {res['adj_r2_M1']:.3f} / {res['adj_r2_M2']:.3f} / {res['adj_r2_M3']:.3f} "
              f"| ΔR² volumen {res['delta_r2_volumen']:+.4f} | ΔR² segmentos {res['delta_r2_segmentos']:+.4f} | ΔR² nulo (perm) media {perm['null_mean']:+.4f}, p95 {perm['null_p95']:+.4f}")
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]["beta_std"]))[:3]
        print("    " + " | ".join(f"{k}: {v['beta_std']:+.3f} (p={v['p']:.2f})" for k, v in top))
        report["outcomes"][label] = res
    # robustez
    variants = {
        "shares_composicion": dict(fe="sector_year", suffix="__share", subset=None),
        "solo_10k": dict(fe="sector_year", suffix="__10k", subset=None),
        "sin_it_comunicaciones": dict(fe="sector_year", suffix="", subset=lambda d: ~d["sic2"].astype(str).isin(IT_COMM_SIC2)),
        "efectos_fijos_empresa": dict(fe="ticker", suffix="", subset=None),
    }
    report["robustez"] = {}
    for vname, v in variants.items():
        print(f"\nROBUSTEZ — {vname}"); print(f"{'outcome':26s} {'n':>5s} {'R²M1':>6s} {'R²M2':>6s} {'ΔR² sem':>8s} {'ΔR²aj':>7s} {'R²parc':>7s} {'Wald p':>7s}")
        report["robustez"][vname] = {}
        outcomes = dict(OUTCOMES); outcomes.update(EXTRA_OUTCOMES if vname == "solo_10k" else {})
        for label, y in outcomes.items():
            dd = m if v["subset"] is None else m[v["subset"](m)]
            try:
                res, d = nested(dd, y, v["fe"], v["suffix"]); wald = wald_semantic(d, y, v["fe"], v["suffix"])
            except Exception as e:  # noqa: BLE001
                print(f"{label:26s} sin muestra ({e})"); continue
            print(f"{label:26s} {res['n']:5d} {res['r2_M1']:6.3f} {res['r2_M2']:6.3f} {res['delta_r2_semantica']:+8.4f} {res['delta_adj_r2_semantica']:+7.4f} {res['partial_r2_semantica']:7.4f} {wald['p']:7.3f}")
            report["robustez"][vname][label] = {**{k: res[k] for k in ("n", "r2_M0", "r2_M1", "r2_M2", "r2_M3", "adj_r2_M1", "adj_r2_M2", "delta_r2_semantica", "delta_adj_r2_semantica", "partial_r2_semantica", "delta_r2_volumen", "delta_r2_segmentos")}, "wald": wald}
    print("\nVOLATILIDAD TOTAL (robustez del outcome, especificación principal)")
    for label, y in EXTRA_OUTCOMES.items():
        res, d = nested(m, y); wald = wald_semantic(d, y, "sector_year", "")
        print(f"{label:26s} {res['n']:5d} {res['r2_M1']:6.3f} {res['r2_M2']:6.3f} {res['delta_r2_semantica']:+8.4f} {res['delta_adj_r2_semantica']:+7.4f} {res['partial_r2_semantica']:7.4f} {wald['p']:7.3f}")
        report["robustez"].setdefault("volatilidad_total", {})[label] = {**{k: res[k] for k in ("n", "r2_M1", "r2_M2", "delta_r2_semantica", "delta_adj_r2_semantica", "partial_r2_semantica")}, "wald": wald}
    (OUT_DIR / "incremental_signal.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR / 'incremental_signal.json'}")


if __name__ == "__main__":
    main()
