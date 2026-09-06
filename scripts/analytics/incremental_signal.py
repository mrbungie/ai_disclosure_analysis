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
sabe qué empresa es y cuánto habla de IA?

Outcomes (cinco): beta, volatilidad realizada pre-filing (proxy de
idiosincrática: no hay descomposición en `firm_year_market_factors`), P/S,
R&D/ventas, crecimiento de ingresos t+1. Fundamentals: log(market cap),
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
OUTCOMES = {"beta": "beta", "volatilidad_pre_60d": "vol_pre_60d", "price_to_sales": "ps_ratio",
            "rd_sobre_ventas": "rd_intensity", "crecimiento_ingresos_t1": "next_revenue_yoy"}
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
              "beta", "vol_pre_60d", "ps_ratio", "next_revenue_yoy"]:
        m[c] = winsor(m[c])
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
        X["volumen"] = np.log1p(d[f"frames_per_1k{suffix}"])
    if block == "M2":
        for name, col in SEMANTIC.items():
            X[name] = np.log1p(d[f"{col}{suffix}"])
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


def nested(d: pd.DataFrame, y: str, fe: str = "sector_year", suffix: str = "") -> dict:
    need = [y] + [c for c in FUNDAMENTALS if c != y] + [f"frames_per_1k{suffix}"] + [f"{c}{suffix}" for c in SEMANTIC.values()] + [fe, "ticker", "segmento"]
    d = d.dropna(subset=need)
    d = d[d.groupby(fe)[y].transform("size") >= 2]
    out = {"n": int(len(d)), "n_firms": int(d["ticker"].nunique())}
    for block in ("M0", "M1", "M2", "M3"):
        X, yy, _ = design(d, y, block, fe, suffix)
        out[f"r2_{block}"] = r2(X, yy)
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
    args = parser.parse_args()
    con = duckdb.connect(str(args.database), read_only=True)
    try:
        m = load(con)
    finally:
        con.close()
    print(f"panel: {len(m):,} empresas-año {YEARS[0]}-{YEARS[-1]}, {m.ticker.nunique()} empresas, "
          f"{m.any_ai.mean()*100:.0f}% con algún frame de IA\n")
    report = {"years": YEARS, "outcomes": {}}
    hdr = f"{'outcome':26s} {'n':>5s} {'R²M0':>6s} {'R²M1':>6s} {'R²M2':>6s} {'R²M3':>6s} {'ΔR² vol':>8s} {'ΔR² sem':>8s} {'IC95':>16s} {'R²parc':>7s} {'ΔR² seg':>8s}"
    print("PRINCIPAL — sector×año, todos los filings"); print(hdr)
    for label, y in OUTCOMES.items():
        res, d = nested(m, y)
        boot = bootstrap(d, y, "sector_year", "", args.bootstrap)
        coefs = coefficients(d, y, "sector_year", "")
        res.update(boot); res["coeficientes_M2"] = coefs
        ci = boot["delta_r2_ci95"]
        print(f"{label:26s} {res['n']:5d} {res['r2_M0']:6.3f} {res['r2_M1']:6.3f} {res['r2_M2']:6.3f} {res['r2_M3']:6.3f} "
              f"{res['delta_r2_volumen']:+8.4f} {res['delta_r2_semantica']:+8.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {res['partial_r2_semantica']:7.4f} {res['delta_r2_segmentos']:+8.4f}")
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]["beta_std"]))[:3]
        print("    " + " | ".join(f"{k}: {v['beta_std']:+.3f} (p={v['p']:.2f})" for k, v in top))
        report["outcomes"][label] = res
    # robustez
    variants = {
        "solo_10k": dict(fe="sector_year", suffix="__10k", subset=None),
        "sin_it_comunicaciones": dict(fe="sector_year", suffix="", subset=lambda d: ~d["sic2"].astype(str).isin(IT_COMM_SIC2)),
        "efectos_fijos_empresa": dict(fe="ticker", suffix="", subset=None),
    }
    report["robustez"] = {}
    for vname, v in variants.items():
        print(f"\nROBUSTEZ — {vname}"); print(f"{'outcome':26s} {'n':>5s} {'R²M1':>6s} {'R²M2':>6s} {'ΔR² sem':>8s} {'R²parc':>7s}")
        report["robustez"][vname] = {}
        for label, y in OUTCOMES.items():
            dd = m if v["subset"] is None else m[v["subset"](m)]
            try:
                res, _ = nested(dd, y, v["fe"], v["suffix"])
            except Exception as e:  # noqa: BLE001
                print(f"{label:26s} sin muestra ({e})"); continue
            print(f"{label:26s} {res['n']:5d} {res['r2_M1']:6.3f} {res['r2_M2']:6.3f} {res['delta_r2_semantica']:+8.4f} {res['partial_r2_semantica']:7.4f}")
            report["robustez"][vname][label] = {k: res[k] for k in ("n", "r2_M0", "r2_M1", "r2_M2", "r2_M3", "delta_r2_semantica", "partial_r2_semantica", "delta_r2_volumen", "delta_r2_segmentos")}
    (OUT_DIR / "incremental_signal.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR / 'incremental_signal.json'}")


if __name__ == "__main__":
    main()
