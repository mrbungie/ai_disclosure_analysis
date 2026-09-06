"""Reproduce, desde los parquets, las tablas numéricas de los documentos de
cruce financiero: `05_senal_incremental.md`,
`04_perfiles_economicos.md`, `apendice/correlaciones_fdr_permutacion.md`
y `04_perfiles_economicos.md`.

MODO DE ANÁLISIS FINAL: margen extensivo. El panel es `firm_year_master_v2`
con TODAS las empresas-año que tienen filings (`build_firm_panels.py`), y las
variables de texto son intensidades por 1.000 párrafos con cero cuando la
empresa no habla de IA (`ai_intensity.py`). Nada condiciona a hablar de IA.

Existe porque esos cuatro documentos reportaban cifras calculadas en consultas
ad hoc de sesión: el dato quedaba, la consulta no. Cualquier cambio de
población obligaba a reconstruir las tablas de memoria. Acá cada número de
esos documentos sale de una función con nombre, y `--json` lo deja en un
archivo para diffear entre corridas.

Cubre:
  §1 talk vs. walk        correlaciones declarado(t) -> real(t+1), crudas,
                          dentro de sector-año y dentro de empresa (efectos
                          fijos por demeaning y por primeras diferencias)
  §2 FDR                  las 11 correlaciones del cruce contable/mercado con
                          Benjamini-Hochberg, más el test de permutación
  §3 perfiles             ratios, valuación, beta/volatilidad y CAR por
                          arquetipo de voz y por cluster de comportamiento
  §4 ROIC-WACC            creación de valor por segmento, con y sin control
                          sectorial (mediana del propio SIC-2 restada)

Determinístico, sin LLM. Requiere haber corrido antes `make analytics`.

Uso:
    uv run python scripts/analytics/report_crosscheck_stats.py
    uv run python scripts/analytics/report_crosscheck_stats.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"
SEED = 42
GROWTH_CLIP = 3.0          # |crecimiento YoY| > 300% se descarta como outlier
PERMUTATIONS = 2000

# Los 11 pares que apendice/correlaciones_fdr_permutacion.md somete a FDR: comportamiento declarado en t contra
# el resultado real, y retórica contra reacción de mercado.
FDR_PAIRS = [
    ("behavior_share_ai_infrastructure", "next_capex_yoy"),
    ("behavior_share_revenue_outcome", "next_revenue_yoy"),
    ("n_frames", "ret_m1_p5"),
    ("promotional_rate", "car_m1_p5"),
    ("specificity_index", "ret_m1_p5"),
    ("specificity_index", "car_m1_p5"),
    ("behavior_share_ai_investment", "next_rd_expense_yoy"),
    ("promotional_rate", "ret_m1_p5"),
    ("behavior_share_ai_investment", "next_capex_yoy"),
    ("behavior_share_ai_infrastructure", "next_rd_expense_yoy"),
    ("behavior_share_cost_outcome", "next_sga_expense_yoy"),
]
# Las letras salen de `build_firm_clusters.py`, que las asigna por PERFIL
# (D reclama promotional_rate, C quantified_rate, A risk_share, B es el
# residuo) y no por el id que devuelve sklearn — así "D" sigue significando
# "líder vocal" después de un re-ajuste.


def _clean(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    data = frame[[x, y]].dropna()
    if y.startswith("next_"):
        data = data[data[y].abs() <= GROWTH_CLIP]
    return data


def correlation(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    """r de Pearson con su p-valor exacto (t de Student), sin scipy."""
    data = _clean(frame, x, y)
    n = len(data)
    if n < 10 or data[x].std() == 0 or data[y].std() == 0:
        return np.nan, np.nan, n
    r = float(data[x].corr(data[y]))
    from scipy import stats
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
    p = float(2 * stats.t.sf(abs(t), n - 2))
    return r, p, n


def within_group_correlation(frame: pd.DataFrame, x: str, y: str,
                             group: list[str]) -> tuple[float, int]:
    """Correlación después de restarle a cada variable la media de su grupo.

    Con `group=['sic2','year']` responde "dentro del mismo sector y año";
    con `group=['ticker']` es exactamente un modelo de efectos fijos de
    empresa — el control más estricto contra confusores fijos."""
    data = frame[[x, y] + group].dropna()
    if y.startswith("next_"):
        data = data[data[y].abs() <= GROWTH_CLIP]
    if data.empty:
        return np.nan, 0
    for column in (x, y):
        data[column] = data[column] - data.groupby(group)[column].transform("mean")
    return float(data[x].corr(data[y])), len(data)


def first_difference_correlation(frame: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    """Primeras diferencias año a año dentro de cada empresa."""
    data = frame[["ticker", "year", x, y]].dropna().sort_values(["ticker", "year"])
    if y.startswith("next_"):
        data = data[data[y].abs() <= GROWTH_CLIP]
    diffs = data.groupby("ticker")[[x, y]].diff().dropna()
    if len(diffs) < 10:
        return np.nan, len(diffs)
    return float(diffs[x].corr(diffs[y])), len(diffs)


def permutation_p(frame: pd.DataFrame, x: str, y: str,
                  permutations: int = PERMUTATIONS) -> float:
    """Placebo: ¿qué tan seguido el azar produce un |r| así de grande?"""
    data = _clean(frame, x, y)
    if len(data) < 10:
        return np.nan
    observed = abs(data[x].corr(data[y]))
    rng = np.random.default_rng(SEED)
    values = data[x].to_numpy()
    target = data[y].to_numpy()
    hits = sum(abs(np.corrcoef(rng.permutation(values), target)[0, 1]) >= observed
               for _ in range(permutations))
    return (hits + 1) / (permutations + 1)


def benjamini_hochberg(pvalues: list[float], q: float = 0.05) -> list[bool]:
    order = np.argsort(pvalues)
    ranked = np.asarray(pvalues)[order]
    m = len(pvalues)
    passing = ranked <= (np.arange(1, m + 1) / m) * q
    keep = np.zeros(m, dtype=bool)
    if passing.any():
        keep[order[: int(np.max(np.flatnonzero(passing))) + 1]] = True
    return keep.tolist()


def sector_demeaned(frame: pd.DataFrame, column: str) -> pd.Series:
    """Valor menos la MEDIANA de su propio SIC-2: separa 'esta empresa se
    comporta distinto' de 'esta empresa está en otra industria'.

    Se reporta después con mediana Y media porque la mediana de la desviación
    cae exactamente en 0 para cualquier grupo grande y repartido entre muchos
    sectores — no porque el grupo esté en la mediana de su industria, sino
    porque la distribución de desviaciones está centrada en 0 por
    construcción. La media es la que distingue segmentos ahí."""
    return frame[column] - frame.groupby("sic2")[column].transform("median")


INTENSITY = {  # la misma pregunta, medida en intensidad por 1.000 párrafos con ceros
    "behavior_share_ai_infrastructure": "ai_infrastructure_per_1k",
    "behavior_share_revenue_outcome": "revenue_outcome_per_1k",
    "behavior_share_ai_investment": "ai_investment_per_1k",
    "behavior_share_cost_outcome": "cost_outcome_per_1k",
    "n_frames": "frames_per_1k",
    "promotional_rate": "promo_per_1k",
    "specificity_index": "spec_per_1k",
}
LEVELS = ["cero", "bajo", "medio", "alto"]
PROFILE_COLUMNS = ["gross_margin", "operating_margin", "net_margin", "roa", "roe", "rd_intensity",
                   "beta", "vol_pre_60d", "momentum_12_1", "pe_ratio", "ps_ratio", "pb_ratio",
                   "market_cap", "car_m1_p5", "ret_m1_p5", "next_revenue_yoy"]


def intensity_level(frame: pd.DataFrame) -> pd.Series:
    """cero = ningún frame de IA en el año; bajo/medio/alto = terciles de
    frames por 1.000 párrafos entre las empresas-año que sí hablan."""
    level = pd.Series("cero", index=frame.index, dtype=object)
    talk = frame["frames_per_1k"] > 0
    level[talk] = pd.qcut(frame.loc[talk, "frames_per_1k"].rank(method="first"), 3,
                          labels=["bajo", "medio", "alto"]).astype(str)
    return level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-dir", type=Path, default=CLUSTERS)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    master = pd.read_parquet(args.clusters_dir / "firm_year_master_v2.parquet")
    full = pd.read_parquet(args.clusters_dir / "firm_year_full_crosscheck.parquet")
    roic = pd.read_parquet(args.clusters_dir / "firm_year_roic_wacc.parquet")
    if "ret_m1_p5" not in master.columns:
        master = master.merge(full[["ticker", "year", "ret_m1_p5"]], on=["ticker", "year"], how="left")
    master["nivel_ia"] = intensity_level(master)
    report: dict = {"n_firm_years": int(len(master)), "n_firms": int(master["ticker"].nunique()),
                    "share_any_ai": float(master["any_ai"].mean()),
                    "n_in_conditional_panel": int(master["in_text_panel"].sum()) if "in_text_panel" in master else None}
    print(f"panel: {len(master):,} empresas-año con filings, {master['ticker'].nunique():,} empresas, "
          f"{master['any_ai'].mean()*100:.0f}% con algún frame de IA\n")

    print("=" * 78); print("§1  TALK VS. WALK — revenue_outcome por 1.000 párrafos (con ceros) contra revenue real en t+1"); print("=" * 78)
    x, y = "revenue_outcome_per_1k", "next_revenue_yoy"
    r_raw, p_raw, n_raw = correlation(master, x, y)
    r_sector, n_sector = within_group_correlation(master, x, y, ["sic2", "year"])
    r_firm, n_firm = within_group_correlation(master, x, y, ["ticker"])
    r_diff, n_diff = first_difference_correlation(master, x, y)
    p_perm_raw = permutation_p(master, x, y)
    sector_frame = master.copy()
    for column in (x, y):
        sector_frame[column] = sector_frame[column] - sector_frame.groupby(["sic2", "year"])[column].transform("mean")
    p_perm_sector = permutation_p(sector_frame, x, y)
    talk = master[master["any_ai"] == 1]
    r_talk, _, n_talk = correlation(talk, x, y)
    rows = [("cruda", r_raw, n_raw, p_perm_raw), ("dentro de sector-año", r_sector, n_sector, p_perm_sector),
            ("dentro de empresa (demeaning = efectos fijos)", r_firm, n_firm, np.nan),
            ("primeras diferencias dentro de empresa", r_diff, n_diff, np.nan),
            ("cruda, sólo empresas-año que hablan de IA", r_talk, n_talk, np.nan)]
    print(f"{'especificación':46s} {'r':>7s} {'n':>6s} {'p(perm)':>9s}")
    for label, r, n, p in rows:
        print(f"{label:46s} {r:7.3f} {n:6d} {'' if np.isnan(p) else f'{p:9.3f}'}")
    report["talk_vs_walk"] = [{"spec": s, "r": r, "n": n, "p_perm": p} for s, r, n, p in rows]

    print("\n" + "=" * 78); print("§2  LAS 11 CORRELACIONES DEL CRUCE, EN INTENSIDAD, CON FDR (Benjamini-Hochberg 5%)"); print("=" * 78)
    results = []
    for x0, y in FDR_PAIRS:
        x = INTENSITY.get(x0, x0)
        if x not in master.columns or y not in master.columns:
            continue
        r, p, n = correlation(master, x, y)
        results.append({"x": x, "y": y, "r": r, "p": p, "n": n})
    results.sort(key=lambda row: (np.inf if np.isnan(row["p"]) else row["p"]))
    flags = benjamini_hochberg([row["p"] for row in results]); m = len(results)
    print(f"{'par':62s} {'r':>7s} {'p':>8s} {'BH':>7s} {'pasa':>5s}")
    for rank, (row, keep) in enumerate(zip(results, flags), start=1):
        row["passes_fdr"] = bool(keep)
        print(f"{row['x'] + ' ~ ' + row['y']:62s} {row['r']:7.3f} {row['p']:8.4f} {rank / m * 0.05:7.4f} {'sí' if keep else 'no':>5s}")
    report["fdr"] = results
    print("\n§2b  ¿Hablar de IA en absoluto predice algo? (any_ai y frames_per_1k)")
    extra = []
    for x in ("any_ai", "frames_per_1k"):
        for y in ("next_revenue_yoy", "next_capex_yoy", "next_rd_expense_yoy", "ret_m1_p5", "car_m1_p5"):
            r, p, n = correlation(master, x, y); extra.append({"x": x, "y": y, "r": r, "p": p, "n": n})
            print(f"  {x:14s} ~ {y:20s} r={r:+.3f} p={p:.4f} n={n}")
    report["any_ai"] = extra

    print("\n" + "=" * 78); print("§3  PERFIL FINANCIERO POR NIVEL DE INTENSIDAD DE IA (cero / terciles de frames por 1.000 párrafos)"); print("=" * 78)
    available = [c for c in PROFILE_COLUMNS if c in master.columns]
    by_level = master.groupby("nivel_ia")[available].median().reindex(LEVELS)
    by_level["n"] = master.groupby("nivel_ia").size().reindex(LEVELS)
    by_level["frames_per_1k_mediana"] = master.groupby("nivel_ia")["frames_per_1k"].median().reindex(LEVELS)
    print(by_level.round(3).T.to_string())
    report["profile_by_level"] = json.loads(by_level.to_json(orient="index"))

    print("\n" + "=" * 78); print("§4  ROIC - WACC POR NIVEL DE INTENSIDAD DE IA (con y sin control sectorial)"); print("=" * 78)
    value = master[["ticker", "year", "nivel_ia"]].merge(
        roic[["ticker", "year", "roic", "wacc", "roic_minus_wacc", "sic2", "erp"]], on=["ticker", "year"])
    usable = value.dropna(subset=["roic_minus_wacc"]).copy()
    usable["spread_vs_sector"] = sector_demeaned(usable, "roic_minus_wacc")
    print(f"ERP usado: {float(value['erp'].dropna().iloc[0]):.4f} | {len(usable):,} empresas-año con ROIC y WACC")
    table = usable.groupby("nivel_ia").agg(
        roic=("roic", "median"), wacc=("wacc", "median"), spread=("roic_minus_wacc", "median"),
        spread_vs_sector_med=("spread_vs_sector", "median"), spread_vs_sector_avg=("spread_vs_sector", "mean"),
        pct_positive=("roic_minus_wacc", lambda s: float((s > 0).mean())), n=("roic_minus_wacc", "size")).reindex(LEVELS)
    print(table.round(4).to_string())
    report["roic_wacc_by_level"] = json.loads(table.to_json(orient="index"))

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=float) + "\n"); print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
