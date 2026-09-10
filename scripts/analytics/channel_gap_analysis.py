"""Brecha entre canales: la misma empresa, el mismo período, earnings call
contra filing SEC.

Es el diseño que `docs/pregunta_identificacion_sec.md` deja planteado y que
el caso Welltower motiva: la SEC no objetó el 10-K por promocional, objetó
que la call y el press release dijeran "industry-leading" sin respaldo
proporcional en el 10-K. El objeto es la BRECHA entre canales, no el nivel
en uno.

    y[i, canal, t] = a[i, t] + b · (post[t] × es_filing[canal]) + e

Con efectos fijos empresa × período y exactamente dos canales por celda,
el estimador es la diferencia dentro de la celda:

    gap[i, t] = y[i, call, t] − y[i, filing, t] = a_i + c · post[t] + e

Todo lo que es común a la empresa en ese período (el boom de IA, su
sector, su ciclo, cuánto habla de IA) se cancela en la resta.

PERÍODO = AÑO FISCAL, alineado por lo que el documento CUBRE, no por la
fecha en que se presenta. El 10-K de febrero de 2025 habla del ejercicio
2024 y las calls de 2024 discuten los trimestres de 2024: parearlos por
año calendario de la fecha mezcla ejercicios. Asignación:
  10-K, 10-Q       año fiscal de `period_end_date`
  earnings call    año fiscal del `document_id` (`TICKER_YYYYQn` = trimestre
                   fiscal discutido)
  8-K, DEF 14A     año fiscal en que se presentan (no cubren un período;
                   el proxy mezcla compensación pasada y gobernanza actual)
El año fiscal de una fecha se calcula con el mes de cierre de cada empresa
(mes de `period_end_date` de sus 10-K). Los filings con frames de IA son
anuales (el 10-Q casi nunca tiene frames), así que la unidad natural es el
año: por trimestre calendario sólo hay celda donde un 10-K coincide con
una call. `--period quarter` conserva esa versión, por fecha calendario.

`post` = año fiscal ≥ 2024. Una celda puede mezclar documentos anteriores
y posteriores al 2024-03-01 (p. ej. el 10-K de FY2023 presentado en febrero
y el proxy de abril): se guarda `share_post_docs` por celda y se reporta la
estimación sin las celdas mixtas como robustez. También se reporta la
estimación ponderada por frames (min de los dos canales): las tasas de
celdas chicas son ruidosas y una regresión sin pesos las trata igual que
las de 100 frames.

Se estima con efectos fijos de empresa sobre la brecha y errores
clusterizados por empresa; se reporta el event study por período (base =
primer período) y la descomposición por canal (¿se mueve la call o el
filing?). `post` = 2024 en adelante (escrutinio de la SEC, marzo 2024).

Salidas (`data/processed/clusters/`):
  channel_gap_cells.parquet      una fila por empresa-ejercicio con ≥1 documento por canal:
                                 intensidad por 1.000 palabras, con ceros (MODO FINAL)
  channel_gap_firm.parquet       brecha promedio por empresa (score por canal)
  channel_gap_analysis.json      estimaciones, event study, descriptivos

Determinístico, sin LLM. Requiere `duckdb/thesis.duckdb` con las tablas de
texto (`build_duckdb.py --with-text-tables`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
# Tres fuentes de transcripciones (ver ai_intensity.py para el detalle): la
# base de Hugging Face (2005-2025, no se actualiza) más dos rellenos de
# huecos que sí cubren 2026.
CALLS_MANIFESTS = [
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet",
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls_equibles.parquet",
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls_stockanalysis.parquet",
]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"


def _calls_manifest_sql() -> str:
    parts = [f"SELECT document_id, ticker, filing_date FROM read_parquet('{p}')" for p in CALLS_MANIFESTS if p.exists()]
    union = " UNION ALL BY NAME ".join(parts)
    return f"""
        SELECT document_id, ticker, TRY_CAST(filing_date AS DATE) AS filing_date FROM (
            SELECT *, row_number() OVER (PARTITION BY document_id ORDER BY 1) AS rn
            FROM ({union})
        ) WHERE rn = 1
    """
FILING_FORMS = ("10-K", "10-Q", "DEF 14A", "8-K")
MIN_FRAMES = 3
EVENT = {"fy": 2024, "quarter": pd.Period("2024Q2", freq="Q")}
EVENT_DATE = "2023-12-05"      # aviso de Gensler sobre AI-washing; el enforcement es del 2024-03-18
OUTCOMES = ["promotional_rate", "quantified_rate", "specificity_index", "realized_share",
            "hypothetical_share", "gov_share"]
NOTORIOUS = ["WELL", "NVDA", "PLTR", "TSLA", "ORCL", "GOOGL", "CRM", "MSFT", "META", "AMZN",
             "HPE", "ANET", "IBM", "PANW", "CRWD"]


def load_frames(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Frames de EE.UU. con empresa, fecha y canal. Los filings resuelven
    ticker/fecha por `filing_manifest` (10-K, DEF 14A, 8-K) y
    `filing_manifest_10q`; las calls por su manifiesto propio, cuyo
    `document_id` es el `accession_number` sintético `TICKER_YYYYQn`."""
    filings = con.execute(f"""
        WITH manifest AS (
            SELECT country_code, accession_number, ticker, filing_date, period_end_date FROM filing_manifest
            WHERE form_type != 'Earnings call transcript'
            UNION ALL
            SELECT country_code, accession_number, ticker, filing_date, period_end_date FROM filing_manifest_10q
        )
        SELECT 'filing' AS channel, f.form, m.ticker, m.filing_date AS fecha,
               TRY_CAST(m.period_end_date AS DATE) AS period_end, f.text_hash,
               f.frame_index, f.rhetoric_promotional, f.specificity_quantified_metric,
               f.specificity_business_process, f.specificity_product_or_system,
               f.specificity_vendor_or_partner, f.specificity_date_or_timeline,
               f.temporal, f.concepts
        FROM gold_ai_frames f
        JOIN manifest m USING (country_code, accession_number)
        WHERE f.country_code = 'us' AND f.has_frame AND f.form IN {FILING_FORMS}
          AND m.ticker IS NOT NULL AND m.filing_date IS NOT NULL
    """).df()
    calls = con.execute(f"""
        SELECT 'call' AS channel, f.form, m.ticker, CAST(m.filing_date AS DATE) AS fecha,
               NULL::DATE AS period_end, CAST(regexp_extract(m.document_id, '_([0-9]{{4}})Q', 1) AS INTEGER) AS call_fy,
               f.text_hash, f.frame_index, f.rhetoric_promotional, f.specificity_quantified_metric,
               f.specificity_business_process, f.specificity_product_or_system,
               f.specificity_vendor_or_partner, f.specificity_date_or_timeline,
               f.temporal, f.concepts
        FROM gold_ai_frames f
        JOIN (SELECT document_id, ticker, filing_date
              FROM ({_calls_manifest_sql()}) WHERE ticker IS NOT NULL) m
          ON m.document_id = f.accession_number
        WHERE f.country_code = 'us' AND f.has_frame AND f.form = 'Earnings call'
    """).df()
    frames = pd.concat([filings, calls], ignore_index=True)
    frames = frames.drop_duplicates(["channel", "ticker", "fecha", "text_hash", "frame_index"])
    frames["quarter"] = pd.to_datetime(frames["fecha"]).dt.to_period("Q")
    frames["year"] = frames["quarter"].dt.year
    # mes de cierre fiscal por empresa: el de sus 10-K (moda); diciembre si no hay
    fye = (frames[(frames.form == "10-K") & frames.period_end.notna()]
           .assign(m=lambda d: pd.to_datetime(d.period_end).dt.month)
           .groupby("ticker")["m"].agg(lambda s: int(s.mode().iloc[0])))
    frames["fye_month"] = frames["ticker"].map(fye).fillna(12).astype(int)
    def fiscal_year(dates: pd.Series, fye_month: pd.Series) -> pd.Series:
        d = pd.to_datetime(dates)
        return (d.dt.year + (d.dt.month > fye_month).astype(int)).astype("Int64")
    fy = pd.Series(pd.NA, index=frames.index, dtype="Int64")
    is_pe = frames.form.isin(["10-K", "10-Q"]) & frames.period_end.notna()
    fy[is_pe] = fiscal_year(frames.loc[is_pe, "period_end"], frames.loc[is_pe, "fye_month"])
    is_call = frames.channel == "call"
    fy[is_call] = frames.loc[is_call, "call_fy"].astype("Int64")
    rest = fy.isna()
    fy[rest] = fiscal_year(frames.loc[rest, "fecha"], frames.loc[rest, "fye_month"])
    frames["fy"] = fy.astype(int)
    frames["post_doc"] = (pd.to_datetime(frames["fecha"]) >= pd.Timestamp(EVENT_DATE)).astype(float)
    spec_cols = ["specificity_business_process", "specificity_product_or_system",
                 "specificity_vendor_or_partner", "specificity_quantified_metric",
                 "specificity_date_or_timeline"]
    frames["specificity"] = frames[spec_cols].astype(float).mean(axis=1)
    frames["is_gov"] = frames["concepts"].apply(
        lambda c: any(str(x).startswith("gov_") for x in (list(c) if c is not None else [])))
    return frames


def load_documents(con: duckdb.DuckDBPyConnection, frames: pd.DataFrame) -> pd.DataFrame:
    """Todos los documentos con su cantidad de párrafos, tengan o no frames de
    IA: una call que no menciona IA es una observación con cero, no una celda
    perdida. Mismo año fiscal que en `load_frames`."""
    docs = con.execute(rf"""
        WITH manifest AS (
            SELECT country_code, accession_number, ticker, filing_date, period_end_date, form_type AS form
            FROM filing_manifest WHERE form_type != 'Earnings call transcript'
            UNION ALL
            SELECT country_code, accession_number, ticker, filing_date, period_end_date, '10-Q' FROM filing_manifest_10q
        ), paras AS (
            SELECT country_code, accession_number, count(*) AS n_paragraphs,
                   sum(list_count(regexp_split_to_array(trim(paragraph_text), '\s+'))) AS n_words
            FROM paragraphs
            WHERE is_scorable GROUP BY 1, 2
        )
        SELECT 'filing' AS channel, m.form, m.ticker, m.filing_date AS fecha, TRY_CAST(m.period_end_date AS DATE) AS period_end,
               NULL::INTEGER AS call_fy, m.accession_number, p.n_paragraphs, p.n_words
        FROM manifest m JOIN paras p USING (country_code, accession_number)
        WHERE m.country_code = 'us' AND m.ticker IS NOT NULL AND m.filing_date IS NOT NULL AND m.form IN {FILING_FORMS}
        UNION ALL
        SELECT 'call', 'Earnings call', m.ticker, m.filing_date, NULL::DATE,
               CAST(regexp_extract(m.document_id, '_([0-9]{{4}})Q', 1) AS INTEGER), m.document_id, p.n_paragraphs, p.n_words
        FROM ({_calls_manifest_sql()}) m
        JOIN paras p ON p.accession_number = m.document_id AND p.country_code = 'us'
        WHERE m.ticker IS NOT NULL
    """).df()
    fye = frames.groupby("ticker")["fye_month"].first()
    docs["fye_month"] = docs["ticker"].map(fye).fillna(12).astype(int)
    d = pd.to_datetime(docs["fecha"]); pe = pd.to_datetime(docs["period_end"])
    fy_date = d.dt.year + (d.dt.month > docs["fye_month"]).astype(int)
    fy_pe = pe.dt.year + (pe.dt.month > docs["fye_month"]).astype(int)
    is_pe = docs["form"].isin(["10-K", "10-Q"]) & pe.notna()
    docs["fy"] = np.where(docs["channel"] == "call", docs["call_fy"], np.where(is_pe, fy_pe, fy_date)).astype(int)
    docs["post_doc"] = (d >= pd.Timestamp(EVENT_DATE)).astype(float)
    # frames por documento (0 si no tiene)
    per_doc = frames.groupby(["channel", "ticker", "fy"]).agg(
        n_frames=("text_hash", "size"), n_promo=("rhetoric_promotional", "sum"),
        n_quant=("specificity_quantified_metric", "sum"), n_gov=("is_gov", "sum")).reset_index()
    cell = docs.groupby(["ticker", "fy", "channel"]).agg(
        n_docs=("accession_number", "nunique"), n_paragraphs=("n_paragraphs", "sum"),
        n_words=("n_words", "sum"),
        share_post_docs=("post_doc", "mean")).reset_index()
    cell = cell.merge(per_doc, on=["channel", "ticker", "fy"], how="left").fillna({"n_frames": 0, "n_promo": 0, "n_quant": 0, "n_gov": 0})
    for k in ("frames", "promo", "quant", "gov"):
        cell[f"{k}_per_1k"] = 1000.0 * cell[f"n_{k}"] / cell["n_words"]
    cell["any_ai"] = (cell["n_frames"] > 0).astype(float)
    return cell


EXT_OUTCOMES = ["frames_per_1k", "promo_per_1k", "quant_per_1k", "gov_per_1k", "any_ai"]


def cells_extensive(cell: pd.DataFrame) -> pd.DataFrame:
    """Una fila por empresa × ejercicio con AL MENOS UN documento en cada canal.
    Sin umbral de frames: los ceros son datos."""
    wide = cell.pivot(index=["ticker", "fy"], columns="channel", values=EXT_OUTCOMES + ["n_docs", "n_paragraphs", "n_frames", "share_post_docs"])
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=["n_docs_call", "n_docs_filing"]).reset_index().rename(columns={"fy": "t"})
    for y in EXT_OUTCOMES:
        wide[f"gap_{y}"] = wide[f"{y}_call"] - wide[f"{y}_filing"]
    wide["post"] = (wide["t"] >= EVENT["fy"]).astype(int)
    wide["share_post_docs"] = (wide["share_post_docs_call"] * wide["n_docs_call"] + wide["share_post_docs_filing"] * wide["n_docs_filing"]) / (wide["n_docs_call"] + wide["n_docs_filing"])
    wide["mixed"] = (wide["share_post_docs"] > 0) & (wide["share_post_docs"] < 1)
    wide["weight"] = np.minimum(wide["n_docs_call"], wide["n_docs_filing"]).astype(float)
    return wide


def cells(frames: pd.DataFrame, period: str) -> pd.DataFrame:
    g = frames.groupby(["ticker", period, "channel"])
    cell = g.agg(
        n_frames=("text_hash", "size"),
        promotional_rate=("rhetoric_promotional", "mean"),
        quantified_rate=("specificity_quantified_metric", "mean"),
        specificity_index=("specificity", "mean"),
        realized_share=("temporal", lambda s: float((s == "realized").mean())),
        hypothetical_share=("temporal", lambda s: float((s == "hypothetical").mean())),
        gov_share=("is_gov", "mean"),
        share_post_docs=("post_doc", "mean"),
    ).reset_index()
    cell = cell[cell["n_frames"] >= MIN_FRAMES]
    wide = cell.pivot(index=["ticker", period], columns="channel", values=OUTCOMES + ["n_frames", "share_post_docs"])
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=["n_frames_call", "n_frames_filing"]).reset_index().rename(columns={period: "t"})
    for y in OUTCOMES:
        wide[f"gap_{y}"] = wide[f"{y}_call"] - wide[f"{y}_filing"]
    wide["post"] = (wide["t"] >= EVENT[period]).astype(int)
    wide["share_post_docs"] = (wide["share_post_docs_call"] * wide["n_frames_call"] + wide["share_post_docs_filing"] * wide["n_frames_filing"]) / (wide["n_frames_call"] + wide["n_frames_filing"])
    wide["mixed"] = (wide["share_post_docs"] > 0) & (wide["share_post_docs"] < 1)
    wide["weight"] = np.minimum(wide["n_frames_call"], wide["n_frames_filing"])
    return wide


def _fe_ols(y: pd.Series, X: pd.DataFrame, groups: pd.Series, weights: pd.Series | None = None):
    """Efectos fijos de empresa por demeaning (ponderado si hay pesos) + OLS/WLS
    con SE cluster por empresa."""
    if weights is None:
        wm = lambda s: s.groupby(groups).transform("mean")
    else:
        w = weights.astype(float)
        wm = lambda s: (s * w).groupby(groups).transform("sum") / w.groupby(groups).transform("sum")
    yd = y - wm(y)
    Xd = X.copy()
    for c in Xd.columns:
        Xd[c] = Xd[c] - wm(Xd[c])
    model = sm.WLS(yd.to_numpy(dtype=float), Xd.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)) if weights is not None \
        else sm.OLS(yd.to_numpy(dtype=float), Xd.to_numpy(dtype=float))
    res = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    return res, list(Xd.columns)


def estimate(paired: pd.DataFrame, outcome: str, weighted: bool = False, drop_mixed: bool = False) -> dict:
    cols = ["ticker", "t", "post", "weight", "mixed", f"gap_{outcome}"]
    d = paired[cols].dropna().rename(columns={f"gap_{outcome}": "y"})
    if drop_mixed:
        d = d[~d["mixed"]]
    both = d.groupby("ticker")["post"].nunique()
    d = d[d["ticker"].isin(both[both == 2].index)]
    out = {"n_obs": int(len(d)), "n_firms": int(d["ticker"].nunique()), "weighted": weighted, "drop_mixed": drop_mixed}
    if out["n_firms"] < 10:
        out["usable"] = False; return out
    wts = d["weight"] if weighted else None
    res, _ = _fe_ols(d["y"], pd.DataFrame({"post": d["post"].astype(float)}), d["ticker"], wts)
    out.update({"did_coef": float(res.params[0]), "did_se": float(res.bse[0]), "did_p": float(res.pvalues[0]),
                "pre_mean_gap": float(d.loc[d.post == 0, "y"].mean()), "post_mean_gap": float(d.loc[d.post == 1, "y"].mean())})
    periods = sorted(d["t"].unique()); base = periods[0]
    X = pd.DataFrame({f"t{p}": (d["t"] == p).astype(float) for p in periods[1:]})
    res2, cols = _fe_ols(d["y"], X, d["ticker"], wts)
    out["event_study"] = [{"t": str(p), "coef": float(res2.params[i]), "se": float(res2.bse[i]), "p": float(res2.pvalues[i])}
                          for i, p in enumerate(periods[1:])]
    out["event_base"] = str(base)
    pre_idx = [i for i, p in enumerate(periods[1:]) if p < EVENT[PERIOD]]
    if pre_idx:
        R = np.zeros((len(pre_idx), len(cols)))
        for r, i in enumerate(pre_idx): R[r, i] = 1.0
        ft = res2.f_test(R)
        out["pretrend_F"] = float(np.squeeze(ft.fvalue)); out["pretrend_p"] = float(np.squeeze(ft.pvalue))
        out["usable"] = out["pretrend_p"] > 0.05
    return out


def channel_levels(paired: pd.DataFrame, outcome: str) -> dict:
    """¿La brecha se mueve porque cambia la call o porque cambia el filing?
    Event study de cada canal por separado, mismas celdas, FE de empresa."""
    d = paired[["ticker", "t", "post", f"{outcome}_call", f"{outcome}_filing"]].dropna()
    both = d.groupby("ticker")["post"].nunique(); d = d[d["ticker"].isin(both[both == 2].index)]
    periods = sorted(d["t"].unique()); out = {}
    for ch in ("call", "filing"):
        X = pd.DataFrame({f"t{p}": (d["t"] == p).astype(float) for p in periods[1:]})
        res, _ = _fe_ols(d[f"{outcome}_{ch}"], X, d["ticker"])
        out[ch] = {str(p): {"coef": float(res.params[i]), "p": float(res.pvalues[i])} for i, p in enumerate(periods[1:])}
        out[f"{ch}_mean_by_t"] = {str(p): float(v) for p, v in d.groupby("t")[f"{outcome}_{ch}"].mean().items()}
    return out


def main() -> None:
    global PERIOD
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--period", choices=("fy", "quarter"), default="fy",
                        help="fy = año fiscal alineado por período cubierto (default); quarter = trimestre calendario de la fecha")
    args = parser.parse_args()
    PERIOD = args.period

    con = duckdb.connect(str(args.database), read_only=True)
    frames = load_frames(con)
    print(f"frames: {len(frames):,} | filings {int((frames.channel == 'filing').sum()):,} "
          f"| calls {int((frames.channel == 'call').sum()):,} | empresas {frames.ticker.nunique():,} "
          f"| calls cubren {frames.loc[frames.channel == 'call', 'quarter'].min()}–{frames.loc[frames.channel == 'call', 'quarter'].max()}")
    paired = cells(frames, PERIOD)
    print(f"celdas empresa×{PERIOD} con ambos canales (≥{MIN_FRAMES} frames cada uno): {len(paired):,} "
          f"| empresas {paired.ticker.nunique():,} | por período {paired.groupby('t').size().to_dict()} "
          f"| celdas mixtas pre/post: {int(paired['mixed'].sum())}")

    desc = {}
    print("\nbrecha call − filing (media sobre celdas):")
    for y in OUTCOMES:
        g = paired[f"gap_{y}"].dropna(); t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
        desc[y] = {"mean_call": float(paired[f"{y}_call"].mean()), "mean_filing": float(paired[f"{y}_filing"].mean()),
                   "mean_gap": float(g.mean()), "median_gap": float(g.median()),
                   "share_gap_positive": float((g > 0).mean()), "t_vs_zero": float(t), "n": int(len(g))}
        print(f"  {y:20s} call {desc[y]['mean_call']:.3f} | filing {desc[y]['mean_filing']:.3f} | "
              f"gap {desc[y]['mean_gap']:+.3f} (t={t:+.1f}, n={len(g)})")

    results, levels = {}, {}
    print(f"\nefecto de post-{EVENT[PERIOD]} sobre la brecha (FE de empresa, empresas a ambos lados del corte):")
    for y in OUTCOMES:
        r = estimate(paired, y); results[y] = r
        if r.get("did_coef") is None:
            print(f"  {y:20s} sin muestra ({r['n_firms']} empresas)"); continue
        flag = "pasa" if r.get("usable") else "FALLA"
        es = " ".join(f"{e['t']}:{e['coef']:+.3f}" for e in r["event_study"])
        print(f"  {y:20s} b={r['did_coef']:+.4f} (se {r['did_se']:.4f}, p={r['did_p']:.3f}) | pretrend {flag} "
              f"(p={r.get('pretrend_p', float('nan')):.3f}) | n={r['n_obs']}, {r['n_firms']} empresas | ES vs {r['event_base']}: {es}")
        levels[y] = channel_levels(paired, y)
    robust = {}
    print("\nrobustez (mismo estimador): ponderado por frames | sin celdas mixtas pre/post | ambas")
    for y in ("promotional_rate", "quantified_rate", "specificity_index"):
        robust[y] = {"weighted": estimate(paired, y, weighted=True), "no_mixed": estimate(paired, y, drop_mixed=True),
                     "weighted_no_mixed": estimate(paired, y, weighted=True, drop_mixed=True)}
        print(f"  {y:18s} " + " | ".join(f"{k}: b={r.get('did_coef', float('nan')):+.4f} (p={r.get('did_p', float('nan')):.3f}, pretrend p={r.get('pretrend_p', float('nan')):.2f}, n={r['n_obs']})"
                                       for k, r in robust[y].items()))
    print("\ndescomposición por canal (event study de cada canal, coef vs primer período):")
    for y in ("promotional_rate", "quantified_rate", "specificity_index"):
        for ch in ("call", "filing"):
            print(f"  {y:18s} {ch:7s} " + " ".join(f"{p}:{v['coef']:+.3f}(p={v['p']:.2f})" for p, v in levels[y][ch].items()))

    ext = None
    if PERIOD == "fy":
        ext = cells_extensive(load_documents(con, frames))
        print(f"\nANÁLISIS PRINCIPAL — todas las celdas empresa×ejercicio con ≥1 transcripción y ≥1 filing, con ceros: "
              f"{len(ext):,} | empresas {ext.ticker.nunique():,} | por ejercicio {ext.groupby('t').size().to_dict()} "
              f"| calls sin ningún frame de IA: {int((ext.n_frames_call == 0).sum())} celdas")
        print("  niveles por canal (media sobre celdas): " + " | ".join(
            f"{y}: call {ext[f'{y}_call'].mean():.2f} filing {ext[f'{y}_filing'].mean():.2f}" for y in EXT_OUTCOMES))
        ext_results = {}
        for y in EXT_OUTCOMES:
            r = estimate(ext, y); rw = estimate(ext, y, weighted=True); rm = estimate(ext, y, drop_mixed=True)
            ext_results[y] = {"main": r, "weighted": rw, "no_mixed": rm}
            es = " ".join(f"{e['t']}:{e['coef']:+.2f}" for e in r.get("event_study", []))
            print(f"  gap {y:14s} b={r['did_coef']:+.3f} (se {r['did_se']:.3f}, p={r['did_p']:.3f}) pretrend p={r.get('pretrend_p', float('nan')):.2f} "
                  f"| pond. {rw['did_coef']:+.3f} (p={rw['did_p']:.3f}) | sin mixtas {rm['did_coef']:+.3f} (p={rm['did_p']:.3f}) "
                  f"| n={r['n_obs']}, {r['n_firms']} empresas | ES: {es}")
        ext_levels = {y: channel_levels(ext, y) for y in ("promo_per_1k", "frames_per_1k")}
        for y, lv in ext_levels.items():
            for ch in ("call", "filing"):
                print(f"  {y:13s} {ch:7s} " + " ".join(f"{p}:{v['coef']:+.2f}(p={v['p']:.2f})" for p, v in lv[ch].items()))

    firm = paired.groupby("ticker").agg(
        n_cells=("t", "size"), gap_promotional=("gap_promotional_rate", "mean"),
        gap_specificity=("gap_specificity_index", "mean"), gap_quantified=("gap_quantified_rate", "mean"),
        promotional_call=("promotional_rate_call", "mean"), promotional_filing=("promotional_rate_filing", "mean"),
    ).reset_index()
    min_cells = 2 if PERIOD == "fy" else 3
    firm = firm[firm["n_cells"] >= min_cells].sort_values("gap_promotional", ascending=False)
    firm["z_gap_promotional"] = (firm["gap_promotional"] - firm["gap_promotional"].mean()) / firm["gap_promotional"].std(ddof=1)
    print(f"\nempresas con ≥{min_cells} celdas: {len(firm)} | brecha promocional mediana {firm.gap_promotional.median():+.3f} | top 10:")
    print(firm.head(10)[["ticker", "n_cells", "promotional_call", "promotional_filing", "gap_promotional"]].round(3).to_string(index=False))

    print("\ncasos notorios, brecha promocional por período (call − filing; c/f = frames por canal):")
    notorious = {}
    for tk in NOTORIOUS:
        q = paired[paired.ticker == tk].sort_values("t")
        if q.empty:
            notorious[tk] = None; print(f"  {tk:5s} sin celdas"); continue
        notorious[tk] = [{"t": str(r.t), "gap": float(r.gap_promotional_rate), "call": float(r.promotional_rate_call),
                          "filing": float(r.promotional_rate_filing), "n_call": int(r.n_frames_call), "n_filing": int(r.n_frames_filing)}
                         for r in q.itertuples()]
        print(f"  {tk:5s} " + " ".join(f"{str(r.t)}:{r.gap_promotional_rate:+.2f}(c{int(r.n_frames_call)}/f{int(r.n_frames_filing)})" for r in q.itertuples()))

    cross = {}
    ws = args.output_dir / "firm_washing_score.parquet"
    if ws.exists():
        w = pd.read_parquet(ws)[["ticker", "w", "washing", "callada", "n_promo"]]
        m = firm.merge(w, on="ticker", how="inner"); m["exceso_rel"] = m["w"]
        cross["spearman_gap_vs_score09"] = float(m["gap_promotional"].corr(m["exceso_rel"], method="spearman")); cross["n"] = int(len(m))
        cross["gap_of_09_washing"] = {t: float(v) for t, v in m.loc[m.washing, ["ticker", "gap_promotional"]].values}
        print(f"\nSpearman(brecha por canal, exceso de 09) = {cross['spearman_gap_vs_score09']:+.3f} (n={cross['n']})")
    seg = args.output_dir / "firm_segments.parquet"
    if seg.exists():
        s = pd.read_parquet(seg)[["ticker", "segmento"]]; m = firm.merge(s, on="ticker", how="inner")
        by = m.groupby("segmento")["gap_promotional"].agg(["mean", "median", "size"])
        cross["gap_by_segment"] = {k: {"mean": float(r["mean"]), "median": float(r["median"]), "n": int(r["size"])} for k, r in by.iterrows()}
        print("brecha promocional por segmento:"); print(by.round(3).to_string())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if ext is not None:
        ext.assign(t=ext["t"].astype(str)).to_parquet(args.output_dir / "channel_gap_cells.parquet", index=False)
    firm.to_parquet(args.output_dir / "channel_gap_firm.parquet", index=False)
    payload = {"period": PERIOD, "event": str(EVENT[PERIOD]), "min_frames": MIN_FRAMES,
               "calls_coverage": [str(frames.loc[frames.channel == 'call', 'quarter'].min()), str(frames.loc[frames.channel == 'call', 'quarter'].max())],
               "n_cells": int(len(paired)), "n_firms": int(paired.ticker.nunique()),
               "cells_by_t": {str(k): int(v) for k, v in paired.groupby("t").size().items()},
               "descriptive": desc, "did_on_gap": results, "robustness": robust, "channel_levels": levels,
               "extensive": ({"n_cells": int(len(ext)), "n_firms": int(ext.ticker.nunique()), "did_on_gap": ext_results, "channel_levels": ext_levels}
                             if ext is not None else None),
               "firm_score": {"n_firms": int(len(firm)), "median_gap_promotional": float(firm.gap_promotional.median()),
                              "top10": firm.head(10)[["ticker", "gap_promotional"]].values.tolist()},
               "notorious": notorious, "cross": cross}
    (args.output_dir / "channel_gap_analysis.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\n-> {args.output_dir / 'channel_gap_analysis.json'}")


PERIOD = "fy"

if __name__ == "__main__":
    main()
