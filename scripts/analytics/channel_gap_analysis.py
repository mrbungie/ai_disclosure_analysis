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

PERÍODO = AÑO CALENDARIO por defecto. Los filings con frames de IA son
anuales (10-K en Q1, DEF 14A en Q2; el 10-Q casi nunca tiene frames), así
que parear por trimestre deja celdas sólo donde un 10-K coincide con una
call: 590 celdas de 173 empresas. Por año, la celda junta el 10-K, la DEF
14A, los 10-Q y las ~3-4 calls del año: 619 celdas de 237 empresas y 115
empresas a ambos lados del corte. `--period quarter` conserva la versión
trimestral.

Se estima con efectos fijos de empresa sobre la brecha y errores
clusterizados por empresa; se reporta el event study por período (base =
primer período) y la descomposición por canal (¿se mueve la call o el
filing?). `post` = 2024 en adelante (escrutinio de la SEC, marzo 2024).

Salidas (`data/processed/clusters/`):
  channel_gap_cells.parquet      una fila por celda: tasas por canal y brecha
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
CALLS_MANIFEST = REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
FILING_FORMS = ("10-K", "10-Q", "DEF 14A", "8-K")
MIN_FRAMES = 3
EVENT = {"year": 2024, "quarter": pd.Period("2024Q2", freq="Q")}
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
            SELECT country_code, accession_number, ticker, filing_date FROM filing_manifest
            WHERE form_type != 'Earnings call transcript'
            UNION ALL
            SELECT country_code, accession_number, ticker, filing_date FROM filing_manifest_10q
        )
        SELECT 'filing' AS channel, f.form, m.ticker, m.filing_date AS fecha, f.text_hash,
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
               f.text_hash, f.frame_index, f.rhetoric_promotional, f.specificity_quantified_metric,
               f.specificity_business_process, f.specificity_product_or_system,
               f.specificity_vendor_or_partner, f.specificity_date_or_timeline,
               f.temporal, f.concepts
        FROM gold_ai_frames f
        JOIN (SELECT document_id, ticker, filing_date
              FROM read_parquet('{CALLS_MANIFEST}') WHERE ticker IS NOT NULL) m
          ON m.document_id = f.accession_number
        WHERE f.country_code = 'us' AND f.has_frame AND f.form = 'Earnings call'
    """).df()
    frames = pd.concat([filings, calls], ignore_index=True)
    frames = frames.drop_duplicates(["channel", "ticker", "fecha", "text_hash", "frame_index"])
    frames["quarter"] = pd.to_datetime(frames["fecha"]).dt.to_period("Q")
    frames["year"] = frames["quarter"].dt.year
    spec_cols = ["specificity_business_process", "specificity_product_or_system",
                 "specificity_vendor_or_partner", "specificity_quantified_metric",
                 "specificity_date_or_timeline"]
    frames["specificity"] = frames[spec_cols].astype(float).mean(axis=1)
    frames["is_gov"] = frames["concepts"].apply(
        lambda c: any(str(x).startswith("gov_") for x in (list(c) if c is not None else [])))
    return frames


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
    ).reset_index()
    cell = cell[cell["n_frames"] >= MIN_FRAMES]
    wide = cell.pivot(index=["ticker", period], columns="channel", values=OUTCOMES + ["n_frames"])
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=["n_frames_call", "n_frames_filing"]).reset_index().rename(columns={period: "t"})
    for y in OUTCOMES:
        wide[f"gap_{y}"] = wide[f"{y}_call"] - wide[f"{y}_filing"]
    wide["post"] = (wide["t"] >= EVENT[period]).astype(int)
    return wide


def _fe_ols(y: pd.Series, X: pd.DataFrame, groups: pd.Series):
    """Demeaning por empresa (efectos fijos) + OLS con SE cluster por empresa."""
    yd = y - y.groupby(groups).transform("mean")
    Xd = X.copy()
    for c in Xd.columns:
        Xd[c] = Xd[c] - Xd[c].groupby(groups).transform("mean")
    res = sm.OLS(yd.to_numpy(dtype=float), Xd.to_numpy(dtype=float)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    return res, list(Xd.columns)


def estimate(paired: pd.DataFrame, outcome: str) -> dict:
    d = paired[["ticker", "t", "post", f"gap_{outcome}"]].dropna().rename(columns={f"gap_{outcome}": "y"})
    both = d.groupby("ticker")["post"].nunique()
    d = d[d["ticker"].isin(both[both == 2].index)]
    out = {"n_obs": int(len(d)), "n_firms": int(d["ticker"].nunique())}
    if out["n_firms"] < 10:
        out["usable"] = False; return out
    res, _ = _fe_ols(d["y"], pd.DataFrame({"post": d["post"].astype(float)}), d["ticker"])
    out.update({"did_coef": float(res.params[0]), "did_se": float(res.bse[0]), "did_p": float(res.pvalues[0]),
                "pre_mean_gap": float(d.loc[d.post == 0, "y"].mean()), "post_mean_gap": float(d.loc[d.post == 1, "y"].mean())})
    periods = sorted(d["t"].unique()); base = periods[0]
    X = pd.DataFrame({f"t{p}": (d["t"] == p).astype(float) for p in periods[1:]})
    res2, cols = _fe_ols(d["y"], X, d["ticker"])
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
    parser.add_argument("--period", choices=("year", "quarter"), default="year")
    args = parser.parse_args()
    PERIOD = args.period

    con = duckdb.connect(str(args.database), read_only=True)
    frames = load_frames(con)
    print(f"frames: {len(frames):,} | filings {int((frames.channel == 'filing').sum()):,} "
          f"| calls {int((frames.channel == 'call').sum()):,} | empresas {frames.ticker.nunique():,} "
          f"| calls cubren {frames.loc[frames.channel == 'call', 'quarter'].min()}–{frames.loc[frames.channel == 'call', 'quarter'].max()}")
    paired = cells(frames, PERIOD)
    print(f"celdas empresa×{PERIOD} con ambos canales (≥{MIN_FRAMES} frames cada uno): {len(paired):,} "
          f"| empresas {paired.ticker.nunique():,} | por período {paired.groupby('t').size().to_dict()}")

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
    print("\ndescomposición por canal (event study de cada canal, coef vs primer período):")
    for y in ("promotional_rate", "quantified_rate", "specificity_index"):
        for ch in ("call", "filing"):
            print(f"  {y:18s} {ch:7s} " + " ".join(f"{p}:{v['coef']:+.3f}(p={v['p']:.2f})" for p, v in levels[y][ch].items()))

    firm = paired.groupby("ticker").agg(
        n_cells=("t", "size"), gap_promotional=("gap_promotional_rate", "mean"),
        gap_specificity=("gap_specificity_index", "mean"), gap_quantified=("gap_quantified_rate", "mean"),
        promotional_call=("promotional_rate_call", "mean"), promotional_filing=("promotional_rate_filing", "mean"),
    ).reset_index()
    min_cells = 2 if PERIOD == "year" else 3
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
        w = pd.read_parquet(ws)[["ticker", "exceso", "washing", "callada", "n_frames"]]
        m = firm.merge(w, on="ticker", how="inner"); m["exceso_rel"] = m["exceso"] / m["n_frames"]
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
    paired.assign(t=paired["t"].astype(str)).to_parquet(args.output_dir / "channel_gap_cells.parquet", index=False)
    firm.to_parquet(args.output_dir / "channel_gap_firm.parquet", index=False)
    payload = {"period": PERIOD, "event": str(EVENT[PERIOD]), "min_frames": MIN_FRAMES,
               "calls_coverage": [str(frames.loc[frames.channel == 'call', 'quarter'].min()), str(frames.loc[frames.channel == 'call', 'quarter'].max())],
               "n_cells": int(len(paired)), "n_firms": int(paired.ticker.nunique()),
               "cells_by_t": {str(k): int(v) for k, v in paired.groupby("t").size().items()},
               "descriptive": desc, "did_on_gap": results, "channel_levels": levels,
               "firm_score": {"n_firms": int(len(firm)), "median_gap_promotional": float(firm.gap_promotional.median()),
                              "top10": firm.head(10)[["ticker", "gap_promotional"]].values.tolist()},
               "notorious": notorious, "cross": cross}
    (args.output_dir / "channel_gap_analysis.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\n-> {args.output_dir / 'channel_gap_analysis.json'}")


PERIOD = "year"

if __name__ == "__main__":
    main()
