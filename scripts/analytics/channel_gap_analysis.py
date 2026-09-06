"""Brecha entre canales: la misma empresa, el mismo trimestre, earnings call
contra filing SEC.

Es el diseño que `docs/pregunta_identificacion_sec.md` deja planteado y que
el caso Welltower motiva: la SEC no objetó el 10-K por promocional, objetó
que la call y el press release dijeran "industry-leading" sin respaldo
proporcional en el 10-K. El objeto es la BRECHA entre canales, no el nivel
en uno.

    y[i, canal, t] = a[i, t] + b · (post[t] × es_filing[canal]) + e

Con efectos fijos empresa × trimestre y exactamente dos canales por celda,
el estimador es la diferencia dentro de la celda:

    gap[i, t] = y[i, call, t] − y[i, filing, t] = c + a_i − b · post[t] + e

Todo lo que es común a la empresa en ese trimestre (el boom de IA, su
sector, su ciclo) se cancela en la resta. `post` es 2024Q2 en adelante
(escrutinio de la SEC, marzo 2024). Se estima con efectos fijos de
empresa sobre la brecha y errores clusterizados por empresa, y se reporta
el event study por trimestre relativo con el test conjunto de tendencias
previas — la misma disciplina que `shock_analysis.py`.

Trimestre = trimestre calendario de la fecha del documento (call o
filing). Celda = empresa × trimestre con ≥ MIN_FRAMES frames en CADA
canal.

Salidas (`data/processed/clusters/`):
  channel_gap_firm_quarter.parquet   una fila por celda: tasas por canal y brecha
  channel_gap_firm.parquet           brecha promedio por empresa (score por canal)
  channel_gap_analysis.json          estimaciones, event study, descriptivos

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
EVENT_QUARTER = pd.Period("2024Q2", freq="Q")
WINDOW = 6          # trimestres a cada lado del evento en el event study
OUTCOMES = ["promotional_rate", "quantified_rate", "specificity_index", "realized_share",
            "hypothetical_share", "gov_share"]


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
    spec_cols = ["specificity_business_process", "specificity_product_or_system",
                 "specificity_vendor_or_partner", "specificity_quantified_metric",
                 "specificity_date_or_timeline"]
    frames["specificity"] = frames[spec_cols].astype(float).mean(axis=1)
    frames["is_gov"] = frames["concepts"].apply(
        lambda c: any(str(x).startswith("gov_") for x in (list(c) if c is not None else [])))
    return frames


def firm_quarter_channel(frames: pd.DataFrame) -> pd.DataFrame:
    g = frames.groupby(["ticker", "quarter", "channel"])
    cell = g.agg(
        n_frames=("text_hash", "size"),
        promotional_rate=("rhetoric_promotional", "mean"),
        quantified_rate=("specificity_quantified_metric", "mean"),
        specificity_index=("specificity", "mean"),
        realized_share=("temporal", lambda s: float((s == "realized").mean())),
        hypothetical_share=("temporal", lambda s: float((s == "hypothetical").mean())),
        gov_share=("is_gov", "mean"),
    ).reset_index()
    return cell[cell["n_frames"] >= MIN_FRAMES]


def paired_cells(cell: pd.DataFrame) -> pd.DataFrame:
    """Una fila por empresa × trimestre con los dos canales presentes."""
    wide = cell.pivot(index=["ticker", "quarter"], columns="channel",
                      values=OUTCOMES + ["n_frames"])
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=["n_frames_call", "n_frames_filing"]).reset_index()
    for y in OUTCOMES:
        wide[f"gap_{y}"] = wide[f"{y}_call"] - wide[f"{y}_filing"]
    wide["post"] = (wide["quarter"] >= EVENT_QUARTER).astype(int)
    wide["event_time"] = (wide["quarter"] - EVENT_QUARTER).apply(lambda d: d.n)
    return wide


def _cluster_ols(y: pd.Series, X: pd.DataFrame, groups: pd.Series):
    model = sm.OLS(y.to_numpy(dtype=float), X.to_numpy(dtype=float))
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def within_firm(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby("ticker")[c].transform("mean")
    return out


def estimate(paired: pd.DataFrame, outcome: str) -> dict:
    """gap = a_i + b·post + e, SE cluster por empresa; más event study."""
    data = paired[["ticker", "quarter", "event_time", "post", f"gap_{outcome}"]].dropna().copy()
    data = data.rename(columns={f"gap_{outcome}": "gap"})
    data = data[data["event_time"].abs() <= WINDOW]
    # cada empresa tiene que estar a los dos lados del corte, si no el FE se come todo
    both = data.groupby("ticker")["post"].nunique()
    data = data[data["ticker"].isin(both[both == 2].index)]
    if data["ticker"].nunique() < 10:
        return {"n_obs": int(len(data)), "n_firms": int(data["ticker"].nunique()), "usable": False}
    d = within_firm(data, ["gap", "post"])
    X = pd.DataFrame({"post": d["post"]})
    res = _cluster_ols(d["gap"], X, data["ticker"])
    out = {"n_obs": int(len(data)), "n_firms": int(data["ticker"].nunique()),
           "did_coef": float(res.params[0]), "did_se": float(res.bse[0]), "did_p": float(res.pvalues[0]),
           "pre_mean_gap": float(data.loc[data["post"] == 0, "gap"].mean()),
           "post_mean_gap": float(data.loc[data["post"] == 1, "gap"].mean())}
    # event study: dummies por trimestre relativo, t-1 omitido, demeaning por empresa
    times = sorted(t for t in data["event_time"].unique() if t != -1)
    dummies = pd.DataFrame({f"t{t}": (data["event_time"] == t).astype(float) for t in times})
    d2 = within_firm(pd.concat([data[["ticker", "gap"]], dummies], axis=1), ["gap"] + list(dummies.columns))
    res2 = _cluster_ols(d2["gap"], d2[list(dummies.columns)], data["ticker"])
    es = [{"event_time": int(t), "coef": float(res2.params[i]), "se": float(res2.bse[i]),
           "p": float(res2.pvalues[i])} for i, t in enumerate(times)]
    pre = [i for i, t in enumerate(times) if t < -1]
    if pre:
        R = np.zeros((len(pre), len(times)))
        for r, i in enumerate(pre):
            R[r, i] = 1.0
        ft = res2.f_test(R)
        out["pretrend_F"] = float(np.squeeze(ft.fvalue)); out["pretrend_p"] = float(np.squeeze(ft.pvalue))
        out["usable"] = out["pretrend_p"] > 0.05
    out["event_study"] = es
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    con = duckdb.connect(str(args.database), read_only=True)
    frames = load_frames(con)
    print(f"frames: {len(frames):,} | filings {int((frames.channel == 'filing').sum()):,} "
          f"| calls {int((frames.channel == 'call').sum()):,} | empresas {frames.ticker.nunique():,}")
    cell = firm_quarter_channel(frames)
    paired = paired_cells(cell)
    print(f"celdas empresa×trimestre con ambos canales (≥{MIN_FRAMES} frames cada uno): "
          f"{len(paired):,} | empresas {paired.ticker.nunique():,} | "
          f"trimestres {paired.quarter.min()}–{paired.quarter.max()}")

    # descriptivo: la brecha promedio y su signo
    desc = {}
    for y in OUTCOMES:
        g = paired[f"gap_{y}"].dropna()
        t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
        desc[y] = {"mean_call": float(paired[f"{y}_call"].mean()), "mean_filing": float(paired[f"{y}_filing"].mean()),
                   "mean_gap": float(g.mean()), "median_gap": float(g.median()),
                   "share_gap_positive": float((g > 0).mean()), "t_vs_zero": float(t), "n": int(len(g))}
    print("\nbrecha call − filing (media sobre celdas):")
    for y, v in desc.items():
        print(f"  {y:20s} call {v['mean_call']:.3f} | filing {v['mean_filing']:.3f} | "
              f"gap {v['mean_gap']:+.3f} (t={v['t_vs_zero']:+.1f}, n={v['n']})")

    # DiD sobre la brecha
    results = {}
    print(f"\nefecto de post-{EVENT_QUARTER} sobre la brecha (FE de empresa, ventana ±{WINDOW}):")
    for y in OUTCOMES:
        r = estimate(paired, y)
        results[y] = r
        if r.get("did_coef") is not None:
            flag = "pasa" if r.get("usable") else "FALLA"
            print(f"  {y:20s} b={r['did_coef']:+.4f} (se {r['did_se']:.4f}, p={r['did_p']:.3f}) | "
                  f"tendencias previas {flag} (p={r.get('pretrend_p', float('nan')):.3f}) | "
                  f"n={r['n_obs']} celdas, {r['n_firms']} empresas")
        else:
            print(f"  {y:20s} sin muestra suficiente ({r['n_firms']} empresas)")

    # score por empresa: brecha promedio pooled (el "washing entre canales")
    firm = paired.groupby("ticker").agg(
        n_cells=("quarter", "size"),
        gap_promotional=("gap_promotional_rate", "mean"),
        gap_specificity=("gap_specificity_index", "mean"),
        gap_quantified=("gap_quantified_rate", "mean"),
        promotional_call=("promotional_rate_call", "mean"),
        promotional_filing=("promotional_rate_filing", "mean"),
    ).reset_index()
    firm = firm[firm["n_cells"] >= 3]
    z = (firm["gap_promotional"] - firm["gap_promotional"].mean()) / firm["gap_promotional"].std(ddof=1)
    firm["z_gap_promotional"] = z
    firm = firm.sort_values("gap_promotional", ascending=False)
    print(f"\nempresas con ≥3 celdas: {len(firm)} | brecha promocional mediana "
          f"{firm.gap_promotional.median():+.3f} | top 10:")
    print(firm.head(10)[["ticker", "n_cells", "promotional_call", "promotional_filing", "gap_promotional"]]
          .round(3).to_string(index=False))
    welltower = firm[firm.ticker == "WELL"]
    if not welltower.empty:
        pct = float((firm["gap_promotional"] < welltower["gap_promotional"].iloc[0]).mean() * 100)
        print(f"WELL: brecha {welltower.gap_promotional.iloc[0]:+.3f}, percentil {pct:.1f}")
    else:
        pct = None

    # cruce con el score de 09 y los segmentos de 11
    cross = {}
    ws = args.output_dir / "firm_washing_score.parquet"
    if ws.exists():
        w = pd.read_parquet(ws)[["ticker", "exceso", "washing", "callada", "n_frames"]]
        m = firm.merge(w, on="ticker", how="inner")
        m["z_exceso"] = (m["exceso"] / m["n_frames"])
        cross["spearman_gap_vs_score09"] = float(m["gap_promotional"].corr(m["z_exceso"], method="spearman"))
        cross["n"] = int(len(m))
        cross["gap_of_09_washing"] = {t: float(v) for t, v in
                                      m.loc[m.washing, ["ticker", "gap_promotional"]].values}
        print(f"\nSpearman(brecha por canal, exceso de 09) = {cross['spearman_gap_vs_score09']:+.3f} (n={cross['n']})")
    seg = args.output_dir / "firm_segments.parquet"
    if seg.exists():
        s = pd.read_parquet(seg)[["ticker", "segmento"]]
        m = firm.merge(s, on="ticker", how="inner")
        by = m.groupby("segmento")["gap_promotional"].agg(["mean", "median", "size"])
        cross["gap_by_segment"] = {k: {"mean": float(r["mean"]), "median": float(r["median"]), "n": int(r["size"])}
                                   for k, r in by.iterrows()}
        print("brecha promocional por segmento:"); print(by.round(3).to_string())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.assign(quarter=paired["quarter"].astype(str)).to_parquet(args.output_dir / "channel_gap_firm_quarter.parquet", index=False)
    firm.to_parquet(args.output_dir / "channel_gap_firm.parquet", index=False)
    payload = {"event_quarter": str(EVENT_QUARTER), "window": WINDOW, "min_frames": MIN_FRAMES,
               "n_cells": int(len(paired)), "n_firms": int(paired.ticker.nunique()),
               "descriptive": desc, "did_on_gap": results,
               "firm_score": {"n_firms": int(len(firm)), "median_gap_promotional": float(firm.gap_promotional.median()),
                              "top10": firm.head(10)[["ticker", "gap_promotional"]].values.tolist(),
                              "welltower_percentile": pct},
               "cross": cross}
    (args.output_dir / "channel_gap_analysis.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\n-> {args.output_dir / 'channel_gap_analysis.json'}")


if __name__ == "__main__":
    main()
