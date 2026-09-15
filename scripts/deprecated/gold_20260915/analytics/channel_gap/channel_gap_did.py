"""Brecha entre canales: la misma empresa, el mismo período, earnings call
contra filing SEC -- la ESTIMACIÓN (DiD, event study, robustez, casos
notorios, cruce con el washing score) sobre las celdas que construye
`scripts/gold/channel_gap/build_channel_gap_cells.py`.

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

`post` = año fiscal ≥ 2024 (escrutinio de la SEC, marzo 2024). Se estima
con efectos fijos de empresa sobre la brecha y errores clusterizados por
empresa; se reporta el event study por período (base = primer período) y
la descomposición por canal (¿se mueve la call o el filing?). También se
reporta la estimación ponderada por frames (min de los dos canales) y sin
las celdas que mezclan documentos pre/post-evento (robustez).

Salida: `data/results/channel_gap/channel_gap_did.json`.

Determinístico, sin LLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "channel_gap"))
import layers as L  # noqa: E402
from build_channel_gap_cells import EVENT, OUTCOMES, EXT_OUTCOMES, firm_gap  # noqa: E402

GOLD_WASHING_SCORE = REPO_ROOT / "data" / "gold" / "predictions" / "firm_year" / "washing_score.parquet"
NOTORIOUS = ["WELL", "NVDA", "PLTR", "TSLA", "ORCL", "GOOGL", "CRM", "MSFT", "META", "AMZN",
             "HPE", "ANET", "IBM", "PANW", "CRWD"]


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


def estimate(paired: pd.DataFrame, outcome: str, period: str, weighted: bool = False, drop_mixed: bool = False) -> dict:
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
    res2, cols2 = _fe_ols(d["y"], X, d["ticker"], wts)
    out["event_study"] = [{"t": str(p), "coef": float(res2.params[i]), "se": float(res2.bse[i]), "p": float(res2.pvalues[i])}
                          for i, p in enumerate(periods[1:])]
    out["event_base"] = str(base)
    pre_idx = [i for i, p in enumerate(periods[1:]) if p < EVENT[period]]
    if pre_idx:
        R = np.zeros((len(pre_idx), len(cols2)))
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", choices=("fy", "quarter"), default="fy")
    args = parser.parse_args()
    period = args.period
    grain = "firm_year" if period == "fy" else "call"

    paired = pd.read_parquet(L.gold_path("covariates", grain, "channel_gap_cells_paired"))
    if period == "fy":
        paired["t"] = paired["t"].astype(int)
    else:
        paired["t"] = paired["t"].apply(lambda s: pd.Period(s, freq="Q"))

    desc = {}
    print("brecha call − filing (media sobre celdas):")
    for y in OUTCOMES:
        g = paired[f"gap_{y}"].dropna(); t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))
        desc[y] = {"mean_call": float(paired[f"{y}_call"].mean()), "mean_filing": float(paired[f"{y}_filing"].mean()),
                   "mean_gap": float(g.mean()), "median_gap": float(g.median()),
                   "share_gap_positive": float((g > 0).mean()), "t_vs_zero": float(t), "n": int(len(g))}
        print(f"  {y:20s} call {desc[y]['mean_call']:.3f} | filing {desc[y]['mean_filing']:.3f} | "
              f"gap {desc[y]['mean_gap']:+.3f} (t={t:+.1f}, n={len(g)})")

    results, levels = {}, {}
    print(f"\nefecto de post-{EVENT[period]} sobre la brecha (FE de empresa, empresas a ambos lados del corte):")
    for y in OUTCOMES:
        r = estimate(paired, y, period); results[y] = r
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
        robust[y] = {"weighted": estimate(paired, y, period, weighted=True), "no_mixed": estimate(paired, y, period, drop_mixed=True),
                     "weighted_no_mixed": estimate(paired, y, period, weighted=True, drop_mixed=True)}
        print(f"  {y:18s} " + " | ".join(f"{k}: b={r.get('did_coef', float('nan')):+.4f} (p={r.get('did_p', float('nan')):.3f}, pretrend p={r.get('pretrend_p', float('nan')):.2f}, n={r['n_obs']})"
                                       for k, r in robust[y].items()))
    print("\ndescomposición por canal (event study de cada canal, coef vs primer período):")
    for y in ("promotional_rate", "quantified_rate", "specificity_index"):
        for ch in ("call", "filing"):
            print(f"  {y:18s} {ch:7s} " + " ".join(f"{p}:{v['coef']:+.3f}(p={v['p']:.2f})" for p, v in levels[y][ch].items()))

    ext = None
    ext_results, ext_levels = {}, {}
    if period == "fy":
        ext = pd.read_parquet(L.gold_path("covariates", "firm_year", "channel_gap_cells_extensive"))
        ext["t"] = ext["t"].astype(int)
        print(f"\nANÁLISIS PRINCIPAL — todas las celdas empresa×ejercicio con ≥1 transcripción y ≥1 filing, con ceros: "
              f"{len(ext):,} | empresas {ext.ticker.nunique():,} | por ejercicio {ext.groupby('t').size().to_dict()} "
              f"| calls sin ningún frame de IA: {int((ext.n_frames_call == 0).sum())} celdas")
        print("  niveles por canal (media sobre celdas): " + " | ".join(
            f"{y}: call {ext[f'{y}_call'].mean():.2f} filing {ext[f'{y}_filing'].mean():.2f}" for y in EXT_OUTCOMES))
        for y in EXT_OUTCOMES:
            r = estimate(ext, y, period); rw = estimate(ext, y, period, weighted=True); rm = estimate(ext, y, period, drop_mixed=True)
            ext_results[y] = {"main": r, "weighted": rw, "no_mixed": rm}
            es = " ".join(f"{e['t']}:{e['coef']:+.2f}" for e in r.get("event_study", []))
            print(f"  gap {y:14s} b={r['did_coef']:+.3f} (se {r['did_se']:.3f}, p={r['did_p']:.3f}) pretrend p={r.get('pretrend_p', float('nan')):.2f} "
                  f"| pond. {rw['did_coef']:+.3f} (p={rw['did_p']:.3f}) | sin mixtas {rm['did_coef']:+.3f} (p={rm['did_p']:.3f}) "
                  f"| n={r['n_obs']}, {r['n_firms']} empresas | ES: {es}")
        ext_levels = {y: channel_levels(ext, y) for y in ("promo_per_1k", "frames_per_1k")}
        for y, lv in ext_levels.items():
            for ch in ("call", "filing"):
                print(f"  {y:13s} {ch:7s} " + " ".join(f"{p}:{v['coef']:+.2f}(p={v['p']:.2f})" for p, v in lv[ch].items()))

    # pooled per-firm gap: a plain groupby-mean over channel_gap_cells_paired
    # (`firm_gap`, no cutoff/time index of its own) -- computed here at read
    # time instead of a persisted firm-grain parquet, per docs/gold_pipeline.md.
    firm = firm_gap(paired, period)
    print(f"\nempresas con brecha pooled: {len(firm)} | brecha promocional mediana {firm.gap_promotional.median():+.3f} | top 10:")
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

    # cruce con el washing score (predictions/firm_year/washing_score.parquet):
    # ¿las empresas con mayor brecha por canal son las que el índice W ya
    # marca como "washing"? La cruza contra el segmento deprecado
    # (firm_segments.parquet) se elimina -- scripts/deprecated/build_segments.py.
    cross = {}
    if GOLD_WASHING_SCORE.exists():
        w = pd.read_parquet(GOLD_WASHING_SCORE)[["ticker", "w", "washing", "callada", "n_promo"]]
        m = firm.merge(w, on="ticker", how="inner"); m["exceso_rel"] = m["w"]
        cross["spearman_gap_vs_score09"] = float(m["gap_promotional"].corr(m["exceso_rel"], method="spearman")); cross["n"] = int(len(m))
        cross["gap_of_09_washing"] = {t: float(v) for t, v in m.loc[m.washing, ["ticker", "gap_promotional"]].values}
        print(f"\nSpearman(brecha por canal, exceso de 09) = {cross['spearman_gap_vs_score09']:+.3f} (n={cross['n']})")

    payload = {"period": period, "event": str(EVENT[period]), "n_cells": int(len(paired)), "n_firms": int(paired.ticker.nunique()),
               "cells_by_t": {str(k): int(v) for k, v in paired.groupby("t").size().items()},
               "descriptive": desc, "did_on_gap": results, "robustness": robust, "channel_levels": levels,
               "extensive": ({"n_cells": int(len(ext)), "n_firms": int(ext.ticker.nunique()), "did_on_gap": ext_results, "channel_levels": ext_levels}
                             if ext is not None else None),
               "firm_score": {"n_firms": int(len(firm)), "median_gap_promotional": float(firm.gap_promotional.median()),
                              "top10": firm.head(10)[["ticker", "gap_promotional"]].values.tolist()},
               "notorious": notorious, "cross": cross}
    destination = L.results_path("channel_gap", "channel_gap_did.json")
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
