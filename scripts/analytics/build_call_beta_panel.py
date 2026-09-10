"""Construye el spine A (call) que `call_car_regressions.py`,
`call_beta_regressions.py` y `call_beta_robustness.py` consumen:
`data/processed/clusters/call_beta_main_panel_10k10q_asof.parquet`,
una fila por earnings call.

POR QUÉ EXISTE ESTE SCRIPT. Ese parquet vivía como artefacto "ya
materializado" (`docs/analytics/analysis_spines.md`) sin ningún script en
el repo que lo reconstruyera -- se construyó una vez, en una sesión que no
dejó el código atrás, y quedó congelado en 2025-05-15 (517 calls menos que
los disponibles hoy: nunca vio ni un trimestre de 2026, ni las calls que
los rellenos de huecos `filing_manifest_earnings_calls_equibles.parquet` /
`..._stockanalysis.parquet` agregaron para años anteriores). Cap. 6
completo corría sobre ese panel congelado sin que nada lo señalara. Este
script reconstruye la misma tabla desde las fuentes actuales del pipeline y
la deja reproducible.

MÉTODO, validado por fila contra el panel congelado antes de reemplazarlo
(sección `validate()` abajo; ver también la nota de sesión que journaleó
esta reconstrucción):
  - Spine de calls: `ai_intensity.document_table()` filtrado a channel=='call'
    (ya usa las tres fuentes de manifest de calls, ver `ai_intensity.py`).
    `disclosure` = frames por 1.000 palabras del call. Coincide con el panel
    congelado a <0.001 de diferencia media en las filas que se solapan.
  - `n_activities`/`grounding`/`substance` por call: mismas cinco banderas
    de concreción y la misma fórmula de shrinkage empírico-Bayes que
    `washing_score.py` usa a nivel empresa-año, aplicadas a nivel
    empresa-call (`firm_activities.parquet` agrupado por accession_number,
    que para una call es su `document_id` sintético). El prior de
    shrinkage se re-ajusta sobre la población de calls actual, así que un
    grounding histórico puede moverse unas centésimas al agregar 2026 --
    es el comportamiento esperado de un prior Bayesiano, no un error.
  - `hist_*`/`surprise_*`: media expansiva de calls ESTRICTAMENTE previas
    del mismo ticker (`expanding().shift(1)`), `surprise = valor actual -
    histórico`. Reproduce el panel congelado exactamente donde el número de
    calls previas coincide; donde no coincide es porque este panel tiene
    MÁS historia (ver `n_prior_calls` más abajo), nunca menos.
  - `beta_pre`/`beta_post_126`/`beta_post_252`/`price_pre`/`return60`:
    mismo estimador de un factor que `build_market_factors.py::window_metrics`,
    pero anclado a la fecha del call en vez de la fecha de filing, con
    ventanas hacia adelante para el beta post-call (esa es la razón de ser
    de `call_beta_regressions.py`, frente al CAR de `call_car_regressions.py`
    que sólo mira [-1,+5]).
  - Covariables contables (`revenue`, `operating_income`, `total_assets`,
    `operating_margin`, `asset_turnover`, `shares_out`, `filing_date_pt`):
    último 10-K conocido ESTRICTAMENTE antes del call
    (`firm_year_financials_ratios.parquet`, `merge_asof` backward,
    `allow_exact_matches=False` -- mismo patrón que `attach_roa()` en
    `call_car_regressions.py`).
  - `sic`/`sic2`: estáticos por ticker, de `firm_universe`.
  - `log_market_cap` = log(`price_pre` × `shares_out`).
  - `fe` = `f"{sic2}_{año del call}"` (SIC2 × año-de-call, la celda de
    efectos fijos que usan los tres scripts de regresión).
  - `n_prior_calls` sube más rápido que en el panel congelado para muchas
    empresas: no es un error, es historia real que el panel congelado no
    tenía (calls de 2021-2024 que sólo estaban en los rellenos de huecos).

Determinístico, sin LLM (reutiliza `firm_activities.parquet`, ya extraído).
Requiere `duckdb/thesis.duckdb`, `data/processed/clusters/firm_activities.parquet`,
`data/processed/clusters/firm_year_financials_ratios.parquet`, precios en
`data/raw/market/prices/`, factores en `data/raw/market/factors/ff3_daily.parquet`.

Salida: `data/processed/clusters/call_beta_main_panel_10k10q_asof.parquet`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_intensity import document_table  # noqa: E402
from activity_profiles import flags  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
OUT_PATH = OUT_DIR / "call_beta_main_panel_10k10q_asof.parquet"
PRICES_DIR = REPO_ROOT / "data" / "raw" / "market" / "prices"
FACTORS_PATH = REPO_ROOT / "data" / "raw" / "market" / "factors" / "ff3_daily.parquet"
FINANCIALS_PATH = OUT_DIR / "firm_year_financials_ratios.parquet"

GROUNDING_COMPONENTS = ["named_function", "deployed_or_scaled", "named_product_or_process",
                        "quantified_outcome", "third_party_named_provider"]
BETA_MIN_OBS = 120


def _shrink(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Idéntico a `washing_score.py::_shrink` -- ver esa docstring."""
    raw = np.where(denominator > 0, numerator / denominator, np.nan)
    s = pd.Series(raw, index=numerator.index)
    mean, var = float(s.mean(skipna=True)), float(s.var(skipna=True, ddof=1))
    strength = max(mean * (1 - mean) / var - 1, 1e-6) if 0 < mean < 1 and var > 0 else 1.0
    alpha, beta_ = mean * strength, (1 - mean) * strength
    return (s.fillna(0) * denominator + alpha) / (denominator + alpha + beta_)


def build_call_spine() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    docs = document_table(con)
    con.close()
    calls = docs[docs.channel == "call"].copy()
    calls["disclosure"] = 1000.0 * calls["n_frames"] / calls["n_words"]
    calls = calls.rename(columns={"accession_number": "call_accession_number"})
    return calls[["ticker", "fecha", "call_accession_number", "n_words", "n_frames", "disclosure"]]


def attach_activities(panel: pd.DataFrame) -> pd.DataFrame:
    act = pd.read_parquet(OUT_DIR / "firm_activities.parquet")
    act_calls = act[act.channel == "call"].copy()
    fl = flags(act_calls)
    for c in GROUNDING_COMPONENTS:
        act_calls[c] = fl[c].astype(float)
    per_call = act_calls.groupby("accession_number").agg(
        n_activities=("text_hash", "size"), **{c: (c, "sum") for c in GROUNDING_COMPONENTS}
    ).reset_index().rename(columns={"accession_number": "call_accession_number"})
    shrunk = pd.concat([_shrink(per_call[c], per_call["n_activities"]) for c in GROUNDING_COMPONENTS], axis=1)
    per_call["grounding"] = shrunk.mean(axis=1)
    panel = panel.merge(per_call[["call_accession_number", "n_activities", "grounding"]], on="call_accession_number", how="left")
    panel["n_activities"] = panel["n_activities"].fillna(0.0)
    panel["grounding"] = panel["grounding"].fillna(0.0)
    panel["substance"] = np.log1p(panel["n_activities"]) * panel["grounding"]
    return panel


def attach_history(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "fecha"]).reset_index(drop=True)
    panel["n_prior_calls"] = panel.groupby("ticker").cumcount()
    panel["hist_disclosure"] = panel.groupby("ticker")["disclosure"].transform(lambda s: s.expanding().mean().shift(1))
    panel["hist_substance"] = panel.groupby("ticker")["substance"].transform(lambda s: s.expanding().mean().shift(1))
    panel["surprise_disclosure"] = panel["disclosure"] - panel["hist_disclosure"]
    panel["surprise_substance"] = panel["substance"] - panel["hist_substance"]
    return panel


def _beta(merged: pd.DataFrame) -> float:
    if len(merged) < BETA_MIN_OBS:
        return np.nan
    excess = merged["ret"].to_numpy() - merged["rf"].to_numpy()
    design = np.column_stack([np.ones(len(merged)), merged["mktrf"].to_numpy()])
    coef = np.linalg.lstsq(design, excess, rcond=None)[0]
    return float(coef[1])


def _load_prices(tickers: set[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        p = PRICES_DIR / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["date", "adj_close", "close"])
        d["date"] = pd.to_datetime(d["date"])
        d = d.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)
        d["ret"] = d["adj_close"].pct_change()
        out[t] = d
    return out


def attach_market(panel: pd.DataFrame) -> pd.DataFrame:
    factors = pd.read_parquet(FACTORS_PATH, columns=["date", "mktrf", "rf"])
    factors["date"] = pd.to_datetime(factors["date"])
    prices = _load_prices(set(panel["ticker"]))
    rows = []
    for row in panel[["ticker", "fecha"]].itertuples(index=False):
        pr = prices.get(row.ticker)
        result = {"ticker": row.ticker, "fecha": row.fecha, "beta_pre": np.nan,
                  "beta_post_126": np.nan, "beta_post_252": np.nan, "price_pre": np.nan, "return60": np.nan}
        if pr is not None:
            dates = pr["date"].values
            idx = int(np.searchsorted(dates, np.datetime64(row.fecha), side="left"))
            if 0 <= idx - 1 and idx < len(pr):
                result["price_pre"] = float(pr["close"].iloc[idx - 1])
                if idx - 60 >= 0 and pr["adj_close"].iloc[idx - 60] > 0:
                    result["return60"] = float(pr["adj_close"].iloc[idx - 1] / pr["adj_close"].iloc[idx - 60] - 1)
                if idx - 252 >= 0:
                    pre = pr.iloc[idx - 252:idx].merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
                    result["beta_pre"] = _beta(pre)
                if idx + 126 < len(pr):
                    post126 = pr.iloc[idx:idx + 126].merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
                    result["beta_post_126"] = _beta(post126)
                if idx + 252 < len(pr):
                    post252 = pr.iloc[idx:idx + 252].merge(factors, on="date", how="inner").dropna(subset=["ret", "mktrf", "rf"])
                    result["beta_post_252"] = _beta(post252)
        rows.append(result)
    return panel.merge(pd.DataFrame(rows), on=["ticker", "fecha"], how="left")


def attach_financials(panel: pd.DataFrame) -> pd.DataFrame:
    # `accession_number` here is the as-of matched 10-K's, not the call's own
    # document_id (that one stays as `call_accession_number`) -- it's what
    # `attach_leverage()` in call_beta_regressions.py joins XBRL facts on.
    fin = pd.read_parquet(FINANCIALS_PATH, columns=["ticker", "filing_date", "accession_number", "revenue",
                                                     "operating_income", "total_assets", "operating_margin",
                                                     "asset_turnover", "shares_out"])
    fin["filing_date"] = pd.to_datetime(fin["filing_date"])
    fin = fin.dropna(subset=["filing_date"]).sort_values(["filing_date", "ticker"])
    left = panel.sort_values(["fecha", "ticker"]).copy()
    merged = pd.merge_asof(left, fin, left_on="fecha", right_on="filing_date", by="ticker",
                           direction="backward", allow_exact_matches=False)
    merged = merged.rename(columns={"filing_date": "filing_date_pt"})
    merged["log_market_cap"] = np.log(merged["price_pre"] * merged["shares_out"])
    return merged


def attach_sic_and_fe(panel: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    sic = con.execute("SELECT ticker, sic FROM firm_universe WHERE country_code = 'us'").df()
    con.close()
    panel = panel.merge(sic, on="ticker", how="left")
    panel["sic2"] = panel["sic"].astype(str).str.zfill(4).str[:2]
    panel["fe"] = panel["sic2"] + "_" + pd.to_datetime(panel["fecha"]).dt.year.astype(str)
    return panel


def validate(panel: pd.DataFrame) -> None:
    """Compara contra el panel previo (si existe) para que un cambio de
    metodología no pase inadvertido -- sólo imprime, nunca bloquea."""
    if not OUT_PATH.exists():
        return
    old = pd.read_parquet(OUT_PATH)
    cmp = panel.merge(old, on=["ticker", "fecha"], suffixes=("_new", "_old"))
    print(f"\nValidación contra el panel anterior: {len(cmp):,} calls en común de {len(panel):,} nuevas / {len(old):,} previas")
    for col in ["disclosure", "n_activities", "substance", "hist_disclosure", "surprise_disclosure", "beta_pre", "beta_post_126", "price_pre", "return60"]:
        if f"{col}_new" not in cmp or f"{col}_old" not in cmp:
            continue
        diff = (cmp[f"{col}_new"] - cmp[f"{col}_old"]).abs()
        print(f"  {col:22s} media |Δ|={diff.mean():.4f}  máx |Δ|={diff.max():.4f}  n comparado={diff.notna().sum():,}")


def main() -> None:
    panel = build_call_spine()
    panel = attach_activities(panel)
    panel = attach_history(panel)
    panel = attach_market(panel)
    panel = attach_financials(panel)
    panel = attach_sic_and_fe(panel)

    validate(panel)

    cols = ["ticker", "fecha", "n_words", "n_frames", "disclosure", "n_activities", "grounding", "substance",
            "n_prior_calls", "hist_disclosure", "surprise_disclosure", "hist_substance", "surprise_substance",
            "beta_pre", "beta_post_126", "beta_post_252", "price_pre", "return60", "filing_date_pt",
            "revenue", "operating_income", "total_assets", "sic", "sic2", "operating_margin", "asset_turnover",
            "shares_out", "log_market_cap", "fe", "accession_number"]
    panel = panel[cols].sort_values(["ticker", "fecha"]).reset_index(drop=True)
    panel["fecha"] = pd.to_datetime(panel["fecha"]).astype("datetime64[ns]")
    panel["filing_date_pt"] = pd.to_datetime(panel["filing_date_pt"]).astype("datetime64[ns]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\n{len(panel):,} calls, {panel['ticker'].nunique()} tickers, "
          f"{pd.to_datetime(panel['fecha']).min().date()} .. {pd.to_datetime(panel['fecha']).max().date()}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
