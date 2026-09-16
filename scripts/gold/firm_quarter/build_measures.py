"""
scripts/gold/firm_quarter/build_measures.py — layer 1 of the firm-quarter covariates.

One row per (ticker, quarter) for every closed calendar quarter. Each quarter
row is usable from `as_of_date` (the day after the quarter closes): it only
contains documents published up to the quarter end. Events attach a quarter
with pit.asof_join(event_date=..., snapshot_date="as_of_date").

Measures (counts are summed over document instances, rates are recomputed
from the summed counts):
  disclosure_volume  document counts, words, AI frames by concept, *_per_1k,
                     *_rate (ai_intensity.aggregate definitions)
  activities         disclosed AI activities by family and grounding marker,
                     *_per_1k (covariates/document/activities)
  posture_rates      the seven posture indicators summed over AI frames and
                     their rates (posture_features.frame_indicators)

Channel families (FAMILIES): all, filings (10-K, 10-Q, 8-K, DEF 14A),
periodic (10-K, 10-Q), posture (10-K, 8-K, DEF 14A), calls (earnings calls).

Windows:
  quarter    documents published within the quarter
  ttm        documents published in the last four quarters (trailing 12 months)
  expanding  every document published up to the quarter end

Rows follow the firm-quarter spine (spines/firm_quarter/firm_quarter: universe
tickers x closed quarters; a delisted firm keeps the quarters starting on or
before its delisting date, so the delisting quarter is the last one, with
as_of_date after the delisting; spine attributes `delisted`,
`delisting_date` and `sic2`, the firm's two-digit SIC). A firm with no document of the family in the
window has null measures (not zero); zeros are real zero counts over existing
documents, and rates with a zero denominator are null. Analytics decide how
to fill.

Documents and activity counts come from the gold document tables; posture
frames from silver.ai_frames (posture_features.load_frames).

Output: spines/firm_quarter/firm_quarter and one covariate family per
measure, covariates/firm_quarter/<measure>, with one column per family,
window and metric: `<family>_<window>_<metric>` (e.g. disclosure_volume:
posture_expanding_frames_per_1k).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "document"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from posture_features import POSTURE, frame_indicators, load_frames  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import layers as L  # noqa: E402
from ai_intensity import COUNT_COLUMNS, FILING_FORMS  # noqa: E402
from build_activities import COUNTS as ACTIVITY_COUNTS  # noqa: E402
from build_document import gold_document_table  # noqa: E402

BUILDER = "scripts/gold/firm_quarter/build_measures.py"
FAMILIES: dict[str, tuple[str, ...] | None] = {
    "all": None,
    "filings": FILING_FORMS,
    "periodic": ("10-K", "10-Q"),
    "posture": ("10-K", "8-K", "DEF 14A"),
    "calls": ("Earnings call",),
}
WINDOWS = ("quarter", "ttm", "expanding")
KEYS = ["ticker", "quarter"]
MEASURES = ("disclosure_volume", "activities", "posture_rates")
ACTIVITY_COLUMNS = ACTIVITY_COUNTS + ["n_activities"]


def quarter_of(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period("Q")


def document_events() -> pd.DataFrame:
    docs = gold_document_table()
    docs["n_docs"] = 1.0
    return docs[["ticker", "form", "fecha", "n_docs", "n_paragraphs", "n_words"] + COUNT_COLUMNS]


def activity_events() -> pd.DataFrame:
    """Activity-instance counts per document (covariates/document/activities)."""
    acts = L.read_gold("document", ("covariates", "activities"))
    return acts[["ticker", "form", "fecha"] + ACTIVITY_COLUMNS]


def frame_events() -> pd.DataFrame:
    frames = frame_indicators(load_frames(forms=None))
    out = frames[["ticker", "form", "available_date"] + POSTURE].rename(columns={"available_date": "fecha"})
    out["n_posture_frames"] = 1.0
    return out


def windowed(events: pd.DataFrame, value_cols: list[str], forms: tuple[str, ...] | None,
             grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sums of `value_cols` per (ticker, quarter) on the full grid, for every window."""
    e = events if forms is None else events[events["form"].isin(forms)]
    e = e.assign(quarter=quarter_of(e["fecha"]))
    sums = e.groupby(KEYS)[value_cols].sum().reset_index()
    q = grid.merge(sums, on=KEYS, how="left").fillna({c: 0.0 for c in value_cols})
    q = q.sort_values(KEYS).reset_index(drop=True)
    x = q.copy()
    x[value_cols] = q.groupby("ticker")[value_cols].cumsum()
    t = q.copy()
    t[value_cols] = (q.groupby("ticker")[value_cols].rolling(4, min_periods=1).sum()
                     .reset_index(level=0, drop=True).sort_index())
    return {"quarter": q, "ttm": t, "expanding": x}


def disclosure_rates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        words = d["n_words"].where(d["n_words"] > 0)
        for c in COUNT_COLUMNS:
            d[c.replace("n_", "", 1) + "_per_1k"] = 1000.0 * d[c] / words
        d["any_ai"] = (d["n_frames"] > 0).astype(float)
        frames = d["n_frames"].where(d["n_frames"] > 0)
        for c in ("n_promo", "n_quant", "n_spec", "n_risk", "n_gov", "n_hyp", "n_realized"):
            d[c.replace("n_", "", 1) + "_rate"] = d[c] / frames
    return d


def activity_rates(d: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(words, on=KEYS, how="left")
    w = d["n_words"].where(d["n_words"] > 0)
    for c in ACTIVITY_COLUMNS:
        d[f"{c}_per_1k"] = 1000.0 * d[c] / w
    return d


def posture_rates(d: pd.DataFrame) -> pd.DataFrame:
    d = d.rename(columns={c: f"sum_{c}" for c in POSTURE})
    n = d["n_posture_frames"].where(d["n_posture_frames"] > 0)
    for c in POSTURE:
        d[c] = d[f"sum_{c}"] / n
    return d


def finish(d: pd.DataFrame, no_documents: pd.Series) -> pd.DataFrame:
    d = d.copy()
    values = [c for c in d.columns if c not in KEYS]
    d.loc[no_documents.values, values] = np.nan
    d["as_of_date"] = (d["quarter"].dt.end_time.dt.normalize() + pd.Timedelta(days=1)).astype("datetime64[ns]")
    d.insert(0, "id", d["ticker"] + "_" + d["quarter"].astype(str))
    d["quarter"] = d["quarter"].astype(str)
    return d


def main() -> None:
    events = {
        "disclosure_volume": (document_events(), ["n_docs", "n_paragraphs", "n_words"] + COUNT_COLUMNS),
        "activities": (activity_events(), ACTIVITY_COLUMNS),
        "posture_rates": (frame_events(), POSTURE + ["n_posture_frames"]),
    }
    first = min(e["fecha"].min() for e, _ in events.values())
    last = max(e["fecha"].max() for e, _ in events.values())
    quarters = pd.period_range(quarter_of(pd.Series([first]))[0], quarter_of(pd.Series([last]))[0], freq="Q")
    quarters = [q for q in quarters if q.end_time.normalize() + pd.Timedelta(days=1) <= last + pd.Timedelta(days=1)]
    tickers = L.read("silver.firm_universe").select("ticker").to_pandas()["ticker"].sort_values()
    grid = pd.MultiIndex.from_product([tickers, quarters], names=KEYS).to_frame(index=False)
    # a delisted firm keeps the quarters it was listed in at some point (the
    # quarter starts on or before its delisting date)
    delist = grid["ticker"].map(L.firm_delistings().set_index("ticker")["delisting_date"])
    grid = grid[delist.isna() | (grid["quarter"].map(lambda q: q.start_time) <= delist)].reset_index(drop=True)
    spine = finish(grid.assign(_=0.0), pd.Series(False, index=grid.index)).drop(columns="_")
    spine = spine.merge(L.firm_delistings(), on="ticker", how="left")
    sic = L.read("silver.firm_universe").select("ticker", "sic").to_pandas()
    spine["sic2"] = spine["ticker"].map(sic.set_index("ticker")["sic"]).astype(str).str.zfill(4).str[:2]
    L.write_gold("spines", "firm_quarter", "firm_quarter", spine, builder=BUILDER,
                 extra={"grain": "firm_quarter (universe ticker x closed calendar quarter in which the firm was listed)",
                        "usable_from": "as_of_date = quarter end + 1 day"})
    print(f"{len(tickers)} tickers x {len(quarters)} closed quarters ({quarters[0]} .. {quarters[-1]})")

    wide = {measure: spine.drop(columns=["delisted", "delisting_date", "sic2"]) for measure in MEASURES}
    for family, forms in FAMILIES.items():
        vol = windowed(*events["disclosure_volume"], forms, grid)
        act = windowed(*events["activities"], forms, grid)
        pos = windowed(*events["posture_rates"], forms, grid)
        for window in WINDOWS:
            outputs = {
                "disclosure_volume": disclosure_rates(vol[window]),
                "activities": activity_rates(act[window], vol[window][KEYS + ["n_words"]]),
                "posture_rates": posture_rates(pos[window]),
            }
            no_documents = vol[window]["n_docs"] == 0
            for measure, df in outputs.items():
                df = finish(df, no_documents).drop(columns=["id", "as_of_date"])
                values = [c for c in df.columns if c not in KEYS]
                df = df.rename(columns={c: f"{family}_{window}_{c}" for c in values})
                wide[measure] = wide[measure].merge(df, on=KEYS, how="left", validate="one_to_one")

    for measure, df in wide.items():
        L.write_gold("covariates", "firm_quarter", measure, df, builder=BUILDER,
                     extra={"columns": "<family>_<window>_<metric>",
                            "families": {f: (list(v) if v else "all") for f, v in FAMILIES.items()},
                            "windows": list(WINDOWS), "usable_from": "as_of_date = quarter end + 1 day"})


if __name__ == "__main__":
    main()
