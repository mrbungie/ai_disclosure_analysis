"""Firm-year spine: one row per (ticker, calendar filing year) with at least
one scorable US filing (10-K, 10-Q, DEF 14A or 8-K) in the document spine.

A ticker-year with only a DEF 14A/8-K/10-Q in a calendar year still enters
the spine; accounting and market families have null values there.

`as_of_date` = 1 January of the following year: every firm-year covariate
uses documents and filings published within the calendar year.

Spine attributes `delisted`, `delisting_date` (silver.firm_universe). A
firm-year is kept when the firm was listed at some point of the calendar year
(delisting_date on or after 1 January); documents after the delisting date are
not in silver, so the delisting year only counts the documents filed before
it.

Output: spines/firm_year/firm_year (id = `{ticker}_{year}`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from ai_intensity import FILING_FORMS  # noqa: E402

import layers as L  # noqa: E402


def main() -> None:
    docs = L.read_gold("document", ("covariates", "disclosure_volume", ["n_paragraphs"]))
    docs = docs[docs["form"].isin(FILING_FORMS) & (docs["n_paragraphs"] >= 1)]
    df = (docs.assign(year=docs["fecha"].dt.year.astype("int32"))[["ticker", "year"]]
          .drop_duplicates().sort_values(["ticker", "year"]).reset_index(drop=True))
    df.insert(0, "id", df["ticker"] + "_" + df["year"].astype(str))
    df["as_of_date"] = pd.to_datetime((df["year"] + 1).astype(str) + "-01-01").astype("datetime64[ns]")
    df = df.merge(L.firm_delistings(), on="ticker", how="left")
    # listed at some point of the calendar year (silver already drops the
    # documents a delisted firm filed after its delisting date)
    listed = df["delisting_date"].isna() | (df["delisting_date"] >= pd.to_datetime(df["year"].astype(str) + "-01-01"))
    df = df[listed].reset_index(drop=True)
    L.write_gold("spines", "firm_year", "firm_year", df, builder="scripts/gold/firm_year/build_spine.py",
                 extra={"grain": "firm_year (ticker x calendar filing year with a scorable filing)",
                        "usable_from": "as_of_date = 1 January of year + 1"})


if __name__ == "__main__":
    main()
