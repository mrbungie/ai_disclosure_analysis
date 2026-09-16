"""
scripts/bronze/manifests.py — bronze.firm_universe, bronze.filing_manifest,
bronze.filing_manifest_10q.

The filing manifest unions the 10-K manifest with the manifests of the other
documents that share its lookup (DEF 14A, 8-K, 20-F, 6-K, earnings calls),
with dates and CIKs normalised to one type. The 10-Q manifest is a separate
instrument and stays its own table. No universe filter here.
"""

from __future__ import annotations

import polars as pl
from _paths import COUNTRY, MANIFESTS, L, files

BUILDER = "scripts/bronze/manifests.py"
EXTRA_MANIFESTS = ("filing_manifest_proxy", "filing_manifest_8k", "filing_manifest_earnings_calls",
                   "filing_manifest_earnings_calls_equibles", "filing_manifest_earnings_calls_stockanalysis",
                   "filing_manifest_earnings_calls_equibles_backfill",
                   "filing_manifest_20f", "filing_manifest_6k")


def manifest_source(stem: str):
    """(LazyFrame, input files) for a manifest stored as one file or as
    append-only run parts (newest row per document_id by updated_at wins)."""
    single = MANIFESTS / f"{stem}.parquet"
    parts = files(MANIFESTS, f"{stem}__run=*__part=*.parquet")
    sources = ([single] if single.exists() else []) + parts
    if not sources:
        return None
    lf = pl.concat([pl.scan_parquet(p) for p in sources], how="diagonal_relaxed")
    if parts:
        lf = lf.sort("updated_at", descending=True, maintain_order=True).unique("document_id", keep="first", maintain_order=True)
    return lf, sources


def normalise(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    casts = []
    if schema.get("filing_date") != pl.Date:
        casts.append(pl.col("filing_date").cast(pl.String).str.to_date("%Y-%m-%d", strict=False))
    # Earnings-call manifests store the fiscal quarter ('2021Q4') where SEC
    # manifests store a period end date; it gets its own column.
    if "period_end_date" in schema and schema["period_end_date"] != pl.Date:
        casts += [pl.col("period_end_date").cast(pl.String).alias("fiscal_period"),
                  pl.lit(None, dtype=pl.Date).alias("period_end_date")]
    if "cik" in schema:
        casts.append(pl.col("cik").cast(pl.String))
    for col in ("created_at", "updated_at"):
        dtype = schema.get(col)
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            casts.append(pl.col(col).dt.convert_time_zone("UTC").dt.replace_time_zone(None))
    return lf.with_columns(casts) if casts else lf


def main() -> None:
    src = MANIFESTS / "firm_universe.parquet"
    L.write_table("bronze.firm_universe", pl.scan_parquet(src).select(pl.lit(COUNTRY).alias("country_code"), pl.all()),
                  keys=["ticker"], inputs=[src], builder=BUILDER)

    base = manifest_source("filing_manifest")
    assert base is not None, "data/interim/manifests/filing_manifest.parquet missing"
    frames, inputs = [normalise(base[0])], list(base[1])
    for stem in EXTRA_MANIFESTS:
        extra = manifest_source(stem)
        if extra:
            frames.append(normalise(extra[0]))
            inputs += extra[1]
    lf = pl.concat(frames, how="diagonal_relaxed").select(pl.lit(COUNTRY).alias("country_code"), pl.all())
    L.write_table("bronze.filing_manifest", lf, keys=["form_type", "document_id", "ticker"], inputs=inputs, builder=BUILDER)

    q = manifest_source("filing_manifest_10q")
    assert q is not None, "data/interim/manifests/filing_manifest_10q.parquet missing"
    L.write_table("bronze.filing_manifest_10q", normalise(q[0]).select(pl.lit(COUNTRY).alias("country_code"), pl.all()),
                  keys=["document_id"], inputs=q[1], builder=BUILDER)


if __name__ == "__main__":
    main()
