"""Call spine: one row per earnings call of the analysis universe.

Calls are the `channel == "call"` documents of the document spine: the
transcripts silver.filing_manifest keeps after the checks of
scripts/bronze/call_transcripts.py (the firm's own results calls, one
transcript per call, dated and labelled from the transcript). Spine
attributes: `fiscal_period` ('2024Q1', the fiscal quarter the call reports),
`call_sequence` (first | next | gap | label_break, the call's step in its
ticker's sequence), `sic`/`sic2` (static per ticker, silver.firm_universe) and
`fe` = `{sic2}_{call year}`, the SIC2 x year fixed-effect cell of the call
regressions, and `delisted`, `delisting_date` (silver.firm_universe; silver
has no call of a delisted firm after its delisting date).

The spine obeys the call-sequence invariant (`sequence_violations`, checked
again by scripts/gold/check_spine_alignment.py): per ticker, consecutive calls
are at least 30 days apart and advance the fiscal period by k >= 1 quarters
within the date window of k quarters, except at a `label_break`.

Output: spines/call/call (id = call_accession_number, the transcript's
`{SOURCE_TICKER}_{YEAR}Q{N}` document id).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

CALL_FORM_TYPE = "Earnings call transcript"
MIN_GAP_DAYS = 30
SEQUENCE_EARLY, SEQUENCE_LATE = 60, 70  # scripts/bronze/call_transcripts.py


def sequence_violations(calls: pd.DataFrame) -> pd.DataFrame:
    """Consecutive calls of a ticker (by fecha) that break the invariant:
    fewer than MIN_GAP_DAYS apart, or a fiscal period that does not advance
    k >= 1 quarters within [91k - SEQUENCE_EARLY, 91k + SEQUENCE_LATE] days
    (a `label_break` call is exempt from the period rule)."""
    c = calls.sort_values(["ticker", "fecha"]).reset_index(drop=True)
    quarter = c["fiscal_period"].str[:4].astype(int) * 4 + c["fiscal_period"].str[-1].astype(int)
    same = c["ticker"].eq(c["ticker"].shift())
    days = (pd.to_datetime(c["fecha"]) - pd.to_datetime(c["fecha"]).shift()).dt.days
    k = quarter - quarter.shift()
    in_window = (k >= 1) & (days >= 91 * k - SEQUENCE_EARLY) & (days <= 91 * k + SEQUENCE_LATE)
    bad = same & ((days < MIN_GAP_DAYS) | ~in_window & (c["call_sequence"] != "label_break"))
    return c.assign(days=days, quarters=k)[bad]


def main() -> None:
    docs = L.read_gold("document", spine_columns=["ticker", "accession_number", "fecha", "channel"])
    calls = docs[docs["channel"] == "call"].rename(columns={"accession_number": "call_accession_number"})
    manifest = (L.scan("silver.filing_manifest").filter(pl.col("form_type") == CALL_FORM_TYPE)
                .select(pl.col("document_id").alias("call_accession_number"), "fiscal_period", "call_sequence")
                .collect().to_pandas())
    calls = calls.merge(manifest, on="call_accession_number", how="left", validate="one_to_one")
    sic = L.scan("silver.firm_universe").select("ticker", "sic").collect().to_pandas()
    calls = calls.merge(sic, on="ticker", how="left")
    calls = calls.merge(L.firm_delistings(), on="ticker", how="left")
    calls["sic2"] = calls["sic"].astype(str).str.zfill(4).str[:2]
    calls["fe"] = calls["sic2"] + "_" + pd.to_datetime(calls["fecha"]).dt.year.astype(str)
    calls.insert(0, "id", calls["call_accession_number"])
    calls["fecha"] = pd.to_datetime(calls["fecha"]).astype("datetime64[ns]")
    bad = sequence_violations(calls)
    if len(bad):
        raise ValueError(f"call-sequence invariant broken by {len(bad)} calls:\n{bad.head(20).to_string()}")
    L.write_gold("spines", "call", "call",
                 calls[L.GOLD_SPINE_COLUMNS["call"] + ["fiscal_period", "call_sequence", "sic", "sic2", "fe", "delisted", "delisting_date"]],
                 builder="scripts/gold/call/build_spine.py", extra={"grain": "call (earnings-call transcript)"})
    print(f"{len(calls):,} calls, {calls['ticker'].nunique()} tickers, "
          f"{calls['fecha'].min().date()} .. {calls['fecha'].max().date()}")


if __name__ == "__main__":
    main()
