"""
scripts/bronze/sec_submissions.py — bronze.sec_filing_index, bronze.sec_company_names:
the EDGAR submissions history of every firm-universe CIK
(data/raw/sec_submissions/, scripts/raw_ingestion/us/06_fetch_submissions.py).

  bronze.sec_filing_index    one row per filing: cik, accession_number, form,
                             filing_date, report_date, items (8-K item codes,
                             e.g. "2.02,9.01")
  bronze.sec_company_names   one row per (cik, name): the current EDGAR name
                             (kind "current") and every former name with the
                             dates it was used (kind "former")

The `recent` block and the older pages of a CIK overlap nothing; a filing seen
twice keeps one row. No universe filter.
"""

from __future__ import annotations

import json

import polars as pl
from _paths import L, files

BUILDER = "scripts/bronze/sec_submissions.py"
SUBMISSIONS = L.DATA / "raw" / "sec_submissions"
INDEX_FIELDS = {"accessionNumber": "accession_number", "form": "form", "filingDate": "filing_date",
                "reportDate": "report_date", "items": "items"}


def main() -> None:
    sources = files(SUBMISSIONS, "CIK*.json")
    index, names = [], []
    for path in sources:
        data = json.loads(path.read_text())
        cik = path.name[3:13]
        block = data["filings"]["recent"] if "filings" in data else data
        index.append(pl.DataFrame({new: block.get(old, [None] * len(block["form"])) for old, new in INDEX_FIELDS.items()},
                                  schema={c: pl.String for c in INDEX_FIELDS.values()}).with_columns(pl.lit(cik).alias("cik")))
        if "name" in data:
            names.append({"cik": cik, "name": data["name"], "kind": "current", "date_from": None, "date_to": None})
            names += [{"cik": cik, "name": f["name"], "kind": "former", "date_from": (f.get("from") or "")[:10] or None,
                       "date_to": (f.get("to") or "")[:10] or None} for f in data.get("formerNames", [])]
    date = lambda c: pl.col(c).replace("", None).str.to_date("%Y-%m-%d", strict=False)  # noqa: E731
    filings = (pl.concat(index).unique("accession_number", keep="first", maintain_order=True)
               .with_columns(date("filing_date"), date("report_date"), pl.col("items").replace("", None))
               .select("cik", "accession_number", "form", "filing_date", "report_date", "items"))
    L.write_table("bronze.sec_filing_index", filings, keys=["accession_number"], sort_by=["cik", "filing_date", "accession_number"],
                  inputs=sources, builder=BUILDER)
    company_names = (pl.DataFrame(names, schema={"cik": pl.String, "name": pl.String, "kind": pl.String,
                                                 "date_from": pl.String, "date_to": pl.String})
                     .with_columns(date("date_from"), date("date_to"))
                     .unique(["cik", "name"], keep="first", maintain_order=True))
    L.write_table("bronze.sec_company_names", company_names, keys=["cik", "name"], inputs=sources, builder=BUILDER)


if __name__ == "__main__":
    main()
