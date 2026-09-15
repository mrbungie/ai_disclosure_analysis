"""
scripts/silver/patents.py — silver.patents_firm_year, silver.patents_preshock:
bronze.patents_firm_year / bronze.patents_preshock restricted to the analysis
universe (silver.firm_universe), the same conforming every other silver table
applies before gold reads it.
"""

from __future__ import annotations

from _paths import L, bronze_inputs

BUILDER = "scripts/silver/patents.py"


def main() -> None:
    tickers = L.scan("silver.firm_universe").select("ticker").lazy()

    firm_year = L.scan("bronze.patents_firm_year").join(tickers, on="ticker", how="semi")
    L.write_table("silver.patents_firm_year", firm_year, keys=["ticker", "year"],
                  inputs=bronze_inputs("silver.firm_universe", "bronze.patents_firm_year"), builder=BUILDER)

    preshock = L.scan("bronze.patents_preshock").join(tickers, on="ticker", how="semi")
    L.write_table("silver.patents_preshock", preshock, keys=["ticker"],
                  inputs=bronze_inputs("silver.firm_universe", "bronze.patents_preshock"), builder=BUILDER)


if __name__ == "__main__":
    main()
