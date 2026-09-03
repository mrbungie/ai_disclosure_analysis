# Stock prices — sources

Both countries use the same tool (`yfinance`) and the same storage layout
(`data/raw/market/prices/{SYMBOL}.parquet`, one file per ticker — see
`scripts/03_market_data/01_collect_market_data.py`'s own docstring for the
schema and incremental-fetch contract). Chile doesn't get a separate
directory: its Yahoo symbols already carry the `.SN` (Bolsa de Santiago)
suffix, so `COPEC.SN.parquet` sits right next to `AAPL.parquet` with zero
collision risk.

## US — already built

`scripts/03_market_data/01_collect_market_data.py`. Universe:
`configs/us/config.yaml:corpus.universe.tickers`. Known gap (documented in
that script already): delisted firms aren't served by yfinance — a CRSP/WRDS
export is the only real fix, and the schema's `source` column is already
built to accept `source='crsp'` as a drop-in upgrade.

## Chile — `scripts/cl/03_fetch_market_data.py`

Same tool, same schema, different ticker mapping: Yahoo Finance serves
Bolsa de Santiago under `<NEMO>.SN` (e.g. `CHILE.SN`, `COPEC.SN`,
`ENELAM.SN`). Universe: `configs/cl/universe.csv`'s `nemo` column — the
SAME nemo already used everywhere else in the Chile pipeline, so no new
mapping table.

Known gap, not yet verified at the time of writing: yfinance's coverage
of the less-liquid names in `configs/cl/universe.csv` (small issuers,
name-only debt issuers with no real equity float) is unconfirmed — the
fetch script logs a miss per ticker the same way the US one does, rather
than failing the whole run, so this surfaces as a normal per-ticker gap
to review after a first run, not a blocker to write around now.

## Actual run results (2026-09-03)

86/100 tickers fetched. 4 misses, all nemos with a SPACE in them —
`AZUL AZUL`, `COLO COLO`, `LAS CONDES`, `ORO BLANCO` — which also
tripped a real bug on the first run: `yf.download()` splits a bare
string on whitespace into MULTIPLE tickers, so `"AZUL AZUL.SN"` as a
plain string arg was parsed as two separate symbols and produced
duplicate OHLCV columns on write. Fixed by passing `tickers=[symbol]`
(a one-element list) instead of a bare string — after the fix these 4
just come back as clean per-ticker misses ("possibly delisted; no
timezone found"), not crashes. Worth checking by hand later whether
Yahoo actually has these 4 under a different symbol (the space in the
nemo itself might not be how Yahoo lists them).
