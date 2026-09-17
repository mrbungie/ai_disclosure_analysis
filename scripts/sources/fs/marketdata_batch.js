// FactSet Workstation - beta / market cap / shares outstanding / dividends / sector.
// Two endpoints, both same-origin fetch() with credentials:'include' from a logged-in tab.
//
// (1) POINT-IN-TIME SNAPSHOT (current value only, NOT historical - a `date=` param is
//     silently ignored): POST https://my.apps.factset.com/services/TS/fund
//     Content-Type: application/x-www-form-urlencoded
//     Body: symbol=<comma-separated ids>&field=<comma-separated field codes>
//     Confirmed working field codes: BETA, MARKET_VALUE, SHARES_OUTSTANDING,
//     DIVIDEND_YIELD, ANNUAL_DIVIDEND, SECTOR, INDUSTRY, COMPANY_NAME.
//     Unknown field codes are silently dropped from the response (no error), which makes
//     this endpoint cheap to probe for more fields. Multi-symbol batching confirmed
//     (comma-separated `symbol=`). Fast (~150-270ms for 2 symbols x 14 fields).
//     NOTE: BETA here (0.7055 for AAPL on 2026-09-17) is NOT the same number as the
//     Snapshot page's "Beta (3Y Adj.)" key stat (1.02 for AAPL same day) - two different
//     beta definitions/windows exist in FactSet; which window/benchmark this BETA field
//     uses has not been confirmed. Do not treat them as interchangeable without checking
//     against a known reference beta.
//
// (2) HISTORICAL TIME SERIES for market cap (and presumably other COMA_MARKET_DATA
//     metrics - see prices_batch.js for the general pattern): POST
//     https://my.apps.factset.com/services/Fql?app=html_reports&string_na=1
//     symbols=<ids>&exprs=COMA_MARKET_DATA(metric='MKT_VAL',sdate='NOW',edate='NOW-1AY',frq='M')
//     Confirmed: MKT_VAL returns a monthly (or daily, with frq='D') historical series,
//     values match the TS/fund MARKET_VALUE snapshot at t=0. Tried and NOT working as
//     COMA_MARKET_DATA metrics (return "@NA" for every period): SHARES_OUTSTANDING, DPS,
//     ANNUAL_DIVIDEND - these may need a different FQL function (FF_* prefix, as seen in
//     the financial-reports formula strings) rather than COMA_MARKET_DATA; not resolved.
//     A historical *shares outstanding* / *dividend* series is therefore NOT yet confirmed
//     - only market cap has a working historical pull via this route. The SHS (Reported
//     Shares) report in fundamentals_batch.js-style calls (rpt='SHS') DOES carry
//     historical Shs Outstanding / Market Capitalization / Dividend / Dividend Return
//     rows (32 annual columns, 60 quarterly columns tested) - see fundamentals_batch.js
//     and use that route for a real shares/dividends panel instead of COMA_MARKET_DATA.
//
// FILING/REPORT DATE: checked the financial-reports fundamentals endpoint (BAL/INC/CF/
// RATIO/SHS) for a per-period filing or report date (needed for point-in-time
// alignment) - NOT FOUND. Each column only carries the fiscal PERIOD END date (e.g.
// "27 SEP '25"); there is no separate "as reported on" / filing date field in the
// columnData or rowData structures inspected so far, only an opaque encrypted `href`
// per cell that opens a source-document viewer (not machine-readable as a date).

async function fetchMarketDataSnapshot(tickers, fields = [
  'BETA', 'MARKET_VALUE', 'SHARES_OUTSTANDING', 'DIVIDEND_YIELD', 'ANNUAL_DIVIDEND',
  'SECTOR', 'INDUSTRY', 'COMPANY_NAME',
]) {
  const symStr = tickers.map((t) => t + '-US').join(',');
  const body = 'symbol=' + encodeURIComponent(symStr) + '&field=' + encodeURIComponent(fields.join(','));
  const r = await fetch('https://my.apps.factset.com/services/TS/fund', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  return r.json(); // { "TICKER-US": { field: value, ... }, ... }
}

async function fetchMarketCapHistory(tickers, sdate = 'NOW', edate = 'NOW-11AY', frq = 'M') {
  const expr = `COMA_MARKET_DATA(metric='MKT_VAL',sdate='${sdate}',edate='${edate}',frq='${frq}')`;
  const symStr = tickers.map((t) => t + '-US').join(',');
  const body = 'symbols=' + encodeURIComponent(symStr) + '&exprs=' + encodeURIComponent(expr);
  const r = await fetch('https://my.apps.factset.com/services/Fql?app=html_reports&string_na=1', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  const arr = await r.json();
  const out = {};
  for (const rec of arr) {
    out[rec.$symbol] = rec.$error === 0 && rec.$value ? rec.$value.map((v) => v[0]) : null;
  }
  return out;
}
