// FactSet Workstation (my.apps.factset.com) - bulk daily price/volume/total-return puller.
// Runs INSIDE a logged-in browser tab (chrome-mcp-server chrome_javascript), same-origin fetch()
// with credentials:'include' reuses the Workstation session cookie. No API key needed.
//
// Endpoint: POST https://my.apps.factset.com/services/Fql?app=html_reports&string_na=1
// Content-Type: application/x-www-form-urlencoded
// Body: symbols=<comma-separated FactSet ids>&exprs=<FQL expression string>
//   - symbols: e.g. "AAPL-US,MSFT-US" (id = TICKER + "-US" for US listings; special cases like
//     BRK.B / BF.B need a different FactSet id form - not yet resolved, see docs/sources/fs.md)
//   - exprs: FactSet Query Language (FQL). Chain multiple metrics in ONE string with ";;":
//       COMA_MARKET_DATA(metric='DATE',sdate='NOW',edate='NOW-11AY',frq='D')
//       COMA_MARKET_DATA(metric='PRICE',sdate='NOW',edate='NOW-11AY',frq='D',curn='LOCAL',split='SPLIT',spinadj='1')
//       COMA_MARKET_DATA(metric='VOLUME',sdate='NOW',edate='NOW-11AY',frq='D',split='SPLIT')
//       COMA_MARKET_DATA(metric='TOTAL_RETURN',sdate='NOW',edate='NOW-11AY',frq='D',curn='LOCAL',split='SPLIT',spinadj='1')
//     IMPORTANT: repeated `exprs=` params in the body do NOT work (server only evaluates one) -
//     you must join metrics with ";;" inside a single exprs value.
//   - sdate/edate accept 'NOW' and relative offsets like 'NOW-11AY' (11 years). 11AY was the widest
//     tested and returned ~2766 daily observations per symbol (back to ~2015).
//   - split='SPLIT' adjusts for stock splits; spinadj='1' adjusts for spinoffs. Omit spinadj for VOLUME.
//
// Response: JSON array, one record per (symbol, expression) pair, in the order
// [expr1 for sym1, expr2 for sym1, ..., expr1 for sym2, expr2 for sym2, ...].
// Each record: {"$error":0|1, "$expression":"...", "$symbol":"AAPL-US", "$value":[[v0],[v1],...]}
// $value is a list of 1-element lists, most-recent-date first.
//
// This function fetches one batch of tickers and returns {ticker-US: {date,price,volume,total_return}}.
// For bulk runs, call in batches of ~25 tickers (response ~2.3MB/batch at 11y history) with a short
// sleep between batches, and Blob-download each batch's JSON rather than returning it inline (browser
// tool output is truncated/sanitized above ~50KB).
async function fetchPricesBatch(tickers) {
  const EXPR =
    "COMA_MARKET_DATA(metric='DATE',sdate='NOW',edate='NOW-11AY',frq='D');;" +
    "COMA_MARKET_DATA(metric='PRICE',sdate='NOW',edate='NOW-11AY',frq='D',curn='LOCAL',split='SPLIT',spinadj='1');;" +
    "COMA_MARKET_DATA(metric='VOLUME',sdate='NOW',edate='NOW-11AY',frq='D',split='SPLIT');;" +
    "COMA_MARKET_DATA(metric='TOTAL_RETURN',sdate='NOW',edate='NOW-11AY',frq='D',curn='LOCAL',split='SPLIT',spinadj='1')";
  const names = ['date', 'price', 'volume', 'total_return'];
  const symStr = tickers.map((t) => t + '-US').join(',');
  const body = 'symbols=' + encodeURIComponent(symStr) + '&exprs=' + encodeURIComponent(EXPR);
  const r = await fetch('https://my.apps.factset.com/services/Fql?app=html_reports&string_na=1', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  const arr = await r.json();
  const out = {};
  let idx = 0;
  for (const rec of arr) {
    const sym = rec.$symbol;
    if (!out[sym]) out[sym] = {};
    const mi = idx % names.length;
    out[sym][names[mi]] = rec.$error === 0 && rec.$value ? rec.$value.map((v) => v[0]) : null;
    idx++;
  }
  return out;
}

// Example driver: fetch all TICKERS in batches of 25, downloading one JSON file per batch as
// fs_prices_batch_<startIndex>.json (move from ~/Downloads into
// data/raw/fs/prices/daily/ afterwards).
async function runAllPricesBatches(TICKERS, batchSize = 25) {
  const log = [];
  function download(name, obj) {
    const blob = new Blob([JSON.stringify(obj)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
  for (let i = 0; i < TICKERS.length; i += batchSize) {
    const chunk = TICKERS.slice(i, i + batchSize);
    try {
      const out = await fetchPricesBatch(chunk);
      download('fs_prices_batch_' + String(i).padStart(3, '0') + '.json', out);
      log.push({ i, n: chunk.length, ok: true });
    } catch (e) {
      log.push({ i, n: chunk.length, ok: false, err: String(e) });
    }
    await new Promise((res) => setTimeout(res, 400));
  }
  return log;
}
