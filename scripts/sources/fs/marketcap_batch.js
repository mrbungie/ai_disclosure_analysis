// FactSet - historical market cap pull via FQL COMA_MARKET_DATA(metric='MKT_VAL')
// Daily values from 2015-09-17 to 2026-09-17 (11 years).
// Runs INSIDE a logged-in my.apps.factset.com browser tab via chrome_javascript.
//
// Uses the verified `factset_id` list from data/raw/fs/reference/id_map.parquet
// (499 ids) — NOT plain TICKER-US — so renamed/delisted/reused tickers resolve to the
// correct security. Same endpoint/response shape as prices_batch.js.

async function fetchMarketCapBatch(ids) {
  const EXPR =
    "COMA_MARKET_DATA(metric='DATE',sdate='NOW',edate='NOW-11AY',frq='D');;" +
    "COMA_MARKET_DATA(metric='MKT_VAL',sdate='NOW',edate='NOW-11AY',frq='D',curn='LOCAL')";
  const names = ['date', 'market_cap'];
  const symStr = ids.join(',');
  const body = 'symbols=' + encodeURIComponent(symStr) + '&exprs=' + encodeURIComponent(EXPR);

  const r = await fetch('https://my.apps.factset.com/services/Fql?app=html_reports&string_na=1', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
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

// Drives ALL_IDS in batches of 25, Blob-downloading one JSON file per batch as
// fs_marketcap_batch_<startIndex>.json (move from ~/Downloads into
// data/raw/fs/market_cap/ afterwards; archive old wrong batches first).
async function runAllMarketCapBatches(ALL_IDS, batchSize = 25) {
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
  for (let i = 0; i < ALL_IDS.length; i += batchSize) {
    const chunk = ALL_IDS.slice(i, i + batchSize);
    try {
      const out = await fetchMarketCapBatch(chunk);
      download('fs_marketcap_batch_' + String(i).padStart(3, '0') + '.json', out);
      log.push({ i, n: chunk.length, ok: true, symbols: Object.keys(out).length });
    } catch (e) {
      log.push({ i, n: chunk.length, ok: false, err: String(e) });
    }
    await new Promise((res) => setTimeout(res, 400));
  }
  return log;
}
