// FactSet - re-pull prices for fixed/renamed tickers that need corrections
// Runs INSIDE a logged-in my.apps.factset.com browser tab via chrome_javascript
// Fixed tickers (10 total, only 9 in universe): SIVBQ-US, INFO.XX10-US, PARAA-US, FRCB-US,
// and renamed: WBD-US, FBIN-US, CPAY-US, GAP-US, DINO-US, WTW-US

async function fetchPricesBatch(tickers) {
  const EXPR =
    "COMA_MARKET_DATA(metric='DATE',sdate='NOW',edate='NOW-11AY',frq='D');;" +
    "COMA_MARKET_DATA(metric='PRICE',sdate='NOW',edate='NOW-11AY',frq='D',curn='LOCAL',split='SPLIT',spinadj='1');;" +
    "COMA_MARKET_DATA(metric='VOLUME',sdate='NOW',edate='NOW-11AY',frq='D',split='SPLIT');;" +
    "COMA_MARKET_DATA(metric='TOTAL_RETURN',sdate='NOW',edate='NOW-11AY',frq='D',curn='LOCAL',split='SPLIT',spinadj='1')";
  const names = ['date', 'price', 'volume', 'total_return'];
  const symStr = tickers.join(',');
  const body = 'symbols=' + encodeURIComponent(symStr) + '&exprs=' + encodeURIComponent(EXPR);

  const r = await fetch('https://my.apps.factset.com/services/Fql?app=html_reports&string_na=1', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body,
  });

  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();

  // Parse response into {factset_id: {date, price, volume, total_return}}
  const result = {};
  const groupBySymbol = {};

  for (const item of data) {
    const sym = item.$symbol;
    if (!groupBySymbol[sym]) {
      groupBySymbol[sym] = {};
    }

    const expr = item.$expression;
    const idx = data.filter(x => x.$symbol === sym).indexOf(item) % names.length;
    const metricName = names[idx];

    // Extract values from the nested array structure [[v0],[v1],...]
    const values = (item.$value || []).map(x => (Array.isArray(x) && x.length > 0) ? x[0] : null);
    groupBySymbol[sym][metricName] = values;
  }

  // Convert symbol keys to lowercase for consistency
  for (const sym of Object.keys(groupBySymbol)) {
    result[sym] = groupBySymbol[sym];
  }

  return result;
}

async function runFixedTickersPullFinal() {
  // The 10 fixed tickers (including FRC which is not in universe, but still document its pull)
  const fixedTickers = [
    'SIVBQ-US',    // SVB Financial Group failed 2023-05-02
    'INFO.XX10-US', // IHS Markit merged 2022-02-28
    'PARAA-US',    // Paramount merged 2025-08-07
    'FRCB-US',     // First Republic Bank failed 2023-05-01
    'WBD-US',      // Discovery -> Warner Bros Discovery renamed 2022-04-11
    'FBIN-US',     // Fortune Brands Home & Security -> Fortune Brands Innovations
    'CPAY-US',     // FleetCor Technologies -> Corpay renamed
    'GAP-US',      // Gap Inc trades under GAP-US (not GPS-US)
    'DINO-US',     // HollyFrontier -> HF Sinclair renamed 2022-03-15
    'WTW-US',      // Willis Towers Watson -> WTW renamed 2022-01
  ];

  console.log(`Pulling prices for ${fixedTickers.length} fixed/renamed tickers...`);

  const batchSize = 5; // 5 per batch for smaller responses
  let batchNum = 1;

  for (let i = 0; i < fixedTickers.length; i += batchSize) {
    const batch = fixedTickers.slice(i, i + batchSize);
    console.log(`\nBatch ${batchNum} (${batch.join(', ')})...`);

    try {
      const result = await fetchPricesBatch(batch);

      // Download as JSON file
      const filename = `fs_prices_batch_fixed_${String(batchNum).padStart(2, '0')}.json`;
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      console.log(`Downloaded ${filename}`);
      console.log(`  Symbols: ${Object.keys(result).join(', ')}`);

      // Small pause before next batch (avoid rate limiting)
      if (i + batchSize < fixedTickers.length) {
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    } catch (err) {
      console.error(`Batch ${batchNum} failed:`, err.message);
      throw err;
    }

    batchNum++;
  }

  console.log(`\nCompleted: Downloaded ${Math.ceil(fixedTickers.length / batchSize)} batch files`);
  console.log('Move files from ~/Downloads to data/raw/fs/prices/daily/');
  return 'done';
}

// Execute
runFixedTickersPullFinal();
