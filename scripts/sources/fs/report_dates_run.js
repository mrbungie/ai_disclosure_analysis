// FactSet Workstation (my.apps.factset.com) - per-period report/public-date puller.
// Runs INSIDE a logged-in browser tab (chrome-mcp-server chrome_javascript), same-origin fetch()
// with credentials:'include' reuses the Workstation session cookie. No API key needed.
//
// Purpose: for every annual and quarterly fundamentals period, get a date that a value could
// legitimately be treated as "public" on, without relying on an EDGAR filing-date match. This
// lets gold stay point-in-time even for ids where the EDGAR filing_manifest match is missing,
// wrong, or (for delisted names) never existed.
//
// Method (see docs/sources/fs.md "Report and filing dates" section for full verification):
//   POST https://my.apps.factset.com/services/Fql?app=html_reports&string_na=1
//   Body: symbols=<comma-separated ids>&exprs=<FQL expression string, ";;"-joined>
//   Functions used per (id, frequency):
//     FF_FISCAL_DATE(<ANN_R|QTR_R>,0,-N)   - fiscal period-end date (MM/DD/YYYY)
//     FF_EPS_RPT_DATE(<ANN_R|QTR_R>,0,-N)  - earnings-release date (YYYYMMDD); this is the
//                                            EARLIEST public date for that period's headline
//                                            numbers, typically same day to ~3 weeks before the
//                                            10-K/10-Q is actually filed (verified against
//                                            filing_manifest + live EDGAR lookups for AAPL, MSFT,
//                                            JPM - see fs.md). NOT the SEC filing date itself; no
//                                            FactSet FQL function that returns the exact 10-K/10-Q
//                                            EDGAR filing date was found (see fs.md "Unavailable
//                                            items" - audit "Source Linking" and filings-UI network
//                                            endpoints were both tried and did not work through this
//                                            driver; see notes there).
//     FF_SOURCE_DOC(<ANN_R|QTR_R>,0,-N)    - source form type ("10-K" or "10-Q")
//   N=-16 for ANN_R (~17 annual periods, back to ~2009-2010), N=-64 for QTR_R (~65 quarterly
//   periods, back to ~2010). Delisted/merged ids return "@NA" for periods after their last
//   reported one rather than erroring - no special-casing needed.
//
// Low concurrency (<=4 in-flight requests) is used out of caution, mirroring the
// financial-reports/api/aggr endpoint's documented sensitivity to concurrency in fs.md, even
// though this /services/Fql endpoint tolerated 25-id batches fine for prices_batch.js.

function downloadJson(name, obj) {
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

async function fetchReportDatesBatch(ids) {
  const EXPR = [
    "FF_FISCAL_DATE(ANN_R,0,-16)",
    "FF_EPS_RPT_DATE(ANN_R,0,-16)",
    "FF_SOURCE_DOC(ANN_R,0,-16)",
    "FF_FISCAL_DATE(QTR_R,0,-64)",
    "FF_EPS_RPT_DATE(QTR_R,0,-64)",
    "FF_SOURCE_DOC(QTR_R,0,-64)",
  ].join(";;");
  const names = [
    'fiscal_date_ann', 'eps_rpt_date_ann', 'source_doc_ann',
    'fiscal_date_qtr', 'eps_rpt_date_qtr', 'source_doc_qtr',
  ];
  const symStr = ids.join(',');
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

// Fire-and-forget driver: fetches report dates for IDS in batches of `batchSize`, running up to
// `concurrency` batches in parallel, Blob-downloading one JSON file per batch as
// fs_report_dates_<startIndex>.json (move from ~/Downloads into data/raw/fs/report_dates/
// afterwards). Progress is exposed on window.__fsDatesProgress so it can be polled without
// blocking the calling tool (call this without awaiting it from the driver side).
async function runReportDatesBatches(IDS, batchSize = 10, concurrency = 4) {
  const batches = [];
  for (let i = 0; i < IDS.length; i += batchSize) {
    batches.push({ start: i, ids: IDS.slice(i, i + batchSize) });
  }
  window.__fsDatesProgress = {
    total: batches.length,
    done: 0,
    failed: [],
    startedAt: new Date().toISOString(),
    finishedAt: null,
  };

  let nextIdx = 0;
  async function worker() {
    while (nextIdx < batches.length) {
      const b = batches[nextIdx++];
      try {
        const out = await fetchReportDatesBatch(b.ids);
        downloadJson('fs_report_dates_' + String(b.start).padStart(3, '0') + '.json', out);
      } catch (e) {
        window.__fsDatesProgress.failed.push({ start: b.start, ids: b.ids, err: String(e) });
      }
      window.__fsDatesProgress.done++;
      await new Promise((res) => setTimeout(res, 400));
    }
  }

  const workers = [];
  for (let w = 0; w < concurrency; w++) workers.push(worker());
  await Promise.all(workers);
  window.__fsDatesProgress.finishedAt = new Date().toISOString();
  return window.__fsDatesProgress;
}

// Usage (from chrome_javascript, fire-and-forget so the tool call returns immediately):
//   runReportDatesBatches(IDS, 10, 4);
// then poll with a separate chrome_javascript call:
//   return window.__fsDatesProgress;
//
// IDS should come from data/raw/fs/reference/id_map.parquet's factset_id column (499 ids for the
// full run - NOT run at scale by this pass; only 5 ids were tested: AAPL-US, MSFT-US, JPM-US,
// INFO.XX10-US, SIVBQ-US).
