// FactSet Workstation - bulk financial-statement puller (balance sheet, income statement,
// cash flow, ratio analysis), annual + quarterly, wide history.
// Runs INSIDE a logged-in browser tab (chrome-mcp-server chrome_javascript); same-origin
// fetch() with credentials:'include' reuses the Workstation session cookie.
//
// Endpoint (GET, one ticker + one report + one frequency per call - no multi-id batching,
// tested and confirmed NOT supported: a comma-joined id list returns an empty/malformed table):
//
//   https://my.apps.factset.com/financial-reports/api/aggr/json?
//     ksFinancialsTableFull%5BcomponentName%5D=ksFinancialsTable&
//     ksFinancialsTableFull%5Boptions%5D=<urlencoded options string>
//
// `options` is itself a second-level urlencoded query string (encode once, outer param
// encodes the whole thing again). Key fields:
//   rpt        - report code: 'BAL' | 'INC' | 'CF' | 'RATIO'
//   stdOrArpt  - 'ARPT' (standardized GAAP/IFRS, matches the on-screen report titles)
//   rptType    - 'ANN' (annual) | 'INTM' (interim/quarterly)
//   speriod    - always '0' (most recent period as the right-hand column)
//   eperiod    - negative period count back from speriod. Widest tested WITHOUT errors:
//                  ANN:  eperiod=-30  -> 20 fiscal years returned (looks capped ~20y of
//                        standardized history, e.g. AAPL 2006-2025)
//                  INTM: eperiod=-59  -> 60 fiscal quarters returned (back to ~2011)
//                Going more negative did not add columns (server appears to cap at the
//                data's actual start, not error) - not fully bisected past -60/-30.
//   id         - FactSet id, e.g. 'AAPL-US'. This is the only thing that needs to vary
//                per ticker in an otherwise-fixed options string.
//   (all other params - restated, curn, decimals, units, isFull=true, etc. - copied
//   verbatim from a captured Workstation request; not individually re-tested.)
//
// RESPONSE SHAPE DIFFERS BY REPORT - important gotcha:
//   - BAL / INC / CF: `ksFinancialsTableFull` is a dict of ONE sub-table keyed by an
//     internal id, e.g. `{"ksBalanceSheetTable0": {title, columns, columnData, rowData, ...}}`.
//     Get the table via `Object.values(j.ksFinancialsTableFull)[0]`.
//   - RATIO: `ksFinancialsTableFull` IS the table object directly - no nested wrapper key.
//     `j.ksFinancialsTableFull.title`, `.columns`, `.columnData`, `.rowData` are at the top level.
//   A naive `Object.keys(j.ksFinancialsTableFull)[0]` (picking the first key) works for
//   BAL/INC/CF but for RATIO picks the string key "title" instead of a table - looks like
//   a silent failure (undefined columns) but the request actually succeeded. Use
//   `getTable()` below, which handles both shapes.
//
// Table structure once extracted:
//   table.columns   = ["col-1", "col-2", ...] in display order (left = oldest usually, or
//                     right = most recent - check columnData date labels per column)
//   table.columnData["col-N"].value = [dateLabelString, ...] (human date label, e.g. "SEP '25")
//   table.rowData["rowK"]["col-0"].value = line-item label (col-0 is the label column)
//   table.rowData["rowK"]["col-N"].value = numeric value for that period
//   table.rowData["rowK"]["col-N"].formula = FactSet formula string, e.g.
//     FF_ARPT_SERIES('TA','QTR_R','0','-14','M','BS') - encodes the underlying field code
//     (here 'TA' = total assets) - useful as a stable field key across tickers/periods.
//   table.rowChildren maps parent row ids to child row ids (line-item hierarchy).
//
// Concurrency: this endpoint is comparatively slow (~1-4s per request) and appears to
// rate-limit/return empty tables under high concurrency (~16+ simultaneous requests saw
// ~40% empty/garbled responses in testing). Use LOW concurrency (3-4 simultaneous requests)
// with retry-on-empty, as implemented in fetchOne() below. At concurrency 4 with up to 2
// retries, 40/40 test requests (5 tickers x BAL/INC/CF/RATIO x ANN/INTM) succeeded.
//
// IMPORTANT (per orchestrator note, 2026-09-17): plain `TICKER + '-US'` is NOT a reliable
// FactSet id for the 2021 S&P 500 universe - renamed/delisted tickers (DISCA, FBHS, FLT,
// GPS, HFC, WLTW, SIVB, ...) return nothing, and some (INFO) resolve to the WRONG, newer
// company. Use data/raw/fs/reference/id_map.parquet (ticker -> factset_id) once built,
// and sanity-check the company name in the response against the expected ticker before
// trusting a batch run at scale.

function buildUrl(factsetId, rpt, rptType, speriod, eperiod) {
  const opts =
    `rpt=${rpt}&stdOrArpt=ARPT&rptType=${rptType}&speriod=${speriod}&eperiod=${eperiod}&` +
    `restated=1&grthType=YOY&showExpand=1&showInlineCalc=0&showSpark=0&showGrowthTable=1&` +
    `showCsizeTable=0&compId=DEFAULT&showComp=0&curn=LOCAL&segType=BUS&keyword=SALES&` +
    `acctStd=DEFAULT&part=INC&decimals=-1&units=AUTO&fperiod=0&reversePeriods=0&` +
    `showUnreported=0&showExpandHdr=0&returnChart=1&chartType=total&showAudit=1&isWeb=true&` +
    `datePeriodsDataCenter=Fiscal%20Years%20%26%20Quarters&numberOfPeriodsDataCenter=10&` +
    `reversePeriodsDataCenter=0&datePeriodsTowers=Fiscal%20Years%20%26%20Quarters&` +
    `numberOfPeriodsTowers=10&reversePeriodsTowers=0&rowSelectedOfTable=Financials&` +
    `chartAreaId=8&chartFieldId=27&accordionOpen%5B0%5D=1&id=${factsetId}&` +
    `dbCategories%5B0%5D=fundamental&dynamicBenchmarkId=localmarketindex&` +
    `currencyDialogOption=LOCAL&isFull=true`;
  return (
    'https://my.apps.factset.com/financial-reports/api/aggr/json?' +
    'ksFinancialsTableFull%5BcomponentName%5D=ksFinancialsTable&' +
    'ksFinancialsTableFull%5Boptions%5D=' +
    encodeURIComponent(opts)
  );
}

function getTable(j) {
  const root = j && j.ksFinancialsTableFull;
  if (!root) return null;
  if (root.columns || root.rowData) return root; // RATIO shape (flat)
  const vals = Object.values(root); // BAL/INC/CF shape (nested)
  return vals.length ? vals[0] : null;
}

async function fetchOne(factsetId, rpt, rptType, speriod, eperiod, retries = 2) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await fetch(buildUrl(factsetId, rpt, rptType, speriod, eperiod), {
        credentials: 'include',
      });
      const txt = await r.text();
      const j = JSON.parse(txt);
      const tbl = getTable(j);
      if (tbl && tbl.title) {
        return { factsetId, rpt, rptType, ok: true, title: tbl.title, table: tbl };
      }
    } catch (e) {
      /* fall through to retry */
    }
    await new Promise((res) => setTimeout(res, 800));
  }
  return { factsetId, rpt, rptType, ok: false };
}

// Bounded-concurrency worker pool - keep concurrency LOW (3-4) for this endpoint.
async function runConcurrent(jobs, concurrency = 4) {
  const results = new Array(jobs.length);
  let idx = 0;
  async function worker() {
    while (idx < jobs.length) {
      const my = idx++;
      const [factsetId, rpt, rptType, sp, ep] = jobs[my];
      results[my] = await fetchOne(factsetId, rpt, rptType, sp, ep);
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  return results;
}

// Standard period windows.
const ANNUAL = ['0', '-30']; // ~20 annual columns
const QUARTERLY = ['0', '-59']; // ~60 quarterly columns
const REPORTS = ['BAL', 'INC', 'CF', 'RATIO'];

// Example driver: for one batch of FactSet ids, fetch all 4 reports x 2 frequencies and
// Blob-download one JSON file per ticker (browser tool output truncates well before a
// full RATIO table's ~450KB JSON, so results must leave the page via download).
async function runFundamentalsBatch(factsetIds, concurrency = 4) {
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
  const jobs = [];
  for (const id of factsetIds) {
    for (const rpt of REPORTS) {
      jobs.push([id, rpt, 'ANN', ...ANNUAL]);
      jobs.push([id, rpt, 'INTM', ...QUARTERLY]);
    }
  }
  const results = await runConcurrent(jobs, concurrency);
  // group by ticker
  const byId = {};
  for (const r of results) {
    if (!byId[r.factsetId]) byId[r.factsetId] = {};
    const key = r.rpt + '_' + r.rptType;
    byId[r.factsetId][key] = r.ok ? r.table : null;
  }
  for (const [id, tables] of Object.entries(byId)) {
    download('fs_fundamentals_' + id.replace(/[^A-Za-z0-9]/g, '_') + '.json', tables);
  }
  return { total: jobs.length, failed: results.filter((r) => !r.ok).length };
}
