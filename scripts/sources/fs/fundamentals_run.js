// FactSet Workstation - single clean driver for the fundamentals pull (balance sheet,
// income statement, cash flow, ratio analysis, reported shares - annual + quarterly).
// Supersedes fundamentals_batch.js (kept for reference/history) and the ad-hoc
// fundamentals_supplement_batch.js idea - this ONE file does both a full pull and a
// CF/RATIO-only "supplement" pull for ids already downloaded.
//
// Runs INSIDE a logged-in browser tab (chrome-mcp-server chrome_javascript); same-origin
// fetch() with credentials:'include' reuses the Workstation session cookie. Fire-and-forget:
// call runFundamentals(ids) and let it run; poll window.__fsFundProgress; each id
// Blob-downloads its own JSON when done, independent of the calling tool's connection.
//
// ============================================================================
// THE BUG THIS FIXES (found 2026-09-17 by driving the Cash Flow / Ratio Analysis report
// pages in the browser with chrome_network_capture + direct fetch() probing):
//
//   The original fundamentals_batch.js used rptType='INTM' for ALL FOUR of
//   BAL/INC/CF/RATIO to get "quarterly". That is correct for BAL and INC (confirmed:
//   60 real fiscal-quarter columns, formula basis 'QTR_R', e.g.
//   FF_ARPT_SERIES('TA','QTR_R','0','-14','M','BS')).
//
//   For CF, rptType='INTM' silently returns the SAME annual table as rptType='ANN'
//   (identical column headers, e.g. AAPL CF_INTM headers = "27 SEP '25", "28 SEP '24", ...
//   fiscal YEAR-ends, same as CF_ANN, formula basis stayed 'ANN_R'). This is not a
//   parameter typo on our part - INTM is simply not a valid interim selector for the
//   CF report on this endpoint.
//
//   THE FIX: for the CF report, use rptType='YTD' (not 'INTM'). Confirmed on AAPL:
//   rpt=CF&rptType=YTD&eperiod=-59 returns 60 real fiscal-quarter columns
//   ("27 JUN '26","28 MAR '26","27 DEC '25","27 SEP '25",...), formula basis 'YTD_R'
//   (e.g. FF_ARPT_SERIES('NET INCOME','YTD_R','0','-64','M','CF')).
//
//   IMPORTANT - this is YTD-CUMULATIVE, not discrete-quarter. 'YTD_R' means each
//   column is the cumulative cash flow from the start of that fiscal year through that
//   period end - the standard US 10-Q convention (10-Q cash flow statements report
//   year-to-date, not the discrete quarter). To get a DISCRETE quarter's cash flow
//   (e.g. Q2 alone, not Q1+Q2), the parser/analysis layer must subtract consecutive
//   same-fiscal-year YTD columns (Q2_discrete = YTD_Q2 - YTD_Q1); Q1 and the full-year
//   Q4 column need no adjustment (Q1 YTD = Q1 discrete; Q4 YTD = the annual total).
//   No separate "discrete quarterly" or "LTM" toggle was found for CF on this endpoint -
//   the Workstation UI's own period-type control for the Cash Flow report offers only
//   two choices, "Fiscal Years" and "YTD" (confirmed by enumerating the report's
//   period-type dropdown in the browser) - there is no third "Fiscal Quarters" option
//   for CF the way there is for BAL/INC.
//
//   RATIO: NOT FIXABLE via this endpoint. Every rptType tried (INTM, YTD, QTR, ANN) and
//   every stdOrArpt tried (ARPT, STND) returns byte-identical column headers/values to
//   RATIO with rptType='ANN' (same "SEP 'XX" annual-only labels, same column count,
//   e.g. 46 cols for AAPL regardless of rptType). The Ratio Analysis report on this
//   endpoint appears to be genuinely annual-only - no quarterly ratio series was found.
//   RATIO_INTM is still fetched (for schema/pipeline symmetry) but will equal RATIO_ANN;
//   the parser must detect and flag this rather than silently treat it as real interim
//   data. Quarterly ratios, if needed, should be computed in-repo from the (now-fixed)
//   quarterly BAL/INC/CF fields rather than pulled from FactSet's RATIO report.
//
// SHS (Reported Shares): unaffected by this bug - SHS_INTM already returns 60 real
// quarters (confirmed in original testing), unchanged here.
// ============================================================================
//
// Endpoint (GET, one ticker + one report + one frequency per call - no multi-id batching):
//   https://my.apps.factset.com/financial-reports/api/aggr/json?
//     ksFinancialsTableFull%5BcomponentName%5D=ksFinancialsTable&
//     ksFinancialsTableFull%5Boptions%5D=<urlencoded options string>
//
// RESPONSE SHAPE DIFFERS BY REPORT (see fundamentals_batch.js header comment for detail):
//   BAL/INC/CF: `ksFinancialsTableFull` wraps ONE sub-table keyed by an internal id.
//   RATIO: `ksFinancialsTableFull` IS the table object directly (no wrapper key).
//   getTable() below handles both.
//
// Concurrency: keep LOW (3-4 simultaneous requests) with retry-on-empty - this endpoint
// rate-limits/returns empty tables under higher concurrency (see fundamentals_batch.js).

const TABLE_SPECS = [
  // [outputKey, rpt, rptType, speriod, eperiod]
  ['BAL_ANN', 'BAL', 'ANN', '0', '-30'],
  ['BAL_INTM', 'BAL', 'INTM', '0', '-59'],
  ['INC_ANN', 'INC', 'ANN', '0', '-30'],
  ['INC_INTM', 'INC', 'INTM', '0', '-59'],
  ['CF_ANN', 'CF', 'ANN', '0', '-30'],
  ['CF_INTM', 'CF', 'YTD', '0', '-59'], // FIX: YTD, not INTM - see header comment
  ['RATIO_ANN', 'RATIO', 'ANN', '0', '-30'],
  ['RATIO_INTM', 'RATIO', 'ANN', '0', '-30'], // NOT FIXABLE - duplicates RATIO_ANN, see header comment
  ['SHS_ANN', 'SHS', 'ANN', '0', '-30'],
  ['SHS_INTM', 'SHS', 'INTM', '0', '-59'],
];

// ============================================================================
// STANDARDIZED (STND) TABLE SPECS - added 2026-09-17 for cross-firm-comparable pull.
//
// stdOrArpt='STND' maps every company's as-reported line items onto FactSet's common
// chart of accounts (this is the view the Field Mapping table in fs.md is built
// from). Re-verified rptType per report directly on the STND endpoint rather than
// assuming the ARPT fix carries over - IT DOES NOT for CF:
//
//   BAL/INC/CF/SHS: rptType='INTM' gives REAL fiscal quarters on STND (month diversity
//   confirmed: JUN/MAR/DEC/SEP all present). Unlike ARPT, CF does NOT need rptType='YTD'
//   here - 'INTM' alone already works for CF on the Standardized view. (Not yet
//   understood why ARPT and STND differ this way for CF; verified empirically on AAPL,
//   not derived from any documented FactSet behavior.)
//   RATIO: NOT FIXABLE here either - rptType='INTM'/'YTD'/'QTR' on STND all return the
//   same annual-only ("SEP" only, no month rotation) series as RATIO_ANN, exactly like
//   ARPT. Same permanent limitation, same downstream answer (compute quarterly ratios
//   from the now-interim BAL/INC/CF fields instead).
// ============================================================================
const TABLE_SPECS_STND = [
  ['BAL_ANN', 'BAL', 'ANN', '0', '-30'],
  ['BAL_INTM', 'BAL', 'INTM', '0', '-59'],
  ['INC_ANN', 'INC', 'ANN', '0', '-30'],
  ['INC_INTM', 'INC', 'INTM', '0', '-59'],
  ['CF_ANN', 'CF', 'ANN', '0', '-30'],
  ['CF_INTM', 'CF', 'INTM', '0', '-59'], // STND: plain INTM works, no YTD fix needed here
  ['RATIO_ANN', 'RATIO', 'ANN', '0', '-30'],
  ['RATIO_INTM', 'RATIO', 'ANN', '0', '-30'], // NOT FIXABLE on STND either - see above
  ['SHS_ANN', 'SHS', 'ANN', '0', '-30'],
  ['SHS_INTM', 'SHS', 'INTM', '0', '-59'],
];

// The two keys the "supplement" mode re-fetches for ids already downloaded with the old,
// buggy CF_INTM (and the never-fixed RATIO_INTM, refetched only for interface symmetry -
// it will be identical to RATIO_ANN, which is expected and documented above).
const SUPPLEMENT_KEYS = ['CF_INTM', 'RATIO_INTM'];

function buildUrl(factsetId, rpt, rptType, speriod, eperiod, stdOrArpt = 'ARPT') {
  const opts =
    `rpt=${rpt}&stdOrArpt=${stdOrArpt}&rptType=${rptType}&speriod=${speriod}&eperiod=${eperiod}&` +
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
  const vals = Object.values(root); // BAL/INC/CF/SHS shape (nested)
  return vals.length ? vals[0] : null;
}

async function fetchTable(factsetId, rpt, rptType, speriod, eperiod, stdOrArpt = 'ARPT', retries = 2) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await fetch(buildUrl(factsetId, rpt, rptType, speriod, eperiod, stdOrArpt), {
        credentials: 'include',
      });
      const txt = await r.text();
      const j = JSON.parse(txt);
      const tbl = getTable(j);
      if (tbl && tbl.title) return tbl;
    } catch (e) {
      /* fall through to retry */
    }
    await new Promise((res) => setTimeout(res, 800));
  }
  return null;
}

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

// Bounded-concurrency worker pool over (id, spec) jobs. Groups results by id so each id
// downloads exactly one JSON file once all its specs have resolved.
async function runJobs(ids, specs, concurrency, onIdDone, stdOrArpt = 'ARPT') {
  const jobs = [];
  for (const id of ids) {
    for (const spec of specs) jobs.push([id, spec]);
  }
  const byId = {};
  for (const id of ids) byId[id] = {};
  const remainingPerId = {};
  for (const id of ids) remainingPerId[id] = specs.length;

  let idx = 0;
  let completed = 0;
  const failed = [];

  async function worker() {
    while (idx < jobs.length) {
      const my = idx++;
      const [id, [key, rpt, rptType, sp, ep]] = jobs[my];
      const tbl = await fetchTable(id, rpt, rptType, sp, ep, stdOrArpt);
      byId[id][key] = tbl; // null on failure - recorded, not silently dropped
      if (!tbl) failed.push(`${id}:${key}`);
      remainingPerId[id]--;
      if (remainingPerId[id] === 0) {
        completed++;
        onIdDone(id, byId[id], completed, ids.length, failed.slice());
      }
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  return { failed };
}

/**
 * Full pull: all 10 tables (with the CF/RATIO fix baked in) for a list of FactSet ids.
 * Downloads one JSON file per id, keyed exactly like the existing files
 * (BAL_ANN, BAL_INTM, INC_ANN, INC_INTM, CF_ANN, CF_INTM, RATIO_ANN, RATIO_INTM,
 * SHS_ANN, SHS_INTM) so the parser needs no schema change.
 *
 * @param {string[]} ids - FactSet ids to fetch. Pass the 499 id_map ids MINUS ids already
 *   present on disk (data/raw/fs/fundamentals/fs_fundamentals_batch_*.json) -
 *   compute that diff before calling this, do not hardcode 650 or any other count here.
 * @param {number} concurrency - default 4 (see rate-limit note above).
 */
async function runFundamentals(ids, concurrency = 4) {
  window.__fsFundProgress = {
    mode: 'full',
    total: ids.length,
    done: [],
    failed: [],
    current: null,
    started: new Date().toISOString(),
  };
  const { failed } = await runJobs(ids, TABLE_SPECS, concurrency, (id, tables, completed, total, failedSoFar) => {
    download('fs_fundamentals_' + id.replace(/[^A-Za-z0-9]/g, '_') + '.json', tables);
    window.__fsFundProgress.done.push(id);
    window.__fsFundProgress.current = id;
    window.__fsFundProgress.failed = failedSoFar;
  });
  window.__fsFundProgress.finished = new Date().toISOString();
  return { total: ids.length, failed };
}

/**
 * Supplement pull: re-fetches ONLY CF_INTM (now correctly YTD-cumulative quarterly) and
 * RATIO_INTM (still a duplicate of RATIO_ANN - see header comment) for ids that were
 * already downloaded with the old, buggy fundamentals_batch.js. Downloads one small JSON
 * file per id containing just these two keys; merge into the existing per-id file
 * (or have the parser prefer the supplement file's CF_INTM over the original's) rather
 * than re-downloading all 10 tables.
 *
 * @param {string[]} ids - FactSet ids already present on disk that need the CF_INTM fix.
 */
async function runSupplement(ids, concurrency = 4) {
  window.__fsFundProgress = {
    mode: 'supplement',
    total: ids.length,
    done: [],
    failed: [],
    current: null,
    started: new Date().toISOString(),
  };
  const specs = TABLE_SPECS.filter((s) => SUPPLEMENT_KEYS.includes(s[0]));
  const { failed } = await runJobs(ids, specs, concurrency, (id, tables, completed, total, failedSoFar) => {
    download('fs_fundamentals_supplement_' + id.replace(/[^A-Za-z0-9]/g, '_') + '.json', tables);
    window.__fsFundProgress.done.push(id);
    window.__fsFundProgress.current = id;
    window.__fsFundProgress.failed = failedSoFar;
  });
  window.__fsFundProgress.finished = new Date().toISOString();
  return { total: ids.length, failed };
}

/**
 * Standardized (STND) full pull: same 10-table shape as runFundamentals(), but on the
 * cross-firm-comparable Standardized view (stdOrArpt='STND') instead of As-Reported.
 * Uses its OWN progress object (window.__fsStndProgress) so it never collides with an
 * in-progress runFundamentals()/runSupplement() ARPT pull on window.__fsFundProgress -
 * safe to run in parallel in the same tab (subject to the endpoint's own rate limits;
 * keep combined concurrency across both pulls modest, e.g. 2+2, not 4+4).
 * Downloads one Blob JSON per id, `fs_fundamentals_stnd_<ID>.json` - move these into
 * data/raw/fs/fundamentals_stnd/ (a SEPARATE directory from the ARPT pull's
 * data/raw/fs/fundamentals/, so the two bases never collide on disk either).
 *
 * @param {string[]} ids - FactSet ids to fetch (Standardized ids are the same FactSet ids
 *   used everywhere else, e.g. 'AAPL-US' - no separate id space).
 * @param {number} concurrency - default 4; use 2 while an ARPT pull is also running.
 */
async function runStandardized(ids, concurrency = 4) {
  window.__fsStndProgress = {
    mode: 'standardized',
    total: ids.length,
    done: [],
    failed: [],
    current: null,
    started: new Date().toISOString(),
  };
  const { failed } = await runJobs(
    ids,
    TABLE_SPECS_STND,
    concurrency,
    (id, tables, completed, total, failedSoFar) => {
      download('fs_fundamentals_stnd_' + id.replace(/[^A-Za-z0-9]/g, '_') + '.json', tables);
      window.__fsStndProgress.done.push(id);
      window.__fsStndProgress.current = id;
      window.__fsStndProgress.failed = failedSoFar;
    },
    'STND'
  );
  window.__fsStndProgress.finished = new Date().toISOString();
  return { total: ids.length, failed };
}

// Fire-and-forget entry points - call ONE of these from the browser console / chrome_javascript
// and then poll window.__fsFundProgress / window.__fsStndProgress from separate tool calls;
// do not await the promise itself across tool calls (the page keeps running independently
// either way).
//
//   runFundamentals(['AAPL-US','AAP-US','ABBV-US', ...]);    // ARPT full pull for missing ids
//   runSupplement(['AAPL-US','AAP-US','ABBV-US', ...]);      // ARPT CF/RATIO-only re-pull
//   runStandardized(['AAPL-US','AAP-US','ABBV-US', ...]);    // STND full pull (own progress var)
