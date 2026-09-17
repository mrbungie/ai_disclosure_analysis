// FactSet - historical (as-of) consensus EPS estimates via the All Estimates report's
// ksAeTable component. Runs INSIDE a logged-in my.apps.factset.com browser tab.
//
// CONFIRMED (2026-09-17), on AAPL-USA, component=PERSHARE (EPS), rptType=ANN/COMBO:
//   - `estDate=<YYYYMMDD>` gives a REAL historical as-of consensus: for the FY Sep'21
//     column, estDate=20200115 -> 3.77345, estDate=20210115 -> 4.02458,
//     estDate=20230115 -> 5.61 (actual, already reported) -- values move plausibly
//     toward the reported actual as estDate advances. This is a genuine as-of series,
//     not a cache-busted no-op.
//   - Column headers (`Sep '20`...`Sep '24` etc) are FIXED relative to the *current*
//     wall-clock date via the `period` offset, NOT relative to `estDate`. `period`
//     shifts which fiscal years appear: empirically (as of 2026-09-17) `period=-1`
//     puts `Sep '24` in the first column, `period=-5` puts `Sep '20` there, i.e.
//     `first_col_fiscal_year = 2025 + period`. This "2025" constant is a function of
//     *today's* date at run time, not of estDate -- recompute it once per run with a
//     calibration call (see `calibrateBaseYear()`), don't hardcode it.
//   - `statuser` (MEAN/MEDIAN/HIGH/LOW/STDDEV/NUMESTS/...) has NO effect on this
//     endpoint -- all returned byte-identical EPS values in testing. No high/low/
//     std-dev/#-of-estimates field was found here. Do not claim these fields exist.
//   - `component=SALES/EBITDA/CAPEX` do NOT work on ksAeTable (title comes back
//     "@NA"). SALES/EBITDA/CAPEX ARE available, but only as a CURRENT (non-historical)
//     consensus via the separate Estimate Summary endpoint (`ksEsmEstimatesLeft`,
//     `kitem=SALES|EBITDA|CAPEX`) -- confirmed the values there do NOT change with
//     `estDate` (byte-identical for estDate=20200115 vs NOW), so that endpoint cannot
//     build a historical panel; it can only give the current snapshot. `kitem=DPS` is
//     NOT a valid code there either -- it silently falls back to the EPS table (title
//     comes back "EPS", not "DPS" or "Dividend"), so DPS is NOT available via this
//     report family. Do not run a historical pull for SALES/EBITDA/CAPEX/DPS -- only
//     EPS has a working as-of mechanism.
//   - id form for this report family is `<ticker>-USA` (not `-US`). Built from
//     `id_map.parquet`'s `factset_id` by replacing a trailing `-US` with `-USA` (holds
//     for suffixed ids too, e.g. `SIVBQ-US` -> `SIVBQ-USA`, `BRK.B-US` -> `BRK.B-USA`,
//     confirmed for SIVBQ/FRCB/BRK.B/WBD).
//
// SCOPE: this driver pulls monthly as-of EPS MEAN consensus (annual FY1/FY2 + the
// visible quarterly columns in the same window) for 2015-01 through 2026-09. It does
// NOT attempt SALES/EBITDA/CAPEX/DPS (not historically available) or high/low/stddev/
// numEst (not found at all on this endpoint).

function usaId(factsetId) {
  return factsetId.replace(/-US$/, '-USA');
}

async function fetchAeTable(ticker, period, estDate, rptType = 'COMBO') {
  const opts = `component=PERSHARE&rptType=${rptType}&statuser=MEAN&period=${period}&speriod=-4&eperiod=5&estDate=${estDate}&ticker=${ticker}`;
  const url = 'https://my.apps.factset.com/estimate-reports/all-estimates/api/component/ksAeTable/json?' + opts;
  const r = await fetch(url, { credentials: 'include' });
  if (!r.ok) return { ok: false, status: r.status };
  const j = await r.json();
  if (!j || !j.rows || !j.rows.length) return { ok: false, empty: true };
  const header = j.rows[0].fragments && j.rows[0].fragments[0] ? j.rows[0].fragments[0].cells.map((c) => c.value) : null;
  const epsRow = j.rows.find((row) => row.fragments && row.fragments[0] && row.fragments[0].cells[0] && row.fragments[0].cells[0].value === 'EPS');
  const epsVals = epsRow ? epsRow.fragments[0].cells.map((c) => c.value) : null;
  return { ok: true, header, eps: epsVals };
}

// Calibrate the `period` <-> fiscal-year offset once per run (depends on wall-clock
// "now", not on estDate). Returns baseYear such that
// first_col_fiscal_year = baseYear + period.
async function calibrateBaseYear(anchorTicker = 'AAPL-USA') {
  const r = await fetchAeTable(anchorTicker, -1, 'NOW', 'ANN');
  if (!r.ok || !r.header) throw new Error('calibration failed: ' + JSON.stringify(r));
  const m = /Sep '(\d{2})/.exec(r.header[1]) || /'(\d{2})/.exec(r.header[1]);
  if (!m) throw new Error('could not parse calibration header: ' + JSON.stringify(r.header));
  const yy = parseInt(m[1], 10);
  const firstColYear = 2000 + yy; // works through 2099
  return firstColYear - (-1); // baseYear
}

function monthRange(startYYYYMM, endYYYYMM) {
  const out = [];
  let [y, m] = startYYYYMM.split('-').map(Number);
  const [ey, em] = endYYYYMM.split('-').map(Number);
  while (y < ey || (y === ey && m <= em)) {
    out.push(`${y}${String(m).padStart(2, '0')}15`);
    m++;
    if (m > 12) { m = 1; y++; }
  }
  return out;
}

// Pull one ticker's full monthly as-of EPS history. `targetFyForMonth(estDateStr,
// baseYear)` picks the `period` offset so the window covers FY1 (the next fiscal year
// end at/after the as-of month) and a couple of years around it.
function periodForEstDate(estDateStr, baseYear) {
  const y = parseInt(estDateStr.slice(0, 4), 10);
  // FY1 as of this date: assume FY ends in Sep (verified for AAPL; a per-ticker fiscal
  // year-end would need the SHS/fundamentals report's period-end month, not attempted
  // here) -- target the fiscal year whose Sep close is the first one at/after estDate.
  const mo = parseInt(estDateStr.slice(4, 6), 10);
  const fy1 = mo <= 9 ? y : y + 1;
  return fy1 - baseYear;
}

async function pullTickerHistory(factsetId, baseYear, months, concurrency = 2) {
  const ticker = usaId(factsetId);
  const results = [];
  let idx = 0;
  async function worker() {
    while (idx < months.length) {
      const my = idx++;
      const estDate = months[my];
      const period = periodForEstDate(estDate, baseYear);
      let r;
      try {
        r = await fetchAeTable(ticker, period, estDate, 'COMBO');
      } catch (e) {
        r = { ok: false, error: String(e) };
      }
      results.push({ factset_id: factsetId, as_of: estDate, period, ...r });
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  return results;
}

// Drives ALL_IDS x monthly as-of dates, Blob-downloading one JSON file PER TICKER
// (not per batch -- ~140 months/ticker is already a sizeable payload) as soon as that
// ticker finishes, and exposing window.__fsEstProgress for polling/resume.
async function runEstimatesBatch(ALL_IDS, { startMonth = '2015-01', endMonth = '2026-09', concurrency = 2, startAt = 0 } = {}) {
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

  const baseYear = await calibrateBaseYear();
  const months = monthRange(startMonth, endMonth);
  const ids = ALL_IDS.slice(startAt);
  window.__fsEstProgress = { total: ids.length, done: 0, failedTickers: [], baseYear };

  for (const factsetId of ids) {
    const rows = await pullTickerHistory(factsetId, baseYear, months, concurrency);
    const okCount = rows.filter((r) => r.ok).length;
    download('fs_estimates_' + factsetId.replace(/[^A-Za-z0-9]/g, '_') + '.json', rows);
    window.__fsEstProgress.done++;
    if (okCount === 0) window.__fsEstProgress.failedTickers.push(factsetId);
  }
  return window.__fsEstProgress;
}

// Quick 3-id smoke test (small month range) -- run this BEFORE the full pull.
async function testEstimatesBatch() {
  const baseYear = await calibrateBaseYear();
  const months = monthRange('2020-01', '2020-03');
  const ids = ['AAPL-US', 'SIVBQ-US', 'BRK.B-US'];
  const out = {};
  for (const id of ids) {
    out[id] = await pullTickerHistory(id, baseYear, months, 2);
  }
  return { baseYear, out };
}
