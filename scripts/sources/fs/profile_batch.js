// FactSet - company profile pull via overview-report-api.
// Runs INSIDE a logged-in my.apps.factset.com browser tab via chrome_javascript.
//
// Endpoint: POST /services/overview-report-api/overview-profile/v1/json
// Content-Type: application/json, body {"id": "<factset_id>"} (plain TICKER-US id
// form for most, but must use the verified `factset_id` from id_map.parquet for
// renamed/delisted/reused tickers). One id per request, no batching supported.
//
// Response: data.business.{description,industry,sector,name}, data.contact.*,
// data.size.{revenue,employeeNumber,ev,mcap}, data.stage.{tradeDateRange,foundedYear,
// exchangePrimary,dualListed}, data.meta.sources. Delisted ids may 404/error or return
// an empty/null data payload — recorded as failures, not silently dropped.

async function fetchProfile(id) {
  try {
    const r = await fetch('https://my.apps.factset.com/services/overview-report-api/overview-profile/v1/json', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    if (!r.ok) return { id, ok: false, status: r.status };
    const j = await r.json();
    if (!j || !j.data) return { id, ok: false, status: r.status, empty: true };
    return { id, ok: true, data: j.data, meta: j.meta };
  } catch (e) {
    return { id, ok: false, error: String(e) };
  }
}

// Bounded-concurrency worker pool (concurrency=2 by default — shared browser budget
// with a concurrent fundamentals pull, keep total concurrency <=4 across agents).
// Downloads one JSON file every `chunkSize` ids and exposes window.__fsProfileProgress
// so a stalled/killed run can be inspected and resumed (pass `startAt` to skip ids
// already downloaded).
// Downloads INCREMENTALLY, one file per `chunkSize` ids completed (not only at the
// end) — a page navigation/reload mid-run only loses the in-flight chunk, not
// everything, since window.* state does not survive a navigation.
async function runProfileBatch(ALL_IDS, { concurrency = 2, chunkSize = 50, startAt = 0 } = {}) {
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

  const ids = ALL_IDS.slice(startAt);
  window.__fsProfileProgress = { total: ids.length, done: 0, failed: [] };

  const okCount = { n: 0 };
  const failedIds = [];
  let idx = 0;
  let nextDumpAt = chunkSize;
  let pendingChunk = {};
  const lock = { busy: false };

  async function maybeDump(force = false) {
    if (lock.busy) return;
    if (!force && window.__fsProfileProgress.done < nextDumpAt) return;
    lock.busy = true;
    const startIdx = nextDumpAt - chunkSize;
    download('fs_profile_batch_' + String(startAt + startIdx).padStart(3, '0') + '.json', pendingChunk);
    pendingChunk = {};
    nextDumpAt += chunkSize;
    lock.busy = false;
  }

  let idxCursor = 0;
  async function worker() {
    while (idxCursor < ids.length) {
      const my = idxCursor++;
      const r = await fetchProfile(ids[my]);
      pendingChunk[r.id] = r;
      window.__fsProfileProgress.done++;
      if (r.ok) okCount.n++;
      else failedIds.push(r.id);
      await maybeDump();
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  // Final partial chunk (< chunkSize ids left).
  if (Object.keys(pendingChunk).length) {
    download('fs_profile_batch_' + String(startAt + (nextDumpAt - chunkSize)).padStart(3, '0') + '.json', pendingChunk);
  }

  return { total: ids.length, ok: okCount.n, failed: failedIds };
}
