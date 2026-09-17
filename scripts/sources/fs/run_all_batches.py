#!/usr/bin/env python3
"""
Automate processing of all 499 FactSet IDs in batches of 10.
This script generates the JavaScript calls needed and guides through execution.
"""

import json
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path("/Users/goviedb/Development/thesis")
DATA_DIR = REPO_ROOT / "data" / "raw" / "fs" / "fundamentals"
PROGRESS_FILE = DATA_DIR / "_progress.json"
DOWNLOADS_DIR = Path.home() / "Downloads"

def load_ids():
    """Load all factset IDs."""
    with open('/tmp/factset_ids.json') as f:
        return json.load(f)

def load_progress():
    """Load progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_ids": [], "batches_completed": 0}

def save_progress(progress):
    """Save progress file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def move_downloads(batch_num):
    """Move downloaded files from Downloads to data dir."""
    moved = []
    for file in DOWNLOADS_DIR.glob('fs_fundamentals_*.json'):
        if file.is_file():
            # Extract ID from filename
            safe_name = file.stem.replace('fs_fundamentals_', '')
            # Move to batch directory
            dest = DATA_DIR / f"fs_fundamentals_batch_{batch_num:03d}_{file.name.replace('fs_fundamentals_', '')}"
            file.rename(dest)
            moved.append(dest.name)
    return moved

def generate_batch_javascript(batch_ids, batch_num):
    """Generate JavaScript code for a batch."""
    ids_json = json.dumps(batch_ids)
    return f"""
function buildUrl(factsetId, rpt, rptType, speriod, eperiod) {{
  const opts = `rpt=${{rpt}}&stdOrArpt=ARPT&rptType=${{rptType}}&speriod=${{speriod}}&eperiod=${{eperiod}}&restated=1&grthType=YOY&showExpand=1&showInlineCalc=0&showSpark=0&showGrowthTable=1&showCsizeTable=0&compId=DEFAULT&showComp=0&curn=LOCAL&segType=BUS&keyword=SALES&acctStd=DEFAULT&part=INC&decimals=-1&units=AUTO&fperiod=0&reversePeriods=0&showUnreported=0&showExpandHdr=0&returnChart=1&chartType=total&showAudit=1&isWeb=true&datePeriodsDataCenter=Fiscal%20Years%20%26%20Quarters&numberOfPeriodsDataCenter=10&reversePeriodsDataCenter=0&datePeriodsTowers=Fiscal%20Years%20%26%20Quarters&numberOfPeriodsTowers=10&reversePeriodsTowers=0&rowSelectedOfTable=Financials&chartAreaId=8&chartFieldId=27&accordionOpen%5B0%5D=1&id=${{factsetId}}&dbCategories%5B0%5D=fundamental&dynamicBenchmarkId=localmarketindex&currencyDialogOption=LOCAL&isFull=true`;
  return 'https://my.apps.factset.com/financial-reports/api/aggr/json?ksFinancialsTableFull%5BcomponentName%5D=ksFinancialsTable&ksFinancialsTableFull%5Boptions%5D=' + encodeURIComponent(opts);
}}

function getTable(j) {{
  const root = j && j.ksFinancialsTableFull;
  if (!root) return null;
  if (root.columns || root.rowData) return root;
  const vals = Object.values(root);
  return vals.length ? vals[0] : null;
}}

async function fetchOne(factsetId, rpt, rptType, speriod, eperiod, retries = 2) {{
  for (let attempt = 0; attempt <= retries; attempt++) {{
    try {{
      const r = await fetch(buildUrl(factsetId, rpt, rptType, speriod, eperiod), {{ credentials: 'include' }});
      const txt = await r.text();
      const j = JSON.parse(txt);
      const tbl = getTable(j);
      if (tbl && tbl.title) return {{ factsetId, rpt, rptType, ok: true, title: tbl.title, table: tbl }};
    }} catch (e) {{}}
    if (attempt < retries) await new Promise((res) => setTimeout(res, 800));
  }}
  return {{ factsetId, rpt, rptType, ok: false }};
}}

async function runConcurrent(jobs, concurrency = 4) {{
  const results = new Array(jobs.length);
  let idx = 0;
  async function worker() {{
    while (idx < jobs.length) {{
      const my = idx++;
      const [factsetId, rpt, rptType, sp, ep] = jobs[my];
      results[my] = await fetchOne(factsetId, rpt, rptType, sp, ep);
    }}
  }}
  await Promise.all(Array.from({{ length: concurrency }}, worker));
  return results;
}}

function download(name, obj) {{
  const blob = new Blob([JSON.stringify(obj)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

const batchIds = {ids_json};
const REPORTS = ['BAL', 'INC', 'CF', 'RATIO', 'SHS'];
const ANNUAL = ['0', '-30'];
const QUARTERLY = ['0', '-59'];
const jobs = [];
for (const id of batchIds) {{
  for (const rpt of REPORTS) {{
    jobs.push([id, rpt, 'ANN', ...ANNUAL]);
    jobs.push([id, rpt, 'INTM', ...QUARTERLY]);
  }}
}}

const results = await runConcurrent(jobs, 4);
const byId = {{}};
for (const r of results) {{
  if (!byId[r.factsetId]) byId[r.factsetId] = {{}};
  const key = r.rpt + '_' + r.rptType;
  byId[r.factsetId][key] = r.ok ? r.table : null;
}}
for (const [id, tables] of Object.entries(byId)) {{
  download('fs_fundamentals_' + id.replace(/[^A-Za-z0-9]/g, '_') + '.json', tables);
}}

const failed = results.filter((r) => !r.ok).length;
return {{
  batch: {batch_num},
  ids: batchIds.length,
  total_requests: results.length,
  successful: results.length - failed,
  failed: failed,
  downloads: Object.keys(byId).length
}};
"""

def main():
    """Main entry point."""
    ids = load_ids()
    progress = load_progress()

    completed = set(progress.get('completed_ids', []))
    remaining = [id for id in ids if id not in completed]

    print(f"FactSet Fundamentals Batch Processor")
    print(f"Total IDs: {len(ids)}")
    print(f"Completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Current batch size: 10 IDs")
    print(f"Batches needed: {(len(remaining) + 9) // 10}\n")

    if not remaining:
        print("All IDs already downloaded!")
        return

    # Show what needs to be done
    batch_num = progress.get('batches_completed', 0) + 1
    for i in range(0, len(remaining), 10):
        batch_ids = remaining[i:i+10]
        print(f"Batch {batch_num}: {len(batch_ids)} IDs")
        print(f"  {', '.join(batch_ids[:3])} ... {', '.join(batch_ids[-1:])}")

        # Save JavaScript for manual execution if needed
        js_code = generate_batch_javascript(batch_ids, batch_num)
        js_file = DATA_DIR / f"batch_{batch_num:02d}.js"
        with open(js_file, 'w') as f:
            f.write(js_code)

        batch_num += 1

    print("\nTo execute batches:")
    print("1. Ensure Chrome tab with FactSet is open and active")
    print("2. Copy and paste JavaScript from batch_*.js files into console")
    print("3. Or use: ./scripts/sources/fs/execute_batches.sh\n")

if __name__ == '__main__':
    main()
