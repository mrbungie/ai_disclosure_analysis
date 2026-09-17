#!/usr/bin/env python
"""
FactSet fundamentals parser (polars): read raw per-id JSON files, output a single
long-format parquet with one row per (id, report, frequency, period, field).

Usage: uv run python scripts/sources/fs/parse_fundamentals.py

Input files (rerunnable - picks up whatever exists on disk each time it's run), two bases:

  ARPT (As-Reported - one company's own statement line items verbatim):
  - data/raw/fs/fundamentals/fs_fundamentals_batch_NNN_<ID>.json   (older naming)
  - data/raw/fs/fundamentals/fs_fundamentals_<ID>.json            (fundamentals_run.js
    runFundamentals() naming, no batch prefix)
  - data/raw/fs/fundamentals/supplement/fs_fundamentals_supplement_<ID>.json
    (fundamentals_run.js runSupplement() output - CF_INTM + RATIO_INTM only, for ids that
    were downloaded before the CF_INTM/rptType=YTD fix; OVERRIDES those two keys in the
    main file for the same id, both files are read regardless of which directory Haiku
    actually drops the supplement files in, see SUPPLEMENT_DIRS below)

  STND (Standardized - FactSet's common chart of accounts, comparable across all 499
  tickers; see fs.md "Field mapping"):
  - data/raw/fs/fundamentals_stnd/fs_fundamentals_stnd_<ID>.json
    (fundamentals_run.js runStandardized() output - a SEPARATE directory from ARPT so the
    two bases never collide on disk; files matching *_stnd_* under the ARPT dir are
    excluded from the ARPT scan and picked up here instead, in case a file-mover drops
    them in the wrong place)
  - data/raw/fs/fundamentals_stnd/fs_fundamentals_stndx_<ID>.json (an
    "STND-extra" file carrying only SHS_ANN, SHS_INTM, RATIO_ANN - the three tables that
    were missing from a batch of earlier STND pulls; checked in
    data/raw/fs/fundamentals_stnd/ then data/raw/fs/fundamentals/, same
    file-mover fallback as the main STND file). When present for an id, its three keys
    OVERRIDE the same keys in that id's main STND file (they are supplementary, not
    additive - a stndx file never contains BAL/INC/CF). Rerunnable: an id with a STND
    file but no stndx file yet is still parsed from the main file alone; the merge picks
    up automatically once its stndx file lands, no code change needed.

Each raw JSON has 10 keys: BAL_ANN, BAL_INTM, INC_ANN, INC_INTM, CF_ANN, CF_INTM,
RATIO_ANN, RATIO_INTM, SHS_ANN, SHS_INTM. Each value is either null (fetch failed) or a
FactSet "financials table" dict: {title, footnotes, columns, columnData, rowData,
rowChildren, rows, ...}.

Output: data/raw/fs/fundamentals/fundamentals_long.parquet, columns:
  ticker, factset_id, basis (ARPT|STND), report (BAL|INC|CF|RATIO|SHS),
  frequency (annual|quarterly), period_end (date), period_status
  (PRELIM/RECAP/Restate/blank), field (row label), field_id (stable FactSet code, see
  extract_field_id below), level (row indent, 0 = top-level), value (float), units,
  scale, currency, frequency_ok (bool - False when this table's periods don't actually
  rotate through distinct fiscal months, i.e. FactSet did not return real interim data
  here - see fs.md "Cash flow and ratio interim bug").

IMPORTANT - the CF_INTM fix is DIFFERENT per basis:
  - ARPT: rptType='INTM' is broken for CF (silently returns the annual table); the fix is
    rptType='YTD', and the result is YTD-CUMULATIVE, not discrete-quarter (US 10-Q
    convention: a 10-Q's cash flow statement reports year-to-date, not the standalone
    quarter).
  - STND: rptType='INTM' already works for CF (real discrete fiscal quarters, verified by
    month diversity) - no YTD substitution needed or used here.
  This parser does NOT convert ARPT's YTD to discrete quarters - it emits exactly what
  FactSet returns (frequency='quarterly', ARPT CF values are YTD-cumulative-through-
  period_end, STND CF values are discrete). Any downstream consumer that wants a discrete
  quarter's cash flow from the ARPT basis must compute
    discrete_Q = ytd_value(Q) - ytd_value(previous Q, same fiscal year)
  (Q1 needs no adjustment; the last quarter of the fiscal year needs no adjustment either
  since ytd_value(Q4) == the annual total) - or just use the STND basis's CF_INTM directly,
  which is already discrete. That derivation belongs in the downstream silver/gold build,
  not here, so the raw YTD series stays available for anyone who needs the As-Reported
  cumulative figure directly (e.g. matching a specific 10-Q's reported YTD operating cash
  flow line item).

RATIO_INTM has no fix on EITHER basis - every rptType tried on the FactSet endpoint
returns the same annual-only series for RATIO (see fs.md). frequency_ok=False for
every RATIO_INTM row; quarterly ratios should be computed downstream from the (correctly
interim) BAL/INC/CF fields, not read from this column, on either basis.
"""

import calendar
import glob
import json
import re
from datetime import date
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).parent.parent.parent.parent
FUND_DIR = REPO_ROOT / "data" / "raw" / "fs" / "fundamentals"
FUND_DIR_STND = REPO_ROOT / "data" / "raw" / "fs" / "fundamentals_stnd"
OUTPUT_FILE = FUND_DIR / "fundamentals_long.parquet"

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

NULL_MARKERS = {"@NaNd", "@NA", "NA", "N/A", "-", "", None}

# Where a supplement (CF_INTM/RATIO_INTM-only) file for a given id might live - checked
# in order, first match wins. Kept as a list because the Haiku downloader's exact drop
# location has moved between sessions; add to this list rather than assuming one path.
SUPPLEMENT_DIRS = [
    FUND_DIR / "supplement",
    FUND_DIR,
]


def parse_period_label(label):
    """
    '27 SEP '25' -> (date(2025,9,27), True)   [day known]
    "SEP '25"    -> (date(2025,9,1), False)   [day unknown - placeholder day=1]
    Returns (None, False) if unparseable.
    """
    if not label or not isinstance(label, str):
        return None, False
    label = label.strip().upper().replace("'", " ")
    parts = label.split()
    if len(parts) == 2:  # "SEP 25"
        mon, yr = parts
        day = None
    elif len(parts) == 3:  # "27 SEP 25"
        day_s, mon, yr = parts
        try:
            day = int(day_s)
        except ValueError:
            return None, False
    else:
        return None, False
    if mon not in MONTHS:
        return None, False
    try:
        year = int(yr)
    except ValueError:
        return None, False
    if year < 100:
        # 2-digit year: FactSet labels never go more than ~2 fiscal years past "now" (the
        # current/next report date) or much earlier than 1990. Pick the century that lands
        # closest to today rather than always assuming 2000s (a raw '+2000' misreads
        # historical "'99"/"'98"/... as 2099/2098).
        from datetime import date as _date
        current_2digit = _date.today().year % 100
        year_2000s = 2000 + year
        year_1900s = 1900 + year
        year = (
            year_2000s
            if abs(year_2000s - (2000 + current_2digit)) <= abs(year_1900s - (2000 + current_2digit))
            else year_1900s
        )
    month = MONTHS[mon]
    if day is None:
        return date(year, month, 1), False
    try:
        return date(year, month, day), True
    except ValueError:
        return None, False


def clean_value(raw):
    """Handle '-', '', 'NA', '@NaNd', '@NA', parenthesis-negatives, '%'-strings, commas."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s in NULL_MARKERS:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", "").replace("%", "").strip()
    if s in NULL_MARKERS:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


FF_ARPT_RE = re.compile(r"^FF_ARPT_(?:SERIES|LABEL)\(\s*'([^']+)'")
FN_NAME_RE = re.compile(r"^([A-Z0-9_]+)\(")


def extract_field_id(cell_formula, alt_formula=None):
    """
    FF_ARPT_SERIES('TA','QTR_R',...) / FF_ARPT_LABEL('NET INCOME','CF') -> the field
    code ('TA', 'NET INCOME', ...) - stable across tickers/periods for BAL/INC/CF/SHS.
    RATIO rows use a different formula family per ratio (FF_SGA_SALES(...), etc.) with
    no shared field-code argument position - use the function name itself as the id
    (e.g. 'FF_SGA_SALES'), which is still stable and unique per ratio line.
    """
    for f in (cell_formula, alt_formula):
        if not f or not isinstance(f, str) or f.strip() in ("", "-"):
            continue
        m = FF_ARPT_RE.match(f.strip())
        if m:
            return m.group(1)
        m = FN_NAME_RE.match(f.strip())
        if m:
            return m.group(1)
    return None


def parse_scale_currency(footnotes, is_ratio):
    """'All figures in millions of U.S. Dollar[...]' -> (scale, currency, units)."""
    if is_ratio:
        return None, None, "percent"
    if not footnotes:
        return None, None, None
    text = footnotes.lower()
    scale = None
    for word in ("billions", "billion", "millions", "million", "thousands", "thousand"):
        if word in text:
            scale = word.rstrip("s") + "s"
            break
    currency = None
    if "u.s. dollar" in text or "usd" in text:
        currency = "USD"
    return scale, currency, "amount"


def compute_row_levels(table):
    """
    Row indent level from rowChildren (parent -> [children]) starting at table['rows']
    (the top-level/root rows), depth 0. Falls back to a cell's own 'level' key (seen on
    RATIO rows) when a row isn't reachable from the root list for some reason.
    """
    row_children = table.get("rowChildren") or {}
    roots = table.get("rows") or []
    levels = {}
    frontier = [(r, 0) for r in roots]
    while frontier:
        rid, lvl = frontier.pop()
        if rid in levels:
            continue
        levels[rid] = lvl
        for child in row_children.get(rid, []):
            frontier.append((child, lvl + 1))
    return levels


def load_merged(json_path, supplement_by_id, factset_id):
    with open(json_path) as f:
        data = json.load(f)
    sup_path = supplement_by_id.get(factset_id)
    if sup_path:
        with open(sup_path) as f:
            sup = json.load(f)
        for key in ("CF_INTM", "RATIO_INTM"):
            if sup.get(key) is not None:
                data[key] = sup[key]
    return data


def _sanitized_id_lookup(known_ids):
    """known_ids (e.g. from id_map.parquet's factset_id column) -> {sanitized: canonical}.
    'Sanitized' collapses BOTH '.' and '-' to '_', matching how the browser Blob-download
    naming (and this parser's own naive id_from_filename '_'->'-' reversal) mangles a
    ticker's own '.' (e.g. 'BF.B-US' -> file 'fs_fundamentals_BF_B_US.json'). Needed
    because the naive reversal alone cannot tell '.' from '-' once both are '_' in a
    filename - 'BF_B_US' round-trips to 'BF-B-US', not 'BF.B-US', and 'INFO_XX10_US'
    round-trips to 'INFO-XX10-US', not 'INFO.XX10-US'. Only 3 of the 499 universe ids
    have a '.' (BF.B-US, BRK.B-US, INFO.XX10-US) - confirmed 2026-09-17 - so this lookup
    is a small, cheap disambiguation table, not a general renaming scheme."""
    return {fsid.replace(".", "_").replace("-", "_"): fsid for fsid in known_ids}


def id_from_filename(name, known_ids=None):
    """
    'fs_fundamentals_batch_003_AMGN_US.json' -> 'AMGN-US'
    'fs_fundamentals_AAPL_US.json'            -> 'AAPL-US'
    'fs_fundamentals_supplement_AAPL_US.json' -> 'AAPL-US'
    'fs_fundamentals_stnd_AAPL_US.json'       -> 'AAPL-US'
    'fs_fundamentals_stndx_AAPL_US.json'      -> 'AAPL-US'
    'fs_fundamentals_BF_B_US.json'            -> 'BF.B-US'   (needs known_ids)
    'fs_fundamentals_INFO_XX10_US.json'       -> 'INFO.XX10-US' (needs known_ids)

    known_ids, when given, is the set of canonical factset_ids (from id_map.parquet).
    The naive '_' -> '-' reversal is ambiguous for the handful of ids that themselves
    contain a '.' (that '.' is also sanitized to '_' in the downloaded filename, same as
    the '-' before 'US'), so when the naive result isn't in known_ids this falls back to
    a sanitized-form lookup against known_ids before giving up.
    """
    stem = Path(name).stem
    # stndx_ must be tried before stnd_ - stnd_ alone can't match "stndx_..." (the 5th
    # char is 'x', not '_'), but if the alternation order were reversed this would still
    # be correct for that reason; kept stndx_ first for clarity/robustness.
    m = re.match(r"fs_fundamentals_(?:batch_\d+_|supplement_|stndx_|stnd_)?(.+)$", stem)
    if not m:
        return None
    raw = m.group(1)
    naive = raw.replace("_", "-")
    if not known_ids or naive in known_ids:
        return naive
    sanitized_lookup = _sanitized_id_lookup(known_ids)
    return sanitized_lookup.get(raw, naive)


def discover_main_files(fund_dir=FUND_DIR, known_ids=None):
    """One entry per factset_id -> path, preferring the batch-numbered file if both
    a batch-numbered and a bare-id file exist for the same id (keep the latest by
    mtime otherwise). Files matching *_stnd_* are always excluded here even when
    fund_dir is the ARPT directory (a file-mover may have dropped a Standardized file
    in the wrong place) - those are only ever picked up by discover_stnd_files()."""
    candidates = {}
    for f in fund_dir.glob("fs_fundamentals_*.json"):
        if "supplement" in f.name or "stnd" in f.name or f.name == "_progress.json":
            continue
        fsid = id_from_filename(f.name, known_ids)
        if not fsid:
            continue
        prev = candidates.get(fsid)
        if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
            candidates[fsid] = f
    return candidates


def discover_stnd_files(known_ids=None):
    """One entry per factset_id -> path for the Standardized-basis pull. Checks the
    dedicated fundamentals_stnd/ directory, and also fundamentals/ as a fallback in case
    a file-mover drops a *_stnd_* file in the wrong (ARPT) directory."""
    candidates = {}
    for fund_dir in (FUND_DIR_STND, FUND_DIR):
        if not fund_dir.exists():
            continue
        for f in fund_dir.glob("fs_fundamentals_stnd_*.json"):
            fsid = id_from_filename(f.name, known_ids)
            if not fsid:
                continue
            prev = candidates.get(fsid)
            if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
                candidates[fsid] = f
    return candidates


def discover_stndx_files(known_ids=None):
    """One entry per factset_id -> path for the STND-extra pull (SHS_ANN, SHS_INTM,
    RATIO_ANN only). Checked in the same two directories as discover_stnd_files(), same
    file-mover fallback rationale."""
    candidates = {}
    for fund_dir in (FUND_DIR_STND, FUND_DIR):
        if not fund_dir.exists():
            continue
        for f in fund_dir.glob("fs_fundamentals_stndx_*.json"):
            fsid = id_from_filename(f.name, known_ids)
            if not fsid:
                continue
            prev = candidates.get(fsid)
            if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
                candidates[fsid] = f
    return candidates


def discover_supplement_files(known_ids=None):
    by_id = {}
    for d in SUPPLEMENT_DIRS:
        if not d.exists():
            continue
        for f in d.glob("fs_fundamentals_supplement_*.json"):
            fsid = id_from_filename(f.name, known_ids)
            if fsid and fsid not in by_id:
                by_id[fsid] = f
    return by_id


def parse_one(factset_id, ticker, data, basis="ARPT"):
    rows = []
    day_lookup = {}  # (report_family_ignored, year, month) -> exact day, from full-date labels

    for report_key, table in data.items():
        if not isinstance(table, dict) or not table.get("columns"):
            continue
        report, rpttype = report_key.split("_")
        frequency = "annual" if rpttype == "ANN" else "quarterly"
        is_ratio = report == "RATIO"

        columns = table["columns"]
        column_data = table.get("columnData", {})
        row_data = table.get("rowData", {})
        levels = compute_row_levels(table)
        scale, currency, units = parse_scale_currency(table.get("footnotes"), is_ratio)

        this_labels = tuple(column_data.get(c, {}).get("value", [None])[0] for c in columns)
        frequency_ok = True
        if rpttype != "ANN":
            # Detect a genuinely-quarterly table by month diversity: real fiscal quarters
            # rotate through (up to) 4 distinct months; an annual table pretending to be
            # interim (the CF_INTM / RATIO_INTM bug) only ever shows the single fiscal
            # year-end month, no matter how many columns it returns (so this is robust to
            # ANN/INTM being requested with different eperiod column counts, unlike a raw
            # tuple-equality check against the ANN table).
            months_seen = set()
            for lbl in this_labels:
                label = lbl[0] if isinstance(lbl, list) else lbl
                d, _ = parse_period_label(label)
                if d:
                    months_seen.add(d.month)
            if len(months_seen) <= 1:
                frequency_ok = False

        # Seed day_lookup from any column with a full "DD MON 'YY" label (BAL/INC/CF).
        for c in columns:
            lbl_pair = column_data.get(c, {}).get("value")
            if not lbl_pair:
                continue
            label = lbl_pair[0] if isinstance(lbl_pair, list) else lbl_pair
            d, day_known = parse_period_label(label)
            if d and day_known:
                day_lookup[(d.year, d.month)] = d.day

        for row_id, row_info in row_data.items():
            col0 = row_info.get("col-0", {})
            field_label = col0.get("value")
            if not field_label or not isinstance(field_label, str):
                continue
            # field_id: try col-0's own formula, else the infobox-embedded formula text,
            # else the first data column's formula (RATIO stores the real formula only on
            # the first non-placeholder data column, e.g. col-1/col-3).
            infobox_formula = None
            infobox = col0.get("infobox", {})
            params = infobox.get("params") if isinstance(infobox, dict) else None
            if isinstance(params, str) and "Formula|" in params:
                infobox_formula = params.split("Formula|", 1)[1].split('");', 1)[0]
            first_data_formula = None
            for c in columns:
                f = row_info.get(c, {}).get("formula")
                if f and f.strip() not in ("", "-"):
                    first_data_formula = f
                    break
            field_id = extract_field_id(col0.get("formula"), infobox_formula or first_data_formula)
            level = levels.get(row_id)
            if level is None:
                # fall back to a cell's own 'level' attribute
                for c in columns:
                    lvl = row_info.get(c, {}).get("level")
                    if lvl is not None:
                        level = lvl
                        break

            for c in columns:
                cell = row_info.get(c, {})
                value = clean_value(cell.get("value"))
                if value is None:
                    continue
                lbl_pair = column_data.get(c, {}).get("value")
                if isinstance(lbl_pair, list):
                    label, status = (lbl_pair + [None, None])[:2]
                else:
                    label, status = lbl_pair, None
                period_end, day_known = parse_period_label(label)
                if period_end and not day_known:
                    exact_day = day_lookup.get((period_end.year, period_end.month))
                    if exact_day:
                        period_end = date(period_end.year, period_end.month, exact_day)
                    else:
                        last_day = calendar.monthrange(period_end.year, period_end.month)[1]
                        period_end = date(period_end.year, period_end.month, last_day)

                rows.append({
                    "ticker": ticker,
                    "factset_id": factset_id,
                    "basis": basis,
                    "report": report,
                    "frequency": frequency,
                    "period_end": period_end,
                    "period_status": (status or "").strip() or None,
                    "field": field_label,
                    "field_id": field_id,
                    "level": level,
                    "value": value,
                    "units": units,
                    "scale": scale,
                    "currency": currency,
                    "frequency_ok": frequency_ok,
                })
    return rows


def main():
    id_map_file = REPO_ROOT / "data" / "raw" / "fs" / "reference" / "id_map.parquet"
    id_to_ticker = {}
    known_ids = None
    if id_map_file.exists():
        id_map = pl.read_parquet(id_map_file)
        id_to_ticker = dict(zip(id_map["factset_id"], id_map["ticker"]))
        known_ids = set(id_map["factset_id"].drop_nulls())
    else:
        print(f"WARNING: {id_map_file} not found - tickers will be null")

    # known_ids disambiguates the 3 ids whose own ticker contains a '.' (BF.B-US,
    # BRK.B-US, INFO.XX10-US) - their downloaded filenames sanitize '.' to '_' same as
    # the '-' before 'US', so a naive '_' -> '-' reversal alone can't tell them apart
    # (see id_from_filename docstring). Without known_ids these 3 ids were mis-derived
    # as 'BF-B-US'/'BRK-B-US'/'INFO-XX10-US' - not in id_map, so their rows carried a
    # null ticker and silently dropped out of any ticker-joined output - found and
    # fixed 2026-09-17 (499 files on disk for both ARPT and STND, all 499 now resolve).
    main_files = discover_main_files(known_ids=known_ids)
    supplement_files = discover_supplement_files(known_ids=known_ids)
    stnd_files = discover_stnd_files(known_ids=known_ids)
    stndx_files = discover_stndx_files(known_ids=known_ids)
    print(f"Found {len(main_files)} ids with an ARPT fundamentals file "
          f"({len(supplement_files)} have a CF_INTM/RATIO_INTM supplement), "
          f"{len(stnd_files)} ids with a STND fundamentals file "
          f"({len(stndx_files)} have a stndx SHS/RATIO_ANN supplement)")

    all_rows = []
    n_failed_load = 0
    n_total = len(main_files) + len(stnd_files)
    i = 0
    for factset_id, path in sorted(main_files.items()):
        i += 1
        try:
            data = load_merged(path, supplement_files, factset_id)
        except Exception as e:
            print(f"  [{i}/{n_total}] FAILED to load ARPT {path.name}: {e}")
            n_failed_load += 1
            continue
        ticker = id_to_ticker.get(factset_id)
        rows = parse_one(factset_id, ticker, data, basis="ARPT")
        all_rows.extend(rows)
        if i % 50 == 0:
            print(f"  [{i}/{n_total}] ARPT {factset_id}: {len(rows)} rows so far {len(all_rows)}")

    n_stndx_merged = 0
    for factset_id, path in sorted(stnd_files.items()):
        i += 1
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [{i}/{n_total}] FAILED to load STND {path.name}: {e}")
            n_failed_load += 1
            continue
        stndx_path = stndx_files.get(factset_id)
        if stndx_path is not None:
            try:
                with open(stndx_path) as f:
                    stndx_data = json.load(f)
                for key in ("SHS_ANN", "SHS_INTM", "RATIO_ANN"):
                    if stndx_data.get(key) is not None:
                        data[key] = stndx_data[key]
                n_stndx_merged += 1
            except Exception as e:
                print(f"  [{i}/{n_total}] FAILED to load stndx {stndx_path.name} for {factset_id}: {e}")
        ticker = id_to_ticker.get(factset_id)
        rows = parse_one(factset_id, ticker, data, basis="STND")
        all_rows.extend(rows)
        if i % 50 == 0 or i == n_total:
            print(f"  [{i}/{n_total}] STND {factset_id}: {len(rows)} rows so far {len(all_rows)}")
    print(f"  merged a stndx SHS/RATIO_ANN supplement into {n_stndx_merged}/{len(stnd_files)} STND ids")

    if not all_rows:
        print("No data parsed - nothing to write.")
        return

    schema = {
        "ticker": pl.Utf8, "factset_id": pl.Utf8, "basis": pl.Utf8, "report": pl.Utf8,
        "frequency": pl.Utf8, "period_end": pl.Date, "period_status": pl.Utf8, "field": pl.Utf8,
        "field_id": pl.Utf8, "level": pl.Int64, "value": pl.Float64, "units": pl.Utf8,
        "scale": pl.Utf8, "currency": pl.Utf8, "frequency_ok": pl.Boolean,
    }
    df = pl.DataFrame(all_rows, schema=schema)
    df = df.unique(subset=["factset_id", "basis", "report", "frequency", "period_end", "field_id", "field"])
    df.write_parquet(OUTPUT_FILE)

    print(f"\n=== Coverage ===")
    print(f"Files loaded: {n_total - n_failed_load}/{n_total} (failed: {n_failed_load})")
    print(f"Total rows: {len(df)}")
    print(f"Unique factset_ids: {df['factset_id'].n_unique()}")
    print(f"Unique tickers: {df['ticker'].n_unique()} (null ticker rows: {df['ticker'].null_count()})")

    cov = (
        df.group_by(["basis", "report", "frequency"])
        .agg(
            pl.col("factset_id").n_unique().alias("ids"),
            pl.len().alias("rows"),
            pl.col("frequency_ok").mean().alias("pct_frequency_ok"),
            pl.col("period_end").min().alias("min_period"),
            pl.col("period_end").max().alias("max_period"),
        )
        .sort(["basis", "report", "frequency"])
    )
    print("\nCoverage by basis x report x frequency (pct_frequency_ok < 1.0 means some/all ids")
    print("got a bugged duplicate-of-annual table for that basis x report x frequency - see")
    print("module docstring):")
    with pl.Config(tbl_rows=30, tbl_cols=20):
        print(cov)

    bad = df.filter(~pl.col("frequency_ok"))
    if len(bad) > 0:
        bad_ids = bad.select(["basis", "report", "factset_id"]).unique().group_by(["basis", "report"]).agg(pl.len())
        print(f"\n{len(bad)} rows flagged frequency_ok=False (duplicate-of-annual):")
        print(bad_ids)

    print(f"\nOutput: {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    main()
