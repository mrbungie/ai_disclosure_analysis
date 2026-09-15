"""
scripts/us/edgar_fetch.py — SEC EDGAR fetch logic, specific to this
country/source. Shared by scripts/us/10k/01_fetch_filings.py and
scripts/us/10q/01_fetch_filings.py. Uses edgartools (Company.get_filings +
Filing.html()) instead of hand-rolled requests calls:

- edgartools' HTTP client (httpx-based, see edgar.httpclient) reuses
  connections across requests — plain module-level `requests.get()` calls
  (the old approach) open a fresh connection every time, paying a TCP+TLS
  handshake per request instead of keep-alive.
- Its /Archives/edgar/data responses are cached FOREVER on disk (see
  edgar.httpclient.clear_empty_cached_responses' docstring) — re-running
  this script never re-hits the network for a filing it already has,
  without us hand-rolling that check ourselves.
- Filing.html() fetches exactly one filing's primary document — NOT
  Filings.download(), which downloads whole daily/quarterly bulk index
  archives (every filer, not just ours) and filters locally; wrong tool
  for a targeted subset of ~500 companies.

10-K and 10-Q are fetched via the SAME function (parameterized by form)
but into SEPARATE output paths — no pooling, per the project's data-scope
decision (10-K panel core + 10-Q shock series, never merged).

Storage: primary documents are gzip-compressed on save (.html.gz) — this
is OUR OWN mirror (what section_segmenter.py reads), on top of edgar's own
cache. iXBRL-era 10-Ks/10-Qs are ~90%+ repeated markup/XBRL-context
boilerplate that compresses hard (a real AAPL 10-K: 1.5MB -> 111KB).
"""

import gzip
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import re

import edgar
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))  # scripts/common/
import pipeline_logger

MAX_WORKERS = 8
_configured = False


def configure(user_agent: str, local_storage_dir: Path):
    """Point edgartools at OUR repo-internal, gitignored cache dir instead
    of its default ~/.edgar_cache — keeps everything the pipeline touches
    inside the repo tree (still gitignored, never committed)."""
    global _configured
    if _configured:
        return
    local_storage_dir.mkdir(parents=True, exist_ok=True)
    edgar.set_identity(user_agent)
    edgar.set_local_storage_path(str(local_storage_dir))
    edgar.use_local_storage(True)
    _configured = True


#: Below this many `<ix:nonFraction>` tags, a primary document is treated as
#: "thin" — some filers (mostly banks: BNY, and others filing the same way)
#: incorporate their actual financial statements BY REFERENCE into a
#: secondary exhibit (EX-13, EX-99.1, filed as `<stem>_d2.htm` alongside the
#: primary `<stem>.htm`) instead of tagging them inline in the primary 10-K
#: document itself. Verified against BNY's FY2025 10-K: the primary document
#: (`bk-20251231.htm`) has 11 nonFraction tags (cover-page items only); the
#: companion `bk-20251231_d2.htm` ("Certain Portions of 2025 Annual Report to
#: Shareholders") has 5,441. Same pattern confirmed for CLX (EX-99.1) and IBM
#: (EX-13). A normal filer with real inline-tagged statements (WMT) has
#: 1,198 in its primary document alone — 50 is well below any real filing's
#: count and well above the handful of cover-page facts a wrapper-only
#: primary document carries.
THIN_PRIMARY_THRESHOLD = 50
_NONFRACTION_RE = re.compile(r"<ix:nonFraction", re.IGNORECASE)


_EXHIBIT_DESC_RE = re.compile(r"^EX(?:HIBIT)?[\s.-]*(13|99(?:\.1)?)\b")


def _find_incorporated_financials(f, primary_document: str) -> "edgar.Attachment | None":
    """Look for the companion exhibit that carries the real financial
    statements when `primary_document` is a thin wrapper (see
    THIN_PRIMARY_THRESHOLD). Naming is NOT consistent across filers, so two
    independent signals are checked, either sufficient:

    1. Same `<ticker>-<date>` stem with/without a trailing `_d2`, whichever
       document ISN'T the primary. Verified against BNY/CLX/IBM (primary
       `bk-20251231.htm`, exhibit `bk-20251231_d2.htm`) AND against WFC,
       where the naming is INVERTED — the primary document ITSELF is
       `wfc-20251231_d2.htm` and the exhibit is the bare `wfc-20251231.htm`.
       Matching by stem symmetry (strip `_d2` from whichever side has it)
       catches both directions.
    2. An EX-13 / EX-99(.1) description — but filers spell this two ways:
       Also verified against WFC, whose exhibit is described "EXHIBIT 13"
       (spelled out) where BNY/CLX/IBM/USB use "EX-13" — the regex accepts
       both."""
    def core_stem(document: str) -> str:
        stem = document.rsplit(".", 1)[0]
        return stem[:-3] if stem.endswith("_d2") else stem

    primary_stem = core_stem(primary_document)
    for a in f.attachments:
        doc = a.document or ""
        if not doc.endswith(".htm") or doc == primary_document:
            continue
        description = (getattr(a, "description", "") or "").upper()
        if core_stem(doc) == primary_stem or _EXHIBIT_DESC_RE.match(description):
            return a
    return None


#: Between the primary/exhibit HTML and a traditional (non-inline) XBRL
#: instance document, when both end up in the same cached file — the
#: extractor splits on this to run the right parser
#: (`filing_xbrl_facts.parse_inline_xbrl` vs `.parse_xbrl_instance`) on
#: each side, never both on the same content (an XBRL instance XML isn't
#: valid HTML and vice versa, and each dialect tags facts completely
#: differently — see `parse_xbrl_instance`'s docstring).
XBRL_INSTANCE_BOUNDARY = "<!--EDGAR_FETCH_XBRL_INSTANCE_BOUNDARY-->"


def _find_xbrl_instance_document(f) -> "edgar.Attachment | None":
    """A handful of filers (verified: DDOG's and PLTR's first post-IPO
    10-Ks) file traditional, non-inline XBRL: `f.is_inline_xbrl` is False,
    and the numeric facts live in a standalone `<ticker>-<date>.xml`
    "XBRL INSTANCE DOCUMENT" attachment instead of `<ix:nonFraction>` tags
    in the primary HTML."""
    for a in f.attachments:
        doc = a.document or ""
        description = (getattr(a, "description", "") or "").upper()
        if doc.endswith(".xml") and "INSTANCE" in description:
            return a
    return None


def _fetch_primary_html(f) -> str:
    """The primary document's HTML, plus — whenever a companion
    incorporated-financials exhibit exists AND itself carries real content —
    that exhibit CONCATENATED onto the same cached file, to be parsed as
    ONE document.

    Verified against BNY's FY2025 10-K: the supplement
    (`bk-20251231_d2.htm`, "Certain Portions of 2025 Annual Report to
    Shareholders") has thousands of `<ix:nonFraction>` facts but ZERO
    `<xbrli:context>` definitions of its own — every `contextRef` on its
    facts (e.g. `c-1`) resolves only against context ids defined in the
    PRIMARY document. The two documents are not independently-authored
    XBRL instances that happen to share short ids; they are ONE XBRL
    instance split across two HTML renderings by the filer's tagging
    software, and have to be parsed together for the supplement's facts to
    resolve to a period at all (parsed separately, the supplement yields
    zero facts — every one of its `contextRef`s is unresolvable on its
    own). `filing_xbrl_facts.parse_inline_xbrl` is called once over the
    combined content.

    Deliberately NOT gated on the primary document looking "thin" first
    (an earlier version required the primary to be under
    THIN_PRIMARY_THRESHOLD before even looking for a supplement): verified
    against IBM's FY2021-2025 10-Ks, whose primary document has 62
    nonFraction tags each — enough to clear a naive low threshold, but
    still only cover-page/summary items, not the real statements, which
    IBM also splits into an EX-13 exhibit. The supplement is fetched and
    used whenever it exists and clears the threshold on its own, regardless
    of what the primary document already has."""
    html = f.html()
    primary_document = getattr(f, "primary_document", None) or getattr(f, "document", None)
    if primary_document:
        supplement = _find_incorporated_financials(f, primary_document)
        if supplement is not None:
            supplement_html = supplement.download()
            if isinstance(supplement_html, str) and len(_NONFRACTION_RE.findall(supplement_html)) >= THIN_PRIMARY_THRESHOLD:
                html = html + "\n" + supplement_html
    if not getattr(f, "is_inline_xbrl", True):
        instance = _find_xbrl_instance_document(f)
        if instance is not None:
            instance_xml = instance.download()
            if isinstance(instance_xml, str) and instance_xml.strip():
                html = html + "\n" + XBRL_INSTANCE_BOUNDARY + "\n" + instance_xml
    return html


def _fetch_one_company(row, form, start_date, end_date, allow_amendments, html_dir, existing_manifest_df):
    """Runs in a worker thread. Returns (firm_info, manifest_rows)."""
    ticker = row["ticker"]
    cik = row["cik"]

    manifest_rows = []
    firm_info = row.to_dict()

    try:
        company = edgar.Company(int(cik))
        filings = company.get_filings(
            form=form, filing_date=f"{start_date}:{end_date}", amendments=allow_amendments,
        )
        firm_info["sic"] = str(getattr(company, "sic", "") or "")
        # `industry` since edgartools 5.55 (was `sic_description`). Read both, and
        # do NOT swallow a miss: a silent "" here is what left industry_group empty
        # for all 517 firms and stayed empty, because script 00 only carries the
        # previous value forward.
        industry = getattr(company, "industry", None) or getattr(company, "sic_description", None)
        if not industry and not firm_info.get("industry_group"):
            pipeline_logger.log_event(
                pipeline_step="edgar_fetch", level="WARNING",
                message=f"No industry/sic_description for {ticker} (CIK {cik}); "
                        f"industry_group left empty",
                ticker=ticker, cik=cik)
        firm_info["industry_group"] = industry or firm_info.get("industry_group", "")
    except Exception as e:
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="ERROR",
            message=f"Error looking up {ticker} (CIK {cik}): {e}",
            ticker=ticker, cik=cik, details={"error": str(e)},
        )
        return firm_info, manifest_rows

    # Iterate the Filing objects directly (EntityFilings is iterable) rather
    # than round-tripping through to_pandas() and re-querying by accession
    # number to fetch content — that re-query (company.get_filings(
    # accession_number=...).latest()) intermittently returned None for a
    # small fraction of filings ("'NoneType' object has no attribute
    # 'download'"), for reasons not worth chasing when the fix is simply not
    # doing a redundant second lookup in the first place.
    for f in filings:
        acc_num = f.accession_number
        filing_date = pd.to_datetime(f.filing_date).date()
        year = filing_date.year
        report_date_raw = getattr(f, "period_of_report", None)
        report_date = pd.to_datetime(report_date_raw).date() if report_date_raw else None

        # f.form, not the outer `form` param -- `form` may be a list (e.g.
        # ["DEF 14A", "DEFC14A"]) when a query covers form-code variants of
        # the same underlying document (a contested-proxy year reclassifies
        # DEF 14A to DEFC14A); each filing keeps its OWN actual form code,
        # both in the manifest and in the filename, or every DEFC14A row
        # would misreport itself as "DEF 14A".
        filing_form = f.form

        # filing_form.replace(" ", "") -- "DEF 14A" has a space, which is
        # otherwise a valid (if annoying) filename character; sanitized so
        # every form gets a clean single-token filename, not just the
        # hyphenated ones (10-K/10-Q) this originally shipped with.
        filename = f"{ticker}_{year}_{filing_form.replace(' ', '')}_{acc_num}.html.gz"
        local_path = html_dir / filename

        existing_row = None
        if existing_manifest_df is not None:
            match = existing_manifest_df[existing_manifest_df["accession_number"] == acc_num]
            if not match.empty:
                existing_row = match.iloc[0]

        download_status = existing_row["download_status"] if existing_row is not None else "pending"
        parse_status = existing_row["parse_status"] if existing_row is not None else "pending"
        created_at = existing_row.get("created_at", datetime.now()) if existing_row is not None else datetime.now()

        # OUR OWN idempotency check, on top of edgartools' own cache: if we
        # already wrote this filing's gzip mirror, don't even ask edgartools
        # for it again.
        if local_path.exists() and local_path.stat().st_size > 0:
            download_status = "completed"
        else:
            try:
                html = _fetch_primary_html(f)
                with gzip.open(local_path, "wt", encoding="utf-8") as out:
                    out.write(html)
                download_status = "completed"
                pipeline_logger.log_event(
                    pipeline_step="edgar_fetch", level="SUCCESS",
                    message="Filing fetched and cached",
                    ticker=ticker, cik=cik, accession_number=acc_num,
                    details={"stored_bytes": local_path.stat().st_size},
                )
            except Exception as e:
                download_status = f"failed: {e}"
                pipeline_logger.log_event(
                    pipeline_step="edgar_fetch", level="ERROR",
                    message=f"Error fetching {ticker} {acc_num}: {e}",
                    ticker=ticker, cik=cik, accession_number=acc_num, details={"error": str(e)},
                )

        manifest_rows.append({
            "document_id": acc_num,
            "cik": cik,
            "ticker": ticker,
            "country": row.get("country", "US"),
            "source": row.get("source", "SEC_EDGAR"),
            "form_type": filing_form,
            "filing_date": filing_date,
            "period_end_date": report_date,
            "accession_number": acc_num,
            "sec_url": getattr(f, "url", ""),
            "local_path": str(local_path),
            "download_status": download_status,
            "parse_status": parse_status,
            "prefilter_status": existing_row.get("prefilter_status", "pending") if existing_row is not None else "pending",
            "llm_status": existing_row.get("llm_status", "pending") if existing_row is not None else "pending",
            "priority_score": existing_row.get("priority_score", 1.0) if existing_row is not None else 1.0,
            "batch_id": existing_row.get("batch_id", "") if existing_row is not None else "",
            "created_at": created_at,
            "updated_at": datetime.now(),
        })

    return firm_info, manifest_rows


def _safe_write_universe(universe_path: Path, full_universe_df: pd.DataFrame, updated_rows: list[dict]) -> None:
    """Writes `universe_path` by MERGING `updated_rows` into the FULL
    original universe (by ticker), never by replacing the file with just
    `updated_rows` outright.

    Real incident this fixes (2026-09-04): `universe_path` (firm_universe.
    parquet) is a single SHARED file every fetch script (10-K/10-Q/proxy/
    8-K) reads AND writes -- unlike every other output in this pipeline
    (prefilter_scores, ai_frames, golden_set labels, ...), which is
    append-only `__run=...__part=...parquet`. The periodic checkpoint used
    to `pd.DataFrame(updated_universe).to_parquet(universe_path)` with
    `updated_universe` containing only the tickers processed SO FAR in
    THIS run -- a killed/interrupted run (Ctrl-C, an exception escaping
    the checkpoint window) truncated the shared 517-firm universe down to
    however many tickers had been processed at that point (confirmed:
    517 -> 150 from one interrupted run). Recovered from the last B2 push,
    but the file being a single mutable target made that recovery
    necessary at all. Merging against the full baseline means an
    interrupted run can only ever update a subset of rows, never drop the
    rest."""
    updated_by_ticker = {row["ticker"]: row for row in updated_rows}
    merged = [updated_by_ticker.get(row["ticker"], row) for row in full_universe_df.to_dict("records")]
    # A ticker in updated_rows but NOT in the original baseline (shouldn't
    # happen -- fetch_filings only ever iterates full_universe_df -- but
    # checked rather than silently dropped if it ever does).
    known_tickers = {row["ticker"] for row in full_universe_df.to_dict("records")}
    merged.extend(row for row in updated_rows if row["ticker"] not in known_tickers)
    pd.DataFrame(merged).to_parquet(universe_path, index=False)


def fetch_filings(universe_df, form, start_date, end_date, allow_amendments, html_dir, manifest_path, universe_path):
    """Fetches `form` filings for every (ticker, cik) in universe_df between
    start_date/end_date, writes gzip'd HTML to html_dir, and builds/updates
    manifest_path. Idempotent: rerun skips any filing whose gzip file
    already exists — no re-fetch, no re-write."""
    html_dir.mkdir(parents=True, exist_ok=True)
    full_universe_df = universe_df.copy()  # see _safe_write_universe

    # `form` may be a single form code or a list of form-code variants of the
    # same document (e.g. ["DEF 14A", "DEFC14A"] -- a contested-proxy year
    # reclassifies the form code). Normalize once so the "already done"
    # checks below can match on membership, not equality.
    form_list = [form] if isinstance(form, str) else list(form)

    # _fully_done(ticker) below matches manifest rows by TICKER alone (not
    # CIK) — if universe_df ever contains the same ticker twice (found once,
    # for real: XOM listed under two different CIKs in configs/us/universe.csv
    # after an untracked corporate-CIK-reassignment addition), each
    # duplicate row independently re-triggers the "already done, carry
    # forward existing rows" branch, appending that ticker's full existing
    # row set again — doubling it every single rerun (2 dupes/run compounds
    # to 4x, 16x... across repeated fetch invocations; confirmed exactly
    # this shape in filing_manifest_10q.parquet). Deduping here is a
    # backstop against that recurring, independent of also fixing the
    # source data in universe.csv — first row wins.
    n_before = len(universe_df)
    universe_df = universe_df.drop_duplicates(subset=["ticker"], keep="first")
    if len(universe_df) < n_before:
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="WARNING",
            message=(
                f"universe_df had {n_before - len(universe_df)} duplicate ticker row(s) "
                f"(same ticker, different CIK) — kept the first, dropped the rest."
            ),
        )

    existing_manifest_df = None
    if manifest_path.exists():
        try:
            existing_manifest_df = pd.read_parquet(manifest_path)
        except Exception:
            pass

    updated_universe = []
    manifest_data = []

    def _fully_done(ticker):
        """True if every known-existing manifest row for this ticker is
        already completed AND its gzip file is actually on disk — lets a
        retry pass skip re-querying edgartools for the whole company's
        filing list just to retry a handful of other companies' failures.
        Without this, a rerun that only needs to retry 14/517 companies
        still re-discovers all 517 from scratch every time."""
        if existing_manifest_df is None:
            return False
        rows = existing_manifest_df[
            (existing_manifest_df["ticker"] == ticker) & (existing_manifest_df["form_type"].isin(form_list))
        ]
        if rows.empty:
            return False
        return bool((rows["download_status"] == "completed").all()) and all(
            Path(p).exists() and Path(p).stat().st_size > 0 for p in rows["local_path"]
        )

    skipped_tickers = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for _, row in universe_df.iterrows():
            ticker = row["ticker"]
            if _fully_done(ticker):
                skipped_tickers.add(ticker)
                updated_universe.append(row.to_dict())
                manifest_data.extend(
                    existing_manifest_df[
                        (existing_manifest_df["ticker"] == ticker) & (existing_manifest_df["form_type"].isin(form_list))
                    ].to_dict("records")
                )
                continue
            futures[executor.submit(
                _fetch_one_company, row, form, start_date, end_date, allow_amendments, html_dir, existing_manifest_df,
            )] = ticker

        if skipped_tickers:
            print(f"Skipping {len(skipped_tickers)} companies already fully fetched (no network call).")

        completed_since_checkpoint = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {form}"):
            ticker = futures[future]
            try:
                firm_info, manifest_rows = future.result()
            except Exception as e:
                # A worker thread's uncaught exception (e.g. a raw httpx
                # timeout escaping _fetch_one_company's own try/except)
                # used to propagate out of future.result() and crash the
                # WHOLE run here — losing every other company's already-
                # fetched manifest rows (the gzip files stayed on disk, but
                # the run never got to write filing_manifest*.parquet at
                # all). One company's failure must never cost everyone
                # else's progress.
                pipeline_logger.log_event(
                    pipeline_step="edgar_fetch", level="ERROR",
                    message=f"Unhandled error processing {ticker}, skipping company: {e}",
                    ticker=ticker, details={"error": str(e)},
                )
                continue
            updated_universe.append(firm_info)
            manifest_data.extend(manifest_rows)

            # Checkpoint periodically so a later crash (or Ctrl-C) doesn't
            # lose everything fetched so far in THIS run either.
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= 25:
                _safe_write_universe(universe_path, full_universe_df, updated_universe)
                pd.DataFrame(manifest_data).to_parquet(manifest_path, index=False)
                completed_since_checkpoint = 0

    _safe_write_universe(universe_path, full_universe_df, updated_universe)

    if manifest_data:
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_parquet(manifest_path, index=False)
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="SUCCESS",
            message=f"Built {form} manifest with {len(manifest_df)} filings at {manifest_path}",
            details={"filing_count": len(manifest_df), "form": form},
        )
    else:
        pipeline_logger.log_event(
            pipeline_step="edgar_fetch", level="WARNING",
            message=f"No {form} filings found for the configured universe/date range.",
        )
