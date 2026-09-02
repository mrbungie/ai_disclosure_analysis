"""
scripts/cl/01_fetch_filings.py — fetches, per firm in configs/cl/universe.csv,
the narrative disclosure documents this project actually analyzes: the
Memoria Anual (annual — business narrative, ~10-K analog) and the Análisis
Razonado at each quarterly close 03/06/09/12 (MD&A analog — see
docs/international_expansion_plan.md's instrument mapping).

Deliberately narrower than "everything CMF publishes for a firm": skips
Estados Financieros (PDF/XBRL — numeric tables, not narrative), Declaración
de responsabilidad, and Hechos Relevantes — same scoping principle as
scripts/us/10k/02_extract_sections.py only extracting Item 1/1A/7, not the
whole 10-K (Item 8's financial statements are numbers, not disclosure
narrative).

Talks to www.cmfchile.cl DIRECTLY (scripts/cl/cmf_direct_client.py, a
Python port of vendor/mcp-cmf-chile's anti-bot/HTTP core) — NOT via the
hosted MCP server's cmf_documento_markdown (scripts/cl/cmf_mcp_client.py,
kept for reference): that server converts PDFs to Markdown server-side
with a hard 4MB input cap and took ~2.5 minutes per large Memoria Anual in
a real test run — Empresas Copec's 2021 Memoria alone is TWO PDF parts,
5.5MB + 4.0MB, i.e. it could never have converted there at all.

Saves the RAW PDF bytes (gzip-compressed), not extracted text — download
and text extraction are deliberately two separate stages/scripts (this one
and scripts/cl/02_extract_text.py), so the (slow, network-bound, hours-
long) download can run to completion independently of however long
getting the text-extraction rules right takes, and so a later change to
extraction NEVER requires re-hitting CMF: it just re-reads the cached
PDFs. See scripts/cl/02_extract_text.py's docstring for why extracting
paragraphs from a PDF needs different rules than scripts/us's HTML-derived
lines in the first place (plain PyMuPDF text comes out with none of HTML's
paragraph-per-line structure and no table markers at all).

No env var or API key needed for anything here — see cmf_direct_client.py's
docstring (CMF_API_KEY only gates the separate macro-indicator API this
project never calls).

Idempotent + resumable, same contract as scripts/us/edgar_fetch.py: a
document whose local .pdf.gz mirror(s) already exist is skipped outright,
so re-running after widening the date range or adding firms only fetches
what's new.

Usage:
    uv run python scripts/cl/01_fetch_filings.py [--limit N] [--rut RUT]
"""

import argparse
import gzip
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import pipeline_logger
from cmf_direct_client import fetch_pdf_parts, new_session, post_legacy, resolve_doc_url

QUARTER_CLOSES = ["03", "06", "09", "12"]

# EEFF tab (pestania=3) document list — each row links straight to a PDF
# via safec_ifrs_verarchivo.php. Verified against the real page structure
# (2026-09-02); see scripts/cl/cmf_mcp_client.py's sibling regex in the
# upstream TypeScript for the origin of this pattern.
_EEFF_DOC_RE = re.compile(
    r'href="(\.\./inc/inf_financiera/ifrs/safec_ifrs_verarchivo\.php\?auth=[^"]+&send=[^"]+)"[^>]*>\s*([^<]{2,60})'
)
# Memoria tab (pestania=49) document table — a plain 3-column
# <th>-headed table (Tipo de documento / Fecha Envío / Ver archivo),
# unlike most CMF tables (see vendor/mcp-cmf-chile/CLAUDE.md's Lesson 4 on
# why a GENERIC table parser there is treated as fragile) — narrow,
# targeted regex instead of porting that generic parser, since this is
# the only shape we need from this tab.
_MEMORIA_ROW_RE = re.compile(
    r'<td>([^<]*)</td>\s*<td>([^<]*)</td>\s*<td><a href="([^"]+)"'
)


def _ficha_url(rut: str, pestania: int) -> str:
    return (f"/institucional/mercados/entidad.php?mercado=V&rut={rut}&grupo=&tipoentidad=RVEMI"
            f"&row=&vig=VI&control=svs&pestania={pestania}")


def _find_eeff_doc_url(html: str, contains: list[str]) -> str | None:
    for href, name in _EEFF_DOC_RE.findall(html):
        if any(term in name.lower() for term in contains):
            return resolve_doc_url(href)
    return None


def _find_memoria_doc_url(html: str) -> str | None:
    for tipo_doc, _fecha, href in _MEMORIA_ROW_RE.findall(html):
        if "memoria" in tipo_doc.lower():
            return resolve_doc_url(href)
    return None


def _save_pdf_parts(parts: list[bytes], base_path: Path) -> list[Path]:
    """Saves each raw PDF part gzip-compressed. A single-part document
    (the common case — e.g. every Análisis Razonado seen so far) gets
    exactly `base_path`; a multi-part one (e.g. a large Memoria Anual —
    see this file's module docstring) gets `base_path` with `.partN`
    inserted before the extension, one file per part, in order."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    if len(parts) == 1:
        paths = [base_path]
    else:
        stem = base_path.name.removesuffix(".pdf.gz")
        paths = [base_path.with_name(f"{stem}.part{i}.pdf.gz") for i in range(len(parts))]
    for path, part in zip(paths, parts):
        with gzip.open(path, "wb") as f:
            f.write(part)
    return paths


def _manifest_row(document_id, rut, nemo, form_type, filing_type, period_end_date, local_paths,
                   download_status, n_bytes, created_at):
    return {
        "document_id": document_id,
        "rut": rut,
        "nemo": nemo,
        # No country_code here — build_duckdb.py's per-country UNION adds
        # its own literal 'country_code' column (see _country_configs()),
        # so a manifest that also carried one would collide with it under
        # UNION BY NAME (hit exactly this: BinderException, duplicate
        # "country_code" in one SELECT list, the moment this manifest was
        # first unioned).
        "source": "CMF",
        "form_type": form_type,
        "filing_type": filing_type,
        "period_end_date": period_end_date,
        # Semicolon-joined — usually one path, more than one only for a
        # multi-part Memoria (see _save_pdf_parts); not worth a separate
        # child table for what is, so far, a 1-or-2-element list.
        "local_path": ";".join(str(p) for p in local_paths) if local_paths else "",
        "format": "pdf",
        "download_status": download_status,
        "n_bytes": n_bytes,
        "created_at": created_at,
        "updated_at": datetime.now(),
    }


def _already_fetched(base_path: Path) -> bool:
    """True if base_path itself exists (single-part case) or any
    `<stem>.partN.pdf.gz` sibling exists (multi-part case — see
    _save_pdf_parts)."""
    if base_path.exists():
        return True
    stem = base_path.name.removesuffix(".pdf.gz")
    return any(base_path.parent.glob(f"{stem}.part*.pdf.gz"))


def fetch_one(session, rut: str, nemo: str, years: list[int], raw_dir: Path, log_dir: Path) -> list[dict]:
    """Idempotent purely on the local .pdf.gz mirror(s) existing — same
    contract as scripts/us/edgar_fetch.py (not on the manifest also
    agreeing, which would require never deleting the manifest between
    runs to stay correct; the file itself is the source of truth for
    whether a document was already fetched successfully, since it's only
    ever written after a successful save)."""
    rows = []

    for anio in years:
        # --- Memoria Anual (annual) ---
        doc_id = f"{rut}_memoria_{anio}"
        base_path = raw_dir / rut / f"memoria_{anio}.pdf.gz"
        if not _already_fetched(base_path):
            try:
                html = post_legacy(session, _ficha_url(rut, 49), {"aa": str(anio)})
                doc_url = _find_memoria_doc_url(html)
                if doc_url is None:
                    rows.append(_manifest_row(doc_id, rut, nemo, "Memoria Anual", "annual",
                                               date(anio, 12, 31), [], "not_found", 0, datetime.now()))
                else:
                    parts = fetch_pdf_parts(session, doc_url)
                    saved_paths = _save_pdf_parts(parts, base_path)
                    n_bytes = sum(len(p) for p in parts)
                    rows.append(_manifest_row(doc_id, rut, nemo, "Memoria Anual", "annual",
                                               date(anio, 12, 31), saved_paths, "completed", n_bytes,
                                               datetime.now()))
                    pipeline_logger.log_event(
                        pipeline_step="cmf_fetch", level="SUCCESS",
                        message="Memoria anual fetched", ticker=nemo,
                        details={"rut": rut, "anio": anio, "pdf_parts": len(parts)}, log_dir=log_dir,
                    )
            except Exception as e:
                rows.append(_manifest_row(doc_id, rut, nemo, "Memoria Anual", "annual",
                                           date(anio, 12, 31), [], f"failed: {e}", 0, datetime.now()))
                pipeline_logger.log_event(
                    pipeline_step="cmf_fetch", level="ERROR",
                    message=f"Memoria anual failed: {e}", ticker=nemo, details={"rut": rut, "anio": anio},
                    log_dir=log_dir,
                )

        # --- Análisis Razonado (quarterly, one per close incl. FY) ---
        for mes in QUARTER_CLOSES:
            period = f"{anio}{mes}"
            doc_id = f"{rut}_analisis_razonado_{period}"
            base_path = raw_dir / rut / f"analisis_razonado_{period}.pdf.gz"
            if _already_fetched(base_path):
                continue
            period_end = date(anio, int(mes), 1)
            try:
                html = post_legacy(session, _ficha_url(rut, 3), {
                    "forma": "F", "mm": mes, "aa": str(anio), "tipo": "C", "tipo_norma": "IFRS",
                })
                doc_url = _find_eeff_doc_url(html, ["analisis", "análisis"])
                if doc_url is None:
                    rows.append(_manifest_row(doc_id, rut, nemo, "Análisis Razonado", "quarterly",
                                               period_end, [], "not_found", 0, datetime.now()))
                else:
                    parts = fetch_pdf_parts(session, doc_url)
                    saved_paths = _save_pdf_parts(parts, base_path)
                    n_bytes = sum(len(p) for p in parts)
                    rows.append(_manifest_row(doc_id, rut, nemo, "Análisis Razonado", "quarterly",
                                               period_end, saved_paths, "completed", n_bytes, datetime.now()))
                    pipeline_logger.log_event(
                        pipeline_step="cmf_fetch", level="SUCCESS",
                        message="Analisis razonado fetched", ticker=nemo,
                        details={"rut": rut, "periodo": period}, log_dir=log_dir,
                    )
            except Exception as e:
                rows.append(_manifest_row(doc_id, rut, nemo, "Análisis Razonado", "quarterly",
                                           period_end, [], f"failed: {e}", 0, datetime.now()))
                pipeline_logger.log_event(
                    pipeline_step="cmf_fetch", level="ERROR",
                    message=f"Analisis razonado failed: {e}", ticker=nemo,
                    details={"rut": rut, "periodo": period}, log_dir=log_dir,
                )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only the first N firms (smoke-testing)")
    parser.add_argument("--rut", type=str, default=None, help="Only this RUT (digits, no DV)")
    args = parser.parse_args()

    with open("configs/cl/config.yaml") as f:
        config = yaml.safe_load(f)

    universe = pd.read_csv("configs/cl/universe.csv", dtype={"rut": str})
    if args.rut:
        universe = universe[universe["rut"] == args.rut]
    if args.limit:
        universe = universe.head(args.limit)

    from_year = int(config["corpus"]["filings"]["filing_date"]["from"][:4])
    to_year = int(config["corpus"]["filings"]["filing_date"]["to"][:4])
    years = list(range(from_year, to_year + 1))

    raw_dir = Path(config["storage"]["raw_documents"])
    manifest_dir = Path(config["storage"]["interim_manifests"])
    manifest_path = manifest_dir / "filing_manifest.parquet"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(manifest_path) if manifest_path.exists() else pd.DataFrame()

    session = new_session()
    all_rows = []
    total = len(universe)
    for i, row in enumerate(universe.itertuples(), 1):
        t0 = time.monotonic()
        rows = fetch_one(session, row.rut, row.nemo, years, raw_dir, manifest_dir)
        all_rows.extend(rows)
        pipeline_logger.log_event(
            pipeline_step="cmf_fetch", level="INFO",
            message=f"[{i}/{total}] {row.nemo} done", ticker=row.nemo,
            duration_seconds=time.monotonic() - t0,
            details={"rut": row.rut, "documents": len(rows)}, log_dir=manifest_dir,
        )
        # Checkpoint after every firm — a long multi-hour run must survive
        # being killed/resumed without losing already-fetched progress,
        # same reasoning as scripts/us/edgar_fetch.py's manifest writes.
        if all_rows:
            new_df = pd.DataFrame(all_rows)
            if len(existing):
                merged = pd.concat([existing[~existing["document_id"].isin(new_df["document_id"])], new_df],
                                    ignore_index=True)
            else:
                merged = new_df
            merged.to_parquet(manifest_path, index=False)
            existing = merged

    print(f"Done. Manifest -> {manifest_path} ({len(existing)} rows)")


if __name__ == "__main__":
    main()
