"""
scripts/cl/04_fetch_accounting_data.py — fetches the "Estados financieros
(XBRL)" package for every firm in configs/cl/universe.csv, at each
quarterly close (03/06/09/12), from the SAME EEFF tab (pestania=3)
01_fetch_filings.py already scrapes for Análisis Razonado — see
docs/sources/accounting_data.md for why this route was picked over CMF's
sa_eeff_ifrs bulk system (that one's `xls=y` export, verified directly,
only returns the roster of firms that reported a period, not the actual
figures).

This is the OUTCOME side of the Chile panel (XBRL outcomes instrument),
parallel to scripts/us/03_fetch_accounting_data.py — separate from
scripts/cl/01_fetch_filings.py + 02_extract_text.py, which handle the
NARRATIVE side (Memoria/Análisis Razonado text).

Saves the raw ZIP as-is (already compressed — no extra gzip step, unlike
the PDF fetch): {rut}_{period}_C.zip, containing the .xbrl instance
document + .xsd schema + -definition.xml (verified directly on a real
filing — Empresas Copec FY2023). Parsing the XBRL itself is a downstream
concern, deliberately not done here — same "download raw, parse later"
split as 01_fetch_filings.py/02_extract_text.py.

Own manifest (xbrl_manifest.parquet), separate from filing_manifest.parquet
— different document type/lineage, same reasoning as filing_manifest_10q
being its own table instead of a form_type filter on one shared manifest.

Same known gap as the narrative fetch: the 4 banks (BCI, Banco Santander-
Chile, Banco de Chile, Banco Itaú Chile) aren't registered under
tipoentidad=RVEMI at all, so this tab has nothing for them either — no
manual-download fallback has been built for XBRL since it wasn't asked
for.

Usage:
    uv run python scripts/cl/04_fetch_accounting_data.py [--limit N] [--rut RUT]
"""

import argparse
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/cl/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # scripts/common/

import pipeline_logger
from cmf_direct_client import fetch_binary, new_session, post_legacy, resolve_doc_url

QUARTER_CLOSES = ["03", "06", "09", "12"]

# Same tab, same row shape as scripts/cl/01_fetch_filings.py's
# _EEFF_DOC_RE — filtering for "xbrl" in the document name instead of
# "analisis"/"análisis".
_EEFF_DOC_RE = re.compile(
    r'href="(\.\./inc/inf_financiera/ifrs/safec_ifrs_verarchivo\.php\?auth=[^"]+&send=[^"]+)"[^>]*>\s*([^<]{2,60})'
)


def _ficha_url(rut: str) -> str:
    return (f"/institucional/mercados/entidad.php?mercado=V&rut={rut}&grupo=&tipoentidad=RVEMI"
            f"&row=&vig=VI&control=svs&pestania=3")


def _find_xbrl_doc_url(html: str) -> str | None:
    for href, name in _EEFF_DOC_RE.findall(html):
        if "xbrl" in name.lower():
            return resolve_doc_url(href)
    return None


def fetch_one(session, rut: str, nemo: str, years: list[int], raw_dir: Path, log_dir: Path) -> list[dict]:
    rows = []
    for anio in years:
        for mes in QUARTER_CLOSES:
            period = f"{anio}{mes}"
            document_id = f"{rut}_xbrl_{period}"
            zip_path = raw_dir / rut / f"{rut}_{period}_C.zip"
            if zip_path.exists():
                continue
            period_end = date(anio, int(mes), 1)
            try:
                html = post_legacy(session, _ficha_url(rut), {
                    "forma": "F", "mm": mes, "aa": str(anio), "tipo": "C", "tipo_norma": "IFRS",
                })
                doc_url = _find_xbrl_doc_url(html)
                if doc_url is None:
                    rows.append(_manifest_row(document_id, rut, nemo, period_end, None, "not_found", 0))
                    continue
                content = fetch_binary(session, doc_url)
                if not content.startswith(b"PK\x03\x04"):
                    rows.append(_manifest_row(document_id, rut, nemo, period_end, None,
                                               "failed: not a zip", 0))
                    continue
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                zip_path.write_bytes(content)
                rows.append(_manifest_row(document_id, rut, nemo, period_end, zip_path, "completed", len(content)))
                pipeline_logger.log_event(
                    pipeline_step="cl_xbrl_fetch", level="SUCCESS",
                    message="XBRL package fetched", ticker=nemo,
                    details={"rut": rut, "periodo": period}, log_dir=log_dir,
                )
            except Exception as e:
                rows.append(_manifest_row(document_id, rut, nemo, period_end, None, f"failed: {e}", 0))
                pipeline_logger.log_event(
                    pipeline_step="cl_xbrl_fetch", level="ERROR",
                    message=f"XBRL fetch failed: {e}", ticker=nemo,
                    details={"rut": rut, "periodo": period}, log_dir=log_dir,
                )
    return rows


def _manifest_row(document_id, rut, nemo, period_end_date, local_path, download_status, n_bytes):
    return {
        "document_id": document_id,
        "rut": rut,
        "nemo": nemo,
        "source": "CMF",
        "form_type": "Estados Financieros XBRL",
        "period_end_date": period_end_date,
        "local_path": str(local_path) if local_path else "",
        "format": "xbrl_zip",
        "download_status": download_status,
        "n_bytes": n_bytes,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


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

    raw_dir = Path(config["storage"]["raw_xbrl"])
    manifest_dir = Path(config["storage"]["interim_manifests"])
    manifest_path = manifest_dir / "xbrl_manifest.parquet"
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
            pipeline_step="cl_xbrl_fetch", level="INFO",
            message=f"[{i}/{total}] {row.nemo} done", ticker=row.nemo,
            duration_seconds=time.monotonic() - t0,
            details={"rut": row.rut, "documents": len(rows)}, log_dir=manifest_dir,
        )
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
