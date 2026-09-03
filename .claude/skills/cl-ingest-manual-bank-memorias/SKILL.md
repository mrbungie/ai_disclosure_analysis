---
name: cl-ingest-manual-bank-memorias
description: Ingest Chilean bank Memoria Anual PDFs the user downloaded by hand into data/manual/ — matches each file to a firm/year, gzips it into the pipeline's raw storage layout, updates filing_manifest.parquet, and moves the original aside. Use when the user says they've dropped files in data/manual/ (Chile pipeline) or asks to process/ingest manual bank filings.
---

# Ingesting manually-downloaded Chile bank Memorias

## Why this exists

`scripts/cl/01_fetch_filings.py` scrapes CMF's `entidad.php` ficha system
(tipoentidad=RVEMI) for every firm in `configs/cl/universe.csv`. Four
firms — **BCI** (rut 97006000), **Banco Santander-Chile** (97036000),
**Banco de Chile** (97004000), **Banco Itaú Chile** (97023000) — are
NOT registered under RVEMI in CMF's real system at all (verified directly
against the live site and against CMF's own catalogs — this is not a bug
in the fetch script, CMF simply doesn't carry these 4 RUTs there). Their
Memoria Anual only exists on each bank's own investor-relations site.
BCI's is fetchable automatically; Banco de Chile, Santander and Itaú are
behind commercial bot-mitigation (Imperva / PerimeterX) this project does
not attempt to defeat. So for those, the user downloads the PDF by hand
(a real browser session gets past the anti-bot fine) and drops it in
`data/manual/` for this skill to pick up.

This is a skill, not a rigid script, on purpose: filenames from a
downloaded browser save are messy and inconsistent ("Memoria_Holding_2025
(3).pdf", "memoria-anual-2023.pdf", etc.) — matching them to the right
firm/year needs judgment (read the PDF's first page/cover if the filename
is ambiguous), which a fixed regex handles badly.

## What to do

1. **List `data/manual/`** for files that aren't already under
   `data/manual/processed/`. Ignore anything already processed.

2. **For each file, figure out (rut, nemo, year):**
   - Try the filename first: look for one of the 4 nemos (`BCI`,
     `BSANTANDER`, `CHILE`, `ITAUCL`) or a recognizable bank name/RUT
     substring, plus a 4-digit year.
   - If the filename doesn't make it obvious, open the PDF (e.g. `pdftotext`
     first page, or read a rendered screenshot) and check the cover /
     title page — it will say "Memoria Anual <year>" or "Memoria Integrada
     <year>" and the bank's name.
   - Cross-check the nemo against `configs/cl/universe.csv` to get the
     canonical `rut` (don't hardcode the 4 above — the user may drop
     memorias for other firms too, e.g. if a normal issuer's CMF fetch
     also failed for some other reason).
   - If you genuinely can't tell, ask the user rather than guessing.

3. **Skip firms that already have downloads from CMF for that document
   type.** These 4 banks only need Memoria Anual (see the project's
   scoping decision — Análisis Razonado equivalent was intentionally left
   out for banks). Don't ingest a non-annual document under this flow
   without checking with the user first.

4. **Gzip into the standard raw layout**, matching
   `scripts/cl/01_fetch_filings.py`'s own convention exactly:
   `data/raw/filings_pdf_cl/<rut>/memoria_<year>.pdf.gz`
   (get `<rut>` zero-padding/format from `configs/cl/universe.csv`, and
   the raw dir root from `configs/cl/config.yaml`'s
   `storage.raw_documents`). If that `.pdf.gz` already exists, don't
   overwrite it — move the source file to `data/manual/processed/` as a
   duplicate and skip (same idempotency contract as the fetch script:
   the file on disk is the source of truth for "already have this").

5. **Update `data/interim/manifests_cl/filing_manifest.parquet`** (read
   with pandas, don't hand-edit): add one row per newly-ingested document
   with the same columns `01_fetch_filings.py` writes — `document_id`
   (`<rut>_memoria_<year>`), `rut`, `nemo`, `form_type` ("Memoria Anual"),
   `filing_type` ("annual"), `period_end_date` (`<year>-12-31`),
   `local_path`, `format` ("pdf"), `download_status` ("completed"),
   `parse_status` ("pending"), `n_bytes`, `created_at`, `updated_at` —
   **except `source`, which must be `"manual_ir"`, not `"CMF"`**: this
   bypassed the regulator entirely and that provenance matters if anyone
   ever audits the corpus. Drop any existing row with the same
   `document_id` before appending (same merge pattern as the fetch
   script) so re-running this skill is idempotent.

6. **Move the original file** from `data/manual/` into
   `data/manual/processed/` (create that dir if needed) once it's safely
   gzipped and in the manifest — never delete the user's original.

7. **Tell the user what you ingested**, and offer to run
   `scripts/cl/02_extract_text.py` next so the new documents actually get
   their paragraphs extracted (it picks up any `parse_status == "pending"`
   row automatically, no flag needed).

## Don't

- Don't build this back into a fixed-regex script — that's exactly what
  this skill replaced, because manually-downloaded filenames aren't
  reliable enough for one.
- Don't try to re-derive the anti-bot bypass for Banco de Chile/Santander/
  Itaú's own sites (Imperva/PerimeterX) — that ground was already covered
  and rejected; manual download is the deliberate answer, not a fallback
  to reach for less scraping effort.
- Don't touch `source="CMF"` rows from the regular fetch pipeline.
