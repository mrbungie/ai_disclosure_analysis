"""
scripts/verif/pdf_audit/baseline_audit.py — how often does the text-layer
PDF backend actually fail, across the corpus rather than on a handful of
pages someone picked?

WHY THIS EXISTS. docs/analytics/pdf-backend-poc.md compared four backends
on 6 hand-picked pages of ONE filing. That is enough to show a failure
MODE exists; it is not enough to say anything about how common it is, and
the POC's write-up leaned on it harder than it could bear. This script
measures rates over a real sample: every page of every sampled document,
classified by whether the cheap backend can be trusted on it.

Each check below is a failure the POC observed on a real page, turned
into something countable:

  invisible_pages     the page has no text layer at all. The text-layer
                      backend returns NOTHING for it and the document
                      silently loses those pages — no error, no empty
                      paragraph, just absence. This is the one that can't
                      be fixed by tuning heuristics.
  spliced_paragraphs  a paragraph whose text has a gutter marker
                      (CMF/GRI reference codes) wedged INSIDE it, i.e.
                      reading order put page furniture in the middle of a
                      real sentence.
  chrome_paragraphs   a paragraph that IS page furniture and reached the
                      corpus anyway.
  ragged_tables       a pipe-delimited table paragraph holding at least
                      one row with fewer cells than the table's own widest
                      row — the shape of the "a whole column vanished"
                      failure, countable without per-table ground truth.
                      Reported as a share of TABLE PARAGRAPHS, not of rows:
                      an earlier version counted rows and produced a number
                      (28,189) that read like a catastrophe next to the
                      34,404 table paragraphs it was printed beside, when
                      the two were not the same denominator at all.
  at_risk_pages       pages that are multi-column or hold a table: the
                      conditions under which the POC found reading-order
                      and cell-loss failures. Not a failure count — an
                      exposure count.

Usage:
    uv run python scripts/verif/pdf_audit/baseline_audit.py \
        --pdf-dir data/raw/filings_pdf_cl --out audit.json [--limit N]
"""

import argparse
import gzip
import json
import re
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pymupdf

from scripts.common.pdf import PdfExtractor
from scripts.common.pdf.backends import get_backend

# CMF / GRI cross-reference codes printed down the gutter of every page of
# a Memoria that follows the CMF's own reporting norm. Anchored to a token
# boundary so "3.1.v" matches but "Nº 20.393" doesn't.
_GUTTER_MARKER = re.compile(
    r"(?<![\w.])(?:\d{1,2}\.\d{1,2}(?:\.[ivxlcdm]+|\.\d+)?(?:\.[a-e])?|\d{3}-\d{1,2}|2-\d{1,2})(?![\w.])")
# Chrome that the POC saw reach the corpus verbatim.
_CHROME = re.compile(
    r"^(?:CARTAS|PERSONAS|CLIENTES|INDICADORES|GOBIERNO|PERFIL DE|ESTRATEGIA Y|"
    r"GESTIÓN DE|COMUNIDAD Y|HECHOS RELEVANTES|INFORMES FINANCIEROS|MEDIO AMBIENTE|"
    r"PROVEEDORES|CORPORATIVO|MODELO DE NEGOCIOS|Memoria (?:Anual|Integrada) \d{4}|"
    r"Página \d+ de \d+)")
_SENTENCE = re.compile(r"[a-záéíóúñ]{3,}\s+$")


def _is_spliced(text: str) -> bool:
    """A gutter marker sitting between two lowercase words is furniture
    that landed inside a sentence — the pág.-36 failure ("...participación
    en los 3.6.ix procesos de adaptación..."). Requiring lowercase on BOTH
    sides is what keeps a legitimate numbered reference ("ver 3.1.v") and a
    list item ("2.3.2 Propiedad") from counting."""
    for match in _GUTTER_MARKER.finditer(text):
        before, after = text[:match.start()], text[match.end():]
        if _SENTENCE.search(before) and re.match(r"\s+[a-záéíóúñ]{3,}", after):
            return True
    return False


def _table_shape(text: str) -> tuple[int, int, int]:
    """(rows, ragged_rows, is_ragged) for one pipe-delimited table
    paragraph. A table whose rows all have the same cell count is
    structurally intact as far as this can tell; one with short rows lost
    cells somewhere, and the damage is not recoverable downstream because
    nothing records which column each surviving value belonged to."""
    rows = [r for r in text.split("\n") if "|" in r]
    if len(rows) < 2:
        return len(rows), 0, 0
    widths = [r.count("|") + 1 for r in rows]
    ragged = sum(1 for w in widths if w < max(widths))
    return len(rows), ragged, int(ragged > 0)


def audit_one(path_str: str) -> dict:
    path = Path(path_str)
    out = {"file": path.name, "firm": path.parent.name, "error": None,
           "pages": 0, "invisible_pages": 0, "blank_pages": 0, "at_risk_pages": 0,
           "table_pages": 0, "multicol_pages": 0, "paragraphs": 0, "prose": 0,
           "spliced_paragraphs": 0, "chrome_paragraphs": 0,
           "table_rows": 0, "ragged_rows": 0, "ragged_tables": 0,
           "tables": 0, "pages_yielding_nothing": 0}
    try:
        pdf_bytes = gzip.open(path, "rb").read()

        backend = get_backend("pymupdf", country_code="cl")
        backend.prepare_document([pdf_bytes])
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            out["pages"] = len(doc)
            for page in doc:
                signals = backend.layout_signals(page)
                if signals["text_chars"] < 40:
                    if page.get_images() or page.get_drawings():
                        out["invisible_pages"] += 1
                    else:
                        out["blank_pages"] += 1
                if signals["n_tables"]:
                    out["table_pages"] += 1
                if signals["n_columns"] > 1:
                    out["multicol_pages"] += 1
                if signals["n_tables"] or signals["n_columns"] > 1:
                    out["at_risk_pages"] += 1

        with PdfExtractor(backend="pymupdf", country_code="cl") as extractor:
            rows = extractor.extract([pdf_bytes])
        out["paragraphs"] = len(rows)
        pages_with_output = {r["page"] for r in rows}
        out["pages_yielding_nothing"] = out["pages"] - len(pages_with_output)
        for row in rows:
            text = row["paragraph_text"]
            if row["content_type"] == "table":
                out["tables"] += 1
                n_rows, ragged_rows, is_ragged = _table_shape(text)
                out["table_rows"] += n_rows
                out["ragged_rows"] += ragged_rows
                out["ragged_tables"] += is_ragged
                continue
            out["prose"] += 1
            if _CHROME.match(text):
                out["chrome_paragraphs"] += 1
            elif _is_spliced(text):
                out["spliced_paragraphs"] += 1
    except Exception:
        out["error"] = traceback.format_exc(limit=3)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    files = sorted(str(p) for p in Path(args.pdf_dir).rglob("*.pdf.gz"))
    if args.limit:
        files = files[:args.limit]
    print(f"auditing {len(files)} documents", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_one, f): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 10 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))

    ok = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    total = Counter()
    for r in ok:
        for k, v in r.items():
            if isinstance(v, int):
                total[k] += v

    def pct(n, d):
        return f"{n / d:.1%}" if d else "n/a"

    print(f"\n{'='*62}\nDOCUMENTS  {len(ok)} audited, {len(failed)} crashed")
    print(f"PAGES      {total['pages']:,}")
    print(f"\n-- pages the text-layer backend cannot see --")
    print(f"  no text layer but has ink : {total['invisible_pages']:6,}  "
          f"{pct(total['invisible_pages'], total['pages'])}")
    print(f"  genuinely blank           : {total['blank_pages']:6,}  "
          f"{pct(total['blank_pages'], total['pages'])}")
    print(f"  yielded zero paragraphs   : {total['pages_yielding_nothing']:6,}  "
          f"{pct(total['pages_yielding_nothing'], total['pages'])}")
    print(f"\n-- pages exposed to the failure modes the POC found --")
    print(f"  multi-column              : {total['multicol_pages']:6,}  "
          f"{pct(total['multicol_pages'], total['pages'])}")
    print(f"  containing a table        : {total['table_pages']:6,}  "
          f"{pct(total['table_pages'], total['pages'])}")
    print(f"  either (at risk)          : {total['at_risk_pages']:6,}  "
          f"{pct(total['at_risk_pages'], total['pages'])}")
    print(f"\n-- damage actually visible in the emitted corpus --")
    print(f"  paragraphs                : {total['paragraphs']:6,} "
          f"({total['prose']:,} prose / {total['tables']:,} table)")
    print(f"  furniture as a paragraph  : {total['chrome_paragraphs']:6,}  "
          f"{pct(total['chrome_paragraphs'], total['prose'])} of prose")
    print(f"  marker spliced mid-sentence: {total['spliced_paragraphs']:5,}  "
          f"{pct(total['spliced_paragraphs'], total['prose'])} of prose")
    print(f"  table paragraphs w/ lost cells: {total['ragged_tables']:3,}  "
          f"{pct(total['ragged_tables'], total['tables'])} of tables")
    print(f"  ragged rows / table rows  : {total['ragged_rows']:6,} / {total['table_rows']:,}  "
          f"{pct(total['ragged_rows'], total['table_rows'])}")

    worst = sorted(ok, key=lambda r: -(r["invisible_pages"] / max(r["pages"], 1)))[:12]
    print(f"\n-- worst documents by share of unreadable pages --")
    for r in worst:
        if not r["invisible_pages"]:
            break
        print(f"  {r['firm']}/{r['file']:34s} {r['invisible_pages']:4d}/{r['pages']:4d} pages "
              f"({r['invisible_pages'] / r['pages']:5.1%}), {r['paragraphs']:5d} paragraphs")
    for r in failed:
        print(f"\nCRASHED {r['firm']}/{r['file']}\n{r['error']}")


if __name__ == "__main__":
    main()
