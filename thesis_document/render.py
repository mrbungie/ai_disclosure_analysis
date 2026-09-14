#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "python-docx>=1.1",
#     "docxcompose>=1.4",
# ]
# ///
"""
Render the thesis into a final, submission-ready .docx.

Pipeline:
  0. Refresh every deterministic analytics output the thesis cites
     (`make analytics`) so the render can never bake in stale numbers from
     a script someone forgot to re-run -- see `refresh_analytics()`.
  1. Render thesis.qmd with Quarto (executing inline Python blocks for all figures).
  2. Post-process all data tables in docx: compact booktabs styling, tight cell
     margins, repeating headers, non-splitting rows, and calibrated typography.
  3. Prepend cover.docx, with one blank page in between cover and content.
  4. Mark the document's fields (TOC entries and page numbers) dirty and set
     <w:updateFields> in settings.xml -- belt-and-suspenders for anyone who
     opens the .docx by hand in Word, since that flag alone doesn't reliably
     force a recompute during the *scripted* PDF export in step 7 (AppleScript
     "save as PDF" doesn't always trigger it in time), which is why that step
     also explicitly updates every field before exporting.
  5. Save as compiled/GermanOviedo_FinalThesis_<YYYYmmdd_HHMMSS>.docx
  6. Remove every intermediate file/dir created along the way.
  7. Convert the final .docx to PDF (headless LibreOffice) and save as
     compiled_pdf/GermanOviedo_FinalThesis_<YYYYmmdd_HHMMSS>.pdf
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.table import Table
from docxcompose.composer import Composer

ROOT = Path(__file__).resolve().parent
QMD = ROOT / "thesis.qmd"
COVER = ROOT / "cover.docx"
COMPILED_DIR = ROOT / "compiled"
COMPILED_PDF_DIR = ROOT / "compiled_pdf"
AUTHOR_SLUG = "GermanOviedo"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if not VENV_PYTHON.exists():
    VENV_PYTHON = ROOT.parent / ".venv" / "bin" / "python"
SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")


def is_numeric_content(text: str) -> bool:
    t = text.strip()
    if not t or t in ("—", "-", "N/A", "--", "–"):
        return False
    if t in ("10-K", "10-Q", "DEF 14A", "8-K", "Calls"):
        return False
    clean = (
        t.replace("$", "")
        .replace("%", "")
        .replace(",", "")
        .replace("+", "")
        .replace("–", "-")
        .replace("−", "-")
        .strip()
    )
    try:
        float(clean)
        return True
    except ValueError:
        pass
    if t.startswith("[") and t.endswith("]"):
        return True
    if re.match(r"^p\s*[<>=]", t):
        return True
    if re.match(r"^\$?\d+(\.\d+)?[BMK]?$", t):
        return True
    return False


def unwrap_captioned_tables(document: Document) -> None:
    """Un-nest every captioned table Quarto wraps in a 1x1 layout table.

    Quarto's docx writer puts the caption paragraph and the real data table
    together inside one cell of an outer, single-row, single-column
    "layout" table -- purely so the caption stays glued to the table it
    describes. That nesting is invisible in Word, but LibreOffice's PDF
    export handles a nested table badly when the outer row does not fit in
    the remaining page space: it corrupts the inner table's rendering
    (empty cells, then the same data dumped as loose paragraph text below)
    instead of just moving the block to the next page. `cantSplit` on the
    outer row does not fix this -- LibreOffice's layout engine apparently
    does not honor it for a row whose only content is a nested table.

    The fix is to remove the nesting entirely: pull the caption paragraph
    and the real table out of the wrapper cell and place them directly in
    the document body, in the same order, then delete the now-empty
    wrapper. Word and LibreOffice both lay out "caption paragraph directly
    followed by a table" correctly without any special-casing.
    """
    body = document.element.body
    for tbl in list(body.iter(qn("w:tbl"))):
        if tbl.getparent() is not body:
            continue  # nested tables get promoted to the body when their wrapper is unwrapped
        rows = tbl.findall(qn("w:tr"))
        if len(rows) != 1:
            continue
        cells = rows[0].findall(qn("w:tc"))
        if len(cells) != 1:
            continue
        cell = cells[0]
        if len(cell.findall(qn("w:tbl"))) != 1:
            continue
        children = [child for child in cell if child.tag != qn("w:tcPr")]
        parent = tbl.getparent()
        idx = list(parent).index(tbl)
        for offset, child in enumerate(children):
            parent.insert(idx + offset, child)
        parent.remove(tbl)


def style_all_tables(document: Document) -> None:
    """Apply publication-grade, compact booktabs styling to all data tables.
    
    Eliminates vertical/horizontal whitespace bloat:
      - Tight cell margins (padding: 20-25 dxa top/bottom, 35-45 dxa left/right)
      - Line spacing: 1.0, space before/after: 0.4 pt
      - Font size: 6.8pt (>=8 cols), 7.5pt (5-7 cols), 8.0pt (<=4 cols)
      - Clean horizontal rules: dark top/bottom, subtle row dividers, no vertical lines
      - Header shading (#F1F5F9) and automatic page-split header repetition (<w:tblHeader/>)
      - Prevent awkward mid-row page splits (<w:cantSplit/>)
      - Right-align numeric columns, left-align textual descriptions

    Quarto wraps every captioned table (docx output) in an outer single-cell,
    single-column table that holds the caption paragraph plus the real data
    table nested inside that same cell -- so the actual data table is never a
    top-level table in the document body. `document.tables` only walks
    top-level `w:tbl` elements, which means it only ever sees the empty
    1x1 wrapper (skipped below) and never reaches the nested table that
    needs this styling. Walking the whole tree for every `w:tbl` element
    finds the nested tables too.
    """
    no_border_xml = """
    <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:top w:val="none"/>
        <w:left w:val="none"/>
        <w:bottom w:val="none"/>
        <w:right w:val="none"/>
        <w:insideH w:val="none"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    """
    for tbl_element in document.element.body.iter(qn("w:tbl")):
        table = Table(tbl_element, document)
        try:
            n_cols = len(table.columns)
        except Exception:
            n_cols = 1
        if len(table.rows) <= 1 or n_cols <= 1:
            # The 1x1 caption wrapper Quarto puts around every captioned
            # table still inherits the "Table" style's colored grid unless
            # explicitly cleared -- that grid is what shows up as a colored
            # box around the caption and the real table nested inside it.
            table._tbl.tblPr.append(parse_xml(no_border_xml))
            # That wrapper cell holds the caption AND the real nested table
            # as siblings. If the wrapper's one row doesn't fit in the
            # remaining page space, a renderer splitting it mid-row has to
            # split the nested table along with it -- LibreOffice's PDF
            # export does this badly and corrupts the nested table's
            # rendering entirely (empty cells above, its data dumped as
            # loose paragraph text below). Marking the wrapper row
            # uncplittable forces the whole caption+table block onto the
            # next page instead, which is what should happen anyway.
            for row in table.rows:
                trPr = row._tr.get_or_add_trPr()
                trPr.append(parse_xml(r'<w:cantSplit xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
            continue

        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        num_cols = len(table.columns)

        if num_cols >= 8:
            font_size = Pt(6.8)
            pad_tb, pad_lr = "20", "35"
        elif num_cols >= 5:
            font_size = Pt(7.5)
            pad_tb, pad_lr = "22", "40"
        else:
            font_size = Pt(8.0)
            pad_tb, pad_lr = "26", "48"

        tblPr = table._tbl.tblPr

        borders_xml = """
        <w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:top w:val="single" w:sz="8" w:space="0" w:color="0F172A"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="0F172A"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
        """
        tblPr.append(parse_xml(borders_xml))

        margins_xml = f"""
        <w:tblCellMar xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:top w:w="{pad_tb}" w:type="dxa"/>
            <w:bottom w:w="{pad_tb}" w:type="dxa"/>
            <w:left w:w="{pad_lr}" w:type="dxa"/>
            <w:right w:w="{pad_lr}" w:type="dxa"/>
        </w:tblCellMar>
        """
        tblPr.append(parse_xml(margins_xml))

        col_numeric = [0] * num_cols
        col_count = [0] * num_cols
        for row_idx in range(1, len(table.rows)):
            row = table.rows[row_idx]
            for c_idx, cell in enumerate(row.cells):
                txt = cell.text.strip()
                if txt:
                    col_count[c_idx] += 1
                    if is_numeric_content(txt):
                        col_numeric[c_idx] += 1

        is_numeric_col = [
            (col_numeric[c] / col_count[c] >= 0.5 if col_count[c] > 0 else False)
            for c in range(num_cols)
        ]

        for r_idx, row in enumerate(table.rows):
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(r'<w:cantSplit xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))
            if r_idx == 0:
                trPr.append(parse_xml(r'<w:tblHeader xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'))

            for c_idx, cell in enumerate(row.cells):
                tcPr = cell._tc.get_or_add_tcPr()
                if r_idx == 0:
                    tcPr.append(parse_xml(r'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="F1F5F9"/>'))
                    tcPr.append(parse_xml(r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:bottom w:val="single" w:sz="8" w:space="0" w:color="475569"/></w:tcBorders>'))

                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(0.4)
                    p.paragraph_format.space_after = Pt(0.4)
                    p.paragraph_format.line_spacing = 1.0

                    if r_idx == 0:
                        if is_numeric_col[c_idx] and c_idx > 0:
                            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                        else:
                            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    else:
                        if is_numeric_content(cell.text) and c_idx > 0:
                            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                        else:
                            p.alignment = WD_ALIGN_PARAGRAPH.LEFT

                    for run in p.runs:
                        run.font.name = "Calibri"
                        run.font.size = font_size
                        if r_idx == 0:
                            run.font.bold = True
                            run.font.color.rgb = RGBColor(15, 23, 42)

                    for math_zone in p._p.iter(qn("m:oMath")):
                        set_math_run_size(math_zone, round(font_size.pt * 2))


def set_math_run_size(math_zone, half_points: int) -> None:
    """Pandoc's LaTeX-to-OMML conversion never writes a `w:sz` on the math
    runs it emits, so an equation object has no explicit font size of its
    own -- Word then renders it at its own default math size, which has no
    reason to match the surrounding text (a table cell's font might be
    styled down to 6.8-8pt by `style_all_tables`, but the equation sitting
    right next to it stays at Word's normal default and looks oversized and
    inconsistent). `paragraph.runs` in python-docx also does not surface
    `m:r` elements at all (they aren't `w:r`), so the sizing loop just above
    silently skips them -- this has to run as a separate, explicit pass.
    Setting `w:sz` directly on each math run's run properties is the fix;
    the math font itself (Cambria Math) is left alone so symbols still
    render as symbols instead of falling back to Calibri glyphs."""
    for m_r in math_zone.iter(qn("m:r")):
        w_rPr = m_r.find(qn("w:rPr"))
        if w_rPr is None:
            w_rPr = m_r.makeelement(qn("w:rPr"), {})
            m_rPr = m_r.find(qn("m:rPr"))
            if m_rPr is not None:
                m_rPr.addnext(w_rPr)
            else:
                m_r.insert(0, w_rPr)
        for existing_sz in w_rPr.findall(qn("w:sz")):
            w_rPr.remove(existing_sz)
        sz = w_rPr.makeelement(qn("w:sz"), {qn("w:val"): str(half_points)})
        w_rPr.append(sz)


def fix_prose_math_font_size(document: Document, half_points: int = 22) -> None:
    """Same fix as inside table cells (see `set_math_run_size`), for every
    equation that sits in ordinary body text rather than a table -- those
    never get touched by `style_all_tables` at all, so they're just as
    liable to render at a mismatched default size. 22 half-points (11pt)
    matches this template's Normal style."""
    tbl_tag = qn("w:tbl")
    body = document.element.body
    for math_zone in body.iter(qn("m:oMath")):
        ancestor, in_table = math_zone.getparent(), False
        while ancestor is not None:
            if ancestor.tag == tbl_tag:
                in_table = True
                break
            ancestor = ancestor.getparent()
        if in_table:
            continue  # already sized to its table's font_size above
        set_math_run_size(math_zone, half_points)


def content_width_emu(document: Document) -> int:
    """Actual printable width of the page, in EMU (914400 per inch).

    Quarto sizes embedded figures to (page width - left margin - right
    margin), but this template's section also reserves a binding `gutter`
    -- extra space added on top of the margins for a bound copy. Quarto's
    width calculation does not know about the gutter, so every figure comes
    out `gutter` wider than the page can actually print, and spills past
    the right margin in Word. Computing the width from the actual section
    properties (not a hardcoded constant) means this keeps working if the
    template's page size or margins ever change.
    """
    section = document.sections[0]
    return section.page_width - section.left_margin - section.right_margin - section.gutter


def resize_oversized_images(document: Document, max_width_emu: int) -> None:
    """Scale down (preserving aspect ratio) any embedded drawing wider than
    the page can actually print. Both `wp:extent` (the drawing's layout
    box) and the nested `a:ext` (the picture transform inside it) have to
    be updated together, or Word stretches/distorts the image to fill
    whichever box is now the odd one out."""
    body = document.element.body
    for extent in body.iter(qn("wp:extent")):
        cx_str = extent.get("cx")
        if not cx_str:
            continue
        cx = int(cx_str)
        if cx <= max_width_emu:
            continue
        cy_str = extent.get("cy")
        if not cy_str:
            continue
        cy = int(cy_str)
        scale = max_width_emu / cx
        new_cx, new_cy = max_width_emu, round(cy * scale)
        extent.set("cx", str(new_cx))
        extent.set("cy", str(new_cy))
        drawing = extent.getparent().getparent()  # wp:inline or wp:anchor -> w:drawing
        for xfrm_ext in drawing.iter(qn("a:ext")):
            xfrm_cx = xfrm_ext.get("cx")
            if xfrm_cx and int(xfrm_cx) == cx:
                xfrm_ext.set("cx", str(new_cx))
                xfrm_ext.set("cy", str(new_cy))


def fix_hyperlink_style(document: Document) -> None:
    """Force the docx's `Hyperlink` character style to bold black, no
    underline.

    Pandoc's docx writer always emits its own `Hyperlink` style (blue,
    single underline) into the output regardless of what `reference-doc`
    defines for that style name -- editing template/reference.docx alone
    does not change it, so it has to be patched here, post-render."""
    try:
        style = document.styles["Hyperlink"]
    except KeyError:
        return
    style.font.bold = True
    style.font.italic = False
    style.font.underline = False
    style.font.color.rgb = RGBColor(0, 0, 0)


def refresh_analytics() -> None:
    """Rebuild every deterministic data/processed/clusters/ output the thesis
    cites (`make analytics`, see the repo Makefile), before Quarto ever
    reads them.

    Without this, a render is only as fresh as whoever last remembered to
    re-run the right scripts by hand -- exactly how `call_beta_*`,
    `strategy_economic_profiles.json`, and `bootstrap_jaccard_200_results.
    json` went stale for days after `activity_profiles.py`/
    `build_strategy_dimensions.py` changed (2026-09-13). No LLM calls, no
    API spend: everything here reads `gold_ai_frames`/`gold_ai_activities`
    (which DID cost LLM calls and are never touched) and the raw XBRL/
    market data, and is safe to re-run on every render.

    Does NOT run `duckdb-text` (rebuilding `gold_ai_frames` itself from the
    raw corpus, minutes-long) or `b2-check` (multi-session B2 sync) -- run
    `make refresh-stale` by hand first if the underlying corpus or the AI
    classify/activities runs themselves changed, not just the analytics
    layer built on top of them.
    """
    repo_root = ROOT.parent
    subprocess.run(["make", "analytics"], check=True, cwd=repo_root)


def render_quarto_content(tmp_dir: Path) -> Path:
    env = os.environ.copy()
    if VENV_PYTHON.exists():
        env["QUARTO_PYTHON"] = str(VENV_PYTHON)

    subprocess.run(
        ["quarto", "render", str(QMD), "--output-dir", str(tmp_dir)],
        check=True,
        cwd=ROOT,
        env=env,
    )
    candidates = list(tmp_dir.glob("*.docx"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one rendered docx in {tmp_dir}, found {candidates}")
    
    doc = Document(str(candidates[0]))
    remove_autogenerated_title_paragraph(doc)
    unwrap_captioned_tables(doc)
    style_all_tables(doc)
    fix_prose_math_font_size(doc)
    resize_oversized_images(doc, content_width_emu(doc))
    doc.save(str(candidates[0]))

    return candidates[0]


def remove_autogenerated_title_paragraph(document: Document) -> None:
    """Deletes the single "Title"-styled paragraph pandoc prepends to the
    body when thesis.qmd's YAML has a `title:` -- which it only does so
    that "Executive Summary" (the qmd's own first `#` heading) doesn't get
    silently promoted into that role itself. Quarto's *notebook* (ipynb)
    rendering path -- the one this whole document goes through, since it
    has executable Python chunks -- auto-detects a document title from the
    first level-1 heading whenever no `title:` metadata is given, styles
    that heading "Title" instead of "Heading 1", and inserts the
    docx-native Table-of-Contents field right after it instead of before
    it. The visible symptom: "Executive Summary" renders as a giant title
    line, immediately followed by the ToC, with the Executive Summary's
    *own* body paragraphs pushed to reappear only after the multi-page
    ToC. Declaring a `title:` heads that off (the heading auto-promotion
    only kicks in when no title is set) but pandoc then dutifully renders
    that title as its own visible "Title" paragraph in the body -- right
    where cover.docx (prepended after this content is produced) already
    puts the real title page. This removes that now-redundant paragraph,
    leaving the ToC as the body's first element and "Executive Summary" as
    an ordinary Heading 1 immediately followed by its own text."""
    body = document.element.body
    for p in body.findall(qn("w:p")):
        p_pr = p.find(qn("w:pPr"))
        if p_pr is None:
            continue
        p_style = p_pr.find(qn("w:pStyle"))
        if p_style is not None and p_style.get(qn("w:val")) == "Title":
            body.remove(p)
            return


def merge_cover_and_content(content_docx: Path, tmp_dir: Path) -> Path:
    merged_path = tmp_dir / "merged.docx"

    master = Document(str(COVER))

    # Blank page between cover and content
    master.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    master.add_paragraph()
    master.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    composer = Composer(master)
    composer.append(Document(str(content_docx)))
    composer.save(str(merged_path))
    return merged_path


def mark_fields_dirty_and_save(document: Document, out_path: Path) -> None:
    fix_hyperlink_style(document)
    for fld_char in document.element.body.iter(qn("w:fldChar")):
        if fld_char.get(qn("w:fldCharType")) == "begin":
            fld_char.set(qn("w:dirty"), "true")

    settings = document.settings.element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = settings.makeelement(qn("w:updateFields"), {})
        settings.insert(0, update_fields)
    update_fields.set(qn("w:val"), "true")

    document.save(str(out_path))


def convert_to_pdf_via_word(docx_path: Path, pdf_path: Path) -> None:
    """Convert via Microsoft Word itself (macOS, AppleScript). Word is the
    application this document's template and styles are actually built
    for, and it lays out tables that need to split across a page break
    correctly. LibreOffice's headless PDF export was tried first and
    rejected: it has a layout bug where a table that must split across a
    page boundary sometimes renders the split corrupted (empty cells, the
    row data dumped afterward as loose paragraph text) instead of just
    repeating the header row on the next page -- confirmed by generating
    the same document both ways and finding the same tables broken only in
    the LibreOffice output."""
    # A ~100-page document with this many embedded figures takes Word
    # longer to open, paginate, and export than AppleScript's default
    # Apple Event timeout (about 120s), which aborts the whole script with
    # error -1712 partway through -- `with timeout of` raises that ceiling
    # for events sent to Word specifically.
    # "save as" needs an actual document reference bound to a variable --
    # inlining the bare expression "active document" as its direct object
    # fails outright ("active document doesn't understand save as", -1708).
    # But the REVERSE substitution, keeping that same bound reference
    # (theDoc) for the later "close", reliably fails too, with a different
    # error ("active document doesn't understand close") -- something about
    # save-as-PDF invalidates the bound reference afterward. So: bind
    # theDoc once and use it for "save as" (the only place it works), then
    # re-fetch "active document" fresh for "close" (the only place that
    # works), wrapped in try/end try so a failure there -- which happens
    # after the PDF is already written -- can't be mistaken for the export
    # itself having failed.
    script = f'''
    with timeout of 900 seconds
        tell application "Microsoft Word"
            set display alerts to false
            open (POSIX file "{docx_path}" as alias)
            set theDoc to active document
            -- "repeat ... in (get X of theDoc)" silently fails on Word's element
            -- specifiers (Word chokes trying to enumerate the un-resolved
            -- reference); resolving it into a plain list first with an explicit
            -- "set ... to get ..." avoids that. The verb also matters: the
            -- one-word "update" command doesn't apply to a Field object (it's
            -- for dialogs/links/TOC-like collections) -- fields need the
            -- two-word "update field" command. Both commands accept a LIST
            -- as their direct parameter and apply to every item in one Apple
            -- Event round trip -- looping "item i of ..." one at a time here
            -- (a few hundred fields in a ~100-page document) turned each one
            -- into its own IPC round trip and made this step take minutes
            -- instead of seconds.
            try
                update field (get fields of theDoc)
            end try
            try
                update (get tables of contents of theDoc)
            end try
            save as theDoc file name (POSIX file "{pdf_path}" as string) file format format PDF
            try
                close active document saving no
            end try
            try
                set display alerts to true
            end try
        end tell
    end timeout
    '''
    subprocess.run(["osascript", "-e", script], check=True, timeout=960)


def convert_to_pdf_via_libreoffice(docx_path: Path, out_dir: Path) -> Path:
    if SOFFICE is None:
        raise RuntimeError("soffice (LibreOffice) not found on PATH; cannot convert to PDF")
    subprocess.run(
        [SOFFICE, "--headless", "--norestore", "--convert-to", "pdf", "--outdir", str(out_dir), str(docx_path)],
        check=True,
        cwd=ROOT,
    )
    pdf_path = out_dir / f"{docx_path.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"Expected {pdf_path} after soffice conversion, not found")
    return pdf_path


def convert_to_pdf(docx_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    pdf_path = out_dir / f"{docx_path.stem}.pdf"
    if sys.platform == "darwin" and shutil.which("osascript"):
        try:
            convert_to_pdf_via_word(docx_path, pdf_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            # Even when the AppleScript call itself raises (e.g. the "close"
            # step failing after a successful "save as"), the PDF may
            # already have been written correctly -- only fall back to
            # LibreOffice if it genuinely was not produced.
            if pdf_path.exists():
                return pdf_path
            print(f"Word PDF export failed ({exc}), falling back to LibreOffice...")
        else:
            if pdf_path.exists():
                return pdf_path
    return convert_to_pdf_via_libreoffice(docx_path, out_dir)


def cleanup_quarto_artifacts() -> None:
    for path in (
        ROOT / ".quarto",
        ROOT / "thesis.docx",
        ROOT / "thesis_files",
    ):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main() -> None:
    import sys

    COMPILED_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = COMPILED_DIR / f"{AUTHOR_SLUG}_FinalThesis_{timestamp}.docx"

    if "--skip-analytics" in sys.argv:
        print("[0/6] Skipping analytics refresh (--skip-analytics) -- numbers may be stale.")
    else:
        print("[0/6] Refreshing analytics outputs (make analytics)...")
        refresh_analytics()

    with tempfile.TemporaryDirectory(prefix="thesis_render_") as tmp:
        tmp_dir = Path(tmp)

        print("[1/6] Rendering Quarto content with inline Python figures...")
        content_docx = render_quarto_content(tmp_dir)

        print("[2/6] Merging cover + blank page + styled content...")
        merged_docx = merge_cover_and_content(content_docx, tmp_dir)

        print("[3/6] Marking fields dirty so Word recomputes the TOC on open...")
        mark_fields_dirty_and_save(Document(str(merged_docx)), final_path)

    print("[4/6] Cleaning up intermediates...")
    cleanup_quarto_artifacts()

    print("[5/6] Converting to PDF...")
    pdf_path = convert_to_pdf(final_path, COMPILED_PDF_DIR)

    print(f"Done: {final_path}")
    print(f"Done: {pdf_path}")


if __name__ == "__main__":
    main()
