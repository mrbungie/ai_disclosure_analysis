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
  1. Render thesis.qmd with Quarto (executing inline Python blocks for all figures).
  2. Post-process all data tables in docx: compact booktabs styling, tight cell
     margins, repeating headers, non-splitting rows, and calibrated typography.
  3. Prepend cover.docx, with one blank page in between cover and content.
  4. Mark the document's fields (TOC entries and page numbers) dirty and set
     <w:updateFields> in settings.xml, so Word recomputes them automatically
     the moment the file is opened.
  5. Save as compiled/GermanOviedo_FinalThesis_<YYYYmmdd_HHMMSS>.docx
  6. Remove every intermediate file/dir created along the way.
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
from docxcompose.composer import Composer

ROOT = Path(__file__).resolve().parent
QMD = ROOT / "thesis.qmd"
COVER = ROOT / "cover.docx"
COMPILED_DIR = ROOT / "compiled"
AUTHOR_SLUG = "GermanOviedo"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


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
    """
    for table in document.tables:
        if len(table.rows) <= 1 or len(table.columns) <= 1:
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
    style_all_tables(doc)
    doc.save(str(candidates[0]))
    
    return candidates[0]


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
    COMPILED_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = COMPILED_DIR / f"{AUTHOR_SLUG}_FinalThesis_{timestamp}.docx"

    with tempfile.TemporaryDirectory(prefix="thesis_render_") as tmp:
        tmp_dir = Path(tmp)

        print("[1/4] Rendering Quarto content with inline Python figures...")
        content_docx = render_quarto_content(tmp_dir)

        print("[2/4] Merging cover + blank page + styled content...")
        merged_docx = merge_cover_and_content(content_docx, tmp_dir)

        print("[3/4] Marking fields dirty so Word recomputes the TOC on open...")
        mark_fields_dirty_and_save(Document(str(merged_docx)), final_path)

    print("[4/4] Cleaning up intermediates...")
    cleanup_quarto_artifacts()

    print(f"Done: {final_path}")


if __name__ == "__main__":
    main()
