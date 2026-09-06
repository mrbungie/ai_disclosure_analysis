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
  1. Render thesis.qmd with Quarto exactly as authored (content + TOC + bibliography).
  2. Prepend cover.docx, with one blank page in between cover and content.
  3. Recompute the Table of Contents (page numbers, entries) via a headless
     LibreOffice pass, so the file that ships already has a correct, baked-in TOC.
  4. Save as compiled/GermanOviedo_FinalThesis_<YYYYmmdd_HHMMSS>.docx
  5. Remove every intermediate file/dir created along the way.

Requires: quarto, soffice (LibreOffice) on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docxcompose.composer import Composer

ROOT = Path(__file__).resolve().parent
QMD = ROOT / "thesis.qmd"
COVER = ROOT / "cover.docx"
COMPILED_DIR = ROOT / "compiled"
AUTHOR_SLUG = "GermanOviedo"

# --- LibreOffice macro bootstrap -------------------------------------------

MACRO_MODULE = (
    Path.home()
    / "Library/Application Support/LibreOffice/4/user/basic/Standard/Module1.xba"
)
MACRO_NAME = "UpdateTocAndSave"
MACRO_CODE = f"""
Sub {MACRO_NAME}
    Dim oDoc As Object
    Dim oIndexes As Object
    Dim i As Integer

    oDoc = ThisComponent
    oDoc.updateLinks()

    oIndexes = oDoc.getDocumentIndexes()
    For i = 0 To oIndexes.getCount() - 1
        oIndexes.getByIndex(i).update()
    Next i

    oDoc.store()
    oDoc.close(False)
End Sub
"""


def ensure_toc_macro() -> None:
    """Install the UpdateTocAndSave Basic macro into LibreOffice's user
    profile if it isn't already there. Idempotent."""
    if not MACRO_MODULE.exists():
        raise RuntimeError(
            f"LibreOffice Basic module not found at {MACRO_MODULE}. "
            "Open LibreOffice once (any app) to initialize the user profile, then retry."
        )
    content = MACRO_MODULE.read_text(encoding="utf-8")
    if f"Sub {MACRO_NAME}" in content:
        return
    new_content = content.replace(
        "</script:module>", MACRO_CODE + "</script:module>"
    )
    MACRO_MODULE.write_text(new_content, encoding="utf-8")


# --- Pipeline steps ----------------------------------------------------------


def render_quarto_content(tmp_dir: Path) -> Path:
    subprocess.run(
        ["quarto", "render", str(QMD), "--output-dir", str(tmp_dir)],
        check=True,
        cwd=ROOT,
    )
    candidates = list(tmp_dir.glob("*.docx"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one rendered docx in {tmp_dir}, found {candidates}")
    return candidates[0]


def merge_cover_and_content(content_docx: Path, tmp_dir: Path) -> Path:
    merged_path = tmp_dir / "merged.docx"

    master = Document(str(COVER))

    # Blank page between cover and content: page break, empty paragraph, page break.
    master.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    master.add_paragraph()
    master.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    composer = Composer(master)
    composer.append(Document(str(content_docx)))
    composer.save(str(merged_path))
    return merged_path


def refresh_toc_in_place(docx_path: Path) -> None:
    ensure_toc_macro()
    lock_file = docx_path.parent / f".~lock.{docx_path.name}#"
    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--invisible",
                "--norestore",
                str(docx_path),
                f"vnd.sun.star.script:Standard.Module1.{MACRO_NAME}"
                "?language=Basic&location=application",
            ],
            check=True,
            timeout=120,
        )
    finally:
        lock_file.unlink(missing_ok=True)


def cleanup_quarto_artifacts() -> None:
    for name in (".quarto", f"{QMD.stem}_files", "_freeze"):
        path = ROOT / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> None:
    for tool in ("quarto", "soffice"):
        if shutil.which(tool) is None:
            sys.exit(f"Required tool '{tool}' not found on PATH.")

    COMPILED_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = COMPILED_DIR / f"{AUTHOR_SLUG}_FinalThesis_{timestamp}.docx"

    with tempfile.TemporaryDirectory(prefix="thesis_render_") as tmp:
        tmp_dir = Path(tmp)

        print("[1/4] Rendering Quarto content...")
        content_docx = render_quarto_content(tmp_dir)

        print("[2/4] Merging cover + blank page + content...")
        merged_docx = merge_cover_and_content(content_docx, tmp_dir)
        shutil.copy(merged_docx, final_path)

        print("[3/4] Refreshing table of contents (LibreOffice headless)...")
        refresh_toc_in_place(final_path)

    print("[4/4] Cleaning up intermediates...")
    cleanup_quarto_artifacts()

    print(f"Done: {final_path}")


if __name__ == "__main__":
    main()
