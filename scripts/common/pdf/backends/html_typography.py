"""
scripts/common/pdf/backends/html_typography.py — blocks out of an HTML
document whose structure lives in its TYPOGRAPHY rather than in its tags.

WHY THIS EXISTS. Italy's ESEF annual reports are XHTML, so the obvious move
was scripts/us/section_segmenter.py's path: markdownify, split on lines,
segment on `#` headings. Measured on real filings, that path is
structurally blind — these documents contain:

    <h1>..<h6>:  0        internal anchors (href="#"):  0
    <a name>:    0        markdown headings recovered:  0

because they are not authored HTML. Half the corpus (36 of 73 files
checked) carries pdf2htmlEX's own fingerprint, and the rest come from other
PDF/DOCX-to-HTML converters: thousands of absolutely-positioned <span>s
with a class each, and the document's entire visual hierarchy encoded in
the stylesheet. One file had 84,033 distinct class names.

So the structure IS there, just not where a tag-based reader looks. Every
one of these generators sets font-size by class, and 25 to 88 distinct
sizes per document survive: body text is the modal size, a heading is
bigger. That signal is the same one scripts/common/pdf/backends/
pymupdf_layout.py already uses on real PDFs (`_is_uniform_heavy_font_label`)
— which is not a coincidence, since these files ARE rendered PDFs.

Deliberately generator-agnostic. Keying off pdf2htmlEX's `.fsN`/`.ffN`
naming would work on half this corpus and silently produce a structureless
document for the other half; resolving the cascade for whatever class names
a file happens to use works on both, and on the next converter nobody has
seen yet.

Its limits, stated rather than discovered later: it reads the class-level
cascade only — no inline `style=`, no id selectors, no media queries, no
specificity resolution. That is what these generators emit, but a
hand-authored page would defeat it, and it makes no attempt to hide that.
"""

from __future__ import annotations

import collections
import re

from bs4 import Comment, Declaration, Doctype, ProcessingInstruction

from scripts.common.pdf import blocks as B

#: Node types that are markup metadata, not rendered text. Without this the
#: XML declaration and each converter's "Created by ..." processing
#: instruction land in the corpus as prose.
_SKIP_NODES = (Comment, Declaration, Doctype, ProcessingInstruction)

_STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
_RULE_RE = re.compile(r"\.([\w-]+)\s*\{([^}]*)\}")
_SIZE_RE = re.compile(r"font-size\s*:\s*([\d.]+)\s*(px|pt|em|rem)", re.I)
_WEIGHT_RE = re.compile(r"font-weight\s*:\s*([\w\d]+)", re.I)
_FAMILY_RE = re.compile(r"font-family\s*:\s*([^;]+)", re.I)
_HEAVY_NAME_RE = re.compile(r"bold|black|heavy|semibold|extrabold", re.I)
#: pt and em are converted to px so one number is comparable across files.
#: The absolute value never matters — only each size's rank against the
#: document's own body size — so a wrong em base shifts everything equally
#: and changes no classification.
_TO_PX = {"px": 1.0, "pt": 96 / 72, "em": 16.0, "rem": 16.0}

_NUMERIC_TOKEN_RE = re.compile(r"^\(?-?[\d.,]+%?\)?$")
_SENTENCE_END_RE = re.compile(r"[.!?:]['\"”’)]*\s*$")


def parse_class_typography(html: str) -> dict[str, dict]:
    """class name -> {size_px, weight, family} from the document's <style>
    blocks. Later rules win, which is the cascade's own rule for equal
    specificity and all these generators emit."""
    out: dict[str, dict] = {}
    for css in _STYLE_RE.findall(html):
        for name, body in _RULE_RE.findall(css):
            entry = out.setdefault(name, {})
            if match := _SIZE_RE.search(body):
                entry["size_px"] = float(match.group(1)) * _TO_PX.get(match.group(2).lower(), 1.0)
            if match := _WEIGHT_RE.search(body):
                entry["weight"] = match.group(1).lower()
            if match := _FAMILY_RE.search(body):
                entry["family"] = match.group(1).strip()
    return {k: v for k, v in out.items() if v}


def _resolve(classes, typography: dict[str, dict]) -> tuple[float | None, bool]:
    """(size_px, is_heavy) for one element's class list. The LARGEST size
    among its classes wins: these generators stack a size class with layout
    classes, and taking the last one declared would depend on attribute
    order rather than on what the reader sees."""
    size, heavy = None, False
    for name in classes:
        entry = typography.get(name)
        if not entry:
            continue
        if (value := entry.get("size_px")) is not None:
            size = value if size is None else max(size, value)
        weight = str(entry.get("weight", ""))
        if weight in {"bold", "bolder", "600", "700", "800", "900"}:
            heavy = True
        if _HEAVY_NAME_RE.search(entry.get("family", "")):
            heavy = True
    return size, heavy


def _is_hidden_markup(tag) -> bool:
    """iXBRL's own header/hidden sections, plus anything the document
    explicitly hides. Matched on the LOCAL tag name so it works whether the
    parser kept the `ix:` prefix or dropped it."""
    name = (tag.name or "").rsplit(":", 1)[-1].lower()
    if name in {"header", "hidden", "references", "resources"} and "ix" in str(tag.name).lower():
        return True
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def _body_size(sizes: list[float]) -> float:
    """The document's body font size: the size that the most CHARACTERS are
    set in, not the most elements. Counting elements lets a few thousand
    one-character positioning spans outvote the running text."""
    if not sizes:
        return 0.0
    return collections.Counter(sizes).most_common(1)[0][0]


def _is_numeric_heavy(text: str, token_threshold: float = 0.4,
                      digit_threshold: float = 0.18) -> bool:
    """Two tests, because these converters defeat the token-based one.

    A PDF-to-HTML converter lays table cells out by absolute position and
    emits no separator between them, so concatenating the DOM produces
    "Costi per servizi vari948991" — one token, not the four cells a reader
    sees. The token test scores that as 0% numeric and lets a whole
    financial table through as prose.

    The character test catches it: real Italian narrative prose runs about
    1-2% digits, while a laid-out table runs well above 18% even when its
    cells are glued to their labels. Both tests are kept — the token one
    still catches clean numeric rows that the character one would miss
    because their numbers are short.
    """
    tokens = text.split()
    if not tokens:
        return False
    if sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t)) / len(tokens) >= token_threshold:
        return True
    letters_and_digits = [c for c in text if c.isalnum()]
    if len(letters_and_digits) < 20:
        return False
    digits = sum(1 for c in letters_and_digits if c.isdigit())
    return digits / len(letters_and_digits) >= digit_threshold


class HtmlTypographyBackend:
    """Reads pixels' worth of layout out of HTML. CPU-only, so it scales
    across processes like the PyMuPDF backend does."""

    name = "html_typography"
    reads_pixels = False

    def __init__(self, country_code: str = "", heading_ratio: float = 1.15,
                 min_body_chars: int = 2, **_):
        #: A block counts as a heading when its size exceeds the body size
        #: by this factor. 1.15 rather than something larger: these
        #: documents step sizes finely (25-88 distinct sizes each), so a
        #: sub-heading is often only ~20% above body text.
        self.heading_ratio = heading_ratio
        self.min_body_chars = min_body_chars
        self.max_workers = 1  # set by the caller's pool, not by this backend

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def parse_html(self, html: str, page: int = 0) -> list[B.Block]:
        from bs4 import BeautifulSoup

        typography = parse_class_typography(html)
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        # iXBRL declares its contexts, units and entity identifiers in a
        # HIDDEN subtree that the reader never sees. It is rendered text as
        # far as the DOM is concerned, so without this it lands in the
        # corpus as a paragraph — a real one looked like
        # "<LEI>-2022-12-31-it.xhtml <LEI> 2022-01-01 2022-12-31 <LEI> ...".
        for tag in soup.find_all(_is_hidden_markup):
            tag.decompose()

        # Weight the size histogram by characters, so running text decides
        # what "body" means (see _body_size).
        weighted: list[float] = []
        units: list[tuple[str, float | None, bool, bool]] = []
        for element in soup.find_all(string=True):
            if isinstance(element, _SKIP_NODES):
                continue  # comments, the XML declaration, the generator PI
            # NOT stripped. These converters emit a real space as its own
            # text node (pdf2htmlEX uses <span class="_ _0"> </span>), so
            # stripping each node and rejoining with " " both deletes the
            # real spaces and invents ones inside words — it turned
            # "calcestruzzo" into "c a l c es t r uz z o" on a real filing.
            # Keeping the nodes verbatim and concatenating reproduces the
            # rendered text exactly; whitespace is normalised once, later.
            text = str(element)
            if not text.strip():
                text = " " if text else ""
            if not text:
                continue
            classes: list[str] = []
            in_table = False
            for parent in element.parents:
                classes.extend(parent.get("class") or [])
                if parent.name in {"table", "td", "th", "tr"}:
                    in_table = True
            size, heavy = _resolve(classes, typography)
            if size is not None:
                weighted.extend([size] * len(text))
            units.append((text, size, heavy, in_table))

        body = _body_size(weighted)

        # Consecutive text nodes at the same size and weight are one visual
        # run — these generators split a single sentence across dozens of
        # spans (one file had 377,955 of them), so emitting a block per node
        # would shred every paragraph.
        merged: list[B.Block] = []
        current: dict | None = None
        for text, size, heavy, in_table in units:
            key = (round(size or 0, 1), heavy, in_table)
            if current is not None and current["key"] == key:
                current["parts"].append(text)
                continue
            if current is not None:
                merged.append(self._to_block(current, body, page))
            current = {"key": key, "parts": [text], "size": size,
                       "heavy": heavy, "in_table": in_table}
        if current is not None:
            merged.append(self._to_block(current, body, page))

        return _merge_prose_runs([b for b in merged if b.text.strip()])

    def _to_block(self, run: dict, body: float, page: int) -> B.Block:
        text = re.sub(r"\s+", " ", "".join(run["parts"])).strip()
        size = run["size"] or body
        if run["in_table"] or _is_numeric_heavy(text):
            block_type, raw = B.TABLE, "table_or_numeric"
        elif body and size >= body * self.heading_ratio and len(text.split()) <= 25:
            # Size AND brevity: a pull-quote or a cover paragraph can be set
            # large without being a heading, and calling it one would put a
            # whole paragraph into the heading stream.
            block_type, raw = B.HEADING, f"size_{size:.0f}_vs_body_{body:.0f}"
        elif run["heavy"] and len(text.split()) <= 12:
            block_type, raw = B.HEADING, "heavy_short"
        else:
            block_type, raw = B.BODY, "body"
        return B.Block(type=block_type, text=text, page=page, raw_type=raw,
                       meta={"size_px": size, "body_px": body, "heavy": run["heavy"]})


def _merge_prose_runs(items: list[B.Block]) -> list[B.Block]:
    """Joins consecutive body blocks that don't end in terminal punctuation.

    Same rule and the same reason as the PyMuPDF backend's
    _merge_prose_fragments: a converter splits one sentence wherever an
    inline style changed — a bolded word mid-sentence is enough — so
    without this an italicised term turns one paragraph into three.
    """
    merged: list[B.Block] = []
    for item in items:
        previous = merged[-1] if merged else None
        if (previous is not None and item.type == B.BODY and previous.type == B.BODY
                and not _SENTENCE_END_RE.search(previous.text)):
            previous.text += " " + item.text
            continue
        # A multi-line title is one heading set across several visual runs
        # at the SAME size — "Bilancio d'esercizio e" / "bilancio
        # consolidato" / "per l'esercizio chiuso" is one title, and leaving
        # it as three makes each fragment useless as a section label.
        if (previous is not None and item.type == B.HEADING and previous.type == B.HEADING
                and previous.meta.get("size_px") == item.meta.get("size_px")
                and not _SENTENCE_END_RE.search(previous.text)):
            previous.text += " " + item.text
            continue
        merged.append(item)
    return merged
