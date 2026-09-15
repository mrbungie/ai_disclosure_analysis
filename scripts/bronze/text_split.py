"""
scripts/bronze/text_split.py — section text -> paragraphs -> sentences, in polars.

This defines `paragraph_index` and `text_hash`, the keys every additive
output (embeddings, prefilter scores, LLM frames and activities, golden set)
was written against. The rules therefore reproduce the corpus those outputs
were computed on byte for byte, including two quirks that must NOT be
"fixed" without recomputing every additive output:

  * Markdown links `[visible](target)` are replaced by the control character
    U+0001, not by their visible text (~3% of paragraphs carry it).
  * "Whitespace" follows the original engine: `trim` strips only Unicode
    space separators (category Zs: U+0020, U+00A0, U+1680, U+2000-U+200A,
    U+202F, U+205F, U+3000), never tabs; regex `\\s` means [\\t\\n\\f\\r ].

Paragraph rules (one row per paragraph):
  1. split section_text on "\\n" (1-based `line_index`), strip image refs,
     replace links (see above), trim;
  2. drop blank lines, bare page numbers and "Table of Contents" /
     "Index to Financial Statements" navigation lines;
  3. classify each remaining line: 'table' (contains "|", or a dash-only row
     next to a "|" line), 'list' (starts with "•", "- " or "* "), 'prose';
  4. drop dash-only rows not adjacent to a "|" line (page-break rules);
  5. consecutive table lines merge into one paragraph, consecutive list lines
     too; every prose line is its own paragraph;
  6. `paragraph_index` = first line_index of the paragraph; `text_hash` =
     BLAKE2b-8 of the paragraph text (lines joined by "\\n"); `is_scorable` =
     trimmed length > 3 and at least one ASCII alphanumeric.

Sentence rules: prose paragraphs split at [.!?]+ followed by whitespace and a
capital letter (common abbreviations masked first); list paragraphs split
into their lines; table paragraphs stay whole unless they are prose packed
into table cells (`resplit_table_text`).
"""

from __future__ import annotations

import hashlib
import re

import polars as pl

KEY = ["country_code", "form", "accession_number", "item_key"]

ZS_CHARS = "\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
RE2_SPACE = r"[\t\n\f\r ]"

STRIP_MD_IMAGES_RE = r"!\[[^\]]*\]\([^)]*\)"
STRIP_MD_LINKS_RE = r"\[([^\]]*)\]\([^)]*\)"
LINK_REPLACEMENT = "\x01"
_WS = r"[\t\n\f\r \xa0]"
TOC_NAV_LINK_RE = (
    rf"(?i)^{_WS}*"
    rf"(Tables?({_WS}*of{_WS}*Contents?)?[.`]?{_WS}*)?"
    rf"(Index{_WS}+to{_WS}+Financial{_WS}+Statements?[.`]?{_WS}*)?"
    r"$"
)

ABBREV_WORDS = (
    "Mr|Mrs|Ms|Dr|Jr|Sr|Prof|Rev|Gen|Sen|Rep|Gov|Lt|Col|Capt|Cmdr|Sgt|"
    "vs|etc|al|Inc|Corp|Co|Ltd|LLC|No|St|Ave|Blvd|Fig|Vol|pp|cf|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    "Mon|Tue|Wed|Thu|Fri|Sat|Sun"
)
# (?-u:\b): ASCII word boundary, as in the original engine.
ABBREV_WORD_RE = rf"(?i)(?-u:\b)({ABBREV_WORDS})\."
ABBREV_COMPOUND_RE = r"(?i)(?-u:\b)(U\.S|U\.K|i\.e|e\.g|a\.m|p\.m)\."
ABBREV_MASK = chr(30)
SENTENCE_BOUNDARY_RE = rf"([.!?]+){RE2_SPACE}+([A-Z])"
SENTENCE_BOUNDARY_MARKER = chr(31)


def text_hash8(text: str | None) -> int:
    """BLAKE2b-8 of paragraph text, big-endian unsigned — the canonical `text_hash`."""
    return int.from_bytes(hashlib.blake2b((text or "").encode("utf-8"), digest_size=8).digest(),
                          "big", signed=False)


def _hash_series(s: pl.Series) -> pl.Series:
    return pl.Series(s.name, [text_hash8(t) for t in s.to_list()], dtype=pl.UInt64)


def text_hash_expr(col: str) -> pl.Expr:
    return pl.col(col).map_batches(_hash_series, return_dtype=pl.UInt64)


def is_scorable_expr(col: str) -> pl.Expr:
    return ((pl.col(col).str.strip_chars(ZS_CHARS).str.len_chars() > 3)
            & pl.col(col).str.contains(r"[A-Za-z0-9]"))


def resplit_table_text(text: str | None) -> list[str]:
    """A 'table' paragraph stays whole unless it is prose packed into table
    cells (bullet lists of risk factors, several sentences in one cell): then
    its cells are extracted and multi-sentence cells split further."""
    text = text or ""
    if len(text) < 300 or not (text.count("|") >= 6 or text.count("---") >= 2 or text.count("●") >= 2):
        return [text]
    cells: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.count("|") >= 2:
            for cell in line.split("|"):
                cell = cell.strip().lstrip("●").strip()
                if len(cell) > 15 and not re.fullmatch(r"-{2,}", cell):
                    cells.append(cell)
        elif len(line) > 15:
            cells.append(line)
    out: list[str] = []
    for cell in cells:
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", cell):
            part = part.strip()
            if len(part) > 10:
                out.append(part)
    deduped: list[str] = []
    for s in out:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    return deduped or [text]


def split_paragraphs(sections: pl.LazyFrame) -> pl.LazyFrame:
    """sections: one row per found section with KEY columns + section_text."""
    lines = (
        sections.filter(pl.col("section_text").is_not_null())
        .select(*KEY, pl.col("section_text").str.split("\n").alias("line_text"))
        .with_columns(pl.int_ranges(1, pl.col("line_text").list.len() + 1).alias("line_index"))
        .explode(["line_text", "line_index"], empty_as_null=True)
        .with_columns(
            pl.col("line_text")
            .str.replace_all(STRIP_MD_IMAGES_RE, "")
            .str.replace_all(STRIP_MD_LINKS_RE, LINK_REPLACEMENT, literal=False)
            .str.strip_chars(ZS_CHARS)
        )
        .filter(
            (pl.col("line_text") != "")
            & ~pl.col("line_text").str.contains(r"^[0-9]+$")
            & ~pl.col("line_text").str.strip_chars("` ").str.contains(TOC_NAV_LINK_RE)
        )
    )
    part = ["form", "accession_number", "item_key"]
    dash_only = pl.col("line_text").str.strip_chars("- ") == ""
    classified = (
        lines.with_columns(pl.col("line_text").str.contains("|", literal=True).alias("has_pipe"))
        .with_columns(
            pl.col("has_pipe").shift(1).over(part, order_by="line_index").fill_null(False).alias("prev_has_pipe"),
            pl.col("has_pipe").shift(-1).over(part, order_by="line_index").fill_null(False).alias("next_has_pipe"),
        )
        .with_columns(
            pl.when(pl.col("has_pipe")).then(pl.lit("table"))
            .when(dash_only & (pl.col("prev_has_pipe") | pl.col("next_has_pipe"))).then(pl.lit("table"))
            .when(pl.col("line_text").str.starts_with("•")
                  | pl.col("line_text").str.contains(rf"^[-*]{RE2_SPACE}")).then(pl.lit("list"))
            .otherwise(pl.lit("prose"))
            .alias("content_type")
        )
        .filter(~(dash_only & ~pl.col("prev_has_pipe") & ~pl.col("next_has_pipe")))
    )
    grouped = (
        classified.with_columns(pl.col("content_type").shift(1).over(part, order_by="line_index").alias("prev_type"))
        .with_columns(
            ((pl.col("content_type") == "prose")
             | pl.col("prev_type").is_null()
             | (pl.col("content_type") != pl.col("prev_type"))).cast(pl.UInt32)
            .cum_sum().over(part, order_by="line_index").alias("group_id")
        )
    )
    return (
        grouped.group_by(*KEY, "group_id", "content_type")
        .agg(
            pl.col("line_index").min().alias("paragraph_index"),
            pl.col("line_text").sort_by("line_index").str.join("\n").alias("paragraph_text"),
        )
        .drop("group_id")
        .with_columns(
            pl.col("paragraph_index").cast(pl.Int64),
            text_hash_expr("paragraph_text").alias("text_hash"),
            is_scorable_expr("paragraph_text").alias("is_scorable"),
        )
        .select(*KEY, "content_type", "paragraph_index", "paragraph_text", "text_hash", "is_scorable")
    )


def split_sentences(paragraphs: pl.LazyFrame) -> pl.LazyFrame:
    keys = [*KEY, "content_type", "paragraph_index"]
    prose = (
        paragraphs.filter(pl.col("content_type") == "prose")
        .select(*keys, pl.col("paragraph_text")
                .str.replace_all(ABBREV_WORD_RE, "${1}" + ABBREV_MASK)
                .str.replace_all(ABBREV_COMPOUND_RE, "${1}" + ABBREV_MASK)
                .str.replace_all(SENTENCE_BOUNDARY_RE, "${1}" + SENTENCE_BOUNDARY_MARKER + "${2}")
                .str.split(SENTENCE_BOUNDARY_MARKER).alias("sentence_text"))
        .with_columns(pl.int_ranges(1, pl.col("sentence_text").list.len() + 1).alias("sentence_index"))
        .explode(["sentence_text", "sentence_index"], empty_as_null=True)
        .with_columns(pl.col("sentence_text").str.replace_all(ABBREV_MASK, ".", literal=True))
        .filter(pl.col("sentence_text").str.strip_chars(ZS_CHARS) != "")
    )
    lists = (
        paragraphs.filter(pl.col("content_type") == "list")
        .select(*keys, pl.col("paragraph_text").str.split("\n").alias("sentence_text"))
        .with_columns(pl.int_ranges(1, pl.col("sentence_text").list.len() + 1).alias("sentence_index"))
        .explode(["sentence_text", "sentence_index"], empty_as_null=True)
    )
    tables = (
        paragraphs.filter(pl.col("content_type") == "table")
        .select(*keys, pl.col("paragraph_text")
                .map_elements(resplit_table_text, return_dtype=pl.List(pl.String)).alias("sentence_text"))
        .with_columns(pl.int_ranges(1, pl.col("sentence_text").list.len() + 1).alias("sentence_index"))
        .explode(["sentence_text", "sentence_index"], empty_as_null=True)
    )
    cols = [*keys, pl.col("sentence_index").cast(pl.Int64), "sentence_text"]
    return pl.concat([prose.select(cols), lists.select(cols), tables.select(cols)])
