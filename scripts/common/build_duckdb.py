"""
scripts/common/build_duckdb.py — (re)creates duckdb/thesis.duckdb: SQL VIEWS
over the pipeline's parquet outputs, so any of them is one query away
instead of a pandas glob-and-concat every time.

Most of these are VIEWs, not materialized tables — DuckDB re-evaluates the
underlying read_parquet(glob) on each query, so they're always current
with whatever's on disk. `paragraphs`/`sentences` are the exception —
materialized TABLEs (see MATERIALIZED_TABLES below), because the regex-
heavy split behind them was too expensive to recompute live on every
query. Safe/cheap to rerun any time (e.g. after a new extraction run adds
part files, or the corpus grows) — just note paragraphs/sentences won't
reflect new data until this script runs again.

MULTI-COUNTRY: every configs/<country>/config.yaml found gets its own set
of per-country tables, UNIONed (BY NAME, so a country whose schema gains
an extra column later doesn't break the others) into one final view per
name — `firm_universe`, `filing_manifest`, etc. always span every country
configured, not just "us". Right now there's only one country, so this is
a no-op in practice, but it's real: add a second `configs/uk/config.yaml`
+ matching `scripts/uk/` pipeline and its data shows up in every view here
with no code change, PROVIDED its config.yaml's storage paths don't
collide with another country's (see _country_configs' docstring).

Usage:
    uv run python scripts/common/build_duckdb.py
    duckdb duckdb/thesis.duckdb   # then: .tables / select * from filing_manifest limit 5;
"""

import glob
import hashlib
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "duckdb" / "thesis.duckdb"
PREFILTER_SCORES_DIR = REPO_ROOT / "data" / "interim" / "prefilter_scores"


def _text_hash8(text: str | None) -> int:
    """BLAKE2b-8 of paragraph text — THE single canonical definition. Every
    other script that ever needs "the hash of this paragraph's text"
    (ai_embed.py, golden_set.py, ai_classify.py) should read `paragraphs.
    text_hash` / `unique_paragraphs.text_hash`, not recompute this
    themselves: a real bug (docs/prefilter_evaluation.md §8.7) came from
    ai_classify.py independently reconstructing paragraph text slightly
    differently and hashing THAT, silently breaking a dedup join. Computing
    it once, here, at the source, removes that entire class of bug."""
    return int.from_bytes(
        hashlib.blake2b((text or "").encode("utf-8"), digest_size=8).digest(),
        "big", signed=False)

# Everything else here is a live VIEW (re-evaluated on every query, always
# current with whatever's on disk). paragraphs/sentences are the
# exception: TABLEs, computed once at build time and stored in the
# .duckdb file — the regex-heavy paragraph/sentence split (especially
# sentences' mask/boundary/split chain) re-run as a live view on every
# query was expensive enough to be genuinely disruptive on a real
# machine, not just "slow". Tradeoff: these two go stale after a new
# extraction run until build_duckdb.py is rerun (every other view here
# doesn't); worth it for queries against them to actually be fast.
MATERIALIZED_TABLES = {"paragraphs", "sentences", "unique_paragraphs"}


def _country_configs() -> list[tuple[str, dict]]:
    """[(country_code, config_dict), ...] for every configs/<country>/config.yaml
    found, sorted by country code. Each country's config.yaml:storage paths
    are assumed to point at THAT country's own data (e.g. "data/interim/
    manifests/us") — nothing here enforces that; a second country reusing
    the first's storage paths would silently collide/overwrite rather than
    union cleanly, so give every new country its own path in `storage`."""
    configs_dir = REPO_ROOT / "configs"
    countries = []
    for country_dir in sorted(configs_dir.iterdir()):
        config_path = country_dir / "config.yaml"
        if country_dir.is_dir() and config_path.exists():
            with open(config_path) as f:
                countries.append((country_dir.name, yaml.safe_load(f)))
    return countries


def _union(selects: list[str]) -> str:
    """One country's SELECT as-is; multiple countries' SELECTs unioned BY
    NAME (column-name alignment, not positional) so a schema difference
    between countries doesn't silently misalign columns or hard-fail."""
    if not selects:
        raise ValueError("_union() got zero SELECTs — every country must have been filtered out by _existing()")
    if len(selects) == 1:
        return selects[0]
    return "\n            UNION ALL BY NAME\n".join(f"({s})" for s in selects)


def _existing(path_pattern: str) -> bool:
    """True if path_pattern (an exact path or a glob) matches at least one
    file on disk. A country appearing under configs/ doesn't mean every
    view's source file exists for it yet — e.g. Chile has
    firm_universe.parquet the moment scripts/cl/00_build_firm_universe.py
    runs, but filing_manifest.parquet only once scripts/cl/
    01_fetch_filings.py has produced it, and it has no
    filing_manifest_10q.parquet at all (no 10-Q-shaped instrument for
    Chile — see docs/international_expansion_plan.md). Without this
    check, build_duckdb.py hard-fails the ENTIRE run (every country, every
    view) the moment ANY one country is mid-rollout — exactly the
    multi-country design this file's docstring promises should be safe."""
    return len(glob.glob(path_pattern)) > 0


def _filing_manifest_selects(countries: list[tuple[str, dict]], dirs) -> list[str]:
    """One SELECT per country for the `filing_manifest` view, UNIONing in
    `filing_manifest_proxy.parquet` (and, later, 8-K/comment-letter
    manifests the same way) alongside the 10-K manifest -- unlike 10-Q,
    a deliberately separate INSTRUMENT never pooled with the 10-K panel
    (the project's data-scope decision), proxy/8-K aren't competing
    analytical panels, just more form types feeding the same
    paragraph-level AI-disclosure pipeline. Sharing one lookup means a
    proxy paragraph's accession_number resolves ticker/filing_date the
    same way a 10-K's already does -- `form_type`/`form` still
    distinguishes them for anything that needs to."""
    selects = []
    for country, cfg in countries:
        manifests_dir = dirs(cfg)[0]
        base_path = f"{manifests_dir}/filing_manifest.parquet"
        if not _existing(base_path):
            continue
        parts = [f"SELECT '{country}' AS country_code, * FROM read_parquet('{base_path}')"]
        proxy_path = f"{manifests_dir}/filing_manifest_proxy.parquet"
        if _existing(proxy_path):
            parts.append(f"SELECT '{country}' AS country_code, * FROM read_parquet('{proxy_path}')")
        eightk_path = f"{manifests_dir}/filing_manifest_8k.parquet"
        if _existing(eightk_path):
            parts.append(f"SELECT '{country}' AS country_code, * FROM read_parquet('{eightk_path}')")
        selects.append("\n            UNION ALL BY NAME\n            ".join(parts))
    return selects


# Every raw LINE of section_text (see _paragraph_select_sql) is classified
# into one of three content types:
# - 'table': contains "|" (a markdown table cell/row) or is a bare
#            "| --- |"-style separator row with the pipes stripped —
#            financial-statement tables are the overwhelming source
#            (checked on the real corpus: 38.5% of ALL raw lines had a
#            "|" in them at all — routine, not an edge case) — PROVIDED
#            it's actually adjacent to real "|" content (see
#            `_ISOLATED_DASH_ROW_IS_NOISE` below for why that qualifier
#            matters).
# - 'list':  starts with "•" (the dominant bullet marker seen — 221,618
#            occurrences in one corpus check) or "- "/"* ".
# - 'prose': everything else.
# `has_pipe`/`prev_has_pipe`/`next_has_pipe` are computed in their own
# CTE step before this CASE runs — a pure-dash line's classification
# needs its NEIGHBORS' pipe status, and DuckDB doesn't allow nesting one
# window function (LAG/LEAD) inside another (a CASE expression used
# directly inside a window PARTITION/computation).
_LINE_TYPE_CASE = r"""
    CASE
        WHEN has_pipe THEN 'table'
        WHEN trim(line_text, '- ') = '' AND (prev_has_pipe OR next_has_pipe) THEN 'table'
        WHEN line_text LIKE '•%' OR regexp_matches(line_text, '^[-*]\s') THEN 'list'
        ELSE 'prose'
    END
"""
# A bare "---" divider NOT next to any real "|" table content is a
# markdown horizontal rule / PAGE-BREAK marker — not a table. Checked on
# the real corpus: 211,817 of 473,200 'table' paragraphs (44.7%!) were
# nothing but this isolated marker, before this fix. Worse than noise:
# because prose paragraphs never merge with EACH OTHER (see
# `_paragraph_select_sql`'s docstring), a page-break marker sandwiched
# between two prose lines that are really one continuous sentence split
# it in two — confirmed on a real filing ("...We are unable to predict
# the" / "---" / "full impact of..."). Dropping isolated dash rows here
# (same treatment as a page-number line) doesn't reunite that sentence
# (prose still doesn't re-merge across paragraph boundaries — a separate,
# bigger change), but it does stop mislabeling near-half of "tables" as
# tables when they're just page furniture.
_ISOLATED_DASH_ROW_IS_NOISE = r"""
    trim(line_text, '- ') = '' AND NOT prev_has_pipe AND NOT next_has_pipe
"""
# Inline markdown image refs ("![img188831189_0.jpg](img188831189_0.jpg)")
# — their "alt text" is invariably just the image filename repeated
# (confirmed on the real corpus: 4,928 lines that were ONLY an image ref,
# plus 561 more with real prose text alongside one), never a caption or
# anything informative. Stripped to '' (not kept like a real link's
# visible text) at the earliest possible point — inside `raw_lines`
# itself, before pipe/TOC/classification logic ever sees the line — so a
# pure-image line collapses to blank and is dropped by `_DROP_LINE_WHERE`
# same as a page-number line, and a mixed image+prose line keeps only the
# prose.
_STRIP_MD_IMAGES_RE = r"!\[[^\]]*\]\([^)]*\)"
# Regular markdown links ([visible text](#anchor)) are reduced to their
# visible text — same place, same pass, right after the image strip —
# because whole LINES that are entirely one link are common and genuinely
# mixed: some are pure TOC/cross-reference noise ("[Index](#a7195)",
# "[Item 1A. Risk Factors](#anchor)"), but others are real content that
# just happens to double as a jump-link — e.g. a "Risk Factor Summary"
# section where each bullet's visible text IS the summary sentence,
# hyperlinked to its full discussion later in the document (confirmed on
# the real corpus: "[Our growth strategy depends, in part, on our
# ability to make acquisitions...](#anchor)"). Dropping such lines
# outright would silently lose that real content; stripping to visible
# text keeps it and only degrades pure-noise lines to short heading-like
# fragments (still harmless — `_DROP_LINE_WHERE`/`_TOC_NAV_LINK_RE` catch
# the "Table of Contents"/"Index to Financial Statements" ones of those).
_STRIP_MD_LINKS_RE = r"\[([^\]]*)\]\([^)]*\)"
# Some filers chain multiple nav links on one line — "Table of Contents"
# immediately followed by "Index to Financial Statements" (990
# occurrences on the real corpus) — so this matches either phrase, any
# combination, in either order, and nothing else. `[\s\xa0]`, not bare
# `\s`: these lines are routinely padded with actual U+00A0 non-breaking-
# space characters between the two link phrases, and RE2's `\s` — unlike
# Python's — does NOT match \xa0 (confirmed directly); the plain `\s*`
# version of this regex silently matched 0 of the 990 real "Table of
# Contents<NBSPs>Index to Financial Statements" lines it was written for.
_WS = r"[\s\xa0]"
_TOC_NAV_LINK_RE = (
    rf"(?i)^{_WS}*"
    rf"(Tables?({_WS}*of{_WS}*Contents?)?[.`]?{_WS}*)?"
    rf"(Index{_WS}+to{_WS}+Financial{_WS}+Statements?[.`]?{_WS}*)?"
    r"$"
)
_DROP_LINE_WHERE = rf"""
    trim(line_text) <> ''
    AND NOT regexp_matches(line_text, '^[0-9]+$')
    AND NOT regexp_matches(trim(line_text, '` '), '{_TOC_NAV_LINK_RE}')
"""
# ORDER matters between this and _ISOLATED_DASH_ROW_IS_NOISE: dropping
# page-number/TOC-nav lines happens FIRST (see `pre_pipe_context` in
# _paragraph_select_sql), and pipe-adjacency for the isolated-dash check
# is computed on what's LEFT after that — so a "---" divider sitting
# right next to a page-number line (dropped) still correctly sees past
# it to whatever real content line is next.


def _paragraph_select_sql(form: str, source_view: str) -> str:
    """One row per PARAGRAPH — a run of consecutive same-content-type raw
    lines of section_text, merged together (see `_LINE_TYPE_CASE`): a
    table's rows merge into ONE paragraph (the whole table, still marked
    `content_type='table'` so it's never mistaken for prose), a bulleted
    list's items merge into ONE paragraph the same way (`content_type=
    'list'`), and prose lines each stay their OWN paragraph (never merged
    with a neighboring prose line — clean_html_to_lines already delimited
    real paragraph/heading breaks one per line; merging those would lose
    that boundary). Earlier version DROPPED table/list lines outright as
    "noise" — wrong: ~48% of raw lines were table/list content, and both
    are real, meaningful content, just not prose sentences (see
    `sentences`' per-content_type branching for how each gets handled at
    that finer granularity).

    `paragraph_index` is the 1-based position of the group's FIRST line in
    the original (pre-grouping) line sequence — not densely renumbered,
    so it stays a stable pointer even where lines were merged away or
    dropped. No byte-offset tracking into `section_text` (char_start/
    char_len) — tried that, it added real complexity (a whole extra
    window-function pass, plus a documented edge case where a dropped
    line sandwiched inside a merged group broke exact substr
    reconstruction) for lineage nobody asked for; `paragraph_index` +
    (accession_number, item_key) is enough to locate a paragraph without
    it. Deliberately NARROW on lineage generally: only the columns
    needed to identify the source (form, country_code, accession_number,
    item_key) are carried — ticker/filing_date/section_name/run_id/...
    live on `source_view` and JOIN back cleanly on (accession_number,
    item_key) when actually needed; duplicating them onto every one of
    millions of paragraph rows was real, wasted memory for no benefit
    over a join.

    `form` ("10-K"/"10-Q") is stamped onto every row as a literal — this
    is ONE building block meant to be UNION ALL BY NAME'd across every
    form (and, via _union, every country) into a single final
    `paragraphs` view (see main()). Paragraphs are the leaf level most
    downstream text-analysis queries actually read from; keeping that
    level split by form/country the way filing_manifest legitimately is
    (different instruments, different panel semantics) would just mean
    reimplementing the same "read text, don't care where it's from"
    query against N near-identical views forever. `form`/`country_code`
    is how a query re-imposes that split if it needs to — a WHERE clause,
    not a different table.

    Grouping is the classic "gaps and islands" pattern: a new group
    starts whenever a line's content_type differs from the previous
    line's (LAG, computed in its own CTE step — DuckDB doesn't allow
    nesting one window function inside another), OR whenever the line is
    'prose' (which ALWAYS starts a new group, even following another
    prose line)."""
    return f"""
        WITH raw_lines AS (
            SELECT '{form}' AS form, country_code, accession_number, item_key,
                   trim(regexp_replace(
                       regexp_replace(
                           unnest(string_split(section_text, chr(10))),
                           '{_STRIP_MD_IMAGES_RE}', '', 'g'
                       ),
                       '{_STRIP_MD_LINKS_RE}', '\1', 'g'
                   )) AS line_text,
                   generate_subscripts(string_split(section_text, chr(10)), 1) AS line_index
            FROM {source_view}
        ),
        pre_pipe_context AS (
            -- Page-number/TOC-nav lines dropped FIRST — see
            -- _DROP_LINE_WHERE's own comment on why this ordering matters
            -- for the isolated-dash check that follows.
            SELECT *, line_text LIKE '%|%' AS has_pipe
            FROM raw_lines
            WHERE {_DROP_LINE_WHERE}
        ),
        with_pipe_context AS (
            SELECT *,
                LAG(has_pipe) OVER (
                    PARTITION BY form, accession_number, item_key ORDER BY line_index
                ) AS prev_has_pipe,
                LEAD(has_pipe) OVER (
                    PARTITION BY form, accession_number, item_key ORDER BY line_index
                ) AS next_has_pipe
            FROM pre_pipe_context
        ),
        classified AS (
            SELECT * EXCLUDE (prev_has_pipe, next_has_pipe),
                COALESCE(prev_has_pipe, false) AS prev_has_pipe,
                COALESCE(next_has_pipe, false) AS next_has_pipe,
                {_LINE_TYPE_CASE} AS line_type
            FROM with_pipe_context
        ),
        not_page_break_noise AS (
            SELECT * FROM classified
            WHERE NOT ({_ISOLATED_DASH_ROW_IS_NOISE})
        ),
        with_prev AS (
            SELECT *,
                LAG(line_type) OVER (
                    PARTITION BY form, accession_number, item_key ORDER BY line_index
                ) AS prev_line_type
            FROM not_page_break_noise
        ),
        grouped AS (
            SELECT *,
                SUM(CASE
                    WHEN line_type = 'prose' THEN 1
                    WHEN prev_line_type IS NULL OR line_type != prev_line_type THEN 1
                    ELSE 0
                END) OVER (
                    PARTITION BY form, accession_number, item_key ORDER BY line_index
                    ROWS UNBOUNDED PRECEDING
                ) AS group_id
            FROM with_prev
        )
        SELECT
            form, country_code, accession_number, item_key,
            line_type AS content_type,
            MIN(line_index) AS paragraph_index,
            string_agg(line_text, chr(10) ORDER BY line_index) AS paragraph_text,
            text_hash8(string_agg(line_text, chr(10) ORDER BY line_index)) AS text_hash,
            -- `is_scorable`: descarta lo que la extracción deja como párrafo
            -- pero no tiene contenido. Medido sobre el corpus: 235.935 filas
            -- (7,2%) tienen 3 caracteres o menos — viñetas sueltas, espacios de
            -- ancho cero, guiones. Un `•` repetido 23.497 veces es el texto más
            -- "frecuente" del corpus.
            --
            -- El criterio es tener al menos un carácter alfanumérico Y más de 3
            -- caracteres útiles. NO se filtra por largo mayor: encabezados como
            -- "Risks Related to Artificial Intelligence" tienen 40 caracteres y
            -- son señal legítima.
            --
            -- No se borran las filas: se marcan. El párrafo sigue existiendo
            -- para reconstruir la sección, y `paragraph_index` mantiene su
            -- correspondencia con el texto original — filtrarlas acá
            -- renumeraría todo y rompería las llaves ya escritas en los
            -- embeddings, los scores y el golden set.
            length(trim(string_agg(line_text, chr(10) ORDER BY line_index))) > 3
                AND regexp_matches(string_agg(line_text, chr(10) ORDER BY line_index),
                                   '[A-Za-z0-9]') AS is_scorable
        FROM grouped
        GROUP BY form, country_code, accession_number, item_key, group_id, line_type
    """


CL_PARAGRAPHS_GLOB = str(
    REPO_ROOT / "data" / "interim" / "sections_cl" / "filing_paragraphs__run=*__part=*.parquet")


def _cl_paragraph_select_sql(source_glob: str) -> str:
    """Chile's PDF pipeline (scripts/cl/cmf_pdf_paragraphs.py) extracts
    PARAGRAPHS directly from Memoria Anual / Análisis Razonado PDFs — there
    is no raw "section text" intermediate the way US 10-K/10-Q HTML has, so
    this does NOT go through `_paragraph_select_sql`'s line-merging (gaps-
    and-islands) logic at all; it's already paragraph-grained on disk.

    Column mapping to the shared `paragraphs` contract (this is what was
    MISSING before: this function didn't exist, so `filing_paragraphs_cl`
    just sat on disk, unUNIONed, and every downstream table — paragraphs,
    unique_paragraphs, the prefilter, the golden set, ai_classify — was
    silently US-only despite 1,077,595 Chilean paragraphs already being
    ready):
      - `country_code` = 'cl' (literal — this glob only ever holds CL data)
      - `form` = `filing_type` ('annual'/'quarterly') — CL's own vocabulary,
        deliberately NOT forced into '10-K'/'10-Q': those are SEC forms,
        Chile doesn't file them. `annual` is CL's 10-K-equivalent panel
        core, `quarterly` its 10-Q-equivalent shock series (see the
        project's two-instruments data-scope decision) — same ROLE,
        different label, on purpose.
      - `accession_number` = `document_id` — verified unique per filing
        (1,948 distinct values, and (document_id, paragraph_index) is
        already globally unique with zero extra work).
      - `item_key` = constant '0' — Memorias/Análisis Razonado have no
        SEC-style Item 1/1A/7 structure to preserve; a constant is honest
        about that rather than fabricating false section semantics. Every
        uniqueness/grouping property the rest of the pipeline relies on
        (PARAGRAPH_KEY, GroupKFold by accession_number) still holds because
        (document_id, paragraph_index) alone is already unique.
      - `is_scorable`: same length/alnum rule as the US branch, computed
        here since CL's own extractor doesn't emit it.
    """
    return f"""
        SELECT
            'cl' AS country_code,
            filing_type AS form,
            document_id AS accession_number,
            '0' AS item_key,
            content_type,
            paragraph_index,
            paragraph_text,
            text_hash8(paragraph_text) AS text_hash,
            length(trim(paragraph_text)) > 3
                AND regexp_matches(paragraph_text, '[A-Za-z0-9]') AS is_scorable
        FROM read_parquet('{source_glob}')
    """


# Cheap, "decent" (not linguistically perfect) sentence-boundary heuristic:
# split after a run of .!? that's followed by whitespace and then a
# CAPITAL letter — the capital-letter gate is what keeps "the U.S. and
# Japan" together (lowercase "and" after the period) while still splitting
# "...in 1973. It operates..." (capital "It" after the period). DuckDB's
# regex engine (RE2) doesn't support lookaround, hence the two-step
# replace-then-split rather than a single lookahead-based split regex.
#
# On its own this still wrongly split "Mr. Smith agreed." at the capital
# "S" — a real, common failure mode, not a hypothetical one. Fixed with a
# MASK/split/UNMASK pass: common abbreviations get their trailing period
# swapped for a placeholder BEFORE the boundary regex ever sees it (so it
# can't be mistaken for a sentence end), then the placeholder is restored
# to "." after splitting. Not exhaustive (a full abbreviation dictionary
# doesn't exist), but covers the common cases (titles, "U.S.", "e.g.",
# "Inc.", months, ...) that actually show up in filings.
_ABBREV_WORDS = (
    "Mr|Mrs|Ms|Dr|Jr|Sr|Prof|Rev|Gen|Sen|Rep|Gov|Lt|Col|Capt|Cmdr|Sgt|"
    "vs|etc|al|Inc|Corp|Co|Ltd|LLC|No|St|Ave|Blvd|Fig|Vol|pp|cf|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    "Mon|Tue|Wed|Thu|Fri|Sat|Sun"
)
_ABBREV_WORD_RE = rf"\b({_ABBREV_WORDS})\."
_ABBREV_COMPOUND_RE = r"\b(U\.S|U\.K|i\.e|e\.g|a\.m|p\.m)\."
_ABBREV_MASK = chr(30)  # ASCII record separator — stands in for a masked "."
_SENTENCE_BOUNDARY_RE = r"([.!?]+)\s+([A-Z])"
_SENTENCE_BOUNDARY_MARKER = chr(31)  # ASCII unit separator — won't collide with real filing text


def _prose_sentence_select_sql() -> str:
    """One row per SENTENCE within a `content_type='prose'` paragraph (see
    the abbreviation-mask and `_SENTENCE_BOUNDARY_RE` comments above for
    the split heuristic).

    The mask/boundary-mark/split regex chain runs ONCE per paragraph, into
    a single `sentence_parts` LIST column, which `unnest` and
    `generate_subscripts` then both read — not once each (computing the
    same 3-regex chain twice per row, as an earlier version did, roughly
    doubled the cost of an already regex-heavy view over 3M+ paragraphs)."""
    return f"""
        WITH prose_paragraphs AS (
            SELECT * FROM paragraphs WHERE content_type = 'prose'
        ),
        masked AS (
            SELECT
                form, country_code, accession_number, item_key, content_type, paragraph_index,
                regexp_replace(
                    regexp_replace(paragraph_text, '{_ABBREV_WORD_RE}', '\\1{_ABBREV_MASK}', 'gi'),
                    '{_ABBREV_COMPOUND_RE}', '\\1{_ABBREV_MASK}', 'gi'
                ) AS masked_text
            FROM prose_paragraphs
        ),
        split_parts AS (
            SELECT
                * EXCLUDE (masked_text),
                string_split(
                    regexp_replace(masked_text, '{_SENTENCE_BOUNDARY_RE}', '\\1{_SENTENCE_BOUNDARY_MARKER}\\2', 'g'),
                    '{_SENTENCE_BOUNDARY_MARKER}'
                ) AS sentence_parts
            FROM masked
        ),
        split AS (
            SELECT
                * EXCLUDE (sentence_parts),
                replace(unnest(sentence_parts), '{_ABBREV_MASK}', '.') AS sentence_text,
                generate_subscripts(sentence_parts, 1) AS sentence_index
            FROM split_parts
        )
        SELECT
            form, country_code, accession_number, item_key, content_type, paragraph_index,
            sentence_index, sentence_text
        FROM split
        WHERE trim(sentence_text) <> ''
    """


def _list_sentence_select_sql() -> str:
    """One row per ITEM within a `content_type='list'` paragraph — split
    back on the SAME chr(10) that `_paragraph_select_sql` used to merge
    the list's original lines together, so each "sentence" here is
    exactly one bullet/list item, not a regex-guessed boundary. No
    abbreviation masking needed (list items don't get sentence-boundary-
    split at all)."""
    return """
        WITH list_paragraphs AS (
            SELECT * FROM paragraphs WHERE content_type = 'list'
        )
        SELECT
            form, country_code, accession_number, item_key, content_type, paragraph_index,
            generate_subscripts(string_split(paragraph_text, chr(10)), 1) AS sentence_index,
            unnest(string_split(paragraph_text, chr(10))) AS sentence_text
        FROM list_paragraphs
    """


def _table_sentence_select_sql() -> str:
    """A `content_type='table'` paragraph is NOT sentence-structured —
    stays whole, as its own single "sentence" (sentence_index=1,
    sentence_text=paragraph_text), so every paragraph — table included —
    has at least one row in `sentences` and a query that just wants "all
    the text, sentence-grain or not" doesn't need a UNION with
    `paragraphs` to avoid silently dropping tables."""
    return """
        SELECT
            form, country_code, accession_number, item_key, content_type, paragraph_index,
            1 AS sentence_index, paragraph_text AS sentence_text
        FROM paragraphs
        WHERE content_type = 'table'
    """


def _sentence_select_sql() -> str:
    """`sentences` = one row per PROSE sentence + one row per LIST item +
    one row per TABLE (whole) — see the three _*_sentence_select_sql
    functions above for why each content_type needs different handling.
    Built directly on `paragraphs` (not re-derived from raw section text)
    — paragraph_index (and, through it, content_type) comes along as
    lineage, joinable back on (form, accession_number, item_key,
    paragraph_index) rather than duplicated here."""
    return (
        f"({_prose_sentence_select_sql()})"
        + "\n        UNION ALL BY NAME\n"
        + f"({_list_sentence_select_sql()})"
        + "\n        UNION ALL BY NAME\n"
        + f"({_table_sentence_select_sql()})"
    )


def main(with_text_tables: bool = False):
    countries = _country_configs()
    if not countries:
        raise SystemExit(f"No configs/<country>/config.yaml found under {REPO_ROOT / 'configs'}")

    market_prices_dir = REPO_ROOT / "data" / "raw" / "market" / "prices"
    market_factors_dir = REPO_ROOT / "data" / "raw" / "market" / "factors"

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.create_function("text_hash8", _text_hash8, ["VARCHAR"], "UBIGINT")
    # `paragraphs`' window functions (LAG/LEAD per accession_number) OOM'd
    # (2026-09-04) once DEF 14A -- large, table-heavy documents -- joined
    # 10-K/10-Q in the UNION: default thread count multiplies the working
    # set per-partition across cores faster than a single machine's RAM
    # grows. `preserve_insertion_order=false` lets DuckDB spill/stream
    # instead of buffering the whole ordered result, and fewer threads
    # means fewer copies of that working set alive at once -- both cheap
    # to try before reaching for a bigger memory_limit.
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=2")
    # (2026-09-05) 8-K joining the same UNION pushed total row volume past
    # what 4 threads' working sets fit in this machine's 16GB even with
    # insertion-order preservation off -- dropping to 2 threads plus an
    # explicit temp_directory (letting DuckDB spill intermediates to disk
    # instead of OOMing) fixed it without needing a bigger machine.
    con.execute(f"SET temp_directory='{REPO_ROOT / 'duckdb' / '.tmp_spill'}'")
    con.execute("SET memory_limit='10GB'")

    def dirs(config):
        return (
            REPO_ROOT / config["storage"]["interim_manifests"],
            REPO_ROOT / config["storage"]["interim_sections"],
        )

    views = {
        # --- firm universe + filing manifest ---
        "firm_universe": _union([
            f"SELECT '{country}' AS country_code, * FROM read_parquet('{dirs(cfg)[0]}/firm_universe.parquet')"
            for country, cfg in countries
            if _existing(f"{dirs(cfg)[0]}/firm_universe.parquet")
        ]),
        # UNIONs in filing_manifest_proxy.parquet (and, later, 8-K/comment-
        # letter manifests) alongside the 10-K manifest -- unlike 10-Q,
        # which is a deliberately separate INSTRUMENT never pooled with the
        # 10-K panel (see the project's data-scope decision), proxy/8-K are
        # not competing analytical panels, just more form types feeding the
        # same paragraph-level AI-disclosure pipeline. Sharing one lookup
        # means a proxy paragraph's accession_number resolves ticker/
        # filing_date the same way a 10-K's already does -- `form_type`/
        # `form` still distinguishes them for anything that needs to.
        "filing_manifest": _union(_filing_manifest_selects(countries, dirs)),
        # 10-Q shock series — a SEPARATE instrument, never pooled with
        # filing_manifest above (see the project's data-scope decision) —
        # hence its own manifest AND its own extraction_trace/
        # filing_sections views below, never a UNION with the 10-K ones.
        # Not every country has one at all (Chile doesn't — see
        # docs/international_expansion_plan.md) — `_existing()` handles
        # both "not built yet" and "doesn't exist for this country".
        "filing_manifest_10q": _union([
            f"SELECT '{country}' AS country_code, * FROM read_parquet('{dirs(cfg)[0]}/filing_manifest_10q.parquet')"
            for country, cfg in countries
            if _existing(f"{dirs(cfg)[0]}/filing_manifest_10q.parquet")
        ]),
        # Full trace: one row per (filing, target item), found or not.
        # See scripts/us/section_segmenter.py:load_extraction_trace — this
        # view is that same contract in SQL. union_by_name=True inside
        # read_parquet matters here (distinct from the cross-COUNTRY
        # UNION ALL BY NAME above): a part file whose "note" column is
        # 100% NULL (every row in that checkpoint's batch was
        # `found=True`) gets that column typed NULL rather than VARCHAR by
        # Parquet's own type inference — without union_by_name,
        # read_parquet's glob takes its schema from just the FIRST file it
        # opens, so a query touching "note" on a glob spanning many runs
        # breaks the moment file ordering picks one of those all-NULL
        # files as the reference schema (hit exactly this: ConversionException
        # reading "note" as NULL after a rerun added a new run_id's parts).
        "extraction_trace": _union([
            f"""
            SELECT '{country}' AS country_code, *
            FROM read_parquet('{dirs(cfg)[1]}/filing_sections__run=*__part=*.parquet', union_by_name=True)
            QUALIFY row_number() OVER (PARTITION BY accession_number, item_key ORDER BY run_date DESC) = 1
            """
            for country, cfg in countries
            if _existing(f"{dirs(cfg)[1]}/filing_sections__run=*__part=*.parquet")
        ]),
        # Just the found sections — the text itself.
        "filing_sections": "SELECT * FROM extraction_trace WHERE found",
        # 10-Q equivalent of the two views above — Item 2 (MD&A) + Item 1A
        # (Part II risk-factor updates), see scripts/us/10q/
        # 02_extract_sections.py for why.
        "extraction_trace_10q": _union([
            f"""
            SELECT '{country}' AS country_code, *
            FROM read_parquet('{dirs(cfg)[1]}/filing_sections_10q__run=*__part=*.parquet', union_by_name=True)
            QUALIFY row_number() OVER (PARTITION BY accession_number, item_key ORDER BY run_date DESC) = 1
            """
            for country, cfg in countries
            if _existing(f"{dirs(cfg)[1]}/filing_sections_10q__run=*__part=*.parquet")
        ]),
        "filing_sections_10q": "SELECT * FROM extraction_trace_10q WHERE found",
        # DEF 14A — the whole document as one "section" (item_key='0', see
        # scripts/us/proxy/proxy_segmenter.py for why there's no real
        # per-heading segmenter yet). Not a separate instrument the way
        # 10-Q is -- just another form type, same paragraph pipeline.
        "extraction_trace_proxy": _union([
            f"""
            SELECT '{country}' AS country_code, *
            FROM read_parquet('{dirs(cfg)[1]}/filing_sections_proxy__run=*__part=*.parquet', union_by_name=True)
            QUALIFY row_number() OVER (PARTITION BY accession_number, item_key ORDER BY run_date DESC) = 1
            """
            for country, cfg in countries
            if _existing(f"{dirs(cfg)[1]}/filing_sections_proxy__run=*__part=*.parquet")
        ]),
        "filing_sections_proxy": "SELECT * FROM extraction_trace_proxy WHERE found",
        # 8-K — same "whole document as one section" choice, see
        # scripts/us/8k/segmenter_8k.py. Also not a separate instrument,
        # just another form type sharing the filing_manifest lookup.
        "extraction_trace_8k": _union([
            f"""
            SELECT '{country}' AS country_code, *
            FROM read_parquet('{dirs(cfg)[1]}/filing_sections_8k__run=*__part=*.parquet', union_by_name=True)
            QUALIFY row_number() OVER (PARTITION BY accession_number, item_key ORDER BY run_date DESC) = 1
            """
            for country, cfg in countries
            if _existing(f"{dirs(cfg)[1]}/filing_sections_8k__run=*__part=*.parquet")
        ]),
        "filing_sections_8k": "SELECT * FROM extraction_trace_8k WHERE found",
        # --- 03_market_data: prices + Fama-French factors ---
        "market_prices": f"""
            SELECT * FROM read_parquet('{market_prices_dir}/*.parquet', filename = true)
        """,
        "market_factors_daily": f"""
            SELECT * FROM read_parquet('{market_factors_dir}/ff3_daily.parquet')
        """,
        "market_factors_monthly": f"""
            SELECT * FROM read_parquet('{market_factors_dir}/ff3_monthly.parquet')
        """,
    }

    if with_text_tables:
        # Two leaf-level text granularities, both ACROSS EVERY FORM AND
        # COUNTRY — not split by source (see _paragraph_select_sql's
        # docstring for why). Full lineage back to the parent section/
        # filing/ticker survives as metadata columns (form, country_code,
        # accession_number, item_key, run_id, ...), not as separate
        # tables. Plain `UNION ALL BY NAME` here (not the per-country
        # `_union` helper): this is unioning FORMS, already-per-country-
        # unioned upstream. MATERIALIZED (see MATERIALIZED_TABLES) — the
        # regex-heavy sentence split in particular is too expensive to
        # re-run as a live VIEW on every query. OPTIONAL (only built when
        # `with_text_tables` is set, e.g. `--with-text-tables` /
        # `make duckdb-text`): took ~52s standalone, real cost on a real
        # machine, not worth paying on every `make duckdb` when most
        # rebuilds just need the fast views refreshed.
        # (2026-09-05) One giant UNION ALL of every form's window-function-
        # heavy branch, planned and executed as a SINGLE CREATE TABLE
        # statement, OOM'd once 8-K (34k more documents) joined 10-K/10-Q/
        # DEF14A: DuckDB's planner kept every branch's line-level CTE chain
        # (unnest of every raw line, LAG/LEAD per accession_number) alive
        # at once rather than freeing one branch's working set before
        # starting the next. Materializing each form's branch into its own
        # physical staging table FIRST (sequential con.execute calls, each
        # one's memory released once its CREATE TABLE finishes) and THEN
        # doing a cheap UNION ALL BY NAME over the now-small staging tables
        # fixed it -- same total rows, same window-function semantics
        # (still partitioned by accession_number, still per-form), just
        # not all resident in memory simultaneously.
        paragraph_stage_names = []
        paragraph_stage_specs = [
            ("_paragraphs_stage_10k", _paragraph_select_sql("10-K", "filing_sections")),
            ("_paragraphs_stage_10q", _paragraph_select_sql("10-Q", "filing_sections_10q")),
        ]
        if "filing_sections_proxy" in views:
            paragraph_stage_specs.append(
                ("_paragraphs_stage_proxy", _paragraph_select_sql("DEF 14A", "filing_sections_proxy"))
            )
        else:
            print("  skipping DEF 14A paragraphs (no filing_sections_proxy files yet)")
        if "filing_sections_8k" in views:
            paragraph_stage_specs.append(
                ("_paragraphs_stage_8k", _paragraph_select_sql("8-K", "filing_sections_8k"))
            )
        else:
            print("  skipping 8-K paragraphs (no filing_sections_8k files yet)")
        if _existing(CL_PARAGRAPHS_GLOB):
            paragraph_stage_specs.append(
                ("_paragraphs_stage_cl", _cl_paragraph_select_sql(CL_PARAGRAPHS_GLOB))
            )
        else:
            print(f"  skipping CL paragraphs (no files matching {CL_PARAGRAPHS_GLOB})")

        # The staging tables' own dependency views (filing_sections_proxy,
        # filing_sections_8k, ...) must already exist in the DB before
        # these CREATE TABLE statements run -- flush every view/table
        # queued in `views` so far (all the fast views built above) before
        # materializing paragraph branches against them.
        for name, query in views.items():
            for drop_kind in ("VIEW", "TABLE"):
                try:
                    con.execute(f"DROP {drop_kind} IF EXISTS {name}")
                except duckdb.CatalogException:
                    pass
            con.execute(f"CREATE OR REPLACE VIEW {name} AS {query}")
            print(f"  view {name} OK")
        views.clear()

        for stage_name, stage_sql in paragraph_stage_specs:
            con.execute(f"DROP TABLE IF EXISTS {stage_name}")
            con.execute(f"CREATE TABLE {stage_name} AS {stage_sql}")
            print(f"  table {stage_name} OK")
            paragraph_stage_names.append(stage_name)

        views["paragraphs"] = "\n            UNION ALL BY NAME\n".join(
            f"SELECT * FROM {name}" for name in paragraph_stage_names
        )
        # Built ON `paragraphs` (not re-derived from raw section text) —
        # see _sentence_select_sql's docstring.
        views["sentences"] = _sentence_select_sql()
        # THE canonical dedup surface (docs/prefilter_evaluation.md §8.7):
        # ~50% of `paragraphs` is literal boilerplate repeated across
        # filings. One row per unique `text_hash`, naming a single
        # deterministic representative instance (smallest natural key) plus
        # `duplicate_count` (how many paragraph instances share this text).
        # Any downstream step whose cost scales with corpus size (LLM
        # classification chief among them, but conceptually also embedding/
        # scoring) should compute over THIS table, then broadcast back to
        # instances via `text_hash` if it needs per-instance output — not
        # reimplement its own group-by-text dedup (that's exactly how
        # ai_classify.py's dedup broke: a second, independently-computed
        # hash of a slightly different text reconstruction).
        views["unique_paragraphs"] = """
            WITH ranked AS (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY text_hash
                        ORDER BY country_code, form, accession_number, item_key, paragraph_index
                    ) AS rn,
                    count(*) OVER (PARTITION BY text_hash) AS duplicate_count
                FROM paragraphs
            )
            SELECT country_code, form, accession_number, item_key, paragraph_index,
                   text_hash, paragraph_text, content_type, is_scorable, duplicate_count
            FROM ranked WHERE rn = 1
        """
    else:
        print("  skipping paragraphs/sentences (pass --with-text-tables to build them)")

    # Terminal retrieval output: one row per scored paragraph, with lexical
    # matches and semantic similarities.  Scores are append-only run files;
    # no threshold/candidate decision belongs in this acquisition-stage view.
    score_glob = f"{PREFILTER_SCORES_DIR}/prefilter_scores__run=*.parquet"
    if _existing(score_glob):
        # Only ONE (model, anchors, dtype) population at a time. Score parts are
        # append-only and are deliberately never deleted, so after retuning the
        # anchors or switching precision the directory holds several complete
        # populations of the SAME paragraphs — a plain union over the glob would
        # silently duplicate every key and mix score scales. The newest run's
        # configuration wins; older parts stay on disk for comparison and are
        # reachable by reading the parquet directly.
        views["ai_prefilter_scores"] = f"""
            WITH all_scores AS (
                SELECT * FROM read_parquet('{score_glob}', union_by_name=True)
            ), current AS (
                SELECT model, anchors_fingerprint, dtype
                FROM all_scores ORDER BY run_id DESC LIMIT 1
            )
            SELECT a.* FROM all_scores a JOIN current c
              ON a.model = c.model
             AND a.anchors_fingerprint = c.anchors_fingerprint
             AND a.dtype = c.dtype
        """
    else:
        print("  skipping ai_prefilter_scores (run scripts/common/ai_prefilter.py first)")

    # --- Gold tables: LLM frame extraction and lexical entity mentions,
    # broadcast from `unique_paragraphs`' one-row-per-text back out to every
    # real paragraph INSTANCE via `text_hash` (docs/prefilter_evaluation.md
    # §8.8/§8.12) — a downstream reader wants "which filings/paragraphs",
    # not "which distinct texts". Both are COUNTRY-AGNOSTIC by construction:
    # neither filters by country_code anywhere, they just broadcast whatever
    # is in `paragraphs` — Chile (or any later country) starts appearing the
    # moment its own paragraphs enter the underlying is_ai_prefiltered
    # population or get an entity-mention run, no view change needed here.
    frames_glob = "data/interim/ai_classify/ai_frames__session=*.parquet"
    if with_text_tables and _existing(frames_glob):
        views["gold_ai_frames"] = f"""
            WITH latest_frames AS (
                SELECT * FROM read_parquet('{frames_glob}', union_by_name=True)
                WHERE error IS NULL
                QUALIFY row_number() OVER (
                    PARTITION BY text_hash, frame_index ORDER BY session_id DESC
                ) = 1
            )
            SELECT p.country_code, p.form, p.accession_number, p.item_key, p.paragraph_index,
                   f.text_hash, up.duplicate_count, f.frame_index, f.has_frame,
                   f.subject, f.ai_type, f.temporal, f.domain, f.concepts,
                   f.specificity_business_process, f.specificity_product_or_system,
                   f.specificity_vendor_or_partner, f.specificity_quantified_metric,
                   f.specificity_date_or_timeline,
                   f.rhetoric_promotional, f.rhetoric_strategic_importance,
                   f.evidence_sentence_ids, f.sentence_indices,
                   f.judge_model, f.prompt_version, f.classified_at
            FROM paragraphs p
            JOIN latest_frames f ON f.text_hash = p.text_hash
            JOIN unique_paragraphs up ON up.text_hash = f.text_hash
        """
    elif with_text_tables:
        print(f"  skipping gold_ai_frames (no files matching {frames_glob} — "
              f"run scripts/common/ai_classify.py first)")

    entity_mentions_glob = "data/interim/ai_entity_mentions/ai_entity_mentions__run=*.parquet"
    if with_text_tables and _existing(entity_mentions_glob):
        views["gold_ai_entity_mentions"] = f"""
            WITH latest_mentions AS (
                SELECT * FROM read_parquet('{entity_mentions_glob}', union_by_name=True)
                QUALIFY row_number() OVER (
                    PARTITION BY text_hash, term ORDER BY run_id DESC
                ) = 1
            )
            SELECT p.country_code, p.form, p.accession_number, p.item_key, p.paragraph_index,
                   m.text_hash, up.duplicate_count, m.term, m.geo, m.modality, m.run_id
            FROM paragraphs p
            JOIN latest_mentions m ON m.text_hash = p.text_hash
            JOIN unique_paragraphs up ON up.text_hash = m.text_hash
        """
    elif with_text_tables:
        print(f"  skipping gold_ai_entity_mentions (no files matching {entity_mentions_glob} — "
              f"run scripts/common/ai_entity_mentions.py first)")

    for name, query in views.items():
        kind = "TABLE" if name in MATERIALIZED_TABLES else "VIEW"
        # DROP before CREATE OR REPLACE: a prior run may have created this
        # name as the OTHER kind (e.g. `make duckdb-text` made paragraphs
        # a TABLE; a later plain `make duckdb` needs to drop that TABLE
        # before anything, since it doesn't rebuild it at all this run) —
        # DuckDB refuses "CREATE OR REPLACE TABLE" over an existing VIEW
        # of the same name (and vice versa) as a type mismatch, AND
        # "DROP VIEW IF EXISTS" over an existing TABLE errors instead of
        # no-op'ing (IF EXISTS only covers "doesn't exist", not "exists as
        # the other kind") — so try both, ignoring whichever doesn't apply.
        for drop_kind in ("VIEW", "TABLE"):
            try:
                con.execute(f"DROP {drop_kind} IF EXISTS {name}")
            except duckdb.CatalogException:
                pass
        con.execute(f"CREATE OR REPLACE {kind} {name} AS {query}")
        print(f"  {kind.lower()} {name} OK")

    if with_text_tables:
        # Staging tables only existed to keep each branch's window-function
        # working set out of memory at the same time as the others (see
        # the paragraphs-materialization comment above) -- once the real
        # `paragraphs` table is built from them, they're dead weight.
        for stage_name in paragraph_stage_names:
            con.execute(f"DROP TABLE IF EXISTS {stage_name}")

    print(f"\nCountries: {', '.join(c for c, _ in countries)}")
    print(f"Wrote -> {DB_PATH}")
    print("Open with: duckdb duckdb/thesis.duckdb   (then .tables, or SELECT * FROM <view> LIMIT 5;)")
    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-text-tables", action="store_true",
        help="Also (re)build the paragraphs/sentences materialized tables (~52s, regex-heavy). "
             "Skipped by default — most reruns only need the fast views refreshed.",
    )
    args = parser.parse_args()
    main(with_text_tables=args.with_text_tables)
