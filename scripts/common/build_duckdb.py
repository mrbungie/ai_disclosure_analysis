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

from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "duckdb" / "thesis.duckdb"

# Everything else here is a live VIEW (re-evaluated on every query, always
# current with whatever's on disk). paragraphs/sentences are the
# exception: TABLEs, computed once at build time and stored in the
# .duckdb file — the regex-heavy paragraph/sentence split (especially
# sentences' mask/boundary/split chain) re-run as a live view on every
# query was expensive enough to be genuinely disruptive on a real
# machine, not just "slow". Tradeoff: these two go stale after a new
# extraction run until build_duckdb.py is rerun (every other view here
# doesn't); worth it for queries against them to actually be fast.
MATERIALIZED_TABLES = {"paragraphs", "sentences"}


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
    if len(selects) == 1:
        return selects[0]
    return "\n            UNION ALL BY NAME\n".join(f"({s})" for s in selects)


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
            string_agg(line_text, chr(10) ORDER BY line_index) AS paragraph_text
        FROM grouped
        GROUP BY form, country_code, accession_number, item_key, group_id, line_type
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
        ]),
        "filing_manifest": _union([
            f"SELECT '{country}' AS country_code, * FROM read_parquet('{dirs(cfg)[0]}/filing_manifest.parquet')"
            for country, cfg in countries
        ]),
        # 10-Q shock series — a SEPARATE instrument, never pooled with
        # filing_manifest above (see the project's data-scope decision) —
        # hence its own manifest AND its own extraction_trace/
        # filing_sections views below, never a UNION with the 10-K ones.
        "filing_manifest_10q": _union([
            f"SELECT '{country}' AS country_code, * FROM read_parquet('{dirs(cfg)[0]}/filing_manifest_10q.parquet')"
            for country, cfg in countries
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
        ]),
        "filing_sections_10q": "SELECT * FROM extraction_trace_10q WHERE found",
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
        views["paragraphs"] = (
            f"({_paragraph_select_sql('10-K', 'filing_sections')})"
            + "\n            UNION ALL BY NAME\n"
            + f"({_paragraph_select_sql('10-Q', 'filing_sections_10q')})"
        )
        # Built ON `paragraphs` (not re-derived from raw section text) —
        # see _sentence_select_sql's docstring.
        views["sentences"] = _sentence_select_sql()
    else:
        print("  skipping paragraphs/sentences (pass --with-text-tables to build them)")

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
