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


def _paragraph_select_sql(form: str, source_view: str) -> str:
    """One row per LINE of `section_text` (clean_html_to_lines already
    collapsed real paragraph/bullet/heading breaks down to one non-empty
    markdown line each — see section_segmenter.py — so a line here IS a
    structural paragraph, not an arbitrary character chunk; it's routinely
    MULTI-sentence — see `sentences`, built on top of this, for that
    granularity). `paragraph_index` is the 1-based position within the
    section; `char_start`/`char_len` are offsets into the PARENT
    `section_text`, so `substr(section_text, char_start+1, char_len) =
    paragraph_text` always holds — full lineage back to the exact byte
    range in the section this paragraph came from, on top of every other
    column already on `source_view` (accession_number, ticker, item_key,
    section_name, run_id, run_date, ...) carried through unchanged,
    EXCEPT `char_len`/`word_count` — `source_view` (filing_sections)
    already has WHOLE-SECTION versions of those, which collided with (and
    got silently shadowed by) the per-paragraph ones this view defines;
    renamed to `section_char_len`/`section_word_count` so both
    granularities survive under distinct names.

    `form` ("10-K"/"10-Q") is stamped onto every row as a literal — this
    is ONE building block meant to be UNION ALL BY NAME'd across every
    form (and, via _union, every country) into a single final
    `paragraphs` view (see main()). Paragraphs are the leaf level most
    downstream text-analysis queries actually read from; keeping that
    level split by form/country the way filing_manifest legitimately is
    (different instruments, different panel semantics) would just mean
    reimplementing the same "read text, don't care where it's from"
    query against N near-identical views forever. `form` (and
    `country_code`, already on `source_view`) is how a query re-imposes
    that split if it needs to — a WHERE clause, not a different table.

    char_start is a running total via a window function (SUM ... ROWS
    UNBOUNDED PRECEDING), not a per-row correlated subquery re-summing a
    list slice — the latter is O(n^2) per section and made this view
    unusably slow (minutes, not seconds) on sections with thousands of
    lines. The window function is O(n)."""
    return f"""
        WITH split AS (
            SELECT '{form}' AS form, *,
                   unnest(string_split(section_text, chr(10))) AS paragraph_text,
                   generate_subscripts(string_split(section_text, chr(10)), 1) AS paragraph_index
            FROM {source_view}
        )
        SELECT
            * EXCLUDE (section_text, paragraph_text, paragraph_index, char_len, word_count),
            char_len AS section_char_len,
            word_count AS section_word_count,
            paragraph_index,
            paragraph_text,
            CAST(
                COALESCE(SUM(LENGTH(paragraph_text) + 1) OVER (
                    PARTITION BY form, accession_number, item_key
                    ORDER BY paragraph_index
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS BIGINT
            ) AS char_start,
            LENGTH(paragraph_text) AS char_len
        FROM split
        WHERE trim(paragraph_text) <> ''
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


def _sentence_select_sql() -> str:
    """One row per SENTENCE within a paragraph (see the abbreviation-mask
    and `_SENTENCE_BOUNDARY_RE` comments above for the split heuristic).
    Built directly on `paragraphs` (not on `filing_sections`/
    `filing_sections_10q` again) — sentences are a finer split of an
    already-defined paragraph, not a separate derivation from the raw
    section text, so paragraph_index/paragraph_char_start (the
    paragraph's own offset into section_text) come along as lineage
    exactly as computed there. `sentence_char_start`/`char_len` are
    offsets into the PARENT `paragraph_text` (so `substr(paragraph_text,
    sentence_char_start+1, char_len) = sentence_text` holds); adding the
    paragraph's own `char_start` gives the sentence's absolute offset into
    `section_text` too, without storing it redundantly.

    The mask/boundary-mark/split regex chain runs ONCE per paragraph, into
    a single `sentence_parts` LIST column, which `unnest` and
    `generate_subscripts` then both read — not once each (computing the
    same 3-regex chain twice per row, as an earlier version did, roughly
    doubled the cost of an already regex-heavy view over 3M+ paragraphs)."""
    return f"""
        WITH masked AS (
            SELECT
                * EXCLUDE (char_start, char_len),
                char_start AS paragraph_char_start,
                regexp_replace(
                    regexp_replace(paragraph_text, '{_ABBREV_WORD_RE}', '\\1{_ABBREV_MASK}', 'gi'),
                    '{_ABBREV_COMPOUND_RE}', '\\1{_ABBREV_MASK}', 'gi'
                ) AS masked_text
            FROM paragraphs
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
            * EXCLUDE (paragraph_text, sentence_text, sentence_index),
            paragraph_text,
            sentence_index,
            sentence_text,
            CAST(
                COALESCE(SUM(LENGTH(sentence_text) + 1) OVER (
                    PARTITION BY form, accession_number, item_key, paragraph_index
                    ORDER BY sentence_index
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS BIGINT
            ) AS sentence_char_start,
            LENGTH(sentence_text) AS char_len
        FROM split
        WHERE trim(sentence_text) <> ''
    """


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
