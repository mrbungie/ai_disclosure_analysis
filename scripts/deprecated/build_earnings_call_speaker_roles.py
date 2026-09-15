"""Classify earnings-call speakers as company (executive/IR) vs external analyst.

Deterministic, no LLM. Reads
data/interim/sections/earnings_call_paragraphs__v=3__run=*__part=*.parquet;
writes data/interim/earnings_call_speaker_roles/speaker_roles.parquet.

Problem this fixes (measured on the v=3 corpus, 11,407 calls): the naive
per-call rule -- "speaks in `prepared` => executive; only in `qa` => analyst"
-- misses executives who are on the call ONLY to field Q&A and never take a
`prepared` turn (e.g. ServiceNow's Amit Zavery, President & Chief Product
Officer, present in all 7 sampled NOW calls, always in `qa` only). Joined
against `gold_ai_frames` (subject='firm', item_key='qa'), the naive rule
misattributes 17.5% of firm-subject Q&A frames to a non-firm speaker
(2,745 / 15,728).

The first fix tried here -- pool a speaker's `prepared`-turn history across
every call of the same ticker, so appearing in `prepared` once anywhere
makes them an executive everywhere -- uncovered a bigger problem: `section`
itself is wrong often enough to poison that pool. Progressive (PGR) is the
clearest case: PGR_2021Q2, PGR_2021Q4 and PGR_2026Q2 all mislabel the
analyst Elyse Greenspan's Q&A turns as `section='prepared'` (the segmenter's
prepared/Q&A boundary rule evidently fails on PGR's call format). Pooling
"ever in prepared" across quarters turned one bad segmentation into a
permanent, cross-quarter false executive -- and this was not a one-off:
190 distinct tickers have at least one speaker wrongly promoted this way
(2,268 speaker instances total), found by cross-checking against the
signal below.

Three deterministic signals, applied in order of confidence:

  1. Operator-announced analyst (highest confidence, and used to CLEAN
     signal 2, not just to label). Every operator transition into a new
     Q&A turn ("Our next question comes from Elyse Greenspan with KBW.")
     names the asker; state is carried forward with SQL window functions
     (`LAST_VALUE ... IGNORE NULLS`) so follow-up turns from the same
     analyst inherit the same announced name without a new operator line.
     A speaker whose name overlaps an announced name for a call is
     positively an outside analyst for that ticker, full stop -- this
     overrides `section` labels entirely and is what strips the Elyse
     Greenspan-style false positives out of the executive roster before
     it is used. Matches ~54% of Q&A turns (analysts who ask a single,
     un-introduced follow-up right after their own prior turn are the
     main miss, since only the operator line carries the name).

  2. Cross-quarter executive roster, MINUS anyone signal 1 ever confirmed
     as an outside analyst for that ticker. A speaker's `prepared`-turn
     history is pooled across every call of the same ticker; if the same
     person (after name normalization) ever opens `prepared` for this
     ticker and was never operator-confirmed as an analyst there, they are
     an executive on every call, including ones where they only speak in
     `qa`.

  3. SEC Form 4 officer-of-record roster (build_sec_officer_roster.py),
     matched by an order-invariant name key so "Zavery Amit" (the SEC's
     Last-First filing order) and "Amit Zavery" (the transcript's First-Last
     order) resolve to the same person. This is what finally closes the
     Amit Zavery case: he never speaks in `prepared` in any of ServiceNow's
     7 sampled calls and the operator never announces him by name (he isn't
     an analyst being introduced), so signals 1-2 have nothing to catch him
     with -- but his Form 4 filings carry `RPTOWNER_RELATIONSHIP="Officer"`,
     `RPTOWNER_TITLE="President, CPO and COO"` directly from SEC's own
     structured data, independent of anything in the call transcript.

  4. Title-keyword fallback, also excluded from signal 1's confirmed
     analysts. For a ticker where the executive never appears in
     `prepared` in the whole sample and has no SEC officer-roster match
     (e.g. too recently promoted to show up in the fetched quarters), an
     explicit corporate title in the raw speaker string (CEO, CFO, "Head of
     Investor Relations", ...) is deterministic evidence, matched with word
     boundaries so it doesn't fire on names that merely contain the letters
     ("Cook", "Spector", "Francfort" do NOT match "coo"/"cto"/"cfo").

Name normalization (strip trailing "- CFO" / "(CEO)" / affiliation suffixes,
collapse whitespace, lowercase) runs before every one of these comparisons,
since sources are inconsistent about whether the speaker string carries a
title or affiliation suffix at all.

`speaker` is call-scoped free text with no company/analyst flag anywhere
upstream, and `section` itself is not fully reliable, so this combination
is the ceiling of what's recoverable without sending transcript text to a
judge model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_GLOB = str(
    REPO_ROOT / "data" / "interim" / "sections" /
    "earnings_call_paragraphs__v=3__run=*__part=*.parquet")
OUT_DIR = REPO_ROOT / "data" / "interim" / "earnings_call_speaker_roles"
OUT_PATH = OUT_DIR / "speaker_roles.parquet"
SUMMARY_PATH = REPO_ROOT / "docs" / "analytics" / "earnings_call_speaker_roles_summary.json"

# Cut the speaker string at the first separator that introduces a title,
# affiliation, or parenthetical -- keeps only the bare name on the left.
_NAME_SEP_RE = re.compile(r"\s*[-–—;|]\s*|\s*\(")
_WS_RE = re.compile(r"\s+")

# Corporate-title fallback. Word-boundaried so short abbreviations don't
# match inside ordinary surnames.
_TITLE_RE = re.compile(
    r"(\bchief\b|\bcfo\b|\bceo\b|\bcoo\b|\bcto\b|\bpresident\b|"
    r"investor relations|\bir\b|\btreasurer\b|\bfounder\b|\bchairman\b|"
    r"general counsel)",
    re.IGNORECASE,
)

# Operator's Q&A hand-off line: "Our next question comes from NAME with FIRM."
# and its many phrasings across transcript vendors.
_ANNOUNCE_RE = re.compile(
    r"(?:next question|next caller|our next question|question is|caller is)"
    r"\D{0,20}?(?:from|is from|comes from|will come from)?\s*"
    r"(?:the line of\s+)?"
    r"([A-Z][a-zA-Z'.\-]+(?:\s+[A-Z][a-zA-Z'.\-]+){0,3})"
    r"(?:\s+(?:with|from|of)\s+[A-Za-z0-9&.,' ]+?)?"
    r"(?:\.|,|\s+Please|\s+Your line|$)"
)

_TOKEN_RE = re.compile(r"[a-z]+")
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "dr", "mr", "mrs", "ms"}

SEC_OFFICER_ROSTER_PATH = REPO_ROOT / "data" / "interim" / "sec_officer_roster" / "officer_roster.parquet"


def normalize_speaker(raw: str) -> str:
    """Strip trailing title/affiliation text, collapse whitespace, lowercase."""
    name_part = _NAME_SEP_RE.split(raw, maxsplit=1)[0]
    return _WS_RE.sub(" ", name_part).strip().lower()


def canonical_name_key(raw: str | None) -> str | None:
    """Order-invariant identity key: lowercase, drop initials/suffixes, sort tokens.

    SEC Form 4 names are "Last First Middle" ("Zavery Amit"); call transcripts
    use "First Last" ("Amit Zavery"). Sorting the token set makes the two
    comparable without guessing which source uses which order, and dropping
    single-letter initials and suffixes (Jr., III, ...) avoids a false
    mismatch when only one source includes them. Requires >=2 remaining
    tokens so a single common surname alone can never satisfy the key.
    """
    if raw is None:
        return None
    tokens = sorted(
        t for t in _TOKEN_RE.findall(raw.lower())
        if len(t) >= 2 and t not in _NAME_SUFFIXES
    )
    return " ".join(tokens) if len(tokens) >= 2 else None


def name_matches_with_nickname(a: str | None, b: str | None) -> bool:
    """Same-length token sets, exact match except for at most one nickname pair.

    Catches cases the exact `canonical_name_key` join misses because SEC
    filings use a legal first name while the transcript uses a nickname
    (Salesforce's Srini Tallapragada is SEC-registered as "Tallapragada
    Srinivas"). Every OTHER token -- crucially the surname -- must match
    exactly; only one token pair may differ, and only via a >=3-letter
    prefix relationship ("srini" -> "srinivas"), which keeps this from
    matching two unrelated same-surname people.
    """
    if a is None or b is None:
        return False
    ta = sorted(t for t in _TOKEN_RE.findall(a.lower()) if len(t) >= 2 and t not in _NAME_SUFFIXES)
    tb = sorted(t for t in _TOKEN_RE.findall(b.lower()) if len(t) >= 2 and t not in _NAME_SUFFIXES)
    if len(ta) < 2 or len(ta) != len(tb):
        return False
    nickname_slots_used = 0
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)):
            nickname_slots_used += 1
            continue
        return False
    return nickname_slots_used <= 1


def extract_announced_name(operator_text: str | None) -> str | None:
    """Pull the analyst name out of an operator Q&A hand-off line, else None."""
    if operator_text is None:
        return None
    m = _ANNOUNCE_RE.search(operator_text)
    return m.group(1) if m else None


def name_token_overlap(announced: str | None, speaker: str) -> bool:
    """True if the announced name and the speaker string share a >=3-letter token."""
    if not announced:
        return False
    announced_tokens = {t for t in _TOKEN_RE.findall(announced.lower()) if len(t) >= 3}
    speaker_tokens = {t for t in _TOKEN_RE.findall(speaker.lower()) if len(t) >= 3}
    return bool(announced_tokens & speaker_tokens)


def main() -> None:
    con = duckdb.connect()
    con.create_function("normalize_speaker", normalize_speaker, ["VARCHAR"], "VARCHAR")
    con.create_function(
        "extract_announced_name", extract_announced_name, ["VARCHAR"], "VARCHAR",
        null_handling="special")
    con.create_function("name_token_overlap", name_token_overlap, ["VARCHAR", "VARCHAR"], "BOOLEAN")
    con.create_function(
        "canonical_name_key", canonical_name_key, ["VARCHAR"], "VARCHAR",
        null_handling="special")
    con.create_function(
        "name_matches_with_nickname", name_matches_with_nickname, ["VARCHAR", "VARCHAR"], "BOOLEAN",
        null_handling="special")

    con.execute(f"""
        CREATE TEMP TABLE ec AS
        SELECT *,
               split_part(document_id, '_', 1) AS ticker,
               normalize_speaker(speaker) AS speaker_norm,
               canonical_name_key(speaker) AS speaker_name_key
        FROM read_parquet('{SOURCE_GLOB}', union_by_name=True)
        WHERE speaker IS NOT NULL AND trim(speaker) <> ''
    """)

    has_officer_roster = SEC_OFFICER_ROSTER_PATH.exists()
    if has_officer_roster:
        con.execute(f"""
            CREATE TEMP TABLE sec_officer_roster AS
            SELECT DISTINCT ticker, officer_name_raw, canonical_name_key(officer_name_raw) AS name_key
            FROM read_parquet('{SEC_OFFICER_ROSTER_PATH.as_posix()}')
            WHERE canonical_name_key(officer_name_raw) IS NOT NULL
        """)
    else:
        print(f"  no SEC officer roster at {SEC_OFFICER_ROSTER_PATH} -- run "
              f"build_sec_officer_roster.py first; skipping that signal")
        con.execute("CREATE TEMP TABLE sec_officer_roster (ticker VARCHAR, officer_name_raw VARCHAR, name_key VARCHAR)")

    # Signal 1: operator hand-off announcements in the Q&A section, with the
    # announced name carried forward to every subsequent turn until the next
    # announcement (follow-up questions from the same analyst don't repeat
    # the operator line).
    con.execute("""
        CREATE TEMP TABLE qa_announced AS
        WITH ann AS (
            SELECT document_id, paragraph_index, block_type, speaker,
                   CASE WHEN block_type = 'operator'
                        THEN extract_announced_name(paragraph_text) END AS announced_raw
            FROM ec
            WHERE section = 'qa'
        )
        SELECT document_id, paragraph_index, block_type, speaker,
               LAST_VALUE(announced_raw IGNORE NULLS) OVER (
                   PARTITION BY document_id ORDER BY paragraph_index
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS current_announced
        FROM ann
    """)

    con.execute("""
        CREATE TEMP TABLE confirmed_analyst AS
        SELECT DISTINCT e.ticker, e.speaker_norm, e.speaker_name_key
        FROM qa_announced q
        JOIN ec e ON e.document_id = q.document_id AND e.paragraph_index = q.paragraph_index
        WHERE q.block_type = 'speaker_turn'
          AND name_token_overlap(q.current_announced, e.speaker)
    """)

    # Signal 2: cross-quarter executive roster, pooled per ticker over every
    # call in the sample, EXCLUDING anyone operator-confirmed as an analyst
    # for that ticker (this is what removes the Elyse Greenspan / PGR-style
    # contamination from bad `section` labels).
    con.execute("""
        CREATE TEMP TABLE exec_roster_raw AS
        SELECT DISTINCT e.ticker, e.speaker_norm
        FROM ec e
        WHERE e.section = 'prepared' AND e.block_type = 'speaker_turn'
    """)
    con.execute("""
        CREATE TEMP TABLE exec_roster AS
        SELECT ticker, speaker_norm FROM exec_roster_raw
        EXCEPT
        SELECT ticker, speaker_norm FROM confirmed_analyst
    """)

    # Signal 3: SEC Form 4 officer-of-record roster (build_sec_officer_roster.py),
    # keyed by an order-invariant name so "Zavery Amit" (SEC) and "Amit
    # Zavery" (transcript) match. Also excludes operator-confirmed analysts,
    # in case a name key were ever shared with an unrelated officer.
    con.execute("""
        CREATE TEMP TABLE sec_officer_roster_clean AS
        SELECT sor.ticker, sor.officer_name_raw, sor.name_key
        FROM sec_officer_roster sor
        LEFT JOIN confirmed_analyst ca
            ON ca.ticker = sor.ticker AND ca.speaker_name_key = sor.name_key
        WHERE ca.speaker_name_key IS NULL
    """)

    con.execute(f"""
        CREATE TABLE speaker_roles AS
        WITH sp AS (
            SELECT document_id, ticker, speaker, speaker_norm, speaker_name_key,
                   bool_or(section = 'prepared' AND block_type = 'speaker_turn') AS in_prepared_this_call,
                   bool_or(section = 'qa' AND block_type = 'speaker_turn') AS in_qa,
                   bool_or(block_type = 'operator') AS ever_operator,
                   count(*) AS n_turns
            FROM ec
            GROUP BY document_id, ticker, speaker, speaker_norm, speaker_name_key
        ), rostered AS (
            SELECT sp.*,
                   ca.speaker_norm IS NOT NULL AS operator_confirmed_analyst,
                   er.speaker_norm IS NOT NULL AS on_exec_roster,
                   so.name_key IS NOT NULL AS on_sec_officer_roster,
                   regexp_matches(lower(sp.speaker), '{_TITLE_RE.pattern}') AS has_title_keyword
            FROM sp
            LEFT JOIN confirmed_analyst ca
                ON ca.ticker = sp.ticker AND ca.speaker_norm = sp.speaker_norm
            LEFT JOIN exec_roster er
                ON er.ticker = sp.ticker AND er.speaker_norm = sp.speaker_norm
            LEFT JOIN sec_officer_roster_clean so
                ON so.ticker = sp.ticker AND so.name_key = sp.speaker_name_key
        ), nickname_matched AS (
            -- Only probe the fuzzy nickname join for speakers the exact
            -- name_key join above already missed, and only against that
            -- ticker's officers -- bounds the UDF's cost to the residual.
            SELECT r.document_id, r.speaker,
                   bool_or(name_matches_with_nickname(sor.officer_name_raw, r.speaker)) AS matched
            FROM rostered r
            JOIN sec_officer_roster_clean sor ON sor.ticker = r.ticker
            WHERE NOT r.on_sec_officer_roster AND NOT r.in_prepared_this_call
              AND NOT r.operator_confirmed_analyst
            GROUP BY r.document_id, r.speaker
        )
        SELECT r.document_id, r.ticker, r.speaker, r.speaker_norm, r.n_turns,
               r.in_prepared_this_call, r.in_qa, r.ever_operator,
               r.operator_confirmed_analyst, r.on_exec_roster, r.on_sec_officer_roster,
               coalesce(nm.matched, false) AS on_sec_officer_roster_nickname,
               r.has_title_keyword,
               CASE
                   WHEN r.ever_operator AND NOT r.in_prepared_this_call AND NOT r.in_qa THEN 'operator_only'
                   WHEN r.operator_confirmed_analyst THEN 'analyst_confirmed_by_operator'
                   WHEN r.in_prepared_this_call THEN 'exec_ir'
                   WHEN r.on_exec_roster THEN 'exec_ir_cross_quarter'
                   WHEN r.on_sec_officer_roster THEN 'exec_ir_sec_officer'
                   WHEN coalesce(nm.matched, false) THEN 'exec_ir_sec_officer_nickname'
                   WHEN r.has_title_keyword AND r.in_qa THEN 'exec_ir_title_keyword'
                   WHEN r.in_qa THEN 'analyst_qa_only'
                   ELSE 'other'
               END AS heuristic_role
        FROM rostered r
        LEFT JOIN nickname_matched nm ON nm.document_id = r.document_id AND nm.speaker = r.speaker
    """)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY speaker_roles TO '{OUT_PATH}' (FORMAT PARQUET)")

    role_counts = con.execute("""
        SELECT heuristic_role, count(*) AS n_speaker_instances, count(DISTINCT document_id) AS n_calls
        FROM speaker_roles GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()

    recovered = con.execute("""
        SELECT count(*) FROM speaker_roles
        WHERE heuristic_role IN ('exec_ir_cross_quarter', 'exec_ir_sec_officer',
                                  'exec_ir_sec_officer_nickname', 'exec_ir_title_keyword')
    """).fetchone()[0]

    cleaned_from_roster = con.execute("""
        SELECT count(DISTINCT ca.ticker || '|' || ca.speaker_norm)
        FROM confirmed_analyst ca
        JOIN exec_roster_raw er ON er.ticker = ca.ticker AND er.speaker_norm = ca.speaker_norm
    """).fetchone()[0]

    # Re-run the misattribution measurement from the exploratory pass with
    # the fixed roles, to quantify the improvement.
    frames_glob = str(REPO_ROOT / "data" / "deprecated" / "ai_classify" / "ai_frames__session=*.parquet")
    misattribution = con.execute(f"""
        WITH frames AS (
            SELECT * FROM read_parquet('{frames_glob}', union_by_name=True)
            WHERE error IS NULL AND form = 'Earnings call' AND item_key = 'qa'
              AND has_frame AND subject = 'firm'
        ),
        joined AS (
            SELECT sr.heuristic_role
            FROM frames f
            JOIN ec e
                ON e.document_id = f.accession_number
               AND e.paragraph_index = f.paragraph_index
               AND e.section = 'qa'
            JOIN speaker_roles sr
                ON sr.document_id = e.document_id AND sr.speaker = e.speaker
        )
        SELECT heuristic_role, count(*) AS n_frames
        FROM joined GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()

    total_frames = sum(n for _, n in misattribution)
    residual_analyst = sum(
        n for role, n in misattribution
        if role in ('analyst_qa_only', 'analyst_confirmed_by_operator')
    )

    total_rows = sum(n_inst for _, n_inst, _ in role_counts)
    print(f"speaker_roles written: {OUT_PATH} ({total_rows} rows)")
    for role, n_inst, n_calls in role_counts:
        print(f"  {role:30s} {n_inst:7d} instances  {n_calls:6d} calls")
    print(f"speakers recovered from naive-rule false negative -> executive: {recovered}")
    print(f"false executives removed from the roster by operator confirmation: {cleaned_from_roster}")
    print(f"firm-subject Q&A frames from a confirmed/likely non-firm speaker: {residual_analyst}/{total_frames} "
          f"({100 * residual_analyst / total_frames:.2f}%)")

    summary = {
        "role_counts": [
            {"heuristic_role": role, "n_speaker_instances": n_inst, "n_calls": n_calls}
            for role, n_inst, n_calls in role_counts
        ],
        "n_speakers_recovered_by_fix": recovered,
        "n_false_executives_removed_by_operator_confirmation": cleaned_from_roster,
        "qa_firm_frames_by_role_after_fix": [
            {"heuristic_role": role, "n_frames": n} for role, n in misattribution
        ],
        "residual_non_firm_speaker_pct": round(100 * residual_analyst / total_frames, 2) if total_frames else None,
        "naive_rule_baseline_analyst_qa_only_pct": 17.45,
        "cross_quarter_roster_only_baseline_pct": 13.33,
        "plus_operator_confirmation_baseline_pct": 22.04,
        "plus_sec_officer_roster_pct": 20.24,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"summary written: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
