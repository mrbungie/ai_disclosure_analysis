"""Classify earnings-call speakers as company (executive/IR) vs external analyst.

Deterministic, no LLM. Reads
data/interim/sections/earnings_call_paragraphs__v=3__run=*__part=*.parquet;
writes data/interim/earnings_call_speaker_roles/speaker_roles.parquet.

Problem this fixes (measured on the v=3 corpus, 11,407 calls): the naive
per-call rule -- "speaks in `prepared` => executive; only in `qa` => analyst"
-- misses executives who are on the call ONLY to field Q&A and never take a
`prepared` turn. Example: ServiceNow's Amit Zavery (President & Chief
Product Officer) appears in all 7 NOW calls in the sample, always
`in_prepared=False`, so the naive rule mislabels him as an analyst in every
one. Joined against `gold_ai_frames` (subject='firm', item_key='qa'), the
naive rule misattributes 17.5% of firm-subject Q&A frames to a non-firm
speaker (2,745 / 15,728).

Two deterministic signals fix most of this without any model call:

  1. Cross-quarter executive roster. A speaker's `prepared`-turn history is
     pooled across EVERY call of the same ticker, not just the current
     quarter. If the same person (after name normalization) ever opens a
     `prepared` section for this ticker -- in any quarter in the sample --
     they are an executive on every other call too, including ones where
     they only speak in `qa`. This alone resolves the Zavery-style case in
     any ticker that has at least one call with that person in `prepared`.

  2. Name normalization before matching. Sources are inconsistent about
     whether the speaker string carries a title suffix (`"Lisa T. Su (CEO)"`
     in `qa` vs `"Lisa Su"` in `prepared` for the same call) -- normalizing
     both to a bare name before the roster lookup and the same-call
     `in_prepared` check catches these without an LLM re-reading the
     transcript.

  3. Title-keyword fallback. For a ticker where the executive NEVER appears
     in `prepared` in the whole sample (so signal 1 has nothing to pool
     from), an explicit corporate title in the raw speaker string (CEO, CFO,
     "Head of Investor Relations", ...) is itself deterministic evidence,
     matched with word boundaries so it doesn't fire on names that merely
     contain the letters (e.g. "Cook", "Spector", "Francfort" do NOT match
     "coo"/"cto"/"cfo" once boundaries are enforced).

`speaker` is call-scoped free text with no company/analyst flag anywhere
upstream, so this is the only way to recover the distinction without
sending transcript text to a judge model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def normalize_speaker(raw: str) -> str:
    """Strip trailing title/affiliation text, collapse whitespace, lowercase."""
    name_part = _NAME_SEP_RE.split(raw, maxsplit=1)[0]
    return _WS_RE.sub(" ", name_part).strip().lower()


def main() -> None:
    con = duckdb.connect()
    con.create_function("normalize_speaker", normalize_speaker, ["VARCHAR"], "VARCHAR")

    con.execute(f"""
        CREATE TEMP TABLE ec AS
        SELECT *,
               split_part(document_id, '_', 1) AS ticker,
               normalize_speaker(speaker) AS speaker_norm
        FROM read_parquet('{SOURCE_GLOB}', union_by_name=True)
        WHERE speaker IS NOT NULL AND trim(speaker) <> ''
    """)

    # Signal 1: cross-quarter executive roster, pooled per ticker over every
    # call in the sample -- NOT scoped to the current call.
    con.execute("""
        CREATE TEMP TABLE exec_roster AS
        SELECT DISTINCT ticker, speaker_norm
        FROM ec
        WHERE section = 'prepared' AND block_type = 'speaker_turn'
    """)

    con.execute(f"""
        CREATE TABLE speaker_roles AS
        WITH sp AS (
            SELECT document_id, ticker, speaker, speaker_norm,
                   bool_or(section = 'prepared' AND block_type = 'speaker_turn') AS in_prepared_this_call,
                   bool_or(section = 'qa' AND block_type = 'speaker_turn') AS in_qa,
                   bool_or(block_type = 'operator') AS ever_operator,
                   count(*) AS n_turns
            FROM ec
            GROUP BY document_id, ticker, speaker, speaker_norm
        ), rostered AS (
            SELECT sp.*,
                   er.speaker_norm IS NOT NULL AS on_exec_roster,
                   regexp_matches(lower(sp.speaker), '{_TITLE_RE.pattern}') AS has_title_keyword
            FROM sp
            LEFT JOIN exec_roster er
                ON er.ticker = sp.ticker AND er.speaker_norm = sp.speaker_norm
        )
        SELECT document_id, ticker, speaker, speaker_norm, n_turns,
               in_prepared_this_call, in_qa, ever_operator,
               on_exec_roster, has_title_keyword,
               CASE
                   WHEN ever_operator AND NOT in_prepared_this_call AND NOT in_qa THEN 'operator_only'
                   WHEN in_prepared_this_call THEN 'exec_ir'
                   WHEN on_exec_roster THEN 'exec_ir_cross_quarter'
                   WHEN has_title_keyword AND in_qa THEN 'exec_ir_title_keyword'
                   WHEN in_qa THEN 'analyst_qa_only'
                   ELSE 'other'
               END AS heuristic_role
        FROM rostered
    """)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY speaker_roles TO '{OUT_PATH}' (FORMAT PARQUET)")

    role_counts = con.execute("""
        SELECT heuristic_role, count(*) AS n_speaker_instances, count(DISTINCT document_id) AS n_calls
        FROM speaker_roles GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()

    recovered = con.execute("""
        SELECT count(*) FROM speaker_roles
        WHERE heuristic_role IN ('exec_ir_cross_quarter', 'exec_ir_title_keyword')
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
    residual_analyst = sum(n for role, n in misattribution if role == 'analyst_qa_only')

    total_rows = sum(n_inst for _, n_inst, _ in role_counts)
    print(f"speaker_roles written: {OUT_PATH} ({total_rows} rows)")
    for role, n_inst, n_calls in role_counts:
        print(f"  {role:24s} {n_inst:7d} instances  {n_calls:6d} calls")
    print(f"speakers recovered from naive-rule false negative -> executive: {recovered}")
    print(f"firm-subject Q&A frames re-attributed to analyst_qa_only: {residual_analyst}/{total_frames} "
          f"({100 * residual_analyst / total_frames:.2f}%)")

    summary = {
        "role_counts": [
            {"heuristic_role": role, "n_speaker_instances": n_inst, "n_calls": n_calls}
            for role, n_inst, n_calls in role_counts
        ],
        "n_speakers_recovered_by_fix": recovered,
        "qa_firm_frames_by_role_after_fix": [
            {"heuristic_role": role, "n_frames": n} for role, n in misattribution
        ],
        "residual_analyst_qa_only_pct": round(100 * residual_analyst / total_frames, 2) if total_frames else None,
        "naive_rule_baseline_analyst_qa_only_pct": 17.45,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"summary written: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
