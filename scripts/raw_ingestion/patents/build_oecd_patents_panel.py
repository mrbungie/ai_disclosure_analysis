"""
Pipeline to compute firm-year AI patent counts for S&P 500 firms using OECD (2025) methodology.
OECD (2025) Reference:
"Identifying emerging AI technologies using patent data: A semi-automated approach"
OECD Science, Technology and Industry Working Papers / OECD Publishing, Paris.

Rules:
1. Core AI: 5 CPC groups (G06N3, G06N5, G06N7, G06N20, G06F18) -> unconditional AI.
2. Related AI: 95 CPC groups -> counts as AI IF AND ONLY IF title or abstract contains >= 1 OECD keyword.
3. 274 AI keywords spanning ML, DL, LLMs, generative AI, transformers, computer vision, etc.
"""

from pathlib import Path
import re
import fitz
import pandas as pd
from google.cloud import bigquery

REPO_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = REPO_ROOT / ".tmp_oecd" / "oecd_ai_patents_2025.pdf"

PROJECT_ID = "skillforge-dev-502010"
DATASET_ID = "thesis_patents"

CORE_CPC = ["G06N3", "G06N5", "G06N7", "G06N20", "G06F18"]

def get_oecd_taxonomies():
    doc = fitz.open(PDF_PATH)
    # Page 30 (0-indexed 29): Table A B.2
    p30 = doc[29]
    cpcs = re.findall(r'\b[A-Z]\d{2}[A-Z]\d+\b', p30.get_text())
    related_cpc = sorted(list(set(c for c in cpcs if c not in CORE_CPC)))

    # Pages 28-29: Table A B.1 (274 keywords)
    keywords = []
    for p_num in [27, 28]:
        tabs = doc[p_num].find_tables()
        for t in tabs.tables:
            for row in t.extract():
                for cell in row:
                    if cell:
                        cleaned = " ".join(cell.split()).strip()
                        if cleaned:
                            keywords.append(cleaned)

    return CORE_CPC, related_cpc, keywords

def compile_keywords_regex(keywords: list[str]) -> list[str]:
    patterns = []
    for k in keywords:
        k_clean = k.lower().strip()
        if "(…)" in k_clean:
            parts = [re.escape(p.strip()) for p in k_clean.split("(…)") if p.strip()]
            pat = r'\b' + r'.{1,30}'.join(parts) + r'\b'
        else:
            words = k_clean.split()
            pat = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
        patterns.append(pat)

    # Group into chunks of 25 to ensure optimal BigQuery regex execution
    chunk_size = 25
    regex_chunks = []
    for i in range(0, len(patterns), chunk_size):
        regex_chunks.append("|".join(patterns[i:i + chunk_size]))
    return regex_chunks

def generate_sql(core_cpc: list[str], related_cpc: list[str], regex_chunks: list[str]) -> str:
    core_cpc_sql = ", ".join(f"'{c}'" for c in core_cpc)
    related_cpc_sql = ", ".join(f"'{c}'" for c in related_cpc)

    keyword_clauses = "\n        OR ".join(
        f"REGEXP_CONTAINS(text_combined, r'(?i){chunk}')"
        for chunk in regex_chunks
    )

    sql = f"""
WITH matched_assignees AS (
  SELECT
    ticker,
    cik,
    company_name,
    assignee_regex
  FROM `{PROJECT_ID}.{DATASET_ID}.sp500_patent_aliases_seed`
),
-- assignee_harmonized.name repeats across every patent a firm holds (millions of rows
-- collapse to ~5M distinct strings in the 2015-2026 filing window). Running the 549-way
-- REGEXP_CONTAINS once per DISTINCT name instead of once per (patent, assignee) row cuts
-- the expensive regex evaluation ~33x; the second pass below re-joins by plain string
-- equality (a hash join), which is cheap regardless of table size.
distinct_assignee_names AS (
  SELECT DISTINCT assignee_harmonized.name AS name
  FROM `patents-public-data.patents.publications`,
  UNNEST(assignee_harmonized) AS assignee_harmonized
  WHERE filing_date >= 20150101 AND filing_date <= 20260901
),
matched_assignee_names AS (
  SELECT
    da.name AS assignee_name,
    m.ticker,
    m.cik,
    m.company_name
  FROM distinct_assignee_names da
  JOIN matched_assignees m
    ON REGEXP_CONTAINS(UPPER(da.name), m.assignee_regex)
),
-- Candidate patents belonging to S&P 500 assignees
candidate_patents AS (
  SELECT
    mn.ticker,
    mn.cik,
    mn.company_name,
    p.publication_number,
    p.filing_date,
    p.publication_date,
    DIV(p.filing_date, 10000) AS filing_year,
    DIV(p.publication_date, 10000) AS publication_year,
    -- Extract distinct CPC main groups for this patent
    ARRAY(
      SELECT DISTINCT SPLIT(code, '/')[OFFSET(0)]
      FROM UNNEST(p.cpc)
      WHERE code IS NOT NULL AND INSTR(code, '/') > 0
    ) AS cpc_groups,
    -- Concatenate title and abstract text
    LOWER(
      CONCAT(
        COALESCE((SELECT STRING_AGG(text, ' ') FROM UNNEST(p.title_localized)), ''),
        ' ',
        COALESCE((SELECT STRING_AGG(text, ' ') FROM UNNEST(p.abstract_localized)), '')
      )
    ) AS text_combined
  FROM `patents-public-data.patents.publications` p,
  UNNEST(p.assignee_harmonized) AS ah
  JOIN matched_assignee_names mn
    ON ah.name = mn.assignee_name
  WHERE p.filing_date >= 20150101 AND p.filing_date <= 20260901
),
-- Deduplicate patent by publication_number and firm
deduped_patents AS (
  SELECT
    ticker,
    cik,
    company_name,
    publication_number,
    filing_year,
    publication_year,
    cpc_groups,
    text_combined
  FROM candidate_patents
  QUALIFY ROW_NUMBER() OVER(PARTITION BY ticker, publication_number) = 1
),
-- Evaluate OECD 2025 AI conditions
patent_evaluation AS (
  SELECT
    ticker,
    cik,
    company_name,
    publication_number,
    filing_year,
    publication_year,
    -- Condition 1: Core AI (at least one CPC in 5 core groups)
    EXISTS(
      SELECT 1 FROM UNNEST(cpc_groups) g
      WHERE g IN ({core_cpc_sql})
    ) AS is_core_ai,
    -- Condition 2: Related AI (at least one CPC in 95 related groups AND >= 1 OECD keyword)
    (
      EXISTS(
        SELECT 1 FROM UNNEST(cpc_groups) g
        WHERE g IN ({related_cpc_sql})
      )
      AND (
        {keyword_clauses}
      )
    ) AS is_related_ai
  FROM deduped_patents
),
classified_patents AS (
  SELECT
    ticker,
    cik,
    company_name,
    publication_number,
    filing_year,
    publication_year,
    is_core_ai,
    is_related_ai,
    (is_core_ai OR is_related_ai) AS is_ai_oecd
  FROM patent_evaluation
)
-- Aggregate by firm and filing year
SELECT
  ticker,
  cik,
  company_name,
  filing_year AS year,
  COUNT(DISTINCT publication_number) AS total_patents,
  COUNT(DISTINCT IF(is_ai_oecd, publication_number, NULL)) AS ai_patents_oecd,
  COUNT(DISTINCT IF(is_core_ai, publication_number, NULL)) AS ai_patents_core,
  COUNT(DISTINCT IF(is_related_ai, publication_number, NULL)) AS ai_patents_related,
  ROUND(SAFE_DIVIDE(
    COUNT(DISTINCT IF(is_ai_oecd, publication_number, NULL)),
    COUNT(DISTINCT publication_number)
  ), 4) AS ai_patent_intensity
FROM classified_patents
WHERE filing_year BETWEEN 2015 AND 2026
GROUP BY 1, 2, 3, 4
ORDER BY ticker, year
"""
    return sql

if __name__ == "__main__":
    core, rel, kw = get_oecd_taxonomies()
    chunks = compile_keywords_regex(kw)
    sql = generate_sql(core, rel, chunks)
    out_sql_path = REPO_ROOT / "scripts" / "raw_ingestion" / "patents" / "sql" / "oecd_ai_patents_query.sql"
    with open(out_sql_path, "w") as f:
        f.write(sql)
    print(f"Generated OECD BigQuery SQL saved to {out_sql_path} ({len(sql)} chars)")
