#!/usr/bin/env python3
"""Build geographic provenance summary of AI-context regional references.

Reads:
  - configs/geo_provenance.yaml (region term lists)
  - duckdb/thesis.duckdb (gold_ai_frames, unique_paragraphs, filing manifests)
  - data/interim/manifests/filing_manifest_earnings_calls*.parquet (calls)
  - data/processed/clusters/firm_strategy_dimensions.parquet (archetype)

Writes:
  - data/processed/clusters/geo_provenance_summary.json

Construct: a "mention" is any lexical, word-boundary match of a term from
`configs/geo_provenance.yaml` against the text of a paragraph that already
carries an AI disclosure frame (`gold_ai_frames.has_frame`). Matching is not
restricted to named external entities or any activity role: a term list
mixes named technology entities (companies, models, platforms) with literal
country/demonym words (e.g. "china", "chinese"), and any match inside
AI-context text counts. This deliberately widens the older, role-filtered
construct (which only counted entities tagged external_provider/
external_model/partner within an extracted activity claim) to also capture
market and geopolitical references to a region within AI disclosure -- e.g.
a firm discussing Chinese AI-market opportunity or export controls, not only
firms naming a Chinese vendor they use.

United States is the one asymmetric region: its term list is company/product
names only, with no literal country term ("united states", "u.s.",
"america"). Every U.S.-domiciled SEC filer routinely names its own
jurisdiction, so a literal-country match there would saturate the United
States share near 100% and erase the measure's ability to discriminate.

Matching uses word-boundary-anchored substring search (not `in` containment),
so a 2-3 letter term (e.g. "yi", the Yi/01.AI model family) does not match
inside unrelated words ("trying", "buying"), and "india" does not match
inside "Indiana".
"""

import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "geo_provenance.yaml"
STRATEGY_PATH = REPO_ROOT / "data" / "processed" / "clusters" / "firm_strategy_dimensions.parquet"
DB_PATH = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_PATH = REPO_ROOT / "data" / "processed" / "clusters" / "geo_provenance_summary.json"

# Tres fuentes de transcripciones -- ver ai_intensity.py: la base de Hugging
# Face no se actualiza desde 2025; los dos rellenos cubren 2026.
CALLS_MANIFESTS = [
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet",
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls_equibles.parquet",
    REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls_stockanalysis.parquet",
]
CATEGORIES = ["United States", "Europe", "China", "Other"]


def _calls_manifest_sql() -> str:
    parts = [f"SELECT document_id, ticker, filing_date FROM read_parquet('{p}')" for p in CALLS_MANIFESTS if p.exists()]
    union = " UNION ALL BY NAME ".join(parts)
    return f"""
        SELECT document_id, ticker, TRY_CAST(filing_date AS DATE) AS filing_date FROM (
            SELECT *, row_number() OVER (PARTITION BY document_id ORDER BY 1) AS rn
            FROM ({union})
        ) WHERE rn = 1
          -- same analysis universe as the views (S&P 500 at 2021-01-01, see build_duckdb.py)
          AND ticker IN (SELECT ticker FROM firm_universe WHERE country_code = 'us')
    """


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    ecosystems = config.get("ecosystems", {})

    def boundary_alternation(terms: list[str]) -> re.Pattern:
        """Un único patrón compilado por zona (alternación de todos sus
        términos) en vez de un patrón por término: una sola pasada de regex
        por párrafo y por zona, no decenas."""
        alts = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
        return re.compile(r"(?<![a-z0-9])(?:" + alts + r")(?![a-z0-9])")

    zone_patterns = {eco: boundary_alternation(ecosystems.get(eco, [])) for eco in CATEGORIES}

    def matched_zones(low: str) -> set[str]:
        """Regiones cuyo término aparece en `low` (ya en minúsculas)."""
        return {eco for eco, pat in zone_patterns.items() if pat.search(low)}

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(f"CREATE OR REPLACE TEMP VIEW calls_manifest AS {_calls_manifest_sql()}")
    ai_paragraphs = con.execute("""
        WITH ai_ctx AS (
            SELECT DISTINCT accession_number, paragraph_index
            FROM gold_ai_frames
            WHERE has_frame
        )
        SELECT a.accession_number, a.paragraph_index, p.paragraph_text
        FROM ai_ctx a
        JOIN unique_paragraphs p USING (accession_number, paragraph_index)
    """).df()
    filings_map = con.execute("""
        WITH m AS (
            SELECT accession_number, ticker, filing_date FROM filing_manifest
            UNION ALL
            SELECT accession_number, ticker, filing_date FROM filing_manifest_10q
        )
        SELECT accession_number, ticker, filing_date FROM m
    """).df()
    calls_map = con.execute("SELECT document_id AS accession_number, ticker, filing_date FROM calls_manifest").df()
    con.close()

    ai_paragraphs = ai_paragraphs.merge(
        pd.concat([filings_map, calls_map], ignore_index=True).drop_duplicates("accession_number"),
        on="accession_number", how="left")
    ai_paragraphs["year"] = pd.to_datetime(ai_paragraphs["filing_date"], errors="coerce").dt.year
    ai_paragraphs = ai_paragraphs.dropna(subset=["ticker", "year"])
    ai_paragraphs = ai_paragraphs[ai_paragraphs["year"].between(2021, 2026)]
    ai_paragraphs["year"] = ai_paragraphs["year"].astype(int)

    # Un match por (párrafo, zona): un párrafo que nombra dos regiones cuenta
    # una mención en cada una; no hay doble conteo dentro de la misma zona
    # aunque aparezcan varios términos de esa zona en el mismo párrafo.
    records = []
    for row, low in zip(ai_paragraphs.itertuples(index=False), ai_paragraphs["paragraph_text"].str.lower()):
        for z in matched_zones(low):
            records.append({"ticker": row.ticker, "year": row.year, "geo": z})

    df_geo = pd.DataFrame(records)
    strat = pd.read_parquet(STRATEGY_PATH)[["ticker", "archetype"]]
    df_geo = df_geo.merge(strat, on="ticker", how="left")

    years = sorted(df_geo["year"].unique().tolist())
    ct_counts = pd.crosstab(df_geo["year"], df_geo["geo"]).reindex(columns=CATEGORIES, fill_value=0)
    ct_shares = (pd.crosstab(df_geo["year"], df_geo["geo"], normalize="index") * 100).reindex(columns=CATEGORIES, fill_value=0)
    panel_a = {
        "years": [int(y) for y in years],
        "shares": {cat: [float(ct_shares.loc[y, cat]) for y in years] for cat in CATEGORIES},
        "counts": {cat: [int(ct_counts.loc[y, cat]) for y in years] for cat in CATEGORIES},
    }

    archetypes = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]
    arch_counts = pd.crosstab(df_geo["archetype"], df_geo["geo"]).reindex(index=archetypes, columns=CATEGORIES, fill_value=0)
    arch_shares = (pd.crosstab(df_geo["archetype"], df_geo["geo"], normalize="index") * 100).reindex(index=archetypes, columns=CATEGORIES, fill_value=0)
    panel_b = {
        "archetypes": archetypes,
        "shares": {cat: [float(arch_shares.loc[a, cat]) for a in archetypes] for cat in CATEGORIES},
        "counts": {cat: [int(arch_counts.loc[a, cat]) for a in archetypes] for cat in CATEGORIES},
    }

    firm_geos = df_geo.groupby("ticker")["geo"].unique()
    n_firms_total = int(ai_paragraphs["ticker"].nunique())
    reach = {
        "n_firms_with_activities": n_firms_total,
        "n_firms_with_external_stack": int(len(firm_geos)),
        "us_firms": int(sum("United States" in g for g in firm_geos)),
        "europe_firms": int(sum("Europe" in g for g in firm_geos)),
        "china_firms": int(sum("China" in g for g in firm_geos)),
        "other_firms": int(sum("Other" in g for g in firm_geos)),
        "total_mentions": int(len(df_geo)),
        "us_mentions_share": float((df_geo["geo"] == "United States").mean() * 100),
        "europe_mentions_share": float((df_geo["geo"] == "Europe").mean() * 100),
        "china_mentions_share": float((df_geo["geo"] == "China").mean() * 100),
        "other_mentions_share": float((df_geo["geo"] == "Other").mean() * 100),
    }

    output_data = {"panel_a": panel_a, "panel_b": panel_b, "reach": reach}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Geographic provenance summary written to {OUT_PATH}")
    print("Reach stats:", reach)


if __name__ == "__main__":
    main()
