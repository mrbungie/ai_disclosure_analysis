"""
scripts/silver/ai_vendors.py — silver.ai_vendor_mentions: which named AI
vendor each AI-disclosure paragraph mentions, one row per mention.

The long, navigable table behind the gold vendor variables. One row per
(paragraph instance, matched term), carrying the term as written, the family
it belongs to, and that family's ecosystem, country and role in the AI stack
(configs/ai_vendor_ecosystems.yaml). Reading it answers "who is named, where,
in which document" without a join to anything but the document spine.

Deterministic and cheap: literal word-boundary regex, the same technique
ai_prefilter.py uses for lexical scoring. No LLM, no GPU, no additive
checkpoint — it recomputes from the config on every run, so a term dropped
from the lexicon disappears from the table.

Only commercial names are matched — companies, cloud platforms, accelerators,
model families, enterprise AI products. Country and demonym words are absent
from the lexicon by construction: naming a market is not evidence that an
ecosystem's technology is present in the disclosure.

Population: paragraphs that already carry an AI disclosure frame
(silver.ai_frames.has_frame), so a match always sits inside AI-context text,
never a generic supply-chain or tariff paragraph. Matching runs once per
distinct text (bronze.unique_paragraphs) and is broadcast to every paragraph
instance carrying that text, with `duplicate_count` kept for weighting.

Ambiguous terms (`context_required` in the config) are ordinary words,
personal names or multi-vendor products — "copilot", "gemini", "claude",
"foundry", "arm". Each is kept only when one of its context terms appears in
the same paragraph, so a paragraph about a robotic arm is not an Arm Holdings
mention.

Lineage: (country_code, form, accession_number, item_key, paragraph_index)
-> bronze.paragraphs -> bronze.extraction_trace; text_hash -> the paragraph
content the term was matched in.
"""

from __future__ import annotations

import re

import polars as pl
import yaml
from _paths import PARAGRAPH_KEY, L, bronze_inputs, log

BUILDER = "scripts/silver/ai_vendors.py"
CONFIG_PATH = L.REPO_ROOT / "configs" / "ai_vendor_ecosystems.yaml"

# Word boundary that does not fire on digits either: "arm" must not match
# inside "alarm", and "gpt-4" must not match inside "gpt-4o".
BOUNDARY_BEFORE = "(?:^|[^a-z0-9])"
BOUNDARY_AFTER = "(?:[^a-z0-9]|$)"

VALUE_COLUMNS = ["term", "family", "ecosystem", "country", "vendor_type", "vendor_ticker", "context_gated"]


def lexicon() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """(term -> family metadata, term -> context terms it requires).

    A term listed by two families resolves to the first in file order; the
    config collapses aliases into one family to keep that from happening."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    excluded = {t.lower() for t in config.get("generic_exclusions", [])}
    terms: dict[str, dict] = {}
    for vendor in config["vendors"]:
        meta = {"family": vendor["family"], "ecosystem": vendor["ecosystem"], "country": vendor["country"],
                "vendor_type": vendor["type"], "vendor_ticker": vendor.get("ticker")}
        for term in vendor["terms"]:
            term = term.lower().strip()
            if term not in excluded and term not in terms:
                terms[term] = meta
    context = {t.lower(): [c.lower() for c in ctx] for t, ctx in config.get("context_required", {}).items()}
    return terms, context


def boundary(term: str) -> str:
    return BOUNDARY_BEFORE + re.escape(term) + BOUNDARY_AFTER


def ai_texts() -> pl.DataFrame:
    """Distinct texts of the paragraphs that carry an AI disclosure frame."""
    return (L.scan("silver.ai_frames").filter(pl.col("has_frame")).select("text_hash").unique()
            .join(L.scan("bronze.unique_paragraphs").select("text_hash", "duplicate_count", "paragraph_text"),
                  on="text_hash")
            .with_columns(pl.col("paragraph_text").fill_null("").str.to_lowercase().alias("lowered"))
            .drop("paragraph_text")
            .collect(engine="streaming"))


def matches(texts: pl.DataFrame) -> pl.DataFrame:
    terms, context = lexicon()
    log(f"{len(terms)} términos en {len({m['family'] for m in terms.values()})} familias")
    context_patterns = {t: [re.compile(boundary(c)) for c in ctx] for t, ctx in context.items()}

    found = []
    for term, meta in terms.items():
        hit = texts.filter(pl.col("lowered").str.contains(boundary(term)))
        if term in context_patterns:
            pats = context_patterns[term]
            hit = hit.filter(pl.Series([any(p.search(low) for p in pats) for low in hit["lowered"]],
                                       dtype=pl.Boolean)) if hit.height else hit
        if not hit.height:
            continue
        found.append(hit.select(
            "text_hash", pl.lit(term).alias("term"), pl.lit(meta["family"]).alias("family"),
            pl.lit(meta["ecosystem"]).alias("ecosystem"), pl.lit(meta["country"]).alias("country"),
            pl.lit(meta["vendor_type"]).alias("vendor_type"),
            pl.lit(meta["vendor_ticker"], dtype=pl.String).alias("vendor_ticker"),
            pl.lit(term in context_patterns).alias("context_gated")))

    schema = {"text_hash": pl.UInt64, "term": pl.String, "family": pl.String, "ecosystem": pl.String,
              "country": pl.String, "vendor_type": pl.String, "vendor_ticker": pl.String,
              "context_gated": pl.Boolean}
    return pl.concat(found) if found else pl.DataFrame(schema=schema)


def panel_documents() -> pl.LazyFrame:
    """Documents of the analysis universe, with the document_id that stands in
    for the accession number of an earnings call."""
    manifest = L.scan("silver.filing_manifest")
    return pl.concat([
        manifest.select(pl.col("accession_number")),
        manifest.filter(pl.col("accession_number").is_null()).select(pl.col("document_id").alias("accession_number")),
        L.scan("silver.filing_manifest_10q").select(pl.col("accession_number")),
    ]).drop_nulls().unique()


def main() -> None:
    texts = ai_texts()
    log(f"{texts.height:,} textos únicos con marco de IA")
    mentions = matches(texts)
    log(f"{mentions.height:,} menciones en {mentions['text_hash'].n_unique():,} textos")

    instances = (L.scan("bronze.paragraphs").select(*PARAGRAPH_KEY, "text_hash")
                 .join(panel_documents(), on="accession_number", how="semi")
                 .join(L.scan("bronze.unique_paragraphs").select("text_hash", "duplicate_count"), on="text_hash"))
    out = (instances.join(mentions.lazy(), on="text_hash")
           .select(*PARAGRAPH_KEY, "text_hash", "duplicate_count", *VALUE_COLUMNS)
           .collect(engine="streaming"))

    L.write_table("silver.ai_vendor_mentions", out, keys=[*PARAGRAPH_KEY, "term"],
                  inputs=bronze_inputs("bronze.paragraphs", "bronze.unique_paragraphs") + [CONFIG_PATH],
                  builder=BUILDER)
    log(f"ai_vendor_mentions: {out.height} rows | {out['accession_number'].n_unique():,} documents | "
        f"{out['family'].n_unique()} families")


if __name__ == "__main__":
    main()
