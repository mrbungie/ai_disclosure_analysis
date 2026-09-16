"""Named AI vendor ecosystems present in a document's AI disclosure.

Two wide document-grain tables over the long silver.ai_vendor_mentions, one
row per document of the gold document spine (10-K, 10-Q, DEF 14A, 8-K and
earnings calls):

  covariates/document/ai_vendors           n_ai_paragraphs (the denominator),
                                           n_vendor_paragraphs, and per
                                           ecosystem a presence flag and a
                                           paragraph count, split between the
                                           AI technology stack and enterprise
                                           software
  covariates/document/ai_vendor_families   one presence flag per provider
                                           family (`vendor_openai`,
                                           `vendor_deepseek`, ...)

The unit is the document and presence is binary: a call naming Azure twenty
times counts once for Microsoft and once for the United States. Ecosystems
and families overlap — a document naming both NVIDIA and DeepSeek counts for
the United States and for China — so the flags are independent indicators,
never shares of a partition.

Two coverages stay separate and are never pooled into one number. The AI
technology stack is model developers, cloud infrastructure and semiconductors
(`vendor_stack_*`); applied AI inside business software is enterprise
software (`vendor_entsw_*`).

A mention of the filer's own products (`vendor_ticker` equal to the
document's ticker) measures own branding, not the reach of an external
ecosystem, and is excluded from every flag; `n_vendor_self_paragraphs` keeps
how many paragraphs were dropped that way.

Deterministic, no LLM.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

BUILDER = "scripts/gold/document/build_ai_vendors.py"

# Ecosystem -> column suffix. The four-way split of the lexicon; the specific
# country and the matched term stay in silver.ai_vendor_mentions.
ECOSYSTEMS = {"United States": "us", "China": "cn", "Europe": "eu", "Other": "other"}
STACK_TYPES = ("model_developer", "cloud_infrastructure", "semiconductor")


def column_name(family: str) -> str:
    return "vendor_" + re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_")


def mentions(spine: pd.DataFrame) -> pd.DataFrame:
    """One row per (document, AI paragraph, family), external mentions flagged."""
    df = (L.scan("silver.ai_vendor_mentions")
          .select("accession_number", "paragraph_index", "family", "ecosystem", "vendor_type", "vendor_ticker")
          .unique(["accession_number", "paragraph_index", "family"])
          .collect(engine="streaming").to_pandas())
    df = df.merge(spine[["accession_number", "ticker"]], on="accession_number", how="inner")
    df["is_self_reference"] = df["vendor_ticker"].eq(df["ticker"])
    df["is_stack"] = df["vendor_type"].isin(STACK_TYPES)
    return df


def main() -> None:
    spine = L.read_gold("document", spine_columns=["id", "ticker", "accession_number", "fecha"])

    n_ai = (L.scan("silver.ai_frames").filter(pl.col("has_frame"))
            .select("accession_number", "paragraph_index").unique()
            .group_by("accession_number").len().rename({"len": "n_ai_paragraphs"})
            .collect().to_pandas())

    found = mentions(spine)
    external = found[~found["is_self_reference"]]

    out = spine.merge(n_ai, on="accession_number", how="left")
    out["n_ai_paragraphs"] = out["n_ai_paragraphs"].fillna(0).astype("int64")

    def paragraph_count(frame: pd.DataFrame) -> pd.Series:
        """Per document, how many distinct AI paragraphs `frame` covers."""
        counts = frame.drop_duplicates(["accession_number", "paragraph_index"]).groupby("accession_number").size()
        return out["accession_number"].map(counts).fillna(0).astype("int64")

    out["n_vendor_paragraphs"] = paragraph_count(external)
    out["n_vendor_self_paragraphs"] = paragraph_count(found[found["is_self_reference"]])

    for ecosystem, key in ECOSYSTEMS.items():
        in_eco = external[external["ecosystem"] == ecosystem]
        out[f"vendor_stack_{key}_paragraphs"] = paragraph_count(in_eco[in_eco["is_stack"]])
        out[f"vendor_entsw_{key}_paragraphs"] = paragraph_count(in_eco[~in_eco["is_stack"]])
        out[f"vendor_stack_{key}"] = out[f"vendor_stack_{key}_paragraphs"] > 0
        out[f"vendor_entsw_{key}"] = out[f"vendor_entsw_{key}_paragraphs"] > 0

    L.write_gold("covariates", "document", "ai_vendors", out, builder=BUILDER)

    families = spine.copy()
    for family in sorted(external["family"].unique()):
        naming = set(external.loc[external["family"] == family, "accession_number"])
        families[column_name(family)] = families["accession_number"].isin(naming)
    L.write_gold("covariates", "document", "ai_vendor_families", families, builder=BUILDER)

    reached = out[out["n_vendor_paragraphs"] > 0]
    print(f"{len(out):,} documents | {len(reached):,} naming an external AI vendor "
          f"({len(reached) / max(len(out), 1) * 100:.1f}%) | {reached['ticker'].nunique():,} firms | "
          f"{len(families.columns) - 4} family columns")
    for ecosystem, key in ECOSYSTEMS.items():
        print(f"  {ecosystem:>14}: stack {out[f'vendor_stack_{key}'].sum():>6,} docs | "
              f"enterprise software {out[f'vendor_entsw_{key}'].sum():>5,} docs")


if __name__ == "__main__":
    main()
