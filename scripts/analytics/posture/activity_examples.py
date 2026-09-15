"""Illustrative statutory filing excerpts across disclosure archetypes and
activity orientations (thesis.qmd `tbl-archetype-activity-examples`,
~2302-2350): one example activity per archetype (AMD/Sell/metric-evidence,
AIG/Control/named-evidence, Eli Lilly/generic-evidence), with its source
paragraph text.

Reads the gold activity grain (covariates/activity/{extraction, taxonomy},
joined with the gold firm archetype, same `domain` classification as
archetype_domain.py) and bronze.paragraphs for the quoted text. Writes
data/results/posture/activity_examples.parquet with lineage keys
(accession_number, item_key, paragraph_index, text_hash) alongside the
quote, so the excerpt can be traced back to the filing. Each excerpt is
pinned by its activity key (text_hash, activity_id), since its diagnostic is
written for that paragraph; the domain/evidence filters check that the
pinned activity still carries the classification it illustrates.
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
from archetype_domain import build_act, get_domain  # noqa: E402


def clean(text: str) -> str:
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    t = re.sub(r"\(see page[s]?\s*[^)]*\)", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return re.sub(r"\s+([.,;:])", r"\1", t).strip()


def quote(act: pd.DataFrame, ticker: str, paragraphs: pl.DataFrame, **filters) -> dict | None:
    sub = act[act["ticker"] == ticker]
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    if len(sub) != 1:
        return None
    r = sub.iloc[0]
    hit = paragraphs.filter((pl.col("accession_number") == r["accession_number"])
                            & (pl.col("paragraph_index") == int(r["paragraph_index"])))
    text = clean(hit["paragraph_text"][0]) if hit.height else ""
    return {"ticker": ticker, "accession_number": r["accession_number"], "item_key": r["item_key"],
           "paragraph_index": int(r["paragraph_index"]), "text_hash": int(r["text_hash"]),
           "archetype": r["archetype"], "domain": r.get("domain"), "action": r["action"],
           "object": r["object"], "evidence_type": r["evidence_type"], "quote": text}


def main() -> None:
    act = build_act()
    act["domain"] = act.apply(get_domain, axis=1)
    paragraphs = (L.scan("bronze.paragraphs").select("accession_number", "paragraph_index", "paragraph_text")
                 .collect())

    examples = []
    ex = quote(act, "AMD", paragraphs, text_hash=11057980077568252630, activity_id=0,
               domain="Sell (Customer products)", evidence_type="metric")
    if ex:
        ex["diagnostic"] = "Specific and verifiable: names the acquisition, product line, and deployment targets."
        examples.append(ex)
    ex = quote(act, "AIG", paragraphs, text_hash=11101146028916835566, activity_id=0,
               domain="Control (Risk & security)", evidence_type="named")
    if ex:
        ex["diagnostic"] = "Institutional oversight language rather than technical or operational deployment."
        examples.append(ex)
    ex = quote(act, "LLY", paragraphs, text_hash=1336536157000345163, activity_id=0, evidence_type="generic")
    if ex:
        ex["diagnostic"] = "Vague and defensive: no named products, operational stages, or concrete metrics."
        examples.append(ex)

    out_df = pd.DataFrame(examples)
    out = L.results_path("posture", "activity_examples.parquet")
    out_df.to_parquet(out, index=False)
    for e in examples:
        print(f"[{e['archetype']}] {e['ticker']} ({e['domain']}): {e['quote'][:180]}")
    print(f"\n-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
