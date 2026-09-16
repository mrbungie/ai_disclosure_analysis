#!/usr/bin/env python3
"""Presence of named AI vendor ecosystems in corporate AI disclosure.

Descriptive report: reads the gold document-grain vendor variables and the
gold archetype prediction for the cross-tab, writes to `data/results/posture/`.
No aggregation another script depends on.

Reads:
  - data/gold/covariates/document/ai_vendors.parquet (ecosystem presence)
  - data/gold/covariates/document/ai_vendor_families.parquet (provider presence)
  - data/gold/covariates/firm/posture_archetype_static.parquet (archetype)
  - configs/ai_vendor_ecosystems.yaml (the ecosystem and type of each family)

Writes:
  - data/results/posture/vendor_ecosystems_summary.json

Construct: a document with AI disclosure "names" an ecosystem when at least
one of its AI-frame paragraphs mentions a commercial name belonging to that
ecosystem. Presence is binary per document, ecosystems overlap, and the
denominator is every document with at least one AI-frame paragraph in the
period. Self-references (a filer naming its own products) are excluded
upstream, in `build_ai_vendors.py`.

Rates do not sum to 100: they are four independent presence rates over the
same denominator, not shares of a partition.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

OUT_PATH = L.results_path("posture", "vendor_ecosystems_summary.json")


def column_name(family: str) -> str:
    """The family's column in covariates/document/ai_vendor_families."""
    return "vendor_" + re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_")


ECOSYSTEMS = {"United States": "us", "China": "cn", "Europe": "eu", "Other": "other"}
ARCHETYPES = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]
YEARS = range(2021, 2027)


def main() -> None:
    docs = L.read_gold("document", ("covariates", "ai_vendors"), ("covariates", "ai_vendor_families"))
    docs = docs[docs["n_ai_paragraphs"] > 0].copy()
    docs["year"] = pd.to_datetime(docs["fecha"]).dt.year
    docs = docs[docs["year"].between(min(YEARS), max(YEARS))]

    archetype = L.read_gold("firm", ("covariates", "posture_archetype_static", ["archetype"]))[["ticker", "archetype"]]
    docs = docs.merge(archetype, on="ticker", how="left")

    years = sorted(docs["year"].unique().tolist())
    panel_a = {
        "years": [int(y) for y in years],
        "n_documents": [int((docs["year"] == y).sum()) for y in years],
        "stack_rate": {eco: [float(docs.loc[docs["year"] == y, f"vendor_stack_{key}"].mean() * 100) for y in years]
                       for eco, key in ECOSYSTEMS.items()},
        "stack_documents": {eco: [int(docs.loc[docs["year"] == y, f"vendor_stack_{key}"].sum()) for y in years]
                            for eco, key in ECOSYSTEMS.items()},
        "entsw_rate": {eco: [float(docs.loc[docs["year"] == y, f"vendor_entsw_{key}"].mean() * 100) for y in years]
                       for eco, key in ECOSYSTEMS.items()},
    }

    panel_b = {
        "archetypes": ARCHETYPES,
        "stack_rate": {eco: [float(docs.loc[docs["archetype"] == a, f"vendor_stack_{key}"].mean() * 100)
                             for a in ARCHETYPES] for eco, key in ECOSYSTEMS.items()},
        "entsw_rate": {eco: [float(docs.loc[docs["archetype"] == a, f"vendor_entsw_{key}"].mean() * 100)
                             for a in ARCHETYPES] for eco, key in ECOSYSTEMS.items()},
        "n_documents": [int((docs["archetype"] == a).sum()) for a in ARCHETYPES],
    }

    # Firm-level reach: a firm names an ecosystem when any of its documents does.
    firm = docs.groupby("ticker")[[f"vendor_{c}_{k}" for c in ("stack", "entsw") for k in ECOSYSTEMS.values()]].any()
    n_firms = int(docs["ticker"].nunique())
    reach = {
        "n_firms_with_ai_disclosure": n_firms,
        "n_documents_with_ai_disclosure": int(len(docs)),
        "n_documents_naming_a_vendor": int((docs["n_vendor_paragraphs"] > 0).sum()),
        "firms_naming_a_vendor": int((docs.groupby("ticker")["n_vendor_paragraphs"].max() > 0).sum()),
        "stack_firms": {eco: int(firm[f"vendor_stack_{key}"].sum()) for eco, key in ECOSYSTEMS.items()},
        "stack_firm_pct": {eco: float(firm[f"vendor_stack_{key}"].mean() * 100) for eco, key in ECOSYSTEMS.items()},
        "entsw_firms": {eco: int(firm[f"vendor_entsw_{key}"].sum()) for eco, key in ECOSYSTEMS.items()},
        "entsw_firm_pct": {eco: float(firm[f"vendor_entsw_{key}"].mean() * 100) for eco, key in ECOSYSTEMS.items()},
        "self_reference_documents": int((docs["n_vendor_self_paragraphs"] > 0).sum()),
    }

    # Which provider each ecosystem's presence runs through: the wide family
    # table, one flag per family, with the lexicon supplying ecosystem and type.
    with open(REPO_ROOT / "configs" / "ai_vendor_ecosystems.yaml") as f:
        lexicon = {v["family"]: v for v in yaml.safe_load(f)["vendors"]}
    family_columns = {column_name(fam): fam for fam in lexicon if column_name(fam) in docs.columns}
    families = []
    for column, family in family_columns.items():
        firms_naming = docs.loc[docs[column], "ticker"].nunique()
        meta = lexicon[family]
        families.append({"family": family, "ecosystem": meta["ecosystem"], "country": meta["country"],
                         "type": meta["type"], "n_firms": int(firms_naming),
                         "pct_firms": float(firms_naming / n_firms * 100),
                         "n_documents": int(docs[column].sum())})
    families.sort(key=lambda f: -f["n_firms"])

    family_trend = {
        "years": [int(y) for y in years],
        "n_documents": [int((docs["year"] == y).sum()) for y in years],
        "rate": {f["family"]: [float(docs.loc[docs["year"] == y, column_name(f["family"])].mean() * 100)
                               for y in years] for f in families[:12]},
    }

    output = {"panel_a": panel_a, "panel_b": panel_b, "reach": reach,
              "families": families, "family_trend": family_trend}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Vendor ecosystem summary written to {OUT_PATH}")
    print(f"{reach['n_documents_with_ai_disclosure']:,} AI-disclosing documents | "
          f"{reach['firms_naming_a_vendor']:,} of {n_firms:,} firms name an external vendor")
    for eco in ECOSYSTEMS:
        print(f"  {eco:>14}: {reach['stack_firm_pct'][eco]:5.1f}% of firms (stack), "
              f"{reach['entsw_firm_pct'][eco]:5.1f}% (enterprise software)")
    for f in families[:10]:
        print(f"    {f['family']:>24} [{f['ecosystem']}] {f['pct_firms']:5.1f}% of firms")


if __name__ == "__main__":
    main()
