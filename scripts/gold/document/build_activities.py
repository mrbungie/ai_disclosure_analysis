"""Disclosed AI activities counted per document.

For every document of the document spine, the number of activity instances
(an activity of the activity spine carried by a text of the document) in each
activity family and grounding marker (`build_activity.flags`), plus
`n_activities`. A document without activity instances has zero counts.
Firm-year, firm-quarter and channel aggregates are sums of this table over
the document spine's ticker, dates, fiscal year and channel.

Output: covariates/document/activities.

Deterministic, no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "activity"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from build_activity import ACTIVITY_FAMILIES, GROUNDING, flags, instance_documents  # noqa: E402

import layers as L  # noqa: E402

BUILDER = "scripts/gold/document/build_activities.py"
COUNTS = ACTIVITY_FAMILIES + ["named_product_or_process", "named_function", "deployed_or_scaled"]
assert set(COUNTS) == set(ACTIVITY_FAMILIES + GROUNDING)


def main() -> None:
    acts = L.read_gold("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))
    # one attribute row per judged text x activity index (identical across tickers)
    acts = acts.drop_duplicates(["text_hash", "activity_id"])
    instances = acts.drop(columns=["ticker", "fecha", "accession_number", "channel"]).merge(
        instance_documents()[["text_hash", "accession_number"]], on="text_hash", how="inner")
    fl = flags(instances)
    for c in COUNTS:
        instances[c] = fl[c].astype(float)
    instances["n_activities"] = 1.0
    per_doc = instances.groupby("accession_number")[COUNTS + ["n_activities"]].sum().reset_index()

    docs = L.read_gold("document")[L.GOLD_SPINE_COLUMNS["document"]]
    out = docs.merge(per_doc, on="accession_number", how="left", validate="one_to_one")
    out[COUNTS + ["n_activities"]] = out[COUNTS + ["n_activities"]].fillna(0.0)
    L.write_gold("covariates", "document", "activities", out, builder=BUILDER)
    print(f"{int(out['n_activities'].sum()):,} activity instances over {int((out['n_activities'] > 0).sum()):,} documents")


if __name__ == "__main__":
    main()
