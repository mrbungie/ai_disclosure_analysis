"""Corporate branding versus external-vendor attribution
(thesis.qmd `fig-tech-provenance` and its lead-in prose, ~2568-2620).

Reads the activity grain of gold (spines/activity/activity joined with
covariates/activity/{extraction, taxonomy}: one row per disclosed AI activity
instance, with the `entities` list of {name, role} dicts already resolved).
No LLM/model call: this only tabulates the named entity roles already
produced by scripts/gold/activity/build_activity.py.

Writes data/results/washing/tech_provenance.parquet, one row with:
  n_activities_total, own_brand_pct, ext_vendor_pct, partner_pct,
  third_party_pct (source == "third_party") and ext_provider_firm_pct.

Attribution is ONE construct built from two detectors that see the same
statement. The extraction tags each named entity with a role (own_brand,
provider, model, partner); the vendor lexicon matches commercial names in the
statement's text and resolves them to a provider family
(covariates/activity/ai_vendors). A statement names an external provider when
either detector finds one, and names its own brand when either the extraction
tags an own_brand entity or the lexicon matches a product of the filer itself.
Neither detector alone is the measure: the extraction catches in-house product
names no lexicon can enumerate, the lexicon catches suppliers the extraction
left untagged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

ECOSYSTEMS = {"United States": "us", "China": "cn", "Europe": "eu", "Other": "other"}


def check_role(entities, roles):
    if entities is None or len(entities) == 0:
        return False
    return any(e.get("role") in roles for e in entities)


def main() -> None:
    df = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))
    n_activities_total = len(df)

    vendors = L.read_gold("activity", ("covariates", "ai_vendors"))
    df = df.merge(vendors[["id", "vendor_any", "vendor_self"] + [f"vendor_{k}" for k in ECOSYSTEMS.values()]],
                  on="id", how="left")
    df[["vendor_any", "vendor_self"]] = df[["vendor_any", "vendor_self"]].fillna(False)

    tagged_own = df["entities"].apply(lambda l: check_role(l, ["own_brand"]))
    tagged_ext = df["entities"].apply(lambda l: check_role(l, ["provider", "model"]))
    tagged_partner = df["entities"].apply(lambda l: check_role(l, ["partner"]))

    own_brand = tagged_own | df["vendor_self"]
    ext_vendor = tagged_ext | df["vendor_any"]
    own_brand_pct = own_brand.mean() * 100
    ext_vendor_pct = ext_vendor.mean() * 100
    partner_pct = tagged_partner.mean() * 100
    third_party_pct = float((df["source"] == "third_party").mean() * 100)
    ext_provider_firm_pct = 100 * (ext_vendor | tagged_partner).groupby(df["ticker"]).any().mean()

    row = {
        "n_activities_total": int(n_activities_total),
        "own_brand_pct": float(own_brand_pct),
        "ext_vendor_pct": float(ext_vendor_pct),
        "partner_pct": float(partner_pct),
        "third_party_pct": third_party_pct,
        "ext_provider_firm_pct": float(ext_provider_firm_pct),
    }
    row["eco_any_pct"] = float(df["vendor_any"].mean() * 100)
    for ecosystem, key in ECOSYSTEMS.items():
        row[f"eco_{key}_pct"] = float(df[f"vendor_{key}"].fillna(False).mean() * 100)
    # The components behind the union, for the figure note.
    row["ext_vendor_tagged_pct"] = float(tagged_ext.mean() * 100)
    row["ext_vendor_lexicon_pct"] = float(df["vendor_any"].mean() * 100)

    # The external side as one indicator: a named vendor or model, a named
    # commercial partner, or the statement declaring the technology itself as
    # externally sourced. The two indicators overlap -- a statement can name its
    # own brand and its supplier -- so they are not a partition.
    external = ext_vendor | tagged_partner | (df["source"] == "third_party")
    row["external_pct"] = float(external.mean() * 100)
    # How the two sides meet: a statement crediting a source alongside its own
    # brand, and one crediting a source without claiming a brand of its own.
    row["both_pct"] = float((own_brand & external).mean() * 100)
    row["external_only_pct"] = float((external & ~own_brand).mean() * 100)

    out_df = pd.DataFrame([row])
    out = L.results_path("washing", "tech_provenance.parquet")
    out_df.to_parquet(out, index=False)
    print(row)
    print(f"-> {out} ({len(out_df):,} row)")


if __name__ == "__main__":
    main()
