"""Corporate branding versus external-vendor attribution
(thesis.qmd `fig-tech-provenance` and its lead-in prose, ~2568-2620).

Reads the activity grain of gold (spines/activity/activity joined with
covariates/activity/{extraction, taxonomy}: one row per disclosed AI activity
instance, with the `entities` list of {name, role} dicts already resolved).
No LLM/model call: this only tabulates the named entity roles already
produced by scripts/gold/activity/build_activity.py.

Writes data/results/washing/tech_provenance.parquet, one row with:
  n_activities_total, own_brand_pct, ext_vendor_pct, partner_pct,
  third_party_pct (source == "third_party", the fourth bar in
  fig-tech-provenance), ext_provider_firm_pct, and one <vendor>_pct column
  per hyperscaler/vendor ecosystem pattern (share of S&P 500 firms naming
  that ecosystem anywhere in their disclosures).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

VENDOR_PATTERNS = {
    "openai_microsoft": r"openai|chatgpt|gpt|copilot|microsoft|azure",
    "alphabet_google": r"google|gemini|vertex|bard|deepmind|palm",
    "nvidia": r"nvidia|cuda",
    "aws": r"amazon|aws|bedrock|sagemaker",
}


def check_role(entities, roles):
    if entities is None or len(entities) == 0:
        return False
    return any(e.get("role") in roles for e in entities)


def main() -> None:
    df = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))
    n_activities_total = len(df)

    own_brand_pct = df["entities"].apply(lambda l: check_role(l, ["own_brand"])).mean() * 100
    ext_vendor_pct = df["entities"].apply(lambda l: check_role(l, ["provider", "model"])).mean() * 100
    partner_pct = df["entities"].apply(lambda l: check_role(l, ["partner"])).mean() * 100
    third_party_pct = float((df["source"] == "third_party").mean() * 100)
    any_ext_provider = df.groupby("ticker")["entities"].apply(
        lambda s: s.apply(lambda l: check_role(l, ["provider", "model", "partner"])).any())
    ext_provider_firm_pct = 100 * any_ext_provider.mean()

    def _vendor_firm_pct(pattern):
        def has_vendor(entities):
            if entities is None or len(entities) == 0:
                return False
            return any(e.get("role") in ("provider", "model", "partner")
                      and re.search(pattern, str(e.get("name", "")).lower()) for e in entities)
        by_firm = df.groupby("ticker")["entities"].apply(lambda s: s.apply(has_vendor).any())
        return 100 * by_firm.mean()

    row = {
        "n_activities_total": int(n_activities_total),
        "own_brand_pct": float(own_brand_pct),
        "ext_vendor_pct": float(ext_vendor_pct),
        "partner_pct": float(partner_pct),
        "third_party_pct": third_party_pct,
        "ext_provider_firm_pct": float(ext_provider_firm_pct),
    }
    for name, pat in VENDOR_PATTERNS.items():
        row[f"{name}_pct"] = float(_vendor_firm_pct(pat))

    out_df = pd.DataFrame([row])
    out = L.results_path("washing", "tech_provenance.parquet")
    out_df.to_parquet(out, index=False)
    print(row)
    print(f"-> {out} ({len(out_df):,} row)")


if __name__ == "__main__":
    main()
