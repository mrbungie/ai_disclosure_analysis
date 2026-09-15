"""Sectoral business purpose and operational action of disclosed AI
activity (thesis.qmd `fig-sector-purpose-action`, ~1915-2075): crosses the
10 aggregated economic sectors with (a) the six-way business-purpose
domain (Sell/Serve/Enable/Control/Build/Operate, row-normalized within
sector) and (b) the seven-way operational action taxonomy (deviation from
the pooled action mix, in percentage points). Sector-level, not
archetype-level -- a distinct cross-tab from archetype_domain.py /
archetype_activity.py, sharing only the domain/action taxonomies.

Reads the gold activity grain (covariates/activity/{extraction, taxonomy}) and
silver.firm_universe (sic -> sector). "Other / Diversified" sector is
excluded, matching the qmd.

Output: data/results/posture/sector_purpose_action.parquet, long format
with a `kind` column ("purpose_pct" or "action_deviation_pp") so both
panels of the figure live in one file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
from archetype_sector import map_sector  # noqa: E402

ROLES = ["Sell", "Operate", "Build", "Enable", "Control", "Serve"]
ACTIONS = ["Deploy / Integrate", "Develop / Build", "Partner / Buy", "Pilot / Explore",
          "Analyze / Measure", "Automate", "Govern / Control"]


def get_domain(row) -> str:
    f, obj, tgt = str(row["function_family"]), str(row["object_family"]), str(row["target"])
    if f in ("product_feature", "content_and_media") or "product" in obj or tgt == "customers":
        return "Sell"
    if f in ("customer_service", "marketing_and_sales"):
        return "Serve"
    if f in ("software_development", "hr_and_talent") or "copilot" in obj or "talent" in obj:
        return "Enable"
    if f in ("it_and_cybersecurity", "fraud_and_risk", "governance") or "security" in obj or "governance" in obj:
        return "Control"
    if f in ("ai_compute_and_models", "research_and_product_development", "data_and_analytics") or "compute" in obj or "analytics" in obj:
        return "Build"
    return "Operate"


def get_action(row) -> str:
    a, obj, stage = str(row["action"]), str(row["object_family"]), str(row["stage"])
    if stage in ("exploring", "piloting"):
        return "Pilot / Explore"
    if a in ("deploy", "scale", "integrate"):
        return "Automate" if "automation" in obj else "Deploy / Integrate"
    if a in ("develop", "invest"):
        return "Develop / Build"
    if a in ("partner", "procure"):
        return "Partner / Buy"
    if a == "measure":
        return "Analyze / Measure"
    if a in ("govern", "restrict"):
        return "Govern / Control"
    return "Deploy / Integrate"


def main() -> None:
    fu = L.scan("silver.firm_universe").filter(pl.col("country_code") == "us") \
        .select("ticker", "sic").collect().to_pandas()
    fu["sic2"] = fu["sic"].astype(str).str.zfill(4).str[:2]
    fu["sector"] = fu["sic2"].apply(map_sector)

    act = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy")).merge(fu[["ticker", "sector"]], on="ticker", how="left")
    act = act[act["sector"] != "Other / Diversified"].copy()
    act["role"] = act.apply(get_domain, axis=1)
    act["action_type"] = act.apply(get_action, axis=1)

    sector_order = act["sector"].value_counts().index.tolist()
    purpose_ct = pd.crosstab(act["sector"], act["role"]).reindex(index=sector_order, columns=ROLES).fillna(0)
    action_ct = pd.crosstab(act["sector"], act["action_type"]).reindex(index=sector_order, columns=ACTIONS).fillna(0)

    purpose_pct = purpose_ct.div(purpose_ct.sum(axis=1), axis=0) * 100
    action_pct_row = action_ct.div(action_ct.sum(axis=1), axis=0) * 100
    action_overall = act["action_type"].value_counts(normalize=True).reindex(ACTIONS) * 100
    action_dev = action_pct_row.sub(action_overall, axis=1)

    rows = []
    for sector in sector_order:
        for role in ROLES:
            rows.append({"kind": "purpose_pct", "sector": sector, "category": role,
                        "value": float(purpose_pct.loc[sector, role])})
        for action in ACTIONS:
            rows.append({"kind": "action_deviation_pp", "sector": sector, "category": action,
                        "value": float(action_dev.loc[sector, action])})
    out_df = pd.DataFrame(rows)
    out = L.results_path("posture", "sector_purpose_action.parquet")
    out_df.to_parquet(out, index=False)

    print("Deploy / Integrate deviation, Energy & Mining:", round(action_dev.loc["Energy & Mining", "Deploy / Integrate"], 1))
    print("Deploy / Integrate deviation, Industrials & Mfg:", round(action_dev.loc["Industrials & Mfg", "Deploy / Integrate"], 1))
    print("Pilot / Explore deviation, Utilities:", round(action_dev.loc["Utilities", "Pilot / Explore"], 1))
    print("Govern / Control deviation, Utilities:", round(action_dev.loc["Utilities", "Govern / Control"], 1))
    print(f"\n-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
