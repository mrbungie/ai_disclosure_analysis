"""Mix of disclosed AI activities by business-purpose domain, within each
posture archetype (thesis.qmd `fig-archetype-activity-mix`, ~2211-2280):
every disclosed activity is classified into one of six domains by its
function/object/target (Sell, Serve, Enable, Control, Build, Operate), and
the mix is normalized within each archetype to a 100% stacked composition.

Reads the gold activity grain (spines/activity/activity with
covariates/activity/{extraction, taxonomy}) joined with the gold firm
archetype (covariates/firm/posture_archetype_static.parquet, "No AI" for
firms below the archetype fit's frame floor). Writes
data/results/posture/archetype_domain.parquet: one row per
(archetype, domain) with n_activities and pct_of_archetype (the qmd's
`ct_mix`, column name "domain" instead of "role" to match the naming this
migration uses elsewhere for the same six-way classification).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

ARCH = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]
DOMAIN_ORDER = ["Sell (Customer products)", "Operate (Internal operations)", "Build (Compute & data)",
               "Enable (Productivity & dev)", "Control (Risk & security)", "Serve (Sales & service)"]


def get_domain(row) -> str:
    f, obj, tgt = str(row["function_family"]), str(row["object_family"]), str(row["target"])
    if f in ("product_feature", "content_and_media") or "product" in obj or tgt == "customers":
        return "Sell (Customer products)"
    if f in ("customer_service", "marketing_and_sales"):
        return "Serve (Sales & service)"
    if f in ("software_development", "hr_and_talent") or "copilot" in obj or "talent" in obj:
        return "Enable (Productivity & dev)"
    if f in ("it_and_cybersecurity", "fraud_and_risk", "governance") or "security" in obj or "governance" in obj:
        return "Control (Risk & security)"
    if f in ("ai_compute_and_models", "research_and_product_development", "data_and_analytics") or "compute" in obj or "analytics" in obj:
        return "Build (Compute & data)"
    return "Operate (Internal operations)"


def build_act() -> pd.DataFrame:
    """The (activities x archetype) join every chunk in this cluster reuses."""
    act = (L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"), columns=["firm__archetype"])
           .rename(columns={"firm__archetype": "archetype"}))
    act["archetype"] = act["archetype"].fillna("No AI")
    return act


def main() -> None:
    act = build_act()
    act["domain"] = act.apply(get_domain, axis=1)
    ct_mix = pd.crosstab(act["archetype"], act["domain"], normalize="index") * 100
    ct_n = pd.crosstab(act["archetype"], act["domain"])

    rows = []
    for a in ct_mix.index:
        for d in DOMAIN_ORDER:
            rows.append({"archetype": a, "domain": d, "n_activities": int(ct_n.loc[a, d]),
                        "pct_of_archetype": float(ct_mix.loc[a, d])})
    out_df = pd.DataFrame(rows)
    out = L.results_path("posture", "archetype_domain.parquet")
    out_df.to_parquet(out, index=False)
    print(pd.DataFrame(rows).pivot(index="archetype", columns="domain", values="pct_of_archetype")[DOMAIN_ORDER].loc[ARCH + ["No AI"]].round(1).to_string())
    print(f"\n-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
