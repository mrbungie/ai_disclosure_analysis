"""Disclosed-AI-activity profile across posture archetypes (thesis.qmd
`fig-archetype-activity-heatmap`, ~2088-2161): five behavioral rate
features (proprietary AI, own brand named, customer-facing, deployed or
scaled, quantified metric), each archetype's raw rate and its deviation
from the pooled (whole-inventory) baseline, plus the median disclosed
activity count per firm relative to the pooled median.

Reads the gold activity grain (covariates/activity/{extraction, taxonomy})
joined with the gold firm archetype
(covariates/firm/posture_archetype_static), same `act`/`get_domain` pattern as
archetype_domain.py (duplicated here rather than imported, since the two
scripts' outputs are independent files with their own verification and
this keeps each runnable standalone).

Output: data/results/posture/archetype_activity.parquet, one row per
(archetype, metric) with pooled_rate, archetype_rate, deviation_pp
(percentage points from pooled), plus a second table's worth of rows
(metric="n_activities_median_ratio") giving each archetype's median
per-firm activity count as a ratio to the pooled median.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

ARCH = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]
RATE_METRICS = [
    ("Proprietary AI share", "proprietary_ai"),
    ("Own brand named", "own_brand_named"),
    ("Customer-facing share", "customer_target"),
    ("Deployed / scaled", "deployed_or_scaled"),
    ("Quantified metric", "quantified_outcome"),
]


def main() -> None:
    pred = L.read_gold("firm", ("covariates", "posture_archetype_static", ["archetype"]))[["ticker", "archetype"]]
    act = (L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"), columns=["firm__archetype"])
           .rename(columns={"firm__archetype": "archetype"}))
    act["archetype"] = act["archetype"].fillna("No AI")

    act["proprietary_ai"] = (act["action"] == "develop") | (act["source"] == "own")
    act["own_brand_named"] = act["own_brands"].map(lambda v: len(v) > 0)
    act["quantified_outcome"] = (act["action"] == "measure") | (act["evidence_type"] == "metric")
    act["deployed_or_scaled"] = act["stage"].isin(["deployed", "scaled"])
    act["customer_target"] = act["target"] == "customers"

    n_activities = act.groupby("ticker").size()
    per_firm = pred.set_index("ticker")["archetype"].to_frame().join(n_activities.rename("n_activities")).fillna(0)
    pooled_median_n = per_firm["n_activities"].median()

    rate_cols = [c for _, c in RATE_METRICS]
    pooled_rates = {c: act[c].mean() * 100 for c in rate_cols}

    rows = []
    for label, col in RATE_METRICS:
        for a in ARCH:
            v = act[act.archetype == a][col].mean() * 100
            rows.append({"metric": label, "archetype": a, "pooled_rate": pooled_rates[col],
                        "archetype_rate": v, "deviation_pp": v - pooled_rates[col]})
    for a in ARCH:
        med = per_firm[per_firm.archetype == a]["n_activities"].median()
        rows.append({"metric": "n_activities_median", "archetype": a, "pooled_rate": pooled_median_n,
                    "archetype_rate": med, "deviation_pp": med / pooled_median_n if pooled_median_n else float("nan")})

    out_df = pd.DataFrame(rows)
    out = L.results_path("posture", "archetype_activity.parquet")
    out_df.to_parquet(out, index=False)
    print(out_df.to_string())
    print(f"\n-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
