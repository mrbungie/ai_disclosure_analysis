"""
14_governance_sensitivity.py — BoW-only vs LLM-only D5 Governance sensitivity check

Motivation: the val_01/val_02 validation against an independent LLM judge found
is_governance_related recall = 0.333 (the pipeline's combined BoW+LLM proxy misses
2 of every 3 governance mentions the judge flags). D5 Governance Maturity and the
AI Washing Index's Hype Score are the two places in the analysis most sensitive to
that kind of systematic miss, since an undercounted D5 could make governance-light
firms look artificially credible, or bias which clusters look like "washing".

This script does NOT re-run the LLM classifier. `firm_year_features.parquet`
already carries both signals independently:
  - the BoW proxy shares (share_board_oversight, share_audit_committee,
    share_ethics_policy, ...) that D5 is built from in script 11
  - share_llm_is_governance, the pure-LLM governance share (script 08 output,
    aggregated in script 10), untouched by the BoW proxy logic

It recomputes D5 using each source alone, reruns the programmatic archetype
classifier (from 11_cluster_archetypes.py) with everything else held fixed, and
reports what fraction of firm-years change archetype and which clusters gain or
lose members. Large, cluster-concentrated churn would mean the AI Washing
narrative depends on which governance signal is trusted; small/diffuse churn
means the D5-recall gap is not currently load-bearing for the clusters.

Usage:
    uv run python scripts/14_governance_sensitivity.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import pipeline_logger
    import variant_utils
except ImportError:
    from scripts import pipeline_logger
    from scripts import variant_utils

# Must stay in sync with scripts/11_cluster_archetypes.py — duplicated here
# rather than imported because numbered script filenames aren't importable
# modules.
THRESHOLDS = {
    'd5_governance_min':        0.10,
    'd1_intensity_min':         0.35,
    'd3_technical_min':         0.08,
    'd2_operational_min':       0.15,
    'd9_substantive_min_fsp':   0.30,
    'd9_substantive_min_oaa':   0.50,
}

NEW_CLUSTER_NAMES = {
    0: 'Non-AI Disclosers',
    1: 'Governance & Compliance-Focused',
    2: 'Boilerplate Risk-Warners',
    3: 'Full-Stack AI Pioneers',
    4: 'Operational Application Adopters',
}

D5_BOW_COLS = [
    'share_board_oversight', 'share_audit_committee', 'share_ethics_policy', 'share_compliance',
    'share_eu_regulation', 'share_us_regulation', 'share_word_board_oversight', 'share_word_audit_committee',
    'share_word_responsible_ai', 'share_word_ethical_ai', 'share_word_gdpr', 'share_word_ccpa',
    'share_word_eu_ai_act', 'share_word_executive_order',
]

NON_D5_DIM_DEFS = {
    'D1 AI Intensity': (['ai_mentions_count', 'share_word_ai', 'share_word_artificial_intelligence',
                          'share_word_machine_learning'], True),
    'D2 Operational': (['share_deployment_verb', 'share_specific_product', 'share_customer_facing',
                         'share_product_integration', 'share_internal_productivity', 'share_safety_critical',
                         'share_word_automate_tasks', 'share_word_customer_facing', 'share_word_back_office',
                         'share_word_internal_operations'], False),
    'D3 Technical': (['share_model_training', 'share_compute_infra', 'share_proprietary_data',
                       'share_academic_research', 'share_gen_ai_mention', 'share_classical_ml_mention',
                       'share_word_gpu', 'share_word_llm', 'share_word_deep_learning',
                       'share_word_generative_ai', 'share_word_large_language_model'], False),
    'D9 Substantive-section': (['sec_pct_business', 'sec_pct_mda'], False),
}


def load_config():
    with open("configs/config.json") as f:
        return json.load(f)


def make_dim(df, cols, log=False):
    avail = [df[c] for c in cols if c in df.columns]
    if not avail:
        return pd.Series(0.0, index=df.index)
    raw = pd.concat(avail, axis=1).mean(axis=1)
    if log:
        raw = np.log1p(raw)
    mm = MinMaxScaler()
    return pd.Series(mm.fit_transform(raw.values.reshape(-1, 1)).flatten(), index=df.index)


def classify_row(row, thresholds):
    if row['ai_mentions_count'] == 0:
        return 0
    if row['D5 Governance'] >= thresholds['d5_governance_min']:
        return 1
    if (row['D1 AI Intensity'] >= thresholds['d1_intensity_min']
            and (row['D3 Technical'] >= thresholds['d3_technical_min']
                 or row['D2 Operational'] >= thresholds['d2_operational_min'])
            and row['D9 Substantive-section'] >= thresholds['d9_substantive_min_fsp']):
        return 3
    if row['D9 Substantive-section'] >= thresholds['d9_substantive_min_oaa']:
        return 4
    return 2


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="D5 Governance sensitivity check: BoW-only vs LLM-only"
    )
    variant_utils.add_variant_arg(parser)
    args = parser.parse_args()

    config = load_config()
    variant = variant_utils.resolve_variant(args.variant, config)
    output_root = config.get("variants", {}).get("output_root", "data/processed")

    # Note: share_llm_is_governance (the pure-LLM D5 signal) only carries real
    # signal when the panel was built from a run where script 08's LLM
    # classifier covered the corpus. Under rule_based, script 09 never merges
    # the LLM mentions, so share_llm_is_governance is ~0 for every firm-year
    # and this comparison degenerates to "BoW vs ~empty". Run with
    # --variant llm_full for a comparison where both sources carry signal.
    features_path = variant_utils.variant_path(variant, "firm_year_features", "parquet", output_root=output_root)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    if not features_path.exists():
        print(f"Error: {features_path} not found. Run scripts 07-10 first.")
        return

    df = pd.read_parquet(features_path)
    for col in ['sec_pct_business', 'sec_pct_mda', 'sec_pct_risk_factors', 'sec_pct_other']:
        if col not in df.columns:
            df[col] = 0.0
    if 'share_llm_is_governance' not in df.columns:
        print("Error: share_llm_is_governance not found in firm_year_features.parquet — "
              "run script 08 (LLM classifier) and rebuild the panel before this check.")
        return

    shared = pd.DataFrame({name: make_dim(df, cols, log) for name, (cols, log) in NON_D5_DIM_DEFS.items()})
    d5_bow = make_dim(df, D5_BOW_COLS, log=False)
    d5_llm = make_dim(df, ['share_llm_is_governance'], log=False)

    base = shared.copy()
    base['ai_mentions_count'] = df['ai_mentions_count']

    bow_variant = base.copy()
    bow_variant['D5 Governance'] = d5_bow
    llm_variant = base.copy()
    llm_variant['D5 Governance'] = d5_llm

    cluster_bow = bow_variant.apply(lambda row: classify_row(row, THRESHOLDS), axis=1)
    cluster_llm = llm_variant.apply(lambda row: classify_row(row, THRESHOLDS), axis=1)

    pct_changed = float((cluster_bow != cluster_llm).mean() * 100)
    n_changed = int((cluster_bow != cluster_llm).sum())

    crosstab = pd.crosstab(
        cluster_bow.map(NEW_CLUSTER_NAMES), cluster_llm.map(NEW_CLUSTER_NAMES),
        rownames=["BoW-only D5"], colnames=["LLM-only D5"],
    )

    lines = [
        "D5 Governance Sensitivity: BoW-only vs LLM-only",
        "=" * 60,
        f"Panel: {len(df)} firm-years",
        f"Firm-years with changed archetype: {n_changed} ({pct_changed:.1f}%)",
        "",
        "Archetype cross-tab (rows=BoW-only D5, cols=LLM-only D5):",
        crosstab.to_string(),
        "",
    ]

    gov_bow = int((cluster_bow == 1).sum())
    gov_llm = int((cluster_llm == 1).sum())
    lines.append(
        f"'Governance & Compliance-Focused' membership: {gov_bow} firm-years (BoW-only D5) "
        f"vs {gov_llm} firm-years (LLM-only D5)"
    )
    if gov_bow > 0:
        rel_diff = abs(gov_bow - gov_llm) / gov_bow * 100
        lines.append(f"Relative difference in cluster size: {rel_diff:.1f}%")

    lines.append("")
    if pct_changed >= 10.0:
        lines.append(
            "WARNING: >=10% of firm-years change archetype depending on which governance "
            "signal (BoW proxy vs LLM) is used. The AI Washing narrative should not rely on "
            "a single governance source without reporting this sensitivity explicitly, or "
            "improving the governance LLM prompt/recall before finalizing cluster assignments."
        )
    else:
        lines.append(
            "Archetype assignment is materially stable across BoW-only vs LLM-only D5: "
            "the known governance-dimension recall gap does not appear to be load-bearing "
            "for cluster membership in this panel."
        )

    summary = "\n".join(lines)
    out_path = reports_dir / f"governance_sensitivity__{variant}.txt"
    out_path.write_text(summary)
    print(summary)
    print(f"\nSaved → {out_path}")

    pipeline_logger.log_event(
        pipeline_step="governance_sensitivity",
        level="WARNING" if pct_changed >= 10.0 else "INFO",
        message=f"Governance sensitivity check: {pct_changed:.1f}% of firm-years change archetype "
                f"between BoW-only and LLM-only D5.",
        details={"variant": variant},
    )


if __name__ == "__main__":
    main()
