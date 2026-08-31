"""
tag_harness_defs.py — Cycle 2 shared definitions: the six classification
dimensions, the per-dimension atom pools the formula search may draw from,
derived meta-atoms, legacy production formulas (kept as baseline candidates),
and apply_formula_spec() so a frozen JSON spec can be re-applied to any
feature dataframe (11 fits, 12 applies — same code path).

The six dimensions answer the thesis proposal's disclosure facets directly:
substance, promotion, risk, governance, use-case specificity, quantification.
"""

import numpy as np
import pandas as pd

DIMENSIONS = [
    "is_substantive",
    "is_promotional",
    "is_risk_related",
    "is_governance_related",
    "is_use_case_specific",
    "is_quantified",
]

# REFERENCE LIBRARY of known-relevant atoms per dimension — the menu the
# meta-optimizer draws from when growing a pool. The ACTIVE pool each fit run
# actually searches lives in configs/config.json (tagging.atom_pools), starts
# deliberately minimal, and every change to it is logged in
# docs/journals/harness2_classification.md.
DIMENSION_FEATURE_POOLS: dict[str, list[str]] = {
    "is_substantive": [
        "has_model_training", "has_workforce_talent", "has_ai_hedge", "has_specific_product",
        "has_proprietary_data", "has_compute_infra", "has_word_ai", "has_word_gpu", "has_word_gpus",
        "has_realized_language", "has_deployment_verb", "has_ai_demand_context", "has_ai_quantified_claim",
        "has_ethics_policy", "has_risk_factor", "has_classical_ml_mention", "has_internal_productivity",
        "has_customer_facing", "has_ai_own_use", "has_competitor_mention", "has_classical_ml_operational",
        "has_ai_acquisition", "has_named_deployment", "has_dated_milestone", "has_ai_use_case_specific",
        "has_product_integration", "has_academic_research", "has_open_source",
    ],
    "is_promotional": [
        "has_word_revolutionize", "has_word_revolutionizing", "has_word_revolutionized", "has_word_transform",
        "has_word_transforming", "has_word_transformative", "has_word_cutting_edge", "has_word_next_generation",
        "has_word_leader", "has_word_leadership", "has_word_empower", "has_word_empowering",
        "has_risk_factor", "has_deployment_verb", "has_specific_product", "has_competitor_ai_mention",
        "has_forward_looking", "has_realized_language",
    ],
    "is_risk_related": [
        "has_risk_factor", "has_regulatory_risk", "has_ethics_bias_risk", "has_supply_infra_risk",
        "has_data_licensing", "has_cyber_privacy_risk", "has_word_ai", "has_ip_copyright_risk",
        "has_labor_displacement_risk", "has_safety_critical", "has_competitor_mention", "has_ai_hedge",
    ],
    "is_governance_related": [
        "has_board_oversight", "has_audit_committee", "has_ethics_policy", "has_compliance", "has_word_ai",
        "has_us_regulation", "has_eu_regulation", "has_risk_factor",
    ],
    "is_use_case_specific": [
        "has_ai_use_case_specific", "has_specific_product", "has_named_deployment", "has_ai_own_use",
        "has_product_integration", "has_customer_facing", "has_deployment_verb",
        "has_classical_ml_operational", "has_dated_milestone", "has_internal_productivity",
        "has_ai_model_name", "has_gen_ai_mention",
    ],
    "is_quantified": [
        "has_ai_quantified_claim", "has_metric_percentage", "has_metric_dollar",
        "has_financial_quantification", "has_dated_milestone", "has_capex_mention",
        "has_opex_mention", "has_rd_mention", "has_revenue_impact",
    ],
}


def build_meta_atoms(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Derived (non-has_*) boolean atoms mirroring the ones the legacy
    production formulas used, so the search space includes the formulas that
    actually shipped, not just raw presence flags."""
    return {
        "vague_gt_negative": df["count_vague_words"] > df["count_negative_words"],
        "sentiment_nonneg": df["bow_sentiment_score"] >= 0,
        "has_metrics": df["has_metric_percentage"] | df["has_metric_dollar"],
    }


def legacy_formula(df: pd.DataFrame, dimension: str) -> pd.Series | None:
    """The pre-strip production formulas (old script 09's build_proxy_sql),
    kept as baseline candidates for the four original dimensions — the search
    either beats them or confirms them. The two new dimensions have no legacy
    formula."""
    if dimension == "is_substantive":
        return (
            (df["has_model_training"] & ~df["has_workforce_talent"] & ~df["has_ai_hedge"])
            | df["has_specific_product"]
            | df["has_proprietary_data"]
            | (df["has_compute_infra"] & df["has_word_ai"]
               & (df["has_word_gpu"] | df["has_word_gpus"] | df["has_realized_language"] | df["has_deployment_verb"])
               & ~df["has_ai_demand_context"])
            | (df["has_compute_infra"] & df["has_word_ai"] & df["has_ai_quantified_claim"] & ~df["has_ai_demand_context"])
            | (df["has_ethics_policy"] & df["has_word_ai"] & ~df["has_risk_factor"])
            | (df["has_classical_ml_mention"] & (df["has_internal_productivity"] | df["has_customer_facing"]) & ~df["has_workforce_talent"])
            | (df["has_ai_own_use"] & ~df["has_ai_hedge"] & ~df["has_competitor_mention"] & ~df["has_ai_demand_context"])
            | (df["has_classical_ml_operational"] & ~df["has_risk_factor"])
            | df["has_ai_acquisition"]
        )
    if dimension == "is_governance_related":
        return (
            df["has_board_oversight"] | df["has_audit_committee"]
            | (df["has_ethics_policy"] & df["has_word_ai"])
            | (df["has_compliance"] & df["has_word_ai"])
        )
    if dimension == "is_risk_related":
        return (
            df["has_risk_factor"] | df["has_regulatory_risk"] | df["has_ethics_bias_risk"]
            | df["has_supply_infra_risk"] | df["has_data_licensing"]
            | (df["has_cyber_privacy_risk"] & df["has_word_ai"])
            | (df["has_ip_copyright_risk"] & df["has_word_ai"])
        )
    if dimension == "is_promotional":
        meta = build_meta_atoms(df)
        return meta["vague_gt_negative"] & ~df["has_risk_factor"] & meta["sentiment_nonneg"]
    return None


def _atom(df: pd.DataFrame, name: str) -> np.ndarray:
    meta = build_meta_atoms(df)
    series = meta[name] if name in meta else df[name]
    return series.astype(bool).to_numpy()


def apply_formula_spec(spec: dict, df: pd.DataFrame) -> np.ndarray:
    """Re-apply a frozen JSON formula spec to any dataframe holding the atom
    columns. Specs are what 11 freezes into config and what 12 applies to the
    corpus — one code path for both."""
    op = spec["op"]
    if op == "single":
        return _atom(df, spec["feature"])
    if op == "not":
        return ~_atom(df, spec["feature"])
    if op == "and":
        a, b = spec["features"]
        return _atom(df, a) & _atom(df, b)
    if op == "or":
        a, b = spec["features"]
        return _atom(df, a) | _atom(df, b)
    if op == "and_not":
        a, b = spec["features"]
        return _atom(df, a) & ~_atom(df, b)
    if op == "legacy":
        formula = legacy_formula(df, spec["dimension"])
        if formula is None:
            raise ValueError(f"no legacy formula for {spec['dimension']}")
        return formula.astype(bool).to_numpy()
    raise ValueError(f"unknown formula op: {op}")
