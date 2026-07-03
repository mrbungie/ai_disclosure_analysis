import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
import hdbscan
import umap

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

SEED = 42
np.random.seed(SEED)

DIM_LABELS = {
    'd1_intensity':           'D1 AI Intensity',
    'd2_operational':         'D2 Operational',
    'd3_technical':           'D3 Technical',
    'd4_quantification':      'D4 Quantification',
    'd5_governance':          'D5 Governance',
    'd6_risk':                'D6 Risk Depth',
    'd7_promotional':         'D7 Promotional',
    'd8_defensive':           'D8 Defensive',
    'd9_risk_section':        'D9 Risk-section',
    'd9_substantive_section': 'D9 Substantive-section',
}

NEW_CLUSTER_NAMES = {
    0: 'Non-AI Disclosers',
    1: 'Governance & Compliance-Focused',
    2: 'Boilerplate Risk-Warners',
    3: 'Full-Stack AI Pioneers',
    4: 'Operational Application Adopters'
}

def load_config():
    config_path = Path("configs/config.json")
    if not config_path.exists():
        config_path = Path("../configs/config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def make_dim(df, label, cols, log=False):
    avail = [df[c] for c in cols if c in df.columns]
    if not avail:
        print(f"  {label}: NO features found")
        return pd.Series(0.0, index=df.index)
    raw = pd.concat(avail, axis=1).mean(axis=1)
    if log:
        raw = np.log1p(raw)
    mm = MinMaxScaler()
    scaled = pd.Series(mm.fit_transform(raw.values.reshape(-1,1)).flatten(), index=df.index)
    return scaled

def main():
    parser = argparse.ArgumentParser(description="AI Disclosure Archetypes Clustering")
    parser.add_argument("--method", type=str, choices=["programmatic", "unsupervised"], default="programmatic",
                        help="Clustering method to use (default: programmatic)")
    args = parser.parse_args()

    config = load_config()
    features_path = Path("data/processed/features/firm_year_features.parquet")
    output_path = Path("data/processed/clusters/firm_year_clusters.parquet")

    if not features_path.exists():
        pipeline_logger.log_event(
            pipeline_step="cluster_archetypes",
            level="ERROR",
            message=f"Features file not found at {features_path}"
        )
        print(f"Error: Features file not found at {features_path}")
        return

    pipeline_logger.log_event(
        pipeline_step="cluster_archetypes",
        level="INFO",
        message=f"Loading features and computing dimensions for method: {args.method}"
    )

    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df)} rows from {features_path}")

    # Add section location features if needed (from candidate chunks)
    # Note: 10_build_features.py already aggregates and includes sec_pct_* cols.
    # In case any are missing, fill with 0.0.
    for col in ['sec_pct_business', 'sec_pct_mda', 'sec_pct_risk_factors', 'sec_pct_other']:
        if col not in df.columns:
            df[col] = 0.0

    # Define dimensions config
    dim_defs = {
        'd1_intensity': (['ai_mentions_count', 'share_word_ai', 'share_word_artificial_intelligence', 'share_word_machine_learning'], True),
        'd2_operational': (['share_deployment_verb', 'share_specific_product', 'share_customer_facing', 'share_product_integration',
                            'share_internal_productivity', 'share_safety_critical', 'share_word_automate_tasks',
                            'share_word_customer_facing', 'share_word_back_office', 'share_word_internal_operations'], False),
        'd3_technical': (['share_model_training', 'share_compute_infra', 'share_proprietary_data', 'share_academic_research',
                          'share_gen_ai_mention', 'share_classical_ml_mention', 'share_word_gpu', 'share_word_gpus',
                          'share_word_a100', 'share_word_h100', 'share_word_h200', 'share_word_fine_tuned', 'share_word_fine_tuning',
                          'share_word_foundation_model', 'share_word_foundation_models', 'share_word_inference',
                          'share_word_embedding', 'share_word_embeddings', 'share_word_dataset', 'share_word_datasets',
                          'share_word_llm', 'share_word_llms', 'share_word_deep_learning', 'share_word_parameter',
                          'share_word_parameters', 'share_word_generative_ai', 'share_word_large_language_model',
                          'share_word_rag', 'share_word_retrieval_augmented_generation', 'share_word_cuda',
                          'share_word_transformer', 'share_word_transformers'], False),
        'd4_quantification': (['share_metric_dollar', 'share_metric_percentage', 'share_capex_mention', 'share_opex_mention',
                              'share_rd_mention', 'share_revenue_impact', 'share_word_capital_expenditures', 'share_word_r_d',
                              'share_word_research_and_development', 'share_word_productivity', 'share_word_efficiency'], False),
        'd5_governance': (['share_board_oversight', 'share_audit_committee', 'share_ethics_policy', 'share_compliance',
                            'share_eu_regulation', 'share_us_regulation', 'share_word_board_oversight', 'share_word_audit_committee',
                            'share_word_responsible_ai', 'share_word_ethical_ai', 'share_word_gdpr', 'share_word_ccpa',
                            'share_word_eu_ai_act', 'share_word_executive_order'], False),
        'd6_risk': (['share_risk_factor', 'share_regulatory_risk', 'share_cyber_privacy_risk', 'share_ethics_bias_risk',
                     'share_ip_copyright_risk', 'share_supply_infra_risk', 'share_labor_displacement_risk',
                     'share_word_cybersecurity', 'share_word_breach', 'share_word_breaches', 'share_word_liability',
                     'share_word_litigation', 'share_word_harm', 'share_word_failure', 'share_word_failures',
                     'share_word_regulation', 'share_word_regulations'], False),
        'd7_promotional': (['share_word_transformative', 'share_word_transform', 'share_word_transforming', 'share_word_revolutionize',
                            'share_word_revolutionizing', 'share_word_empower', 'share_word_empowering', 'share_word_seamless',
                            'share_word_cutting_edge', 'share_word_breakthrough', 'share_word_groundbreaking', 'share_word_pioneer',
                            'share_word_pioneering', 'share_word_state_of_the_art', 'share_word_world_class', 'share_word_next_generation',
                            'share_word_next_gen', 'share_word_unlock', 'share_word_unlocking', 'share_word_unprecedented',
                            'share_word_leader', 'share_word_leadership', 'share_word_innovation', 'share_word_innovative',
                            'share_word_opportunities', 'share_word_enhance', 'share_word_enhancing'], False),
        'd8_defensive': (['share_competitor_mention', 'share_word_disrupt', 'share_word_disruption', 'share_word_disruptions',
                          'share_word_disruptive', 'share_word_disrupting', 'share_word_threat', 'share_word_threats',
                          'share_word_challenge', 'share_word_challenges', 'share_word_adversely', 'share_word_uncertainty',
                          'share_word_uncertainties'], False),
        'd9_risk_section': (['sec_pct_risk_factors'], False),
        'd9_substantive_section': (['sec_pct_business', 'sec_pct_mda'], False)
    }

    # Compute dimensions
    dims = pd.DataFrame()
    for dim_key, (cols, log_scale) in dim_defs.items():
        dims[DIM_LABELS[dim_key]] = make_dim(df, DIM_LABELS[dim_key], cols, log=log_scale)

    # Scale for clustering & UMAP
    X = StandardScaler().fit_transform(dims.values)

    # Run GMM and HDBSCAN (for schema compatibility)
    gmm_scores = {}
    for k in range(3, 8):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=SEED, max_iter=500)
        gl = gmm.fit_predict(X)
        gmm_scores[k] = gmm.bic(X)
    best_gmm = min(gmm_scores, key=gmm_scores.get)
    gmm = GaussianMixture(n_components=best_gmm, covariance_type='full', random_state=SEED, max_iter=500)
    gmm_labels = gmm.fit_predict(X)

    hdb = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=4)
    hdb_labels = hdb.fit_predict(X)

    # Project via UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=SEED)
    X_umap = reducer.fit_transform(X)

    result = df[['ticker', 'year', 'industry_group', 'post_sec_2024', 'post_deepseek']].copy()

    if args.method == "programmatic":
        # Implement programmatic rule-based classification
        def classify_row(row):
            # 1. Non-AI Disclosers
            if row['ai_mentions_count'] == 0:
                return 0 # 'Non-AI Disclosers'
            
            # 2. Governance & Compliance-Focused
            if row['D5 Governance'] >= 0.10:
                return 1 # 'Governance & Compliance-Focused'
            
            # 3. Full-Stack AI Pioneers
            if row['D1 AI Intensity'] >= 0.35 and (row['D3 Technical'] >= 0.08 or row['D2 Operational'] >= 0.15) and row['D9 Substantive-section'] >= 0.30:
                return 3 # 'Full-Stack AI Pioneers'
            
            # 4. Operational Application Adopters
            if row['D9 Substantive-section'] >= 0.50:
                return 4 # 'Operational Application Adopters'
            
            # 5. Boilerplate Risk-Warners
            return 2 # 'Boilerplate Risk-Warners'

        # Combine df metadata and dimensions to run classification
        df_for_rules = result.copy()
        df_for_rules['ai_mentions_count'] = df['ai_mentions_count']
        for col in dims.columns:
            df_for_rules[col] = dims[col]

        result['cluster'] = df_for_rules.apply(classify_row, axis=1)
        result['cluster_name'] = result['cluster'].map(NEW_CLUSTER_NAMES)
    else:
        # Unsupervised Ward clustering
        Z = linkage(X, method='ward')
        ward_labels = fcluster(Z, 5, criterion='maxclust') - 1
        result['cluster'] = ward_labels
        result['cluster_name'] = result['cluster'].map(NEW_CLUSTER_NAMES)

    # Schema compatibility variables
    result['cluster_hdbscan'] = hdb_labels
    result['cluster_gmm'] = gmm_labels

    # Add D-dimensions to output dataframe
    for col in dims.columns:
        result[col] = dims[col]

    result['umap1'] = X_umap[:, 0]
    result['umap2'] = X_umap[:, 1]

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="cluster_archetypes",
        level="SUCCESS",
        message=f"Clustering complete. Method: {args.method}. Saved {len(result)} rows to {output_path}."
    )
    print(f"Success: Saved {len(result)} rows to {output_path}")
    print("\nCluster counts:")
    print(result['cluster_name'].value_counts())

if __name__ == "__main__":
    main()
