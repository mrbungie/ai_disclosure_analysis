import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

def load_config():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Score chunks with LLM booleans or BoW proxies")
    # Design Decision: The LLM classifier is made optional using the --bow-only flag.
    # This allows running the entire pipeline using purely deterministic BoW proxies,
    # ensuring data homogeneity and avoiding temporal anomalies (such as the 2023 dip)
    # caused by incomplete LLM classifications across years.
    parser.add_argument("--bow-only", action="store_true", help="Force BoW-only mode, ignoring LLM classifications")
    args = parser.parse_args()

    config = load_config()
    
    chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"
    bow_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_bow_features.parquet"
    mentions_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_mentions.parquet"
    output_path = Path(config["paths"]["candidate_chunks"]) / "ai_scored_chunks.parquet"
    
    # Check if inputs exist
    missing_files = []
    for p in [chunks_path, bow_path]:
        if not p.exists():
            missing_files.append(p.name)
            
    if missing_files:
        pipeline_logger.log_event(
            pipeline_step="combined_scoring",
            level="ERROR",
            message=f"Missing input parquets: {', '.join(missing_files)}"
        )
        print(f"Error: Missing input parquets: {', '.join(missing_files)}")
        return

    pipeline_logger.log_event(
        pipeline_step="combined_scoring",
        level="INFO",
        message="Loading input parquets for combined scoring..."
    )
    
    chunks_df = pd.read_parquet(chunks_path)
    bow_df = pd.read_parquet(bow_path)
    
    # If mentions parquet doesn't exist or is empty, or if --bow-only is specified, create an empty one
    if args.bow_only or not mentions_path.exists():
        mode_msg = "Running in forced BoW-only mode." if args.bow_only else "ai_disclosure_mentions.parquet not found."
        pipeline_logger.log_event(
            pipeline_step="combined_scoring",
            level="WARNING",
            message=f"{mode_msg} Scoring will rely entirely on BoW features."
        )
        print(f"Warning: {mode_msg} Scoring will rely entirely on BoW features.")
        mentions_df = pd.DataFrame(columns=[
            "chunk_id", "is_ai_related", "is_substantive", "is_promotional",
            "is_risk_related", "is_governance_related", "mentions_copilot", "mentions_cloud",
            "mentions_vendor", "mentions_training", "is_financial_impact", "rationale_short",
            "amounts", "entities", "sentiment"
        ])
    else:
        mentions_df = pd.read_parquet(mentions_path)
        if len(mentions_df) == 0:
            pipeline_logger.log_event(
                pipeline_step="combined_scoring",
                level="WARNING",
                message="No LLM classifications found. Scoring will rely entirely on BoW features."
            )
            print("Warning: No LLM classifications found. Scoring will rely entirely on BoW features.")
        
    pipeline_logger.log_event(
        pipeline_step="combined_scoring",
        level="INFO",
        message=f"Merging chunks ({len(chunks_df)}), BoW features ({len(bow_df)}), and LLM classifications ({len(mentions_df)})..."
    )
    
    # Ensure mentions_df has all expected columns so DuckDB query runs without schema errors
    expected_cols = [
        "chunk_id", "is_ai_related", "is_substantive", "is_promotional",
        "is_risk_related", "is_governance_related", "mentions_copilot", "mentions_cloud",
        "mentions_vendor", "mentions_training", "is_financial_impact", "rationale_short",
        "amounts", "entities", "sentiment", "accession_number"
    ]
    for col in expected_cols:
        if col not in mentions_df.columns:
            mentions_df[col] = None

    import re
    # Explicit AI terms to filter out false-positive candidate chunks
    ai_explicit_keywords = [
        "ai", "artificial intelligence", "generative ai", "gen ai", "machine learning",
        "large language model", "llm", "llms", "deep learning", "neural network", "neural networks",
        "openai", "anthropic", "claude", "deepseek", "chatgpt", "copilot", "gemini",
        "natural language processing", "computer vision", "machine translation"
    ]
    ai_explicit_patterns = []
    for kw in ai_explicit_keywords:
        pattern = r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b"
        ai_explicit_patterns.append(pattern)
    regex_pattern = "|".join(ai_explicit_patterns)

    import duckdb
    con = duckdb.connect()
    
    # Register dataframes to query them in SQL
    con.register("chunks_df", chunks_df)
    con.register("bow_df", bow_df)
    con.register("mentions_df", mentions_df)

    sql_query = f"""
    WITH chunks AS (
        SELECT * FROM chunks_df
    ),
    bow AS (
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY chunk_id) as rn
            FROM bow_df
        ) WHERE rn = 1
    ),
    mentions AS (
        SELECT * FROM (
            SELECT * EXCLUDE (accession_number) REPLACE (
                COALESCE(is_ai_related, FALSE) as is_ai_related,
                COALESCE(is_substantive, FALSE) as is_substantive,
                COALESCE(is_promotional, FALSE) as is_promotional,
                COALESCE(is_risk_related, FALSE) as is_risk_related,
                COALESCE(is_governance_related, FALSE) as is_governance_related,
                COALESCE(mentions_training, FALSE) as mentions_training,
                COALESCE(mentions_copilot, FALSE) as mentions_copilot,
                COALESCE(mentions_cloud, FALSE) as mentions_cloud,
                COALESCE(mentions_vendor, FALSE) as mentions_vendor,
                COALESCE(is_financial_impact, FALSE) as is_financial_impact
            ), row_number() OVER (PARTITION BY chunk_id) as rn
            FROM mentions_df
        ) WHERE rn = 1
    ),
    merged AS (
        SELECT 
            c.*,
            b.* EXCLUDE (chunk_id, rn),
            m.* EXCLUDE (chunk_id, rn)
        FROM chunks c
        LEFT JOIN bow b ON c.chunk_id = b.chunk_id
        LEFT JOIN mentions m ON c.chunk_id = m.chunk_id
    ),
    scored_intermediate AS (
        SELECT
            *,
            -- has_metrics
            (has_metric_percentage OR has_metric_dollar) as has_metrics,
            
            -- substantive_proxy
            -- has_compute_infra: gated on NOT has_ai_demand_context (excludes AMD/KLA/utilities
            --   where AI is a customer demand signal, not internal deployment)
            -- has_compute_infra+ai_quantified_claim: alternate gate for named-AI-product revenue
            --   disclosures (e.g. "NVIDIA AI cloud service offerings, Data Center +41%")
            -- has_model_training: excludes workforce_talent AND has_ai_hedge (risk boilerplate)
            -- has_ethics_policy: gated on NOT has_risk_factor to avoid pure regulatory sections
            -- has_ai_own_use: "we incorporate AI" / "our AI-powered X" — gated to exclude chunks
            --   that are risk-hedged, about competitors, or are AI-demand-context supply chain
            -- has_classical_ml_operational: "ML to detect/predict [task]" — gated on NOT
            --   has_risk_factor to exclude cybersecurity-threat descriptions
            -- has_ai_acquisition: M&A of AI company (Featurespace-type deals)
            (
                (has_model_training AND NOT has_workforce_talent AND NOT has_ai_hedge)
                OR has_specific_product
                OR has_proprietary_data
                OR (has_compute_infra AND has_word_ai
                    AND (has_word_gpu OR has_word_gpus OR has_realized_language OR has_deployment_verb)
                    AND NOT has_ai_demand_context)
                OR (has_compute_infra AND has_word_ai AND has_ai_quantified_claim AND NOT has_ai_demand_context)
                OR (has_ethics_policy AND has_word_ai AND NOT has_risk_factor)
                OR (has_classical_ml_mention AND (has_internal_productivity OR has_customer_facing) AND NOT has_workforce_talent)
                OR (has_ai_own_use AND NOT has_ai_hedge AND NOT has_competitor_mention AND NOT has_ai_demand_context)
                OR (has_classical_ml_operational AND NOT has_risk_factor)
                OR has_ai_acquisition
            ) as substantive_proxy,

            -- is_governance_related_proxy
            -- has_compliance removed: fires on any "compliance with laws" text, not AI-specific
            -- has_ethics_policy gated on AI co-occurrence to avoid general ESG FPs
            (
                has_board_oversight
                OR has_audit_committee
                OR (has_ethics_policy AND has_word_ai)
            ) as is_governance_related_proxy,

            -- is_risk_related_proxy
            -- has_ip_copyright_risk and has_cyber_privacy_risk gated on AI co-occurrence
            -- to avoid generic IP/cybersecurity sections with no AI context
            (
                has_risk_factor
                OR has_regulatory_risk
                OR has_ethics_bias_risk
                OR has_supply_infra_risk
                OR has_data_licensing
                OR (has_cyber_privacy_risk AND has_word_ai)
                OR (has_ip_copyright_risk AND has_word_ai)
            ) as is_risk_related_proxy,
            
            -- is_promotional_proxy
            -- extended beyond deployment_verb/specific_product to capture boosterish AI language:
            -- customer_facing + AI, gen-AI forward-looking, and marketing superlatives about AI
            (
                has_deployment_verb
                OR has_specific_product
                OR (has_customer_facing AND has_word_ai)
                OR (has_gen_ai_mention AND has_forward_looking)
                OR (has_word_ai AND (
                    has_word_breakthrough OR has_word_groundbreaking
                    OR has_word_pioneer OR has_word_pioneering
                    OR has_word_game_changer OR has_word_game_changing
                ))
            ) as is_promotional_proxy,
            
            -- mentions_training_proxy
            has_model_training as mentions_training_proxy,
            
            -- is_ai_related_proxy (always True for candidate chunks in fallback)
            TRUE as is_ai_related_proxy,
            
            -- has_vendor_mention
            (has_vendor_nvidia OR has_vendor_openai OR has_vendor_microsoft OR has_vendor_google OR has_vendor_deepseek OR has_vendor_amazon OR has_vendor_meta OR has_vendor_anthropic OR has_vendor_amd) as has_vendor_mention
        FROM merged
    ),
    scored_final AS (
        SELECT
            chunk_id,
            accession_number,
            ticker,
            filing_date,
            section_name,
            chunk_text,
            
            -- Combine LLM and Proxy variables (fallback to proxy if LLM is NULL)
            COALESCE(is_substantive, substantive_proxy) as is_substantive,
            COALESCE(is_promotional, is_promotional_proxy) as is_promotional,
            COALESCE(is_risk_related, is_risk_related_proxy) as is_risk_related,
            COALESCE(is_governance_related, is_governance_related_proxy) as is_governance_related,
            COALESCE(is_ai_related, is_ai_related_proxy) as is_ai_related,
            
            -- Keep sentiment/amounts/entities, with defaults if NULL
            COALESCE(sentiment, 'Neutral') as sentiment,
            amounts,
            entities,
            
            -- specificity score
            LEAST(COALESCE(is_substantive, substantive_proxy)::INT + 
                  (has_metric_percentage OR has_metric_dollar)::INT + 
                  COALESCE(mentions_training, mentions_training_proxy)::INT, 3) as final_specificity,
            
            -- governance score
            LEAST(COALESCE(is_governance_related, is_governance_related_proxy)::INT + 
                  has_board_oversight::INT + 
                  has_audit_committee::INT, 3) as final_governance_score,
            
            -- risk score
            LEAST(COALESCE(is_risk_related, is_risk_related_proxy)::INT + 
                  (COALESCE(is_risk_related, is_risk_related_proxy) AND (has_board_oversight OR has_audit_committee))::INT + 
                  (COALESCE(is_risk_related, is_risk_related_proxy) AND has_vendor_mention)::INT, 3) as final_risk_score,
            
            -- promotional score
            LEAST(COALESCE(is_promotional, is_promotional_proxy)::INT + 
                  (COALESCE(is_promotional, is_promotional_proxy) AND NOT COALESCE(is_substantive, substantive_proxy))::INT + 
                  (COALESCE(is_promotional, is_promotional_proxy) AND NOT (has_metric_percentage OR has_metric_dollar))::INT, 3) as final_promotional_score,
            
            -- has_ai_disclosure
            (COALESCE(is_ai_related, is_ai_related_proxy) AND (COALESCE(is_substantive, substantive_proxy) OR COALESCE(is_risk_related, is_risk_related_proxy))) as has_ai_disclosure,
            
            -- Raw LLM flags (coalesced to FALSE if NULL/missing/BoW-only)
            COALESCE(is_ai_related, FALSE) as llm_is_ai_related,
            COALESCE(is_substantive, FALSE) as llm_is_substantive,
            COALESCE(is_promotional, FALSE) as llm_is_promotional,
            COALESCE(is_risk_related, FALSE) as llm_is_risk_related,
            COALESCE(is_governance_related, FALSE) as llm_is_governance_related,
            COALESCE(is_financial_impact, FALSE) as llm_is_financial_impact,
            COALESCE(mentions_training, FALSE) as llm_mentions_training,
            COALESCE(mentions_vendor, FALSE) as llm_mentions_vendor,
            COALESCE(mentions_cloud, FALSE) as llm_mentions_cloud,
            COALESCE(mentions_copilot, FALSE) as llm_mentions_copilot,
            
            -- Keep all remaining columns dynamically
            * EXCLUDE (
                chunk_id, accession_number, ticker, filing_date, section_name, chunk_text,
                is_substantive, is_promotional, is_risk_related, is_governance_related, is_ai_related,
                sentiment, amounts, entities,
                has_metrics, substantive_proxy, is_governance_related_proxy, is_risk_related_proxy, is_promotional_proxy,
                mentions_training_proxy, is_ai_related_proxy, has_vendor_mention, mentions_training,
                is_financial_impact, mentions_vendor, mentions_cloud, mentions_copilot
            )
        FROM scored_intermediate
    )
    SELECT * FROM scored_final;
    """

    print(f"Scoring {len(chunks_df)} chunks using BoW features + LLM features (as available) via DuckDB SQL...")
    df_duckdb = con.execute(sql_query).df()

    # Reorder/select columns to output to match expected schema exactly
    bow_cols = [c for c in bow_df.columns if c != "chunk_id"]
    out_cols = [
        "chunk_id",
        "accession_number",
        "ticker",
        "filing_date",
        "section_name",
        "chunk_text",
        "final_specificity",
        "final_governance_score",
        "final_risk_score",
        "final_promotional_score",
        "has_ai_disclosure",
        "is_substantive",
        "is_promotional",
        "is_risk_related",
        "is_governance_related",
        "is_ai_related",
        "llm_is_ai_related",
        "llm_is_substantive",
        "llm_is_promotional",
        "llm_is_risk_related",
        "llm_is_governance_related",
        "llm_is_financial_impact",
        "llm_mentions_training",
        "llm_mentions_vendor",
        "llm_mentions_cloud",
        "llm_mentions_copilot",
        "amounts",
        "entities",
        "sentiment"
    ] + bow_cols

    # Deduplicate out_cols
    seen = set()
    out_cols = [x for x in out_cols if not (x in seen or seen.add(x))]
    out_cols = [c for c in out_cols if c in df_duckdb.columns]

    scored_df = df_duckdb[out_cols]
    scored_df.to_parquet(output_path, index=False)
    
    pipeline_logger.log_event(
        pipeline_step="combined_scoring",
        level="SUCCESS",
        message=f"Combined scoring complete. Saved {len(scored_df)} scored chunks to {output_path}."
    )
    print(f"Success: Saved {len(scored_df)} scored chunks to {output_path}")

if __name__ == "__main__":
    main()
