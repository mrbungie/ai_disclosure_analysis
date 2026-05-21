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
    
    # If mentions parquet doesn't exist or is empty, create an empty one
    if not mentions_path.exists():
        pipeline_logger.log_event(
            pipeline_step="combined_scoring",
            level="WARNING",
            message="ai_disclosure_mentions.parquet not found. Scoring will rely entirely on BoW features."
        )
        print("Warning: ai_disclosure_mentions.parquet not found. Scoring will rely entirely on BoW features.")
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
    
    # Deduplicate bow and mentions on chunk_id to prevent row count inflation during the join.
    # BoW features are identical for the same chunk_id since it is a hash of the text.
    bow_df = bow_df.drop_duplicates(subset=["chunk_id"])
    mentions_df = mentions_df.drop(columns=["accession_number"], errors="ignore").drop_duplicates(subset=["chunk_id"])
    
    # Perform a LEFT JOIN starting from chunks/bow to preserve ALL candidate chunks
    merged_df = chunks_df.merge(bow_df, on="chunk_id", how="left")
    merged_df = merged_df.merge(mentions_df, on="chunk_id", how="left")
    
    if len(merged_df) == 0:
        pipeline_logger.log_event(
            pipeline_step="combined_scoring",
            level="WARNING",
            message="Merge resulted in 0 rows. Check chunk_id consistency."
        )
        print("Warning: Merge resulted in 0 rows. Check chunk_ids.")
        return
        
    print(f"Scoring {len(merged_df)} chunks using BoW features + LLM features (as available)...")
    
    # Set up deterministic proxies for LLM features when they are missing (LEFT JOIN result is NaN)
    has_metrics = (merged_df["has_metric_percentage"] | merged_df["has_metric_dollar"])
    
    # Substantive proxy: mentions policies, compliance, training, compute infra, proprietary data, or products
    substantive_proxy = (
        merged_df["has_ethics_policy"] | 
        merged_df["has_compliance"] | 
        merged_df["has_model_training"] | 
        merged_df["has_compute_infra"] | 
        merged_df["has_proprietary_data"] | 
        merged_df["has_specific_product"]
    )
    
    is_governance_related_proxy = (
        merged_df["has_board_oversight"] | 
        merged_df["has_audit_committee"] | 
        merged_df["has_ethics_policy"] | 
        merged_df["has_compliance"]
    )
    
    is_risk_related_proxy = (
        merged_df["has_risk_factor"] | 
        merged_df["has_regulatory_risk"] | 
        merged_df["has_cyber_privacy_risk"] | 
        merged_df["has_ethics_bias_risk"] | 
        merged_df["has_ip_copyright_risk"] | 
        merged_df["has_supply_infra_risk"] |
        merged_df["has_data_licensing"]
    )
    
    is_promotional_proxy = (
        merged_df["has_deployment_verb"] | 
        merged_df["has_specific_product"]
    )
    
    mentions_training_proxy = merged_df["has_model_training"]
    
    is_ai_related_proxy = pd.Series(True, index=merged_df.index)
    
    # Combine LLM and Proxy variables (fallback to proxy if LLM is NaN/Null)
    is_substantive = merged_df["is_substantive"].fillna(substantive_proxy).astype(bool)
    is_governance_related = merged_df["is_governance_related"].fillna(is_governance_related_proxy).astype(bool)
    is_risk_related = merged_df["is_risk_related"].fillna(is_risk_related_proxy).astype(bool)
    is_promotional = merged_df["is_promotional"].fillna(is_promotional_proxy).astype(bool)
    mentions_training = merged_df["mentions_training"].fillna(mentions_training_proxy).astype(bool)
    is_ai_related = merged_df["is_ai_related"].fillna(is_ai_related_proxy).astype(bool)
    
    # Apply combined scoring rules
    # 1. final_specificity (0-3)
    merged_df["final_specificity"] = (
        is_substantive.astype(int) +
        has_metrics.astype(int) +
        mentions_training.astype(int)
    ).clip(0, 3)
    
    # 2. final_governance_score (0-3)
    merged_df["final_governance_score"] = (
        is_governance_related.astype(int) +
        merged_df["has_board_oversight"].astype(int) +
        merged_df["has_audit_committee"].astype(int)
    ).clip(0, 3)
    
    # 3. final_risk_score (0-3)
    has_vendor_mention = (
        merged_df["has_vendor_nvidia"] | 
        merged_df["has_vendor_openai"] | 
        merged_df["has_vendor_microsoft"] | 
        merged_df["has_vendor_google"] | 
        merged_df["has_vendor_deepseek"] |
        merged_df["has_vendor_amazon"] |
        merged_df["has_vendor_meta"] |
        merged_df["has_vendor_anthropic"] |
        merged_df["has_vendor_amd"]
    )
    merged_df["final_risk_score"] = (
        is_risk_related.astype(int) +
        (is_risk_related & (merged_df["has_board_oversight"] | merged_df["has_audit_committee"])).astype(int) +
        (is_risk_related & has_vendor_mention).astype(int)
    ).clip(0, 3)
    
    # 4. final_promotional_score (0-3)
    merged_df["final_promotional_score"] = (
        is_promotional.astype(int) +
        (is_promotional & ~is_substantive).astype(int) +
        (is_promotional & ~has_metrics).astype(int)
    ).clip(0, 3)
    
    # 5. has_ai_disclosure (bool)
    merged_df["has_ai_disclosure"] = is_ai_related & (is_substantive | is_risk_related)
    
    # Update the component columns in the merged dataframe with filled values
    merged_df["is_substantive"] = is_substantive
    merged_df["is_promotional"] = is_promotional
    merged_df["is_risk_related"] = is_risk_related
    merged_df["is_governance_related"] = is_governance_related
    merged_df["is_ai_related"] = is_ai_related

    # Fill NaN/None for amounts and entities with None, and sentiment with "Neutral"
    if "amounts" in merged_df.columns:
        merged_df["amounts"] = merged_df["amounts"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else None)
    else:
        merged_df["amounts"] = None

    if "entities" in merged_df.columns:
        merged_df["entities"] = merged_df["entities"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else None)
    else:
        merged_df["entities"] = None

    if "sentiment" in merged_df.columns:
        # Convert float NaN (from LEFT JOIN) to "Neutral"
        merged_df["sentiment"] = merged_df["sentiment"].apply(lambda x: x if isinstance(x, str) and pd.notna(x) else "Neutral")
    else:
        merged_df["sentiment"] = "Neutral"
    
    # Select columns to output in ai_scored_chunks.parquet
    # We want to preserve metadata, computed scores, filled LLM components, and all BoW features
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
        # Keep components for transparent downstream auditing
        "is_substantive",
        "is_promotional",
        "is_risk_related",
        "is_governance_related",
        "is_ai_related",
        "amounts",
        "entities",
        "sentiment"
    ] + bow_cols
    
    # Deduplicate out_cols to avoid any potential duplicates (e.g. if a column is in both lists)
    seen = set()
    out_cols = [x for x in out_cols if not (x in seen or seen.add(x))]
    
    out_cols = [c for c in out_cols if c in merged_df.columns]
    
    scored_df = merged_df[out_cols]
    scored_df.to_parquet(output_path, index=False)
    
    pipeline_logger.log_event(
        pipeline_step="combined_scoring",
        level="SUCCESS",
        message=f"Combined scoring complete. Saved {len(scored_df)} scored chunks to {output_path}."
    )
    print(f"Success: Saved {len(scored_df)} scored chunks to {output_path}")

if __name__ == "__main__":
    main()
