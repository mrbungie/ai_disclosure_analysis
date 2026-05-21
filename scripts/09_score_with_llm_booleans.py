import os
import json
import pandas as pd
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
    rules_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_rules.parquet"
    mentions_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_mentions.parquet"
    output_path = Path(config["paths"]["candidate_chunks"]) / "ai_scored_chunks.parquet"
    
    # Check if inputs exist
    missing_files = []
    for p in [chunks_path, rules_path, mentions_path]:
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
    rules_df = pd.read_parquet(rules_path)
    mentions_df = pd.read_parquet(mentions_path)
    
    if len(mentions_df) == 0:
        pipeline_logger.log_event(
            pipeline_step="combined_scoring",
            level="WARNING",
            message="No LLM classifications found. Cannot score."
        )
        print("Warning: No LLM classifications found in ai_disclosure_mentions.parquet. Exiting.")
        return
        
    pipeline_logger.log_event(
        pipeline_step="combined_scoring",
        level="INFO",
        message=f"Merging chunks ({len(chunks_df)}), rules ({len(rules_df)}), and LLM classifications ({len(mentions_df)})..."
    )
    
    # We perform an inner join on chunk_id to only score fully classified chunks
    # Note: Keep the text, metadata, rules, and mentions columns
    merged_df = chunks_df.merge(rules_df, on="chunk_id", how="inner")
    merged_df = merged_df.merge(mentions_df, on=["chunk_id", "accession_number"], how="inner")
    
    if len(merged_df) == 0:
        pipeline_logger.log_event(
            pipeline_step="combined_scoring",
            level="WARNING",
            message="Merge resulted in 0 rows. Check chunk_id consistency."
        )
        print("Warning: Merge resulted in 0 rows. Check chunk_ids.")
        return
        
    print(f"Scoring {len(merged_df)} matched chunks...")
    
    # 1. final_specificity (0-3)
    # is_substantive (LLM) + (has_metric_percentage | has_metric_dollar) (Rule) + mentions_training (LLM)
    has_metrics = (merged_df["has_metric_percentage"] | merged_df["has_metric_dollar"])
    merged_df["final_specificity"] = (
        merged_df["is_substantive"].astype(int) +
        has_metrics.astype(int) +
        merged_df["mentions_training"].astype(int)
    ).clip(0, 3)
    
    # 2. final_governance_score (0-3)
    # is_governance_related (LLM) + has_board_oversight (Rule) + has_audit_committee (Rule)
    merged_df["final_governance_score"] = (
        merged_df["is_governance_related"].astype(int) +
        merged_df["has_board_oversight"].astype(int) +
        merged_df["has_audit_committee"].astype(int)
    ).clip(0, 3)
    
    # 3. final_risk_score (0-3)
    # is_risk_related (LLM) + (is_risk_related & (has_board_oversight | has_audit_committee)) + (is_risk_related & vendor rules)
    has_vendor_mention = (
        merged_df["has_vendor_nvidia"] | 
        merged_df["has_vendor_openai"] | 
        merged_df["has_vendor_microsoft"] | 
        merged_df["has_vendor_google"] | 
        merged_df["has_vendor_deepseek"]
    )
    merged_df["final_risk_score"] = (
        merged_df["is_risk_related"].astype(int) +
        (merged_df["is_risk_related"] & (merged_df["has_board_oversight"] | merged_df["has_audit_committee"])).astype(int) +
        (merged_df["is_risk_related"] & has_vendor_mention).astype(int)
    ).clip(0, 3)
    
    # 4. final_promotional_score (0-3)
    # is_promotional (LLM) + (is_promotional & ~is_substantive) + (is_promotional & ~has_metrics)
    merged_df["final_promotional_score"] = (
        merged_df["is_promotional"].astype(int) +
        (merged_df["is_promotional"] & ~merged_df["is_substantive"]).astype(int) +
        (merged_df["is_promotional"] & ~has_metrics).astype(int)
    ).clip(0, 3)
    
    # 5. has_ai_disclosure (bool)
    # true if chunk has a valid substantive or risk AI disclosure
    merged_df["has_ai_disclosure"] = merged_df["is_ai_related"] & (merged_df["is_substantive"] | merged_df["is_risk_related"])
    
    # Select columns to output in ai_scored_chunks.parquet
    # We keep key metadata and the scoring columns as specified in docs
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
        "is_governance_related"
    ]
    
    # Filter to only keep columns that exist in merged_df just in case
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
