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
    
    universe_path = Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet"
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    scored_chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_scored_chunks.parquet"
    output_path = Path("data/processed/features/firm_year_features.parquet")
    
    # Check if files exist
    if not universe_path.exists() or not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step="build_features",
            level="ERROR",
            message=f"Missing metadata files. universe={universe_path.exists()}, manifest={manifest_path.exists()}"
        )
        print("Error: Missing universe or manifest files.")
        return
        
    pipeline_logger.log_event(
        pipeline_step="build_features",
        level="INFO",
        message="Loading firm universe, filing manifest, and scored chunks..."
    )
    
    universe_df = pd.read_parquet(universe_path)
    manifest_df = pd.read_parquet(manifest_path)
    
    # Load scored chunks if they exist, otherwise create an empty dataframe with correct columns
    if scored_chunks_path.exists():
        scored_chunks_df = pd.read_parquet(scored_chunks_path)
        print(f"Loaded {len(scored_chunks_df)} scored chunks.")
    else:
        print("Warning: ai_scored_chunks.parquet not found. Creating empty features panel.")
        scored_chunks_df = pd.DataFrame(columns=[
            "chunk_id", "accession_number", "ticker", "filing_date", "section_name", 
            "chunk_text", "final_specificity", "final_governance_score", "final_risk_score", 
            "final_promotional_score", "has_ai_disclosure", "is_substantive", "is_promotional", 
            "is_risk_related", "is_governance_related", "bow_sentiment_score", "sentiment"
        ])
        
    # Extract year from filing_manifest to build complete panel backbone
    manifest_df["year"] = pd.to_datetime(manifest_df["filing_date"]).dt.year
    
    # Keep unique firm-years from manifest
    firm_years = manifest_df[["ticker", "year"]].drop_duplicates().copy()
    
    # Calculate chunk-level aggregates per firm-year
    # Convert dates and extract year in scored chunks
    if len(scored_chunks_df) > 0:
        scored_chunks_df["year"] = pd.to_datetime(scored_chunks_df["filing_date"]).dt.year
        
        # Map sentiment strings to numeric values
        sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1, "Mixed": 0}
        sentiment_col = scored_chunks_df["sentiment"] if "sentiment" in scored_chunks_df.columns else pd.Series("Neutral", index=scored_chunks_df.index)
        scored_chunks_df["llm_sentiment_numeric"] = sentiment_col.map(sentiment_map).fillna(0)
        
        # Ensure bow_sentiment_score exists
        bow_sent_col = scored_chunks_df["bow_sentiment_score"] if "bow_sentiment_score" in scored_chunks_df.columns else pd.Series(0.0, index=scored_chunks_df.index)
        scored_chunks_df["bow_sentiment_score_filled"] = bow_sent_col.fillna(0.0)
        
        # Group by ticker, year
        grouped = scored_chunks_df.groupby(["ticker", "year"])
        
        # Aggregations
        aggregates = pd.DataFrame({
            "ai_mentions_count": grouped.apply(lambda g: int(g["has_ai_disclosure"].sum())),
            "avg_specificity": grouped["final_specificity"].mean(),
            "avg_operational_grounding": grouped["is_substantive"].mean().astype(float),
            "avg_promotional_score": grouped["final_promotional_score"].mean(),
            "avg_risk_score": grouped["final_risk_score"].mean(),
            "avg_governance_score": grouped["final_governance_score"].mean(),
            
            # Sentiment averages
            "avg_bow_sentiment": grouped["bow_sentiment_score_filled"].mean(),
            "avg_llm_sentiment": grouped["llm_sentiment_numeric"].mean(),
            
            # Shares
            "share_promotional": grouped.apply(lambda g: float(g["is_promotional"].mean())),
            "share_substantive": grouped.apply(lambda g: float(g["is_substantive"].mean())),
            "share_governance": grouped.apply(lambda g: float(g["is_governance_related"].mean())),
            "share_risk": grouped.apply(lambda g: float(g["is_risk_related"].mean()))
        }).reset_index()
    else:
        aggregates = pd.DataFrame(columns=[
            "ticker", "year", "ai_mentions_count", "avg_specificity", "avg_operational_grounding",
            "avg_promotional_score", "avg_risk_score", "avg_governance_score",
            "avg_bow_sentiment", "avg_llm_sentiment",
            "share_promotional", "share_substantive", "share_governance", "share_risk"
        ])
        
    # Merge panel backbone with aggregates
    panel_df = firm_years.merge(aggregates, on=["ticker", "year"], how="left")
    
    # Fill NaN values for firms/years that had no candidate chunks (which means 0 mentions, 0 scores)
    fill_cols = [
        "ai_mentions_count", "avg_specificity", "avg_operational_grounding",
        "avg_promotional_score", "avg_risk_score", "avg_governance_score",
        "avg_bow_sentiment", "avg_llm_sentiment",
        "share_promotional", "share_substantive", "share_governance", "share_risk"
    ]
    for col in fill_cols:
        panel_df[col] = panel_df[col].fillna(0)
        
    # Cast ai_mentions_count to integer
    panel_df["ai_mentions_count"] = panel_df["ai_mentions_count"].astype(int)
    
    # Merge with firm universe to get industry group
    panel_df = panel_df.merge(universe_df[["ticker", "industry_group"]], on="ticker", how="left")
    
    # If a ticker is not in universe, set to Unclassified
    panel_df["industry_group"] = panel_df["industry_group"].fillna("Unclassified")
    
    # Compute event-study and time flags
    panel_df["post_sec_2024"] = panel_df["year"] >= 2024
    panel_df["post_deepseek"] = panel_df["year"] >= 2025
    
    # Ensure correct column ordering
    cols = [
        "ticker", "year", "industry_group", "ai_mentions_count", "avg_specificity", 
        "avg_operational_grounding", "avg_promotional_score", "avg_risk_score", 
        "avg_governance_score", "avg_bow_sentiment", "avg_llm_sentiment",
        "share_promotional", "share_substantive", "share_governance", "share_risk",
        "post_sec_2024", "post_deepseek"
    ]
    panel_df = panel_df[cols]
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(output_path, index=False)
    
    pipeline_logger.log_event(
        pipeline_step="build_features",
        level="SUCCESS",
        message=f"Panel feature dataset built successfully. Saved {len(panel_df)} firm-year observations to {output_path}."
    )
    print(f"Success: Saved {len(panel_df)} firm-year observations to {output_path}")

if __name__ == "__main__":
    main()
