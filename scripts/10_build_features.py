import os
import json
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Suppress PerformanceWarning due to working with highly wide DataFrames
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

try:
    import pipeline_logger
    import variant_utils
except ImportError:
    from scripts import pipeline_logger
    from scripts import variant_utils

def load_config():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build firm-year feature panel from scored chunks")
    variant_utils.add_variant_arg(parser)
    args = parser.parse_args()

    config = load_config()
    variant = variant_utils.resolve_variant(args.variant, config)
    output_root = config.get("variants", {}).get("output_root", "data/processed")

    universe_path = Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet"
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    scored_chunks_path = variant_utils.variant_path(variant, "ai_scored_chunks", "parquet", output_root=output_root)
    output_path = variant_utils.variant_path(variant, "firm_year_features", "parquet", output_root=output_root)

    # Check if files exist
    if not universe_path.exists() or not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step="build_features",
            level="ERROR",
            message=f"Missing metadata files. universe={universe_path.exists()}, manifest={manifest_path.exists()}",
            details={"variant": variant}
        )
        print("Error: Missing universe or manifest files.")
        return

    pipeline_logger.log_event(
        pipeline_step="build_features",
        level="INFO",
        message="Loading firm universe, filing manifest, and scored chunks...",
        details={"variant": variant}
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
        
    import duckdb
    con = duckdb.connect()
    
    # Check if scored chunks exist
    if scored_chunks_path.exists() and os.path.getsize(scored_chunks_path) > 0:
        # Get all has_ columns dynamically
        cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{scored_chunks_path}')").df()
        has_cols = [row['column_name'] for idx, row in cols_info.iterrows() if row['column_name'].startswith('has_') and row['column_name'] != 'has_ai_disclosure']
        
        # Build SQL parts for has_ cols
        inner_selects = []
        outer_selects = []
        for col in has_cols:
            share_name = f"share_{col[4:]}"
            inner_selects.append(f"AVG(CASE WHEN {col} THEN 1.0 ELSE 0.0 END) as {share_name}")
            outer_selects.append(f"COALESCE(c.{share_name}, 0.0) as {share_name}")
            
        inner_select_str = ",\n        " + ",\n        ".join(inner_selects) if inner_selects else ""
        outer_select_str = ",\n    " + ",\n    ".join(outer_selects) if outer_selects else ""
        
        # SQL Query to build complete panel backbone and aggregate scored chunks
        sql_query = f"""
        WITH backbone AS (
            SELECT DISTINCT 
                ticker,
                year(CAST(filing_date AS DATE)) as year
            FROM read_parquet('{manifest_path}')
        ),
        chunk_aggregates AS (
            SELECT
                ticker,
                year(CAST(filing_date AS DATE)) as year,
                SUM(CASE WHEN has_ai_disclosure THEN 1 ELSE 0 END) as ai_mentions_count,
                AVG(final_specificity) as avg_specificity,
                AVG(CASE WHEN is_substantive THEN 1.0 ELSE 0.0 END) as avg_operational_grounding,
                AVG(CASE WHEN is_substantive THEN 1.0 ELSE 0.0 END) as share_substantive,
                AVG(final_promotional_score) as avg_promotional_score,
                AVG(final_risk_score) as avg_risk_score,
                AVG(final_governance_score) as avg_governance_score,
                AVG(COALESCE(bow_sentiment_score, 0.0)) as avg_bow_sentiment,
                AVG(CASE sentiment WHEN 'Positive' THEN 1.0 WHEN 'Negative' THEN -1.0 ELSE 0.0 END) as avg_llm_sentiment,
                AVG(CASE WHEN is_promotional THEN 1.0 ELSE 0.0 END) as share_promotional,
                AVG(CASE WHEN is_governance_related THEN 1.0 ELSE 0.0 END) as share_governance,
                AVG(CASE WHEN is_risk_related THEN 1.0 ELSE 0.0 END) as share_risk,
                
                -- LLM-derived aggregates
                AVG(CASE WHEN llm_is_ai_related THEN 1.0 ELSE 0.0 END) as share_llm_is_ai_related,
                AVG(CASE WHEN llm_is_substantive THEN 1.0 ELSE 0.0 END) as share_llm_is_substantive,
                AVG(CASE WHEN llm_is_promotional THEN 1.0 ELSE 0.0 END) as share_llm_is_promotional,
                AVG(CASE WHEN llm_is_risk_related THEN 1.0 ELSE 0.0 END) as share_llm_is_risk_related,
                AVG(CASE WHEN llm_is_governance_related THEN 1.0 ELSE 0.0 END) as share_llm_is_governance,
                AVG(CASE WHEN llm_is_financial_impact THEN 1.0 ELSE 0.0 END) as share_llm_is_financial_impact,
                AVG(CASE WHEN llm_mentions_training THEN 1.0 ELSE 0.0 END) as share_llm_mentions_training,
                AVG(CASE WHEN llm_mentions_vendor THEN 1.0 ELSE 0.0 END) as share_llm_mentions_vendor,
                AVG(CASE WHEN llm_mentions_cloud THEN 1.0 ELSE 0.0 END) as share_llm_mentions_cloud,
                AVG(CASE WHEN llm_mentions_copilot THEN 1.0 ELSE 0.0 END) as share_llm_mentions_copilot,
                
                -- Section location aggregates
                AVG(CASE WHEN section_name = 'Item 1' THEN 1.0 ELSE 0.0 END) as sec_pct_business,
                AVG(CASE WHEN section_name = 'Item 7' THEN 1.0 ELSE 0.0 END) as sec_pct_mda,
                AVG(CASE WHEN section_name = 'Item 1A' THEN 1.0 ELSE 0.0 END) as sec_pct_risk_factors,
                AVG(CASE WHEN section_name NOT IN ('Item 1', 'Item 1A', 'Item 7') THEN 1.0 ELSE 0.0 END) as sec_pct_other,
                
                -- Use case diversity
                AVG(count_ai_use_case_types) as avg_count_ai_use_case_types,

                -- Ratio features not captured by has_* scan
                AVG(COALESCE(ratio_vague_words, 0.0)) as avg_ratio_vague_words,
                AVG(COALESCE(ratio_forward_to_realized, 0.5)) as avg_ratio_forward_to_realized
                {inner_select_str}
            FROM read_parquet('{scored_chunks_path}')
            WHERE has_ai_disclosure = true
            GROUP BY ticker, year
        )
        SELECT
            b.ticker,
            b.year,
            COALESCE(u.industry_group, 'Unclassified') as industry_group,
            CAST(COALESCE(c.ai_mentions_count, 0) AS INTEGER) as ai_mentions_count,
            COALESCE(c.avg_specificity, 0.0) as avg_specificity,
            COALESCE(c.avg_operational_grounding, 0.0) as avg_operational_grounding,
            COALESCE(c.share_substantive, 0.0) as share_substantive,
            COALESCE(c.avg_promotional_score, 0.0) as avg_promotional_score,
            COALESCE(c.avg_risk_score, 0.0) as avg_risk_score,
            COALESCE(c.avg_governance_score, 0.0) as avg_governance_score,
            COALESCE(c.avg_bow_sentiment, 0.0) as avg_bow_sentiment,
            COALESCE(c.avg_llm_sentiment, 0.0) as avg_llm_sentiment,
            COALESCE(c.share_promotional, 0.0) as share_promotional,
            COALESCE(c.share_governance, 0.0) as share_governance,
            COALESCE(c.share_risk, 0.0) as share_risk,
            
            -- LLM aggregates coalesced
            COALESCE(c.share_llm_is_ai_related, 0.0) as share_llm_is_ai_related,
            COALESCE(c.share_llm_is_substantive, 0.0) as share_llm_is_substantive,
            COALESCE(c.share_llm_is_promotional, 0.0) as share_llm_is_promotional,
            COALESCE(c.share_llm_is_risk_related, 0.0) as share_llm_is_risk_related,
            COALESCE(c.share_llm_is_governance, 0.0) as share_llm_is_governance,
            COALESCE(c.share_llm_is_financial_impact, 0.0) as share_llm_is_financial_impact,
            COALESCE(c.share_llm_mentions_training, 0.0) as share_llm_mentions_training,
            COALESCE(c.share_llm_mentions_vendor, 0.0) as share_llm_mentions_vendor,
            COALESCE(c.share_llm_mentions_cloud, 0.0) as share_llm_mentions_cloud,
            COALESCE(c.share_llm_mentions_copilot, 0.0) as share_llm_mentions_copilot,
            
            -- Section location aggregates coalesced
            COALESCE(c.sec_pct_business, 0.0) as sec_pct_business,
            COALESCE(c.sec_pct_mda, 0.0) as sec_pct_mda,
            COALESCE(c.sec_pct_risk_factors, 0.0) as sec_pct_risk_factors,
            COALESCE(c.sec_pct_other, 0.0) as sec_pct_other,
            
            -- Use case diversity coalesced
            COALESCE(c.avg_count_ai_use_case_types, 0.0) as avg_count_ai_use_case_types,

            -- Ratio features
            COALESCE(c.avg_ratio_vague_words, 0.0) as avg_ratio_vague_words,
            COALESCE(c.avg_ratio_forward_to_realized, 0.5) as avg_ratio_forward_to_realized
            {outer_select_str},
            (b.year >= 2024) as post_sec_2024,
            (b.year >= 2025) as post_deepseek
        FROM backbone b
        LEFT JOIN chunk_aggregates c ON b.ticker = c.ticker AND b.year = c.year
        LEFT JOIN read_parquet('{universe_path}') u ON b.ticker = u.ticker
        ORDER BY b.ticker, b.year;
        """
        
        print("Aggregating scored chunks per firm-year via DuckDB SQL...")
        panel_df = con.execute(sql_query).df()
    else:
        print("Warning: ai_scored_chunks.parquet not found or empty. Creating empty features panel.")
        manifest_df["year"] = pd.to_datetime(manifest_df["filing_date"]).dt.year
        firm_years = manifest_df[["ticker", "year"]].drop_duplicates().copy()
        panel_df = firm_years.merge(universe_df[["ticker", "industry_group"]], on="ticker", how="left")
        panel_df["industry_group"] = panel_df["industry_group"].fillna("Unclassified")
        
        metrics = [
            "ai_mentions_count", "avg_specificity", "avg_operational_grounding", "share_substantive",
            "avg_promotional_score", "avg_risk_score", "avg_governance_score", "avg_bow_sentiment",
            "avg_llm_sentiment", "share_promotional", "share_governance", "share_risk",
            "share_llm_is_ai_related", "share_llm_is_substantive", "share_llm_is_promotional",
            "share_llm_is_risk_related", "share_llm_is_governance", "share_llm_is_financial_impact",
            "share_llm_mentions_training", "share_llm_mentions_vendor", "share_llm_mentions_cloud",
            "share_llm_mentions_copilot", "sec_pct_business", "sec_pct_mda", "sec_pct_risk_factors",
            "sec_pct_other", "avg_count_ai_use_case_types",
            "share_financial_quantification", "share_ai_use_case_specific", "share_competitor_ai_mention",
            "share_deepseek_impact", "share_ai_model_name"
        ]
        for met in metrics:
            panel_df[met] = 0 if met == "ai_mentions_count" else 0.0
            
        panel_df["post_sec_2024"] = panel_df["year"] >= 2024
        panel_df["post_deepseek"] = panel_df["year"] >= 2025

    # Ensure correct column ordering dynamically
    base_cols = ["ticker", "year", "industry_group", "ai_mentions_count"]
    flags = ["post_sec_2024", "post_deepseek"]
    metric_cols = [c for c in panel_df.columns if c not in base_cols + flags]
    metric_cols.sort()
    
    cols = base_cols + metric_cols + flags
    panel_df = panel_df[cols].copy()
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(output_path, index=False)
    
    pipeline_logger.log_event(
        pipeline_step="build_features",
        level="SUCCESS",
        message=f"Panel feature dataset built successfully. Saved {len(panel_df)} firm-year observations to {output_path}.",
        details={"variant": variant}
    )
    print(f"Success: Saved {len(panel_df)} firm-year observations to {output_path}")

if __name__ == "__main__":
    main()
