import os
import re
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

def run_rules(text):
    # Regex patterns (compiled inside or outside, here compiled on first use or simple compiled variables)
    # 1. Board oversight: matches board and oversight within 50 characters, not crossing sentence boundaries
    board_oversight_pat = re.compile(
        r"\bboard(?:'s)?\b[^.!?]{0,50}\boversight\b|\boversight\b[^.!?]{0,50}\bboard(?:'s)?\b", 
        re.IGNORECASE
    )
    # 2. Audit committee
    audit_comm_pat = re.compile(r"\baudit\s+committees?\b", re.IGNORECASE)
    # 3. Vendors/Models
    nvidia_pat = re.compile(r"\bnvidia\b", re.IGNORECASE)
    openai_pat = re.compile(r"\bopenai\b|\bchatgpt\b", re.IGNORECASE)
    microsoft_pat = re.compile(r"\bmicrosoft\b|\bcopilot\b|\bazure\b", re.IGNORECASE)
    google_pat = re.compile(r"\bgoogle\b|\bgemini\b|\bbard\b|\balphabet\b", re.IGNORECASE)
    deepseek_pat = re.compile(r"\bdeepseek\b", re.IGNORECASE)
    # 4. Metrics (using non-word boundary support for %)
    percentage_pat = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)
    dollar_pat = re.compile(
        r"\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?\b|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)?\s*(?:dollars|usd)\b", 
        re.IGNORECASE
    )

    return {
        "has_board_oversight": bool(board_oversight_pat.search(text)),
        "has_audit_committee": bool(audit_comm_pat.search(text)),
        "has_vendor_nvidia": bool(nvidia_pat.search(text)),
        "has_vendor_openai": bool(openai_pat.search(text)),
        "has_vendor_microsoft": bool(microsoft_pat.search(text)),
        "has_vendor_google": bool(google_pat.search(text)),
        "has_vendor_deepseek": bool(deepseek_pat.search(text)),
        "has_metric_percentage": bool(percentage_pat.search(text)),
        "has_metric_dollar": bool(dollar_pat.search(text)),
    }

def main():
    config = load_config()
    
    chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"
    output_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_rules.parquet"
    
    if not chunks_path.exists():
        pipeline_logger.log_event(
            pipeline_step="rule_scoring",
            level="ERROR",
            message=f"Candidate chunks Parquet not found at {chunks_path}"
        )
        print(f"Error: Candidate chunks not found at {chunks_path}")
        return

    pipeline_logger.log_event(
        pipeline_step="rule_scoring",
        level="INFO",
        message="Loading candidate chunks..."
    )
    
    chunks_df = pd.read_parquet(chunks_path)
    
    if len(chunks_df) == 0:
        pipeline_logger.log_event(
            pipeline_step="rule_scoring",
            level="WARNING",
            message="No candidate chunks found in the parquet file."
        )
        print("Warning: No candidate chunks to process.")
        return

    pipeline_logger.log_event(
        pipeline_step="rule_scoring",
        level="INFO",
        message=f"Processing {len(chunks_df)} chunks with deterministic text rules..."
    )
    
    rules_list = []
    for _, row in chunks_df.iterrows():
        chunk_id = row["chunk_id"]
        text = row["chunk_text"]
        
        features = run_rules(text)
        features["chunk_id"] = chunk_id
        rules_list.append(features)
        
    rules_df = pd.DataFrame(rules_list)
    
    # Reorder columns to have chunk_id first
    cols = ["chunk_id"] + [c for c in rules_df.columns if c != "chunk_id"]
    rules_df = rules_df[cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rules_df.to_parquet(output_path, index=False)
    
    pipeline_logger.log_event(
        pipeline_step="rule_scoring",
        level="SUCCESS",
        message=f"Rule scoring complete. Saved {len(rules_df)} rule records to {output_path}."
    )
    print(f"Success: Saved {len(rules_df)} rule records to {output_path}")

if __name__ == "__main__":
    main()
