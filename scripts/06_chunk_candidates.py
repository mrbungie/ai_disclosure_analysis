import os
import json
import re
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    import harness_fit
    import pipeline_logger
except ImportError:
    from scripts import harness_fit, pipeline_logger

# Keyword family mapping
KEYWORD_FAMILIES = {
    "generative ai": "GenAI",
    "gen ai": "GenAI",
    "llm": "GenAI",
    "large language model": "GenAI",
    "artificial intelligence": "AI",
    "machine learning": "ML",
    "deep learning": "ML",
    "neural network": "ML",
    "computer vision": "ML",
    "natural language processing": "ML",
    "predictive analytics": "ML",
    "algorithmic": "Algo",
    "automation": "Automation"
}

def get_keyword_families(text, keywords, false_positives):
    # Clean false positives before matching families
    cleaned_text = text
    for fp in false_positives:
        cleaned_text = re.sub(r"\b" + re.escape(fp) + r"\b", "", cleaned_text, flags=re.IGNORECASE)
        
    families = set()
    matches_count = 0
    
    for kw, family in KEYWORD_FAMILIES.items():
        pattern = r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b"
        matches = re.findall(pattern, cleaned_text, flags=re.IGNORECASE)
        if matches:
            families.add(family)
            matches_count += len(matches)
            
    return list(families), matches_count

def main():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    detection_state = harness_fit.load_detection_state()
    keywords = detection_state["ai_keywords"]
    false_positives = detection_state["false_positives"]
    
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    sections_path = Path(config["paths"]["interim_sections"]) / "filing_sections.parquet"
    output_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists() or not sections_path.exists():
        pipeline_logger.log_event(
            pipeline_step="chunking",
            level="ERROR",
            message=f"Required inputs missing. manifest_exists={manifest_path.exists()}, sections_exists={sections_path.exists()}"
        )
        return
        
    manifest_df = pd.read_parquet(manifest_path)
    sections_df = pd.read_parquet(sections_path)
    
    # Load existing candidate chunks if they exist
    existing_chunks_df = None
    if output_path.exists():
        try:
            existing_chunks_df = pd.read_parquet(output_path)
            pipeline_logger.log_event(
                pipeline_step="chunking",
                level="INFO",
                message=f"Loaded {len(existing_chunks_df)} existing candidate chunks."
            )
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="chunking",
                level="WARNING",
                message=f"Could not load existing chunks parquet: {e}"
            )
            
    # We only process filings that have been matched in the prefilter stage
    matched_filings = manifest_df[manifest_df["prefilter_status"] == "matched"]
    
    # Identify which accession numbers have already been chunked
    chunked_accessions = set()
    if existing_chunks_df is not None:
        chunked_accessions = set(existing_chunks_df["accession_number"].unique())
        
    # Filter to only filings that are not yet chunked
    pending_chunking = matched_filings[~matched_filings["accession_number"].isin(chunked_accessions)]
    
    if len(pending_chunking) == 0:
        pipeline_logger.log_event(
            pipeline_step="chunking",
            level="INFO",
            message="No pending prefiltered filings to chunk."
        )
        return
        
    print("Compiling regex patterns...")
    # Patterns for finding matches in individual paragraphs
    patterns = [r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b" for kw in keywords]
    ai_regex = re.compile("|".join(patterns), re.IGNORECASE)
    
    chunks_list = []
    
    pipeline_logger.log_event(
        pipeline_step="chunking",
        level="INFO",
        message=f"Creating candidate chunks for {len(pending_chunking)} filings..."
    )
    
    for _, row in tqdm(pending_chunking.iterrows(), total=len(pending_chunking)):
        acc_num = row["accession_number"]
        ticker = row["ticker"]
        filing_date = row["filing_date"]
        
        # Get sections
        filing_sections = sections_df[sections_df["accession_number"] == acc_num]
        
        for _, sec_row in filing_sections.iterrows():
            sec_name = sec_row["section_name"]
            text = sec_row["section_text"]
            
            # Split section into paragraphs (lines)
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            
            # Find matching paragraph indices
            matched_indices = []
            for i, p in enumerate(paragraphs):
                # Remove false positives before checking match
                cleaned_p = p
                for fp in false_positives:
                    cleaned_p = re.sub(r"\b" + re.escape(fp) + r"\b", "", cleaned_p, flags=re.IGNORECASE)
                
                if ai_regex.search(cleaned_p):
                    matched_indices.append(i)
                    
            if not matched_indices:
                continue
                
            # Merge overlapping windows
            # Each match i has a window [i-1, i, i+1] (clamped to bounds)
            windows = []
            for idx in matched_indices:
                start = max(0, idx - 1)
                end = min(len(paragraphs) - 1, idx + 1)
                windows.append((start, end))
                
            # Merge intervals
            merged_windows = []
            for start, end in sorted(windows):
                if not merged_windows or merged_windows[-1][1] < start:
                    merged_windows.append((start, end))
                else:
                    merged_windows[-1] = (merged_windows[-1][0], max(merged_windows[-1][1], end))
                    
            # Extract chunks
            for w_start, w_end in merged_windows:
                chunk_paras = paragraphs[w_start:w_end + 1]
                chunk_text = "\n\n".join(chunk_paras)
                
                # Check keyword families and count in the whole chunk
                families, kw_count = get_keyword_families(chunk_text, keywords, false_positives)
                family_str = ", ".join(families) if families else "Unclassified"
                
                # Compute SHA-256 hash of chunk text
                text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                chunk_id = text_hash[:16] # Keep a short unique ID
                
                chunks_list.append({
                    "chunk_id": chunk_id,
                    "accession_number": acc_num,
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "section_name": sec_name,
                    "chunk_text": chunk_text,
                    "ai_keyword_count": kw_count,
                    "keyword_family": family_str,
                    "rule_score": 0.0, # Will be computed in the scoring script
                    "text_hash": text_hash
                })
                
    if chunks_list:
        new_chunks_df = pd.DataFrame(chunks_list)
        if existing_chunks_df is not None:
            # Drop any existing chunks for the accession numbers we just processed
            processed_acc_nums = new_chunks_df["accession_number"].unique()
            existing_chunks_df = existing_chunks_df[~existing_chunks_df["accession_number"].isin(processed_acc_nums)]
            combined_df = pd.concat([existing_chunks_df, new_chunks_df], ignore_index=True)
            pipeline_logger.log_event(
                pipeline_step="chunking",
                level="INFO",
                message=f"Combined existing and new chunks. Total chunks: {len(combined_df)}"
            )
        else:
            combined_df = new_chunks_df
            pipeline_logger.log_event(
                pipeline_step="chunking",
                level="INFO",
                message=f"Created new chunks table with {len(combined_df)} chunks."
            )
            
        combined_df.to_parquet(output_path, index=False)
        pipeline_logger.log_event(
            pipeline_step="chunking",
            level="SUCCESS",
            message=f"Successfully saved {len(new_chunks_df)} new chunks. Total candidate chunks: {len(combined_df)} at {output_path}",
            details={"new_chunks_count": len(new_chunks_df), "total_chunks_count": len(combined_df)}
        )
    else:
        pipeline_logger.log_event(
            pipeline_step="chunking",
            level="INFO",
            message="No new chunks created."
        )

if __name__ == "__main__":
    main()
