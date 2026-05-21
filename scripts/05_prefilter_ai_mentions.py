import os
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def build_keyword_regex(keywords):
    # Escape keywords and join with OR, wrapping in word boundaries
    patterns = []
    for kw in keywords:
        # Replace spaces with \s+ to match formatting/newlines inside terms
        pattern = r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b"
        patterns.append(pattern)
    return re.compile("|".join(patterns), re.IGNORECASE)

def main():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    keywords = config["prefiltering"]["ai_keywords"]
    false_positives = config["prefiltering"]["false_positives"]
    
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    sections_path = Path(config["paths"]["interim_sections"]) / "filing_sections.parquet"
    
    if not manifest_path.exists() or not sections_path.exists():
        print("Error: Required inputs (manifest or sections parquet) are missing.")
        return
        
    manifest_df = pd.read_parquet(manifest_path)
    sections_df = pd.read_parquet(sections_path)
    
    # We only process filings that have been parsed successfully
    parsed_filings = manifest_df[manifest_df["parse_status"] == "completed"]
    
    if len(parsed_filings) == 0:
        print("No parsed filings available to prefilter.")
        return
        
    print("Compiling regex patterns...")
    ai_regex = build_keyword_regex(keywords)
    
    print(f"Prefiltering {len(parsed_filings)} filings for AI mentions...")
    
    matched_filings = 0
    skipped_filings = 0
    
    for idx, row in tqdm(parsed_filings.iterrows(), total=len(parsed_filings)):
        acc_num = row["accession_number"]
        
        # Get sections for this filing
        filing_sections = sections_df[sections_df["accession_number"] == acc_num]
        
        has_ai_mention = False
        for _, sec_row in filing_sections.iterrows():
            text = sec_row["section_text"]
            
            # Remove known false positives before scanning (e.g. Adobe Illustrator)
            cleaned_text = text
            for fp in false_positives:
                cleaned_text = re.sub(r"\b" + re.escape(fp) + r"\b", "", cleaned_text, flags=re.IGNORECASE)
                
            # Find matches in cleaned text
            matches = ai_regex.findall(cleaned_text)
            if matches:
                has_ai_mention = True
                break
                
        if has_ai_mention:
            manifest_df.at[idx, "prefilter_status"] = "matched"
            matched_filings += 1
        else:
            manifest_df.at[idx, "prefilter_status"] = "no_matches"
            skipped_filings += 1
            
        manifest_df.at[idx, "updated_at"] = datetime.now()
        
    manifest_df.to_parquet(manifest_path, index=False)
    print(f"Prefiltering completed. Matched: {matched_filings}, Skipped: {skipped_filings}")

if __name__ == "__main__":
    main()
