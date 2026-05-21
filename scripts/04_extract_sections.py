import os
import json
import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime

# Regex patterns for section start and end
PATTERNS = {
    "Item 1": {
        "start": r"^\s*Item\s+1\.?\s+(?:Business|BUSINESS)\b",
        "end": r"^\s*Item\s+1A\.?\s+(?:Risk|RISK)\s+(?:Factors|FACTORS)\b"
    },
    "Item 1A": {
        "start": r"^\s*Item\s+1A\.?\s+(?:Risk|RISK)\s+(?:Factors|FACTORS)\b",
        "end": r"^\s*Item\s+1B\b|^\s*Item\s+2\b"
    },
    "Item 7": {
        "start": r"^\s*Item\s+7\.?\s+(?:Management|MANAGEMENT)\b",
        "end": r"^\s*Item\s+7A\b|^\s*Item\s+8\b"
    }
}

def clean_html_to_lines(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
        
    soup = BeautifulSoup(html_content, "lxml")
    for script in soup(["script", "style"]):
        script.decompose()
        
    # Append newlines to block tags to preserve layout
    for block in soup.find_all(['p', 'div', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br']):
        block.append('\n')
        
    text = soup.get_text()
    
    # Split into clean lines
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = [line for line in lines if line]
    return cleaned_lines

def extract_section(lines, start_re, end_re):
    starts = [i for i, line in enumerate(lines) if re.search(start_re, line, re.IGNORECASE)]
    if not starts:
        return ""
        
    # Check from last match to first to bypass Table of Contents (TOC)
    for start_idx in reversed(starts):
        ends = [i for i, line in enumerate(lines) if i > start_idx and re.search(end_re, line, re.IGNORECASE)]
        if ends:
            end_idx = ends[0]
            section_text = "\n".join(lines[start_idx:end_idx])
            if len(section_text) > 1000:
                return section_text
                
    # Fallback to taking from the last start line up to 8000 lines
    last_start = starts[-1]
    section_text = "\n".join(lines[last_start:last_start + 8000])
    if len(section_text) > 1000:
        return section_text
        
    return ""

def main():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    sections_path = Path(config["paths"]["interim_sections"]) / "filing_sections.parquet"
    
    # Make parent dirs if not existing
    sections_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}. Run download script first.")
        return
        
    df = pd.read_parquet(manifest_path)
    
    # Process completed downloads that haven't been parsed yet
    pending_parse = df[(df["download_status"] == "completed") & (df["parse_status"] == "pending")]
    
    if len(pending_parse) == 0:
        print("No pending filings to parse.")
        return
        
    # Load existing sections if file exists
    if sections_path.exists():
        existing_sections = pd.read_parquet(sections_path)
        # Identify accession numbers we are processing to remove duplicates
        acc_to_remove = pending_parse["accession_number"].tolist()
        existing_sections = existing_sections[~existing_sections["accession_number"].isin(acc_to_remove)]
        sections_list = existing_sections.to_dict("records")
    else:
        sections_list = []
        
    print(f"Extracting sections from {len(pending_parse)} filings...")
    
    for idx, row in tqdm(pending_parse.iterrows(), total=len(pending_parse)):
        html_path = Path(row["local_path"])
        acc_num = row["accession_number"]
        ticker = row["ticker"]
        filing_date = row["filing_date"]
        
        if not html_path.exists():
            df.at[idx, "parse_status"] = "failed: raw file missing"
            df.at[idx, "updated_at"] = datetime.now()
            continue
            
        try:
            lines = clean_html_to_lines(html_path)
            extracted_any = False
            
            for sec_name, pat in PATTERNS.items():
                sec_text = extract_section(lines, pat["start"], pat["end"])
                if sec_text:
                    char_len = len(sec_text)
                    word_count = len(sec_text.split())
                    
                    sections_list.append({
                        "accession_number": acc_num,
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "section_name": sec_name,
                        "section_text": sec_text,
                        "char_len": char_len,
                        "word_count": word_count
                    })
                    extracted_any = True
                    
            if extracted_any:
                df.at[idx, "parse_status"] = "completed"
            else:
                df.at[idx, "parse_status"] = "failed: no sections extracted"
                
            df.at[idx, "updated_at"] = datetime.now()
            
        except Exception as e:
            df.at[idx, "parse_status"] = f"failed: {str(e)}"
            df.at[idx, "updated_at"] = datetime.now()
            
        # Incremental save of manifest and sections
        df.to_parquet(manifest_path, index=False)
        if sections_list:
            pd.DataFrame(sections_list).to_parquet(sections_path, index=False)
            
    print(f"Finished section extraction. Extracted sections saved to {sections_path}")

if __name__ == "__main__":
    main()
