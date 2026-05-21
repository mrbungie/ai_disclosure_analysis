import os
import json
import re
import warnings
import pandas as pd
from pathlib import Path
from bs4 import XMLParsedAsHTMLWarning
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm
from datetime import datetime

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

# Suppress BS4 XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Space-tolerant and Markdown-tolerant regex patterns for section starts and ends
PATTERNS = {
    "Item 1": {
        "start": r"^\s*(?:[|#*_\s-]*)\s*(?:Item\s+1\.?\s*(?:[|#*_\s-]*)\s*(?:Business|BUSINESS)|ABOUT\s+HONEYWELL)\b",
        "end": r"^\s*(?:[|#*_\s-]*)\s*(?:Item\s+1A\.?\s*(?:[|#*_\s-]*)\s*(?:Risk|RISK)\s*(?:[|#*_\s-]*)\s*(?:Factors|FACTORS)|RISK\s+FACTORS)\b"
    },
    "Item 1A": {
        "start": r"^\s*(?:[|#*_\s-]*)\s*(?:Item\s+1A\.?\s*(?:[|#*_\s-]*)\s*(?:Risk|RISK)\s*(?:[|#*_\s-]*)\s*(?:Factors|FACTORS)|RISK\s+FACTORS)\b",
        "end": r"^\s*(?:[|#*_\s-]*)\s*(?:Item\s+1B\b|Item\s+2\b|UNRESOLVED\s+STAFF\s+COMMENTS\b|PROPERTIES\b)"
    },
    "Item 7": {
        "start": r"^\s*(?:[|#*_\s-]*)\s*(?:Item\s+7\.?\s*(?:[|#*_\s-]*)\s*(?:Management|MANAGEMENT)|MANAGEMENT[’']S\s+DISCUSSION\s+AND\s+ANALYSIS)\b",
        "end": r"^\s*(?:[|#*_\s-]*)\s*(?:Item\s+7A\b|Item\s+8\b|QUANTITATIVE\s+AND\s+QUALITATIVE\b|FINANCIAL\s+STATEMENTS\b)"
    }
}

def clean_html_to_lines(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
        
    # Convert HTML structure to Markdown, stripping scripts and styling blocks
    markdown_text = md(html_content, heading_style="ATX", strip=['script', 'style'])
    
    # Split into clean lines
    lines = [line.strip() for line in markdown_text.split('\n')]
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
        pipeline_logger.log_event(
            pipeline_step="extract_sections",
            level="ERROR",
            message=f"Manifest file not found at {manifest_path}. Run download script first."
        )
        return
        
    df = pd.read_parquet(manifest_path)
    
    # Load existing sections if they exist
    existing_sections_df = None
    if sections_path.exists():
        try:
            existing_sections_df = pd.read_parquet(sections_path)
            pipeline_logger.log_event(
                pipeline_step="extract_sections",
                level="INFO",
                message=f"Loaded {len(existing_sections_df)} existing parsed sections."
            )
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="extract_sections",
                level="WARNING",
                message=f"Could not load existing sections parquet: {e}"
            )
            
    pending_parse = df[(df["download_status"] == "completed") & (df["parse_status"] == "pending")]
    
    if len(pending_parse) == 0:
        pipeline_logger.log_event(
            pipeline_step="extract_sections",
            level="INFO",
            message="No pending filings to parse."
        )
        return
        
    sections_list = []
        
    pipeline_logger.log_event(
        pipeline_step="extract_sections",
        level="INFO",
        message=f"Extracting narrative sections as Markdown from {len(pending_parse)} filings..."
    )
    
    for idx, row in tqdm(pending_parse.iterrows(), total=len(pending_parse)):
        html_path = Path(row["local_path"])
        acc_num = row["accession_number"]
        ticker = row["ticker"]
        filing_date = row["filing_date"]
        
        if not html_path.exists():
            df.at[idx, "parse_status"] = "failed: raw file missing"
            df.at[idx, "updated_at"] = datetime.now()
            df.to_parquet(manifest_path, index=False)
            pipeline_logger.log_event(
                pipeline_step="extract_sections",
                level="ERROR",
                message=f"Raw filing html file missing at {html_path}",
                ticker=ticker,
                cik=row.get("cik"),
                accession_number=acc_num
            )
            continue
            
        try:
            lines = clean_html_to_lines(html_path)
            extracted_any = False
            extracted_count = 0
            extracted_details = {}
            
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
                    extracted_count += 1
                    extracted_details[sec_name] = {"char_len": char_len, "word_count": word_count}
                    
            if extracted_any:
                df.at[idx, "parse_status"] = "completed"
                pipeline_logger.log_event(
                    pipeline_step="extract_sections",
                    level="SUCCESS",
                    message=f"Successfully extracted {extracted_count} sections",
                    ticker=ticker,
                    cik=row.get("cik"),
                    accession_number=acc_num,
                    details=extracted_details
                )
            else:
                df.at[idx, "parse_status"] = "failed: no sections extracted"
                pipeline_logger.log_event(
                    pipeline_step="extract_sections",
                    level="WARNING",
                    message="No sections (Item 1, 1A, 7) extracted from filing",
                    ticker=ticker,
                    cik=row.get("cik"),
                    accession_number=acc_num
                )
                
            df.at[idx, "updated_at"] = datetime.now()
            
        except Exception as e:
            df.at[idx, "parse_status"] = f"failed: {str(e)}"
            df.at[idx, "updated_at"] = datetime.now()
            pipeline_logger.log_event(
                pipeline_step="extract_sections",
                level="ERROR",
                message=f"Exception extracting sections: {e}",
                ticker=ticker,
                cik=row.get("cik"),
                accession_number=acc_num,
                details={"error": str(e)}
            )
            
        # Incremental save of manifest to keep progress
        df.to_parquet(manifest_path, index=False)
        
    if sections_list:
        new_sections_df = pd.DataFrame(sections_list)
        if existing_sections_df is not None:
            # Drop any existing sections for the accession numbers we just processed
            processed_acc_nums = new_sections_df["accession_number"].unique()
            existing_sections_df = existing_sections_df[~existing_sections_df["accession_number"].isin(processed_acc_nums)]
            combined_df = pd.concat([existing_sections_df, new_sections_df], ignore_index=True)
            pipeline_logger.log_event(
                pipeline_step="extract_sections",
                level="INFO",
                message=f"Combined existing and new sections. Total sections: {len(combined_df)}"
            )
        else:
            combined_df = new_sections_df
            pipeline_logger.log_event(
                pipeline_step="extract_sections",
                level="INFO",
                message=f"Created new sections table with {len(combined_df)} sections."
            )
            
        combined_df.to_parquet(sections_path, index=False)
            
    pipeline_logger.log_event(
        pipeline_step="extract_sections",
        level="SUCCESS",
        message=f"Finished section extraction. Extracted sections saved to {sections_path}",
        details={"sections_count": len(combined_df) if 'combined_df' in locals() else 0}
    )

if __name__ == "__main__":
    main()
