import os
import sys
import time
import json
import asyncio
import argparse
from enum import Enum
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger

# Load environment variables from .env file
load_dotenv()

class AmountKind(str, Enum):
    CapEx = "CapEx"
    OpEx = "OpEx"
    Revenue = "Revenue"
    Investment = "Investment"
    R_D = "R&D"
    Other = "Other"

class EntityType(str, Enum):
    Partner = "Partner"
    Competitor = "Competitor"
    Vendor = "Vendor"
    Customer = "Customer"
    RegulatoryBody = "RegulatoryBody"
    CloudProvider = "CloudProvider"
    Other = "Other"

class SentimentKind(str, Enum):
    Positive = "Positive"
    Negative = "Negative"
    Neutral = "Neutral"
    Mixed = "Mixed"

class AmountExtraction(BaseModel):
    value: str = Field(
        description="The raw string/text representing the financial amount as written (e.g. '$10 million', '500,000 dollars', '€50k')."
    )
    numeric_value: float | None = Field(
        default=None,
        description="The parsed numeric value of the amount, normalized to a float (e.g. 10000000.0 for $10M). Null if it cannot be determined."
    )
    kind: AmountKind = Field(
        description="The kind of financial amount (CapEx, OpEx, Revenue, Investment, R&D, Other)."
    )

class EntityExtraction(BaseModel):
    name: str = Field(
        description="The name of the entity (e.g. 'NVIDIA', 'OpenAI', 'Microsoft', 'SEC')."
    )
    type: EntityType = Field(
        description="The type of entity (Partner, Competitor, Vendor, Customer, RegulatoryBody, CloudProvider, Other)."
    )

class AIDisclosureAnalysis(BaseModel):
    is_ai_related: bool = Field(
        description="True if the text specifically discusses artificial intelligence (AI), machine learning (ML), deep learning, neural networks, or LLMs."
    )
    is_substantive: bool = Field(
        description="True if the mention contains concrete operational details, specific use cases, deployed systems, metrics, or investments. False if it is only generic boilerplate, passing phrases, or vague/hypothetical comments."
    )
    is_promotional: bool = Field(
        description="True if the language focuses on marketing, buzzwords, or optimistic claims about AI without specific operational grounding."
    )
    is_risk_related: bool = Field(
        description="True if discussing risk factors, cybersecurity threats, intellectual property issues, legal liability, model hallucinations, or regulation of AI."
    )
    is_governance_related: bool = Field(
        description="True if discussing internal AI policies, board oversight, risk committees, compliance procedures, or ethical frameworks."
    )
    mentions_copilot: bool = Field(
        description="True if specifically mentioning Microsoft Copilot, GitHub Copilot, or similar assistant tools."
    )
    mentions_cloud: bool = Field(
        description="True if mentioning cloud computing providers or cloud infrastructure (e.g. AWS, Azure, Google Cloud) in the context of running/deploying AI."
    )
    mentions_vendor: bool = Field(
        description="True if mentioning external AI vendors, models, or partnerships (e.g. OpenAI, ChatGPT, Microsoft, NVIDIA, Google, Anthropic)."
    )
    mentions_training: bool = Field(
        description="True if discussing training models, dataset collection, data curation, fine-tuning, or compute cluster setup."
    )
    is_financial_impact: bool = Field(
        description="True if discussing financial metrics related to AI (e.g. revenues generated, capital expenditure/CapEx, R&D costs, budgets)."
    )
    amounts: list[AmountExtraction] = Field(
        default_factory=list,
        description="List of specific financial amounts (monetary figures) mentioned in the text related to AI activities."
    )
    entities: list[EntityExtraction] = Field(
        default_factory=list,
        description="List of specific organizations, partners, competitors, vendors, or regulatory bodies mentioned in the context of AI."
    )
    sentiment: SentimentKind = Field(
        description="The sentiment of the text chunk regarding AI adoption, deployment, or risks (Positive, Negative, Neutral, Mixed)."
    )
    rationale_short: str = Field(
        description="A short (1-2 sentences) explanation of the classification decisions."
    )

def load_config():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        return json.load(f)

def update_filing_manifest_status(manifest_path, chunks_df, mentions_df):
    """
    Update the llm_status in the filing manifest.
    - filings with prefilter_status == 'no_matches' are marked 'completed'.
    - filings where all candidate chunks have classifications are marked 'completed'.
    - filings with partial classifications are marked 'partial'.
    - filings with no classifications are marked 'pending' or left as is.
    """
    if not manifest_path.exists():
        return
        
    manifest_df = pd.read_parquet(manifest_path)
    
    # 1. Handle no_matches filings
    no_matches_mask = (manifest_df["prefilter_status"] == "no_matches")
    manifest_df.loc[no_matches_mask, "llm_status"] = "completed"
    
    # 2. Group candidate chunks and mentions by accession number
    if len(chunks_df) > 0:
        chunks_count = chunks_df.groupby("accession_number").size().to_dict()
    else:
        chunks_count = {}
        
    if len(mentions_df) > 0:
        mentions_count = mentions_df.groupby("accession_number").size().to_dict()
    else:
        mentions_count = {}
        
    # 3. Update status for matched filings
    matched_filings = manifest_df[manifest_df["prefilter_status"] == "matched"]
    for idx, row in matched_filings.iterrows():
        acc_num = row["accession_number"]
        total_chunks = chunks_count.get(acc_num, 0)
        processed_chunks = mentions_count.get(acc_num, 0)
        
        if total_chunks == 0:
            manifest_df.at[idx, "llm_status"] = "completed"
        elif processed_chunks == total_chunks:
            manifest_df.at[idx, "llm_status"] = "completed"
        elif processed_chunks > 0:
            manifest_df.at[idx, "llm_status"] = "partial"
        else:
            manifest_df.at[idx, "llm_status"] = "pending"
            
    manifest_df.to_parquet(manifest_path, index=False)

async def save_results(new_records, mentions_path, write_lock):
    async with write_lock:
        if not new_records:
            return
        try:
            if mentions_path.exists():
                existing_df = pd.read_parquet(mentions_path)
            else:
                existing_df = pd.DataFrame()
        except Exception:
            existing_df = pd.DataFrame()
            
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["chunk_id"], keep="last")
        
        mentions_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(mentions_path, index=False)

async def classify_chunk_single(row, agent, write_lock, mentions_path, progress, delay):
    chunk_id = row["chunk_id"]
    text = row["chunk_text"]
    ticker = row["ticker"]
    acc_num = row["accession_number"]
    
    max_retries = 5
    retry_delay = 5.0  # Start with 5 seconds for rate limit recovery
    
    for attempt in range(max_retries):
        try:
            result = await agent.run(text)
            output = result.output
            
            # Construct record
            record = output.model_dump()
            record["chunk_id"] = chunk_id
            record["accession_number"] = acc_num
            
            # Save incrementally
            await save_results([record], mentions_path, write_lock)
            
            progress["success"] += 1
            completed = progress["success"] + progress["fail"]
            print(f"[{completed}/{progress['total']}] Classifying chunk {chunk_id} for {ticker}... SUCCESS")
            
            if delay > 0:
                await asyncio.sleep(delay)
            return True
            
        except Exception as e:
            error_str = str(e)
            is_rate_limit = False
            
            # Check for 429 rate limit indicators
            if "429" in error_str or "Too Many Requests" in error_str or "rate limit" in error_str.lower():
                is_rate_limit = True
            elif hasattr(e, 'status_code') and getattr(e, 'status_code') == 429:
                is_rate_limit = True
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    print(f"Rate limited (429) on chunk {chunk_id} for {ticker}. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2.0
                    continue
                else:
                    # Exhausted retries
                    progress["fail"] += 1
                    completed = progress["success"] + progress["fail"]
                    print(f"[{completed}/{progress['total']}] Classifying chunk {chunk_id} for {ticker}... FAILED: Exhausted retries (429)")
                    pipeline_logger.log_event(
                        pipeline_step="llm_extraction",
                        level="WARNING",
                        message=f"Failed to classify chunk {chunk_id} for {ticker}: Exhausted retries due to rate limiting (429).",
                        ticker=ticker,
                        accession_number=acc_num
                    )
                    return False
            else:
                # Non-rate-limit error (e.g. credential, token validation, bad input) - fail immediately without retrying
                progress["fail"] += 1
                completed = progress["success"] + progress["fail"]
                print(f"[{completed}/{progress['total']}] Classifying chunk {chunk_id} for {ticker}... FAILED: {e}")
                pipeline_logger.log_event(
                    pipeline_step="llm_extraction",
                    level="WARNING",
                    message=f"Failed to classify chunk {chunk_id} for {ticker}: {e}",
                    ticker=ticker,
                    accession_number=acc_num
                )
                return False


async def main_async(limit=None, concurrency=1, delay=1.0):
    config = load_config()
    
    # Setup paths
    manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
    chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"
    mentions_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_mentions.parquet"
    
    if not chunks_path.exists():
        pipeline_logger.log_event(
            pipeline_step="llm_extraction",
            level="ERROR",
            message=f"Candidate chunks Parquet not found at {chunks_path}"
        )
        print(f"Error: Candidate chunks not found at {chunks_path}")
        return
        
    # Load candidate chunks
    chunks_df = pd.read_parquet(chunks_path)
    if len(chunks_df) == 0:
        pipeline_logger.log_event(
            pipeline_step="llm_extraction",
            level="WARNING",
            message="No candidate chunks found to classify."
        )
        print("Warning: No candidate chunks to process.")
        return
        
    # Load or initialize existing mentions
    existing_df = None
    processed_ids = set()
    if mentions_path.exists():
        try:
            existing_df = pd.read_parquet(mentions_path)
            processed_ids = set(existing_df["chunk_id"].unique())
            print(f"Loaded {len(existing_df)} existing classified chunks.")
        except Exception as e:
            pipeline_logger.log_event(
                pipeline_step="llm_extraction",
                level="WARNING",
                message=f"Could not load existing mentions parquet: {e}"
            )
            
    # Filter outstanding chunks
    outstanding_df = chunks_df[~chunks_df["chunk_id"].isin(processed_ids)]
    
    if len(outstanding_df) == 0:
        pipeline_logger.log_event(
            pipeline_step="llm_extraction",
            level="INFO",
            message="All candidate chunks have already been classified."
        )
        print("All candidate chunks have already been classified.")
        # Ensure manifest status is updated
        if existing_df is not None:
            update_filing_manifest_status(manifest_path, chunks_df, existing_df)
        return
        
    # Apply limit if specified
    if limit is not None:
        outstanding_df = outstanding_df.head(limit)
        print(f"Limiting execution to first {limit} outstanding chunks.")
        
    # Setup LLM Provider via environment
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        pipeline_logger.log_event(
            pipeline_step="llm_extraction",
            level="ERROR",
            message="NVIDIA_API_KEY environment variable is not set."
        )
        raise ValueError("NVIDIA_API_KEY is not set. Please configure it in your .env file.")
        
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model_name = os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    
    pipeline_logger.log_event(
        pipeline_step="llm_extraction",
        level="INFO",
        message=f"Starting LLM classifier. model={model_name}, base_url={base_url}, pending={len(outstanding_df)}"
    )
    print(f"Starting LLM classification using model: {model_name} (concurrency={concurrency}, delay={delay}s)...")
    
    # Initialize Pydantic AI
    provider = OpenAIProvider(
        base_url=base_url,
        api_key=api_key
    )
    model = OpenAIChatModel(
        model_name,
        provider=provider
    )
    agent = Agent(
        model, 
        output_type=AIDisclosureAnalysis,
        retries=3,
        system_prompt=(
            "You are an expert NLP classifier analyzing corporate disclosures in SEC filings.\n"
            "Your task is to analyze the provided chunk of text and:\n"
            "1. Determine which of the semantic boolean flags are present.\n"
            "2. Extract any specific financial amounts mentioned related to AI (e.g., Capex, research costs, investments, etc.) and parse their numeric values.\n"
            "3. Extract specific entities (partners, competitors, vendors, customers, regulatory bodies, cloud providers) mentioned in the context of AI.\n"
            "4. Determine the overall sentiment (Positive, Negative, Neutral, Mixed) of the text regarding AI adoption, deployment, or risks.\n"
            "Set all fields accurately based on the text. Keep the rationale short (maximum 2 sentences)."
        )
    )
    
    # Run async classification with a worker pool (which directly controls concurrency)
    write_lock = asyncio.Lock()
    progress = {"success": 0, "fail": 0, "total": len(outstanding_df)}
    
    # Create queue and populate it
    queue = asyncio.Queue()
    for _, row in outstanding_df.iterrows():
        queue.put_nowait(row)
        
    async def worker():
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            
            await classify_chunk_single(
                row=row,
                agent=agent,
                write_lock=write_lock,
                mentions_path=mentions_path,
                progress=progress,
                delay=delay
            )
            queue.task_done()
            
    # Start concurrency number of workers (max concurrency)
    concurrency_val = max(1, concurrency)
    workers = [asyncio.create_task(worker()) for _ in range(concurrency_val)]
    
    # Wait for all workers to finish
    await asyncio.gather(*workers)
    
    # Update manifestation status
    if mentions_path.exists():
        final_mentions_df = pd.read_parquet(mentions_path)
        update_filing_manifest_status(manifest_path, chunks_df, final_mentions_df)
        
    pipeline_logger.log_event(
        pipeline_step="llm_extraction",
        level="SUCCESS",
        message=f"LLM classification round finished. Success: {progress['success']}, Failures: {progress['fail']}",
        details={"success_count": progress['success'], "fail_count": progress['fail']}
    )
    print(f"LLM classification finished. Success: {progress['success']}, Failures: {progress['fail']}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM classifier on candidate chunks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limit number of chunks to process (alias for --limit)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests to LLM")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds after each request (default: 1.0)")
    args = parser.parse_args()
    
    limit = args.max_jobs if args.max_jobs is not None else args.limit
    asyncio.run(main_async(limit=limit, concurrency=args.concurrency, delay=args.delay))

if __name__ == "__main__":
    main()

