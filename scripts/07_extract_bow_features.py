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

# Define word lists for Bag of Words (BoW) analysis
VAGUE_WORDS = [
    "revolutionize", "revolutionizing", "revolutionized",
    "transform", "transforming", "transformed", "transformative",
    "cutting-edge", "next-generation", "next-gen", "infuse", "infusing", "infused",
    "leader", "leadership", "empower", "empowering", "empowered",
    "unlock", "unlocking", "unlocked", "innovative", "innovation",
    "accelerate", "accelerating", "accelerated", "ecosystem", "seamless",
    "strategic wave", "breakthrough", "world-class", "pioneer", "pioneering",
    "reimagine", "reimagining", "reimagined"
]

SPECIFIC_WORDS = [
    "gpu", "gpus", "tpu", "tpus", "h100", "a100", "blackwell", "mi300", "cuda",
    "dataset", "datasets", "curation", "curating", "fine-tuning", "fine-tuned",
    "pre-training", "pre-trained", "parameter", "parameters", "compliance", "regulations",
    "regulation", "eu ai act", "gdpr", "ccpa", "cybersecurity", "breach", "breaches",
    "copyright", "copyrights", "patent", "patents", "board oversight", "audit committee",
    "fiduciary", "ethics committee", "responsible ai", "ethical ai"
]

AI_WORDS = [
    "ai", "artificial intelligence", "generative ai", "gen ai", "machine learning",
    "large language model", "llm", "llms", "deep learning", "neural network", "neural networks"
]

def run_bow_extraction(text):
    # --- PHASE 1: Boolean / Presence Features ---
    # 1. Governance
    board_oversight_pat = re.compile(
        r"\bboard(?:'s)?\b[^.!?]{0,50}\boversight\b|\boversight\b[^.!?]{0,50}\bboard(?:'s)?\b", 
        re.IGNORECASE
    )
    audit_comm_pat = re.compile(r"\baudit\s+committees?\b", re.IGNORECASE)
    ethics_policy_pat = re.compile(
        r"\bethics?\s+committee\b|\bethical\s+standards\b|\bethical\s+guidelines\b|\bcode\s+of\s+conduct\b|\bresponsible\s+ai\b|\bethical\s+ai\b|\bai\s+safety\b|\bai\s+policy\b|\bai\s+policies\b|\bai\s+governance\b|\btrust\s+(?:and|&)\s+safety\b",
        re.IGNORECASE
    )
    compliance_pat = re.compile(
        r"\bcompliance\b|\bregulatory\s+compliance\b|\bdata\s+governance\b|\bregulatory\s+frameworks?\b|\bcompliance\s+requirements?\b|\binternal\s+controls?\b|\bauditing\b",
        re.IGNORECASE
    )
    
    # 2. Risk
    risk_factor_pat = re.compile(
        r"\brisk\s+factors?\b|\brisks\s+and\s+uncertainties\b|\badversely\s+affect\b|\bmaterially\s+affect\b|\bnegative\s+impact\b|\bcould\s+be\s+harmed\b|\bsubject\s+to\s+risks?\b|\breputational\s+damage\b",
        re.IGNORECASE
    )
    regulatory_risk_pat = re.compile(
        r"\bregulatory\s+scrutiny\b|\bregulations?\b|\bregulatory\s+compliance\b|\blegal\s+frameworks?\b|\beu\s+ai\s+act\b|\bfederal\s+trade\s+commission\b|\bftc\b|\bsec\s+scrutiny\b|\blegislative\s+efforts?\b",
        re.IGNORECASE
    )
    cyber_privacy_risk_pat = re.compile(
        r"\bprivacy\b|\bdata\s+protection\b|\bcybersecurity\b|\bcyber-attacks?\b|\bdata\s+breach(?:es)?\b|\bexfiltration\b|\bunauthorized\s+access\b|\bdata\s+leakage\b|\bgdpr\b|\bccpa\b|\bcpra\b|\bpipl\b",
        re.IGNORECASE
    )
    ethics_bias_risk_pat = re.compile(
        r"\bbias(?:ed)?\b|\bdiscrimination\b|\btoxic(?:ity)?\b|\bhallucination\s+concern\b|\bhallucinate\b|\berrors?\b|\binaccurac(?:y|ies)\b|\bfairness\b|\bethical\s+concerns?\b|\bsocietal\s+harm\b|\bdeepfakes?\b",
        re.IGNORECASE
    )
    ip_copyright_risk_pat = re.compile(
        r"\bcopyright\b|\bcopyright\s+infringement\b|\bfair\s+use\b|\blicensing\b|\bintellectual\s+property\b|\bproprietary\b|\bpatents?\b|\btrade\s+secrets?\b|\btrademarks?\b",
        re.IGNORECASE
    )
    supply_infra_risk_pat = re.compile(
        r"\bgpu\s+shortages?\b|\bgpu\s+supply\b|\bchip\s+shortages?\b|\bcompute\s+capacity\b|\bcompute\s+constraints?\b|\bsemiconductor\s+supply\b|\benergy\s+consumption\b|\bpower\s+constraints?\b|\belectricity\b",
        re.IGNORECASE
    )
    
    # 3. Vendors/Models
    nvidia_pat = re.compile(r"\bnvidia\b|\brtx\b|\bdgx\b|\bh100\b|\ba100\b|\bblackwell\b", re.IGNORECASE)
    openai_pat = re.compile(r"\bopenai\b|\bchatgpt\b|\bgpt-4\b|\bgpt-3\b|\bopen\s+ai\b|\bchat\s+gpt\b", re.IGNORECASE)
    microsoft_pat = re.compile(r"\bmicrosoft\b|\bazure\b|\bcopilot\b", re.IGNORECASE)
    google_pat = re.compile(r"\bgoogle\b|\bgemini\b|\bbard\b|\bdeepmind\b|\balphabet\b", re.IGNORECASE)
    deepseek_pat = re.compile(r"\bdeepseek\b", re.IGNORECASE)
    amazon_pat = re.compile(r"\bamazon\b|\baws\b|\bbedrock\b", re.IGNORECASE)
    meta_pat = re.compile(r"\bmeta\b|\bllama\b", re.IGNORECASE)
    anthropic_pat = re.compile(r"\banthropic\b|\bclaude\b", re.IGNORECASE)
    amd_pat = re.compile(r"\bamd\b|\badvanced\s+micro\s+devices\b|\binstinct\b|\bmi300\b|\bryzen\b|\bepyc\b", re.IGNORECASE)
    
    # 4. Metrics / Product / Specificity
    percentage_pat = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)
    dollar_pat = re.compile(
        r"\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?\b|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)?\s*(?:dollars|usd)\b", 
        re.IGNORECASE
    )
    specific_product_pat = re.compile(
        r"\bcopilot\b|\bchatgpt\b|\bfirefly\b|\bsensei\b|\beinstein\b|\baip\b|\bwatson\b|\brtx\b|\bdgx\b|\bcuda\b|\binstinct\b|\bmi300\b", 
        re.IGNORECASE
    )
    deployment_verb_pat = re.compile(
        r"\blaunched\b|\bintroduced\b|\bintegrated\b|\brollout\b|\brolling\s+out\b|\brelease\s+of\b|\bcommercially\s+available\b|\bmonetize\b|\bmonetization\b|\bsubscription\b",
        re.IGNORECASE
    )
    
    # 5. Operational / Technical
    model_training_pat = re.compile(
        r"\btraining\b|\bfine-tuning\b|\bpre-training\b|\bparameter\s+size\b|\bdatasets?\b|\bdataset\s+curation\b|\bpre-train\b|\brlhf\b",
        re.IGNORECASE
    )
    compute_infra_pat = re.compile(
        r"\bgpus?\b|\bh100\b|\ba100\b|\bblackwell\b|\bcompute\b|\btpus?\b|\bdata\s+centers?\b|\bdatacenters?\b|\bservers?\b|\bsupercomputer\b",
        re.IGNORECASE
    )
    proprietary_data_pat = re.compile(
        r"\bproprietary\s+data\b|\bproprietary\s+datasets?\b|\binternal\s+database\b|\bcustomer\s+data\b",
        re.IGNORECASE
    )

    # 6. Newly identified dimensions
    workforce_talent_pat = re.compile(
        r"\btalent\b|\brecruiting\b|\bworkforce\b|\bheadcount\b|\bhiring\b|\bskills?\b|\bemployees?\b|\blabor\b",
        re.IGNORECASE
    )
    partnership_pat = re.compile(
        r"\bstrategic\s+alliance\b|\bpartnership\b|\bjoint\s+venture\b|\bcollaboration\b",
        re.IGNORECASE
    )
    customer_facing_pat = re.compile(
        r"\bcustomer\s+service\b|\bchatbot\b|\bvirtual\s+assistant\b|\bconversational\s+agent\b|\bcustomer\s+support\b|\bchatbots\b|\bagents?\b",
        re.IGNORECASE
    )
    safety_critical_pat = re.compile(
        r"\bself-driving\b|\bautonomous\s+vehicles?\b|\bautopilot\b|\bfsd\b|\bweapons?\b|\bdefense\b|\bmilitary\b|\baerospace\b",
        re.IGNORECASE
    )
    data_licensing_pat = re.compile(
        r"\bcontent\s+licensing\b|\blicensing\s+agreements?\b|\bpublisher\s+agreements?\b|\bscraping\b|\bfair\s+use\b|\bintellectual\s+property\s+claims\b",
        re.IGNORECASE
    )

    # Presences (bool)
    presences = {
        "has_board_oversight": bool(board_oversight_pat.search(text)),
        "has_audit_committee": bool(audit_comm_pat.search(text)),
        "has_ethics_policy": bool(ethics_policy_pat.search(text)),
        "has_compliance": bool(compliance_pat.search(text)),
        "has_risk_factor": bool(risk_factor_pat.search(text)),
        "has_regulatory_risk": bool(regulatory_risk_pat.search(text)),
        "has_cyber_privacy_risk": bool(cyber_privacy_risk_pat.search(text)),
        "has_ethics_bias_risk": bool(ethics_bias_risk_pat.search(text)),
        "has_ip_copyright_risk": bool(ip_copyright_risk_pat.search(text)),
        "has_supply_infra_risk": bool(supply_infra_risk_pat.search(text)),
        "has_vendor_nvidia": bool(nvidia_pat.search(text)),
        "has_vendor_openai": bool(openai_pat.search(text)),
        "has_vendor_microsoft": bool(microsoft_pat.search(text)),
        "has_vendor_google": bool(google_pat.search(text)),
        "has_vendor_deepseek": bool(deepseek_pat.search(text)),
        "has_vendor_amazon": bool(amazon_pat.search(text)),
        "has_vendor_meta": bool(meta_pat.search(text)),
        "has_vendor_anthropic": bool(anthropic_pat.search(text)),
        "has_vendor_amd": bool(amd_pat.search(text)),
        "has_metric_percentage": bool(percentage_pat.search(text)),
        "has_metric_dollar": bool(dollar_pat.search(text)),
        "has_specific_product": bool(specific_product_pat.search(text)),
        "has_deployment_verb": bool(deployment_verb_pat.search(text)),
        "has_model_training": bool(model_training_pat.search(text)),
        "has_compute_infra": bool(compute_infra_pat.search(text)),
        "has_proprietary_data": bool(proprietary_data_pat.search(text)),
        "has_workforce_talent": bool(workforce_talent_pat.search(text)),
        "has_partnership": bool(partnership_pat.search(text)),
        "has_customer_facing": bool(customer_facing_pat.search(text)),
        "has_safety_critical": bool(safety_critical_pat.search(text)),
        "has_data_licensing": bool(data_licensing_pat.search(text)),
    }

    # --- PHASE 2: Counts (Integer Features) ---
    # Total Words (whitespace split)
    words = text.split()
    count_total_words = len(words)
    
    # Counts of specific word categories
    vague_regex = re.compile(r"\b(" + "|".join(VAGUE_WORDS) + r")\b", re.IGNORECASE)
    specific_regex = re.compile(r"\b(" + "|".join(SPECIFIC_WORDS) + r")\b", re.IGNORECASE)
    ai_regex = re.compile(r"\b(" + "|".join(AI_WORDS) + r")\b", re.IGNORECASE)
    
    count_vague_words = len(vague_regex.findall(text))
    count_specific_words = len(specific_regex.findall(text))
    count_ai_mentions = len(ai_regex.findall(text))

    counts = {
        "count_total_words": count_total_words,
        "count_vague_words": count_vague_words,
        "count_specific_words": count_specific_words,
        "count_ai_mentions": count_ai_mentions,
    }

    # --- PHASE 3: Ratios (Float Features) ---
    denom = max(count_total_words, 1)
    ratios = {
        "ratio_vague_words": count_vague_words / denom,
        "ratio_specific_words": count_specific_words / denom,
        "ratio_ai_mentions": count_ai_mentions / denom,
    }

    # --- PHASE 4: Formulas / Scores ---
    # Ratio of vague words to specific words (vagueness ratio)
    vagueness_ratio = count_vague_words / max(count_specific_words, 1)
    
    # Specificity density score
    specificity_score = (count_specific_words - count_vague_words) / denom
    
    # Total distinct risks mentioned
    total_risk_indicators_count = int(
        presences["has_risk_factor"] + 
        presences["has_regulatory_risk"] + 
        presences["has_cyber_privacy_risk"] + 
        presences["has_ethics_bias_risk"] + 
        presences["has_ip_copyright_risk"] + 
        presences["has_supply_infra_risk"] + 
        presences["has_data_licensing"]
    )
    
    # Total distinct governance indicators
    total_governance_indicators_count = int(
        presences["has_board_oversight"] + 
        presences["has_audit_committee"] + 
        presences["has_ethics_policy"] + 
        presences["has_compliance"]
    )

    formulas = {
        "vagueness_ratio": vagueness_ratio,
        "specificity_score": specificity_score,
        "total_risk_indicators_count": total_risk_indicators_count,
        "total_governance_indicators_count": total_governance_indicators_count,
    }

    # Combine all features
    features = {}
    features.update(presences)
    features.update(counts)
    features.update(ratios)
    features.update(formulas)
    
    return features

def main():
    config = load_config()
    
    chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"
    output_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_bow_features.parquet"
    
    if not chunks_path.exists():
        pipeline_logger.log_event(
            pipeline_step="bow_feature_extraction",
            level="ERROR",
            message=f"Candidate chunks Parquet not found at {chunks_path}"
        )
        print(f"Error: Candidate chunks not found at {chunks_path}")
        return

    pipeline_logger.log_event(
        pipeline_step="bow_feature_extraction",
        level="INFO",
        message="Loading candidate chunks..."
    )
    
    chunks_df = pd.read_parquet(chunks_path)
    
    if len(chunks_df) == 0:
        pipeline_logger.log_event(
            pipeline_step="bow_feature_extraction",
            level="WARNING",
            message="No candidate chunks found in the parquet file."
        )
        print("Warning: No candidate chunks to process.")
        return

    pipeline_logger.log_event(
        pipeline_step="bow_feature_extraction",
        level="INFO",
        message=f"Extracting Bag-of-Words features for {len(chunks_df)} chunks in phases..."
    )
    
    features_list = []
    for _, row in chunks_df.iterrows():
        chunk_id = row["chunk_id"]
        text = row["chunk_text"]
        
        features = run_bow_extraction(text)
        features["chunk_id"] = chunk_id
        features_list.append(features)
        
    features_df = pd.DataFrame(features_list)
    
    # Reorder columns to have chunk_id first
    cols = ["chunk_id"] + [c for c in features_df.columns if c != "chunk_id"]
    features_df = features_df[cols]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    
    pipeline_logger.log_event(
        pipeline_step="bow_feature_extraction",
        level="SUCCESS",
        message=f"BoW feature extraction complete. Saved {len(features_df)} records to {output_path}."
    )
    print(f"Success: Saved {len(features_df)} records to {output_path}")

if __name__ == "__main__":
    main()
