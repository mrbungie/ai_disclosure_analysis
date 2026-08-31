"""
09_extract_keyword_atoms.py — Cycle 2, atom extraction: boolean keyword/regex
"atoms" (has_* presence flags, counts, and a few derived scores) for every AI
candidate chunk. These atoms are the SEARCH SPACE of the classification
harness: 11_fit_tag_harness.py searches boolean formulas over them, per
dimension, against the LLM judge's labels (10).

Adapted from the pre-strip BoW extraction (9e6138c^:scripts/07_extract_bow_features.py)
with the NLTK/WordNet synonym expansion removed — synonym drift added opaque
matches that the formula search can't audit; the heuristic inflection
generator (s/es/ed/ing forms) is kept so word lists still match plural and
verb forms.

Usage:
    uv run python scripts/09_extract_keyword_atoms.py
"""

import json
import re
from pathlib import Path

import pandas as pd

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger


def load_config():
    with open(Path("configs/config.json")) as f:
        return json.load(f)

def get_inflections(word):
    """
    Simple heuristic rule-based inflection generator for common English suffixes.
    Ensures standard verb forms (s/es, ed, ing) and noun forms (s/es) are matched.
    """
    inflections = {word.lower()}
    w = word.lower()
    
    if not w.isalpha() or len(w) <= 2:
        return inflections

    # Ends in 'y' (but not 'ey', 'ay', 'oy', 'uy') -> 'ies', 'ied', 'ying'
    if w.endswith('y') and not w.endswith(('ay', 'ey', 'oy', 'uy')):
        base = w[:-1]
        inflections.add(base + 'ies')
        inflections.add(base + 'ied')
        inflections.add(w + 'ing')
    # Ends in 'e' -> 's', 'd', 'ing' (dropping 'e')
    elif w.endswith('e'):
        inflections.add(w + 's')
        inflections.add(w + 'd')
        if w.endswith('ee'):
            inflections.add(w + 'ing')
        elif w.endswith('ie'):
            inflections.add(w[:-2] + 'ying')
        else:
            inflections.add(w[:-1] + 'ing')
    # Ends in 's', 'x', 'z', 'ch', 'sh' -> 'es', 'ed', 'ing'
    elif w.endswith(('s', 'x', 'z', 'ch', 'sh')):
        inflections.add(w + 'es')
        inflections.add(w + 'ed')
        inflections.add(w + 'ing')
    else:
        # Regular verbs/nouns
        inflections.add(w + 's')
        inflections.add(w + 'ed')
        inflections.add(w + 'ing')
        
        # Double consonant rule (for short words ending in CVC, e.g., run -> running, fit -> fitted)
        vowels = 'aeiou'
        if len(w) >= 3 and w[-1] not in vowels + 'wxy' and w[-2] in vowels and w[-3] not in vowels:
            inflections.add(w + w[-1] + 'ing')
            inflections.add(w + w[-1] + 'ed')
            
    return inflections
# Define word lists for Bag of Words (BoW) analysis
VAGUE_WORDS = [
    "revolutionize", "revolutionizing", "revolutionized",
    "transform", "transforming", "transformed", "transformative",
    "cutting-edge", "next-generation", "next-gen", "infuse", "infusing", "infused",
    "leader", "leadership", "empower", "empowering", "empowered",
    "unlock", "unlocking", "unlocked", "innovative", "innovation",
    "accelerate", "accelerating", "accelerated", "ecosystem", "seamless",
    "strategic wave", "breakthrough", "world-class", "pioneer", "pioneering",
    "reimagine", "reimagining", "reimagined", "paradigm shift", "disrupt",
    "disruptive", "disrupting", "unprecedented", "game-changer", "game-changing",
    "groundbreaking", "state-of-the-art"
]

SPECIFIC_WORDS = [
    "gpu", "gpus", "tpu", "tpus", "h100", "a100", "blackwell", "mi300", "cuda",
    "dataset", "datasets", "curation", "curating", "fine-tuning", "fine-tuned",
    "pre-training", "pre-trained", "parameter", "parameters", "compliance", "regulations",
    "regulation", "eu ai act", "gdpr", "ccpa", "cybersecurity", "breach", "breaches",
    "copyright", "copyrights", "patent", "patents", "board oversight", "audit committee",
    "fiduciary", "ethics committee", "responsible ai", "ethical ai",
    "h200", "b200", "mi300x", "inflection", "llama-3", "gpt-4o", "claude-3", "gemini-1.5",
    "rlhf", "dpo", "inference", "quantization", "tensor", "retrieval-augmented generation",
    "rag", "vector database", "embedding", "embeddings", "neural engine", "tensor core"
]

AI_WORDS = [
    "ai", "artificial intelligence", "generative ai", "gen ai", "machine learning",
    "large language model", "llm", "llms", "deep learning", "neural network", "neural networks"
]

POSITIVE_WORDS = [
    "growth", "growth-driving", "efficiency", "productivity", "benefit", "benefits",
    "advantage", "advantages", "accelerate", "accelerates", "accelerated", "improve",
    "improves", "improved", "improving", "success", "successful", "opportunity",
    "opportunities", "innovate", "innovating", "innovation", "optimize", "optimizing",
    "optimized", "strengthen", "strengthening", "strengthened", "outperform",
    "outperforming", "enhance", "enhancing", "enhanced", "value-creation", "competitiveness"
]

NEGATIVE_WORDS = [
    "risk", "risks", "threat", "threats", "adversely", "harm", "harms", "harmed",
    "decline", "declining", "declined", "breach", "breaches", "expense", "expenses",
    "costly", "expensive", "loss", "losses", "challenge", "challenges", "penalty",
    "penalties", "litigation", "lawsuit", "lawsuits", "liability", "liabilities",
    "disruption", "disruptions", "failure", "failures", "failed", "failing",
    "uncertain", "uncertainty", "uncertainties"
]

GEN_AI_WORDS = [
    "generative ai", "gen ai", "large language model", "llm", "llms", "gpt", 
    "transformer", "transformers", "diffusion model", "diffusion models", 
    "foundation model", "foundation models", "chatgpt"
]

CLASSICAL_ML_WORDS = [
    "machine learning", "predictive modeling", "neural network", "neural networks", 
    "deep learning", "supervised learning", "unsupervised learning", "reinforcement learning", 
    "regression", "classification", "random forest", "gradient boosting"
]

INTERNAL_PRODUCTIVITY_WORDS = [
    "productivity gains", "back-office", "efficiency", "automate tasks", 
    "automation", "internal operations", "workflow optimization", "copilot for work", 
    "software developers", "developer velocity"
]

PRODUCT_INTEGRATION_WORDS = [
    "monetize", "monetization", "customer-facing", "product offering", 
    "ai-powered service", "pricing tier", "new features", "revenue streams", 
    "product integration", "integration in our products"
]

CAPEX_WORDS = [
    "capital expenditure", "capital expenditures", "capex", "datacenter investment", 
    "datacenter investments", "infrastructure spending", "compute spend", "equipment purchases"
]

OPEX_WORDS = [
    "operating expense", "operating expenses", "opex", "running costs", 
    "maintenance costs", "subscription fees", "cloud costs"
]

RD_WORDS = [
    "research and development", "r&d", "r & d", "r&d expense", "r&d expenses", 
    "r&d costs", "research expense"
]

REVENUE_IMPACT_WORDS = [
    "top-line growth", "new revenue", "revenue streams", "pricing power", 
    "sales growth", "market share gains"
]

US_REGULATION_WORDS = [
    "ftc", "federal trade commission", "sec", "securities and exchange commission", 
    "executive order", "doj", "department of justice", "us congress"
]

EU_REGULATION_WORDS = [
    "eu ai act", "gdpr", "european union", "european commission", "brussels", "ai act"
]

LABOR_DISPLACEMENT_WORDS = [
    "workforce reduction", "workforce reductions", "headcount optimization", 
    "job displacement", "labor unions", "union concerns", "layoffs", "layoff", 
    "job loss", "job losses", "redundancies"
]

ALL_WORD_LISTS = {
    "vague": VAGUE_WORDS,
    "specific": SPECIFIC_WORDS,
    "ai": AI_WORDS,
    "positive": POSITIVE_WORDS,
    "negative": NEGATIVE_WORDS,
    "gen_ai": GEN_AI_WORDS,
    "classical_ml": CLASSICAL_ML_WORDS,
    "internal_productivity": INTERNAL_PRODUCTIVITY_WORDS,
    "product_integration": PRODUCT_INTEGRATION_WORDS,
    "capex": CAPEX_WORDS,
    "opex": OPEX_WORDS,
    "rd": RD_WORDS,
    "revenue_impact": REVENUE_IMPACT_WORDS,
    "us_regulation": US_REGULATION_WORDS,
    "eu_regulation": EU_REGULATION_WORDS,
    "labor_displacement": LABOR_DISPLACEMENT_WORDS
}

def normalize_word_to_feature(word):
    s = word.lower()
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return f"has_word_{s}"

# Precompiled regex patterns for standard categories
BOARD_OVERSIGHT_PAT = re.compile(
    r"\bboard(?:'s)?\b[^.!?]{0,50}\boversight\b|\boversight\b[^.!?]{0,50}\bboard(?:'s)?\b", 
    re.IGNORECASE
)
AUDIT_COMM_PAT = re.compile(r"\baudit\s+committees?\b", re.IGNORECASE)
ETHICS_POLICY_PAT = re.compile(
    r"\bethics?\s+committee\b|\bethical\s+standards\b|\bethical\s+guidelines\b|\bcode\s+of\s+conduct\b|\bresponsible\s+ai\b|\bethical\s+ai\b|\bai\s+safety\b|\bai\s+policy\b|\bai\s+policies\b|\bai\s+governance\b|\btrust\s+(?:and|&)\s+safety\b",
    re.IGNORECASE
)
COMPLIANCE_PAT = re.compile(
    r"\bcompliance\b|\bregulatory\s+compliance\b|\bdata\s+governance\b|\bregulatory\s+frameworks?\b|\bcompliance\s+requirements?\b|\binternal\s+controls?\b|\bauditing\b",
    re.IGNORECASE
)
RISK_FACTOR_PAT = re.compile(
    r"\brisk\s+factors?\b|\brisks\s+and\s+uncertainties\b|\badversely\s+affect\b|\bmaterially\s+affect\b|\bnegative\s+impact\b|\bcould\s+be\s+harmed\b|\bsubject\s+to\s+risks?\b|\breputational\s+damage\b",
    re.IGNORECASE
)
REGULATORY_RISK_PAT = re.compile(
    r"\bregulatory\s+scrutiny\b|\bregulations?\b|\bregulatory\s+compliance\b|\blegal\s+frameworks?\b|\beu\s+ai\s+act\b|\bfederal\s+trade\s+commission\b|\bftc\b|\bsec\s+scrutiny\b|\blegislative\s+efforts?\b",
    re.IGNORECASE
)
CYBER_PRIVACY_RISK_PAT = re.compile(
    r"\bprivacy\b|\bdata\s+protection\b|\bcybersecurity\b|\bcyber-attacks?\b|\bdata\s+breach(?:es)?\b|\bexfiltration\b|\bunauthorized\s+access\b|\bdata\s+leakage\b|\bgdpr\b|\bccpa\b|\bcpra\b|\bpipl\b",
    re.IGNORECASE
)
ETHICS_BIAS_RISK_PAT = re.compile(
    r"\bbias(?:ed)?\b|\bdiscrimination\b|\btoxic(?:ity)?\b|\bhallucination\s+concern\b|\bhallucinate\b|\berrors?\b|\binaccurac(?:y|ies)\b|\bfairness\b|\bethical\s+concerns?\b|\bsocietal\s+harm\b|\bdeepfakes?\b",
    re.IGNORECASE
)
IP_COPYRIGHT_RISK_PAT = re.compile(
    r"\bcopyright\b|\bcopyright\s+infringement\b|\bfair\s+use\b|\blicensing\b|\bintellectual\s+property\b|\bproprietary\b|\bpatents?\b|\btrade\s+secrets?\b|\btrademarks?\b",
    re.IGNORECASE
)
SUPPLY_INFRA_RISK_PAT = re.compile(
    r"\bgpu\s+shortages?\b|\bgpu\s+supply\b|\bchip\s+shortages?\b|\bcompute\s+capacity\b|\bcompute\s+constraints?\b|\bsemiconductor\s+supply\b|\benergy\s+consumption\b|\bpower\s+constraints?\b|\belectricity\b",
    re.IGNORECASE
)
NVIDIA_PAT = re.compile(r"\bnvidia\b|\brtx\b|\bdgx\b|\bh100\b|\ba100\b|\bblackwell\b", re.IGNORECASE)
OPENAI_PAT = re.compile(r"\bopenai\b|\bchatgpt\b|\bgpt-4\b|\bgpt-3\b|\bopen\s+ai\b|\bchat\s+gpt\b", re.IGNORECASE)
MICROSOFT_PAT = re.compile(r"\bmicrosoft\b|\bazure\b|\bcopilot\b", re.IGNORECASE)
GOOGLE_PAT = re.compile(r"\bgoogle\b|\bgemini\b|\bbard\b|\bdeepmind\b|\balphabet\b", re.IGNORECASE)
DEEPSEEK_PAT = re.compile(r"\bdeepseek\b", re.IGNORECASE)
AMAZON_PAT = re.compile(r"\bamazon\b|\baws\b|\bbedrock\b", re.IGNORECASE)
META_PAT = re.compile(r"\bmeta\b|\bllama\b", re.IGNORECASE)
ANTHROPIC_PAT = re.compile(r"\banthropic\b|\bclaude\b", re.IGNORECASE)
AMD_PAT = re.compile(r"\bamd\b|\badvanced\s+micro\s+devices\b|\binstinct\b|\bmi300\b|\bryzen\b|\bepyc\b", re.IGNORECASE)
PERCENTAGE_PAT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)
DOLLAR_PAT = re.compile(
    r"\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?\b|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)?\s*(?:dollars|usd)\b", 
    re.IGNORECASE
)
SPECIFIC_PRODUCT_PAT = re.compile(
    r"\bcopilot\b|\bchatgpt\b|\bfirefly\b|\bsensei\b|\beinstein\b|\baip\b|\bwatson\b|\brtx\b|\bdgx\b|\bcuda\b|\binstinct\b|\bmi300\b", 
    re.IGNORECASE
)
DEPLOYMENT_VERB_PAT = re.compile(
    r"\blaunched\b|\bintroduced\b|\bintegrated\b|\brollout\b|\brolling\s+out\b|\brelease\s+of\b|\bcommercially\s+available\b|\bmonetize\b|\bmonetization\b|\bsubscription\b",
    re.IGNORECASE
)
MODEL_TRAINING_PAT = re.compile(
    r"\btraining\b|\bfine-tuning\b|\bpre-training\b|\bparameter\s+size\b|\bdatasets?\b|\bdataset\s+curation\b|\bpre-train\b|\brlhf\b",
    re.IGNORECASE
)
COMPUTE_INFRA_PAT = re.compile(
    r"\bgpus?\b|\bh100\b|\ba100\b|\bblackwell\b|\bcompute\b|\btpus?\b|\bdata\s+centers?\b|\bdatacenters?\b|\bservers?\b|\bsupercomputer\b",
    re.IGNORECASE
)
PROPRIETARY_DATA_PAT = re.compile(
    r"\bproprietary\s+data\b|\bproprietary\s+datasets?\b|\binternal\s+database\b|\bcustomer\s+data\b",
    re.IGNORECASE
)
WORKFORCE_TALENT_PAT = re.compile(
    r"\btalent\b|\brecruiting\b|\bworkforce\b|\bheadcount\b|\bhiring\b|\bskills?\b|\bemployees?\b|\blabor\b|\bengineers\b|\bscientists\b|\bphd\b|\bdevelopers\b",
    re.IGNORECASE
)
PARTNERSHIP_PAT = re.compile(
    r"\bstrategic\s+alliance\b|\bpartnership\b|\bjoint\s+venture\b|\bcollaboration\b|\bconsortium\b|\bcollaboration\s+agreement\b|\balliance\b",
    re.IGNORECASE
)
CUSTOMER_FACING_PAT = re.compile(
    r"\bcustomer\s+service\b|\bchatbot\b|\bvirtual\s+assistant\b|\bconversational\s+agent\b|\bcustomer\s+support\b|\bchatbots\b|\bagents?\b",
    re.IGNORECASE
)
SAFETY_CRITICAL_PAT = re.compile(
    r"\bself-driving\b|\bautonomous\s+vehicles?\b|\bautopilot\b|\bfsd\b|\bweapons?\b|\bdefense\b|\bmilitary\b|\baerospace\b",
    re.IGNORECASE
)
DATA_LICENSING_PAT = re.compile(
    r"\bcontent\s+licensing\b|\blicensing\s+agreements?\b|\bpublisher\s+agreements?\b|\bscraping\b|\bfair\s+use\b|\bintellectual\s+property\s+claims\b",
    re.IGNORECASE
)
COMPETITOR_PAT = re.compile(
    r"\bcompetitors?\b|\bcompetition\b|\bcompeting\b|\bcompete\b|\brival(?:s|ry)?\b|\bmarket\s+share\b",
    re.IGNORECASE
)
ACADEMIC_PAT = re.compile(
    r"\bacadem(?:ic|ia)\b|\buniversity\b|\buniversities\b|\bresearch\s+paper\b|\bscientific\b|\bstudy\b|\bstudies\b",
    re.IGNORECASE
)
OPEN_SOURCE_PAT = re.compile(
    r"\bopen\s+source\b|\bopen-source\b|\bhugging\s*face\b|\bopen\s+weights?\b|\bgithub\b|\bgit\b",
    re.IGNORECASE
)

FORWARD_LOOKING_PAT = re.compile(
    r"\b(?:will|expect(?:s|ed)?|plan(?:s|ned|ning)?|intend(?:s|ed|ing)?|anticipate(?:s|d)?|target(?:s|ed|ing)?|aim(?:s|ed|ing)?|expect(?:s|ed)?|goal(?:s)?|objective(?:s)?|outlook)\b",
    re.IGNORECASE
)
REALIZED_LANGUAGE_PAT = re.compile(
    r"\b(?:have|has|had|delivered|launched|achieved|completed|deployed|released|shipped|enabled|generated|produced|drove|drove|realized|demonstrated|built|implemented|introduced|adopted)\b",
    re.IGNORECASE
)

# Sentence-level co-occurrence: AI term within the same sentence as a dollar/% metric
_AI_SENT = r"(?:ai|artificial\s+intelligence|machine\s+learning|llm|generative\s+ai|gen\s+ai|large\s+language\s+model)"
_METRIC_SENT = r"(?:\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?|\d+(?:\.\d+)?\s*(?:%|percent\b))"
AI_QUANTIFIED_CLAIM_PAT = re.compile(
    rf"(?:{_AI_SENT}[^.!?\n]{{0,120}}{_METRIC_SENT}|{_METRIC_SENT}[^.!?\n]{{0,120}}{_AI_SENT})",
    re.IGNORECASE
)

# Named model/product + deployment verb within same sentence
_NAMED = r"(?:gpt-4o?|gpt-3(?:\.\d+)?|chatgpt|llama(?:-\d+)?|gemini(?:-\d+(?:\.\d+)?)?|claude(?:-\d+)?|mistral|deepseek|dall-e|whisper|copilot|einstein|watson|firefly|sensei)"
_DEPLOY = r"(?:launch(?:ed)?|integrat(?:ed|ing)|deploy(?:ed|ing)|roll(?:ed)?\s+out|released|built|powered\s+by|enabled|ship(?:ped)?)"
NAMED_DEPLOYMENT_PAT = re.compile(
    rf"(?:{_NAMED}[^.!?\n]{{0,100}}{_DEPLOY}|{_DEPLOY}[^.!?\n]{{0,100}}{_NAMED})",
    re.IGNORECASE
)

DATED_MILESTONE_PAT = re.compile(
    r"\b(?:by\s+(?:Q[1-4]\s*)?\d{4}|in\s+(?:fiscal\s+)?\d{4}|by\s+(?:end\s+of|mid|late|early)\s+(?:fiscal\s+)?\d{4}|by\s+(?:end\s+of\s+)?(?:the\s+)?(?:year|quarter|fiscal\s+year)|(?:Q[1-4]\s*\d{4})|(?:first|second|third|fourth)\s+quarter\s+of\s+\d{4})\b",
    re.IGNORECASE
)

COMPETITOR_AI_PAT = re.compile(
    r"\b(?:competitor|competitors|competition|competing|compete|rival|rivals|rivalry)\b[^.!?]{0,60}\b(?:ai|artificial\s+intelligence|llms?|large\s+language\s+models?|machine\s+learning|generative\s+ai|gen\s+ai|models?)\b|\b(?:ai|artificial\s+intelligence|llms?|large\s+language\s+models?|machine\s+learning|generative\s+ai|gen\s+ai|models?)\b[^.!?]{0,60}\b(?:competitor|competitors|competition|competing|compete|rival|rivals|rivalry)\b",
    re.IGNORECASE
)

DEEPSEEK_IMPACT_PAT = re.compile(
    r"\bdeepseek\b|\b(?:open[- ]source|open[- ]weights?|cost[- ](?:efficient|efficiency)|low[- ]cost|cheaper)\s+(?:ai\s+)?models?\s+[^.!?]{0,50}\b(?:disruption|disrupt|pressure|threat|challenge|impact)\b|\b(?:disruption|disrupt|pressure|threat|challenge|impact)\b[^.!?]{0,50}\b(?:open[- ]source|open[- ]weights?|cost[- ](?:efficient|efficiency)|low[- ]cost|cheaper)\s+(?:ai\s+)?models?\b",
    re.IGNORECASE
)

# Company actively using AI in its own products/operations (strong substantive signal)
# Requires first-person ("we [verb] AI") or possessive ownership ("our AI-powered X",
# "our X powered by AI") to avoid firing on competitor products or customer AI demand.
AI_OWN_USE_PAT = re.compile(
    # "we [verb] AI/ML" — first-person agent with realized verb (not hedged)
    r"\bwe\s+(?:have\s+)?(?:incorporat(?:e[sd]?|ing)|integrat(?:e[sd]?|ing)|built|developed?\b|deploy(?:ed)?|implemented?\b|leveraged?\b|utiliz(?:e[sd]?|ing)|adopted?\b|embed(?:ded?|ding))\s+(?:[a-z]+\s+){0,6}(?:ai\b|artificial\s+intelligence|machine\s+learning|generative\s+ai|gen\s+ai|large\s+language\s+models?|llms?)\b"
    # "our [product] powered by AI" or "our AI-powered [product]" — company owns it
    r"|\bour\s+(?:[a-z]+\s+){0,10}(?:powered\s+by|enabled\s+by)\s+(?:ai\b|artificial\s+intelligence|machine\s+learning|generative\s+ai)\b"
    r"|\bour\s+(?:[a-z]+\s+){0,5}(?:ai[- ]powered|ai[- ]enabled|ai[- ]driven)\b"
    # "powered by AI" standalone — specific enough to indicate real AI use
    r"|\bpowered\s+by\s+(?:ai\b|artificial\s+intelligence|machine\s+learning|generative\s+ai)\b",
    re.IGNORECASE
)

# AI mentioned as external demand/market signal rather than internal use (FP exclusion)
# Catches: "AI-related demand", "AI end market", "demand for GPUs", "AI chip customers"
AI_DEMAND_CONTEXT_PAT = re.compile(
    r"\bai[- ]related\s+demand\b"
    r"|\bai\s+end[- ]?market\b"
    r"|\bdemand\s+for\s+(?:high[- ]end\s+)?(?:gpus?|ai\s+chips?|ai\s+hardware)\b"
    r"|\bai\s+chip[s]?\s+(?:demand|market|customers?|adoption)\b"
    r"|\bai[- ]driven\s+(?:demand\b|(?:\w+\s+){0,3}(?:data\s+center|infrastructure))\b"
    r"|\b(?:customers?|cloud\s+providers?|hyperscal\w*)\s+(?:[a-z]+\s+){0,6}ai\s+workloads?\b"
    r"|\bai\s+computing\s+(?:demand|chips?|market)\b"
    r"|\bbenefited\s+from\s+(?:\w+\s+){0,4}ai[- ]related\b",
    re.IGNORECASE
)

# ML/DL model performing a specific operational task in the same sentence (stronger than has_classical_ml_mention)
# Catches: "machine learning to detect threats", "ML models designed to estimate", "uses ML to predict"
CLASSICAL_ML_OPERATIONAL_PAT = re.compile(
    r"\b(?:machine\s+learning|deep\s+learning|neural\s+networks?|ml\b)[^.!?\n]{0,100}"
    r"(?:to\s+)?(?:detect|predict|classif|identif|estimat|analyz|correlat|process|optimiz|forecast|monitor|score|assess|automat)\b"
    r"|\b(?:use[sd]?|using|leverages?|employ(?:ed|s)?)\s+(?:machine\s+learning|ml\b|deep\s+learning|neural\s+networks?)"
    r"\s+(?:to\s+)?(?:[a-z]+\s+){0,6}(?:detect|predict|classif|identif|estimat|analyz|process|optimiz|forecast|automat)\b",
    re.IGNORECASE
)

# Acquisition of an AI company or AI technology (strong substantive signal)
# Catches: "definitive agreement to acquire [X], a developer of AI"
# Negative lookahead excludes non-M&A uses of "acquire" (licenses, permits, rights, approvals)
AI_ACQUISITION_PAT = re.compile(
    r"(?:definitive\s+agreement\s+to\s+acquire|agreement\s+to\s+acquire"
    r"|acqui(?:re[ds]?|ring|sition\s+of))\s+"
    r"(?!(?:licenses?\b|permits?\b|approval\b|the\s+rights?\b|market\s+share\b|new\b|additional\b|any\b|these\b|our\b|customer))"
    r"[^.!?\n]{0,200}"
    r"(?:ai\b|artificial\s+intelligence|machine\s+learning|deep\s+learning)",
    re.IGNORECASE
)

# Forward-looking or risk-hedged AI language (exclusion signal for substantive proxy)
# Catches: "we may use AI", "AI could enable attacks", "subject to AI risks"
AI_HEDGE_PAT = re.compile(
    r"\b(?:we\s+)?(?:may|might|could|expect\s+to|plan\s+to|intend\s+to|anticipate|aim\s+to|seek\s+to)\s+"
    r"(?:[a-z]+\s+){0,6}(?:ai\b|artificial\s+intelligence|machine\s+learning|generative\s+ai|llms?)\b"
    r"|\b(?:ai\b|artificial\s+intelligence|machine\s+learning)\s+(?:[a-z]+\s+){0,4}"
    r"(?:may|could|might|can)\s+(?:[a-z]+\s+){0,4}(?:enable|cause|result\s+in|lead\s+to|harm|disrupt|threaten)",
    re.IGNORECASE
)

AI_MODEL_NAME_PAT = re.compile(
    r"\b(?:gpt-4o?|gpt-3\.5|gpt-3|chatgpt|llama(?:-\d+)?|gemini(?:-\d+(?:\.\d+)?)?|claude(?:-\d+)?|mistral|deepseek|bert|t5|stable\s+diffusion|dall-e|whisper|midjourney)\b",
    re.IGNORECASE
)

USE_CASE_PATTERNS = {
    "customer_support": re.compile(r"\bcustomer\s+(?:service|support|care|experience)\b|\bvirtual\s+assistants?\b|\bchatbots?\b|\bconversational\s+(?:agent|assistant|ai)\b", re.IGNORECASE),
    "fraud_underwriting": re.compile(r"\bfraud\s+(?:detection|prevention|mitigation|analysis)\b|\bunderwriting\b|\bloan\s+origination\b|\bcredit\s+(?:scoring|assessment)\b", re.IGNORECASE),
    "predictive_maintenance": re.compile(r"\bpredictive\s+maintenance\b|\bequipment\s+monitoring\b|\banomaly\s+detection\b|\bpredictive\s+failure\b", re.IGNORECASE),
    "supply_chain": re.compile(r"\bsupply\s+chain\s+(?:optimization|forecasting|planning)\b|\binventory\s+(?:management|optimization)\b|\blogistics\s+(?:planning|optimization)\b|\bdemand\s+forecasting\b", re.IGNORECASE),
    "document_processing": re.compile(r"\bdocument\s+processing\b|\bocr\b|\boptical\s+character\s+recognition\b|\btext\s+extraction\b|\bcontract\s+(?:analysis|intelligence)\b|\bsemantic\s+search\b", re.IGNORECASE),
    "marketing_personalization": re.compile(r"\bpersonalization\b|\brecommendation\s+(?:engine|system)s?\b|\btargeted\s+marketing\b|\bcontent\s+generation\b|\bproduct\s+recommendations?\b", re.IGNORECASE),
    "algorithmic_trading": re.compile(r"\balgorithmic\s+trading\b|\bquantitative\s+trading\b|\bautomated\s+trading\b|\bportfolio\s+optimization\b", re.IGNORECASE),
    "code_generation": re.compile(r"\bcode\s+(?:generation|autocomplete|completion)\b|\bdeveloper\s+(?:productivity|velocity)\b|\bsoftware\s+development\b", re.IGNORECASE)
}


# Helper to build a regex matching any word in a list with word boundaries
def build_list_regex(words_list):
    escaped_words = []
    for w in words_list:
        escaped_words.append(re.escape(w))
    # Sort descending by length so that longer phrases are matched first
    escaped_words.sort(key=len, reverse=True)
    pattern = r"\b(" + "|".join(escaped_words) + r")\b"
    return re.compile(pattern, re.IGNORECASE)

# Expand each word with its heuristic inflections (no synonym expansion).
EXPANDED_WORD_LISTS = {}
for category, word_list in ALL_WORD_LISTS.items():
    expanded_set = set()
    for word in word_list:
        expanded_set.update(get_inflections(word))
    EXPANDED_WORD_LISTS[category] = list(expanded_set)

# Category regexes built using expanded lists
VAGUE_REGEX = build_list_regex(EXPANDED_WORD_LISTS["vague"])
SPECIFIC_REGEX = build_list_regex(EXPANDED_WORD_LISTS["specific"])
AI_REGEX = build_list_regex(EXPANDED_WORD_LISTS["ai"])
POS_REGEX = build_list_regex(EXPANDED_WORD_LISTS["positive"])
NEG_REGEX = build_list_regex(EXPANDED_WORD_LISTS["negative"])

GEN_AI_REGEX = build_list_regex(EXPANDED_WORD_LISTS["gen_ai"])
CLASSICAL_ML_REGEX = build_list_regex(EXPANDED_WORD_LISTS["classical_ml"])
INTERNAL_PRODUCTIVITY_REGEX = build_list_regex(EXPANDED_WORD_LISTS["internal_productivity"])
PRODUCT_INTEGRATION_REGEX = build_list_regex(EXPANDED_WORD_LISTS["product_integration"])
CAPEX_REGEX = build_list_regex(EXPANDED_WORD_LISTS["capex"])
OPEX_REGEX = build_list_regex(EXPANDED_WORD_LISTS["opex"])
RD_REGEX = build_list_regex(EXPANDED_WORD_LISTS["rd"])
REVENUE_IMPACT_REGEX = build_list_regex(EXPANDED_WORD_LISTS["revenue_impact"])
US_REGULATION_REGEX = build_list_regex(EXPANDED_WORD_LISTS["us_regulation"])
EU_REGULATION_REGEX = build_list_regex(EXPANDED_WORD_LISTS["eu_regulation"])
LABOR_DISPLACEMENT_REGEX = build_list_regex(EXPANDED_WORD_LISTS["labor_displacement"])

# Per-word presence patterns: each source word gets a column, folding its
# inflections back to the parent word's column name.
words_by_col = {}
for category, word_list in ALL_WORD_LISTS.items():
    for word in word_list:
        col_name = normalize_word_to_feature(word)
        words_by_col.setdefault(col_name, set()).update(get_inflections(word))

WORD_PATTERNS = {}
for col_name, words in words_by_col.items():
    sorted_words = sorted(words, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(w) for w in sorted_words) + r")\b"
    WORD_PATTERNS[col_name] = re.compile(pattern, re.IGNORECASE)

def run_bow_extraction(text):
    # --- PHASE 1: Boolean / Presence Features ---
    presences = {
        "has_board_oversight": bool(BOARD_OVERSIGHT_PAT.search(text)),
        "has_audit_committee": bool(AUDIT_COMM_PAT.search(text)),
        "has_ethics_policy": bool(ETHICS_POLICY_PAT.search(text)),
        "has_compliance": bool(COMPLIANCE_PAT.search(text)),
        "has_risk_factor": bool(RISK_FACTOR_PAT.search(text)),
        "has_regulatory_risk": bool(REGULATORY_RISK_PAT.search(text)),
        "has_cyber_privacy_risk": bool(CYBER_PRIVACY_RISK_PAT.search(text)),
        "has_ethics_bias_risk": bool(ETHICS_BIAS_RISK_PAT.search(text)),
        "has_ip_copyright_risk": bool(IP_COPYRIGHT_RISK_PAT.search(text)),
        "has_supply_infra_risk": bool(SUPPLY_INFRA_RISK_PAT.search(text)),
        "has_vendor_nvidia": bool(NVIDIA_PAT.search(text)),
        "has_vendor_openai": bool(OPENAI_PAT.search(text)),
        "has_vendor_microsoft": bool(MICROSOFT_PAT.search(text)),
        "has_vendor_google": bool(GOOGLE_PAT.search(text)),
        "has_vendor_deepseek": bool(DEEPSEEK_PAT.search(text)),
        "has_vendor_amazon": bool(AMAZON_PAT.search(text)),
        "has_vendor_meta": bool(META_PAT.search(text)),
        "has_vendor_anthropic": bool(ANTHROPIC_PAT.search(text)),
        "has_vendor_amd": bool(AMD_PAT.search(text)),
        "has_metric_percentage": bool(PERCENTAGE_PAT.search(text)),
        "has_metric_dollar": bool(DOLLAR_PAT.search(text)),
        "has_specific_product": bool(SPECIFIC_PRODUCT_PAT.search(text)),
        "has_deployment_verb": bool(DEPLOYMENT_VERB_PAT.search(text)),
        "has_model_training": bool(MODEL_TRAINING_PAT.search(text)),
        "has_compute_infra": bool(COMPUTE_INFRA_PAT.search(text)),
        "has_proprietary_data": bool(PROPRIETARY_DATA_PAT.search(text)),
        "has_workforce_talent": bool(WORKFORCE_TALENT_PAT.search(text)),
        "has_partnership": bool(PARTNERSHIP_PAT.search(text)),
        "has_customer_facing": bool(CUSTOMER_FACING_PAT.search(text)),
        "has_safety_critical": bool(SAFETY_CRITICAL_PAT.search(text)),
        "has_data_licensing": bool(DATA_LICENSING_PAT.search(text)),
        "has_competitor_mention": bool(COMPETITOR_PAT.search(text)),
        "has_academic_research": bool(ACADEMIC_PAT.search(text)),
        "has_open_source": bool(OPEN_SOURCE_PAT.search(text)),
        
        # New Category features
        "has_gen_ai_mention": bool(GEN_AI_REGEX.search(text)),
        "has_classical_ml_mention": bool(CLASSICAL_ML_REGEX.search(text)),
        "has_internal_productivity": bool(INTERNAL_PRODUCTIVITY_REGEX.search(text)),
        "has_product_integration": bool(PRODUCT_INTEGRATION_REGEX.search(text)),
        "has_capex_mention": bool(CAPEX_REGEX.search(text)),
        "has_opex_mention": bool(OPEX_REGEX.search(text)),
        "has_rd_mention": bool(RD_REGEX.search(text)),
        "has_revenue_impact": bool(REVENUE_IMPACT_REGEX.search(text)),
        "has_us_regulation": bool(US_REGULATION_REGEX.search(text)),
        "has_eu_regulation": bool(EU_REGULATION_REGEX.search(text)),
        "has_labor_displacement_risk": bool(LABOR_DISPLACEMENT_REGEX.search(text)),
        "has_financial_quantification": bool(PERCENTAGE_PAT.search(text) or DOLLAR_PAT.search(text)),
        "has_ai_use_case_specific": any(bool(pat.search(text)) for pat in USE_CASE_PATTERNS.values()),
        "has_competitor_ai_mention": bool(COMPETITOR_AI_PAT.search(text)),
        "has_deepseek_impact": bool(DEEPSEEK_IMPACT_PAT.search(text)),
        "has_ai_model_name": bool(AI_MODEL_NAME_PAT.search(text)),
        # Stronger substance signals: sentence-level co-occurrence
        "has_ai_quantified_claim": bool(AI_QUANTIFIED_CLAIM_PAT.search(text)),
        "has_named_deployment": bool(NAMED_DEPLOYMENT_PAT.search(text)),
        "has_dated_milestone": bool(DATED_MILESTONE_PAT.search(text)),
        "has_forward_looking": bool(FORWARD_LOOKING_PAT.search(text)),
        "has_realized_language": bool(REALIZED_LANGUAGE_PAT.search(text)),
        # New precision signals
        "has_ai_own_use": bool(AI_OWN_USE_PAT.search(text)),
        "has_ai_demand_context": bool(AI_DEMAND_CONTEXT_PAT.search(text)),
        "has_classical_ml_operational": bool(CLASSICAL_ML_OPERATIONAL_PAT.search(text)),
        "has_ai_acquisition": bool(AI_ACQUISITION_PAT.search(text)),
        "has_ai_hedge": bool(AI_HEDGE_PAT.search(text)),
    }
    
    # Add presence features for each individual word/phrase
    for col_name, pat in WORD_PATTERNS.items():
        presences[col_name] = bool(pat.search(text))

    # --- PHASE 2: Counts (Integer Features) ---
    words = text.split()
    count_total_words = len(words)
    count_forward_looking = len(FORWARD_LOOKING_PAT.findall(text))
    count_realized_language = len(REALIZED_LANGUAGE_PAT.findall(text))
    
    count_vague_words = len(VAGUE_REGEX.findall(text))
    count_specific_words = len(SPECIFIC_REGEX.findall(text))
    count_ai_mentions = len(AI_REGEX.findall(text))
    count_positive_words = len(POS_REGEX.findall(text))
    count_negative_words = len(NEG_REGEX.findall(text))
    count_ai_use_case_types = sum(1 for pat in USE_CASE_PATTERNS.values() if pat.search(text))

    counts = {
        "count_total_words": count_total_words,
        "count_vague_words": count_vague_words,
        "count_specific_words": count_specific_words,
        "count_ai_mentions": count_ai_mentions,
        "count_positive_words": count_positive_words,
        "count_negative_words": count_negative_words,
        "count_ai_use_case_types": count_ai_use_case_types,
        "count_forward_looking": count_forward_looking,
        "count_realized_language": count_realized_language,
    }

    # --- PHASE 3: Ratios (Float Features) ---
    denom = max(count_total_words, 1)
    ratios = {
        "ratio_vague_words": count_vague_words / denom,
        "ratio_specific_words": count_specific_words / denom,
        "ratio_ai_mentions": count_ai_mentions / denom,
        "ratio_positive_words": count_positive_words / denom,
        "ratio_negative_words": count_negative_words / denom,
    }

    # --- PHASE 4: Formulas / Scores ---
    # Ratio of vague words to specific words (vagueness ratio)
    vagueness_ratio = count_vague_words / max(count_specific_words, 1)
    
    # Specificity density score
    specificity_score = (count_specific_words - count_vague_words) / denom
    
    # Bag of Words sentiment score (-1 to 1)
    pos_neg_sum = count_positive_words + count_negative_words
    bow_sentiment_score = (count_positive_words - count_negative_words) / max(pos_neg_sum, 1) if pos_neg_sum > 0 else 0.0
    
    # Total distinct risks mentioned
    total_risk_indicators_count = int(
        presences["has_risk_factor"] + 
        presences["has_regulatory_risk"] + 
        presences["has_cyber_privacy_risk"] + 
        presences["has_ethics_bias_risk"] + 
        presences["has_ip_copyright_risk"] + 
        presences["has_supply_infra_risk"] + 
        presences["has_data_licensing"] +
        presences["has_labor_displacement_risk"]
    )
    
    # Total distinct governance indicators
    total_governance_indicators_count = int(
        presences["has_board_oversight"] + 
        presences["has_audit_committee"] + 
        presences["has_ethics_policy"] + 
        presences["has_compliance"]
    )

    # Ratio of forward-looking to realized language — high ratio = promise-heavy, low = evidence-heavy
    fl_total = count_forward_looking + count_realized_language
    ratio_forward_to_realized = count_forward_looking / fl_total if fl_total > 0 else 0.5

    formulas = {
        "vagueness_ratio": vagueness_ratio,
        "specificity_score": specificity_score,
        "bow_sentiment_score": bow_sentiment_score,
        "total_risk_indicators_count": total_risk_indicators_count,
        "total_governance_indicators_count": total_governance_indicators_count,
        "ratio_forward_to_realized": ratio_forward_to_realized,
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
    output_path = Path(config["paths"]["candidate_chunks"]) / "keyword_atom_features.parquet"

    if not chunks_path.exists():
        pipeline_logger.log_event(
            pipeline_step="keyword_atom_extraction",
            level="ERROR",
            message=f"Candidate chunks Parquet not found at {chunks_path}",
        )
        print(f"Error: Candidate chunks not found at {chunks_path}")
        return

    chunks_df = pd.read_parquet(chunks_path)
    if len(chunks_df) == 0:
        print("Warning: No candidate chunks to process.")
        return

    print(f"Extracting keyword atoms for {len(chunks_df)} chunks...")
    features_list = []
    for _, row in chunks_df.iterrows():
        features = run_bow_extraction(row["chunk_text"])
        features["chunk_id"] = row["chunk_id"]
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    cols = ["chunk_id"] + [c for c in features_df.columns if c != "chunk_id"]
    features_df = features_df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="keyword_atom_extraction",
        level="SUCCESS",
        message=f"Keyword atom extraction complete. Saved {len(features_df)} records to {output_path}.",
        details={"n_chunks": len(features_df), "n_atoms": len(features_df.columns) - 1},
    )
    print(f"Success: saved {len(features_df)} rows x {len(features_df.columns) - 1} atoms -> {output_path}")


if __name__ == "__main__":
    main()
