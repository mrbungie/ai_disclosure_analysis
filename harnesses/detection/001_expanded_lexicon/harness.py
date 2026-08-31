"""
detection/001_expanded_lexicon — expanded vocabulary candidate for TASK 1.

TASK 1 contract:
    classify(text: str) -> bool

Expands beyond basic "ai/artificial intelligence/machine learning" to capture
common AI/ML concepts, architectures, assistants, and applications appearing in
10-K filings (e.g. Copilot, ADAS, predictive models, data science, deep learning,
LLMs, natural language processing, computer vision, foundation models, etc.).
"""

import re

_PATTERNS = [
    # Core AI and ML terms
    r"\bai\b",
    r"\ba\.i\b",
    r"\bartificial\s+intelligence\b",
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\bneural\s+nets?(?:work)?s?\b",
    r"\bgenerative\s+(?:ai|artificial\s+intelligence)\b",
    r"\bgenai\b",
    r"\bllms?\b",
    r"\blarge\s+language\s+models?\b",
    r"\bnatural\s+language\s+processing\b",
    r"\bnlp\b",
    r"\bcomputer\s+vision\b",

    # AI technologies, architectures, and algorithms
    r"\btransformer\s+models?\b",
    r"\bdiffusion\s+models?\b",
    r"\breinforcement\s+learning\b",
    r"\bsupervised\s+learning\b",
    r"\bunsupervised\s+learning\b",
    r"\bfoundation(?:al)?\s+models?\b",
    r"\bsemantic\s+search\b",
    r"\bvector\s+embeddings?\b",

    # Prominent AI products, systems, and entities
    r"\bcopilot\b",
    r"\bchatgpt\b",
    r"\bgpt-?[34o]\b",
    r"\bopenai\b",
    r"\banthropic\b",

    # AI application domains & 10-K terminology
    r"\badas\b",
    r"\badvanced\s+driver-assistance\b",
    r"\bautonomous\s+(?:driving|vehicles?|cars?|trucks?|flight|navigation)\b",
    r"\bself-driving\b",
    r"\bdata\s+science\b",
    r"\bcognitive\s+computing\b",
    r"\bpredictive\s+models?\b",
    r"\bpredictive\s+modeling\b",
    r"\bspeech\s+recognition\b",
    r"\bimage\s+recognition\b",
    r"\bfacial\s+recognition\b",
    r"\bintelligent\s+automation\b",
    r"\brobotic\s+process\s+automation\b",
]

_AI_REGEX = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def classify(text: str) -> bool:
    return bool(_AI_REGEX.search(text))
