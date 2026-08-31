"""
classification/000_seed — the minimal starting candidate for TASK 2.

TASK 2 (fixed contract for every candidate under harnesses/classification/):
classification — given one unit of AI-related 10-K text (task 1's positives),
tag it on six binary dimensions:

    classify(text: str) -> dict[str, bool] with keys:
        is_substantive, is_promotional, is_risk_related,
        is_governance_related, is_use_case_specific, is_quantified

How a candidate tags — keyword patterns, boolean formulas, staging — is
internal to the candidate. One self-contained stdlib-only file, auditable at
a glance, free for the proposer to rewrite.

This seed: one obvious pattern per dimension. Its scores are the floor.
"""

import re

_SUBSTANTIVE = re.compile(r"\bwe\s+(?:use|deploy|deployed|built|developed|implemented)\b", re.IGNORECASE)
_PROMOTIONAL = re.compile(r"\btransform\w*\b|\brevolution\w*\b|\bleader(?:ship)?\b", re.IGNORECASE)
_RISK = re.compile(r"\brisks?\b|\badversely\b|\bthreats?\b", re.IGNORECASE)
_GOVERNANCE = re.compile(r"\bboard\b|\boversight\b|\bcompliance\b|\bgovernance\b", re.IGNORECASE)
_USE_CASE = re.compile(r"\bchatbots?\b|\bfraud\s+detection\b|\brecommendation\b|\bcustomer\s+service\b", re.IGNORECASE)
_QUANTIFIED = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent\b)|\$\s*\d", re.IGNORECASE)


def classify(text: str) -> dict[str, bool]:
    return {
        "is_substantive": bool(_SUBSTANTIVE.search(text)),
        "is_promotional": bool(_PROMOTIONAL.search(text)),
        "is_risk_related": bool(_RISK.search(text)),
        "is_governance_related": bool(_GOVERNANCE.search(text)),
        "is_use_case_specific": bool(_USE_CASE.search(text)),
        "is_quantified": bool(_QUANTIFIED.search(text)),
    }
