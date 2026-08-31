"""
detection/000_seed — the minimal starting candidate for TASK 1.

TASK 1 (fixed contract for every candidate under harnesses/detection/):
pre-classification — given one unit of 10-K filing text, decide whether it
discusses AI/ML/LLMs or closely related technology at all:

    classify(text: str) -> bool

How a candidate decides — keyword lists, other patterns, staging, whatever —
is internal to the candidate. One self-contained stdlib-only file, auditable
at a glance, free for the proposer to rewrite.

This seed: three keywords. Its scores are the floor.
"""

import re

_AI = re.compile(r"\bai\b|\bartificial\s+intelligence\b|\bmachine\s+learning\b", re.IGNORECASE)


def classify(text: str) -> bool:
    return bool(_AI.search(text))
