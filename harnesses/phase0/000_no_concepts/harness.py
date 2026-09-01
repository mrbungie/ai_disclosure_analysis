"""
phase0/000_no_concepts — the floor for TASK phase0 (docs/
distillation_map.html §0): a phase0 candidate that contributes nothing.

TASK phase0 contract (fixed for every candidate under harnesses/phase0/):
    classify(text: str) -> bool

Same contract as detection (this task reads the SAME eval set — the
underlying question, "is this AI-related," is identical; only the
implementation family differs). Scored by scripts/eval_harness.py
--task phase0 as unique recall gain over the current detection ACTIVE,
subject to a volume ceiling (configs/config.json:
phase0.max_added_volume_frac) — see the harness.py header comment
pattern in harnesses/detection/000_seed for how a candidate exposes this
contract; a phase0 candidate typically loads a concept_seed.json sitting
alongside it and calls phase0_discovery.semantic_hit() (see
harnesses/phase0/README or the next real candidate for that pattern).

This seed always returns False — zero unique recall gain, zero added
volume. Every later phase0 candidate must beat 0.0 unique_recall_gain
(while staying under the volume ceiling) to matter.
"""


def classify(text: str) -> bool:
    return False
