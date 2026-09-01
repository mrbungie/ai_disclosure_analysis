# phase0/000_no_concepts

Proposer rationale: none — this is the floor. Always returns False, so
unique_recall_gain=0.0 and added_volume_frac=0.0 on any split. Every
later phase0 candidate (an embedding/ConceptSeed-backed harness.py) must
beat 0.0 unique recall gain, while staying under
configs/config.json: phase0.max_added_volume_frac, to matter.
