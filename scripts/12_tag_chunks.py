"""
12_tag_chunks.py — Cycle 2, application: tag every AI candidate chunk with the
frozen classification formulas (configs/config.json tagging.formulas, written
by 11 after its single holdout look) and write the chunk-level tagged panel.

This is the harness running at corpus scale — the deterministic stand-in for
the LLM judge whose agreement with it was measured in 11's holdout report and
whose agreement with a human was measured by agreement_check.py. Firm-year
aggregation is analysis and lives downstream, not here.

Usage:
    uv run python scripts/12_tag_chunks.py
"""

import json
from pathlib import Path

import pandas as pd

try:
    import pipeline_logger
    import tag_harness_defs as defs
except ImportError:
    from scripts import pipeline_logger
    from scripts import tag_harness_defs as defs

CONFIG_PATH = Path("configs/config.json")
OUTPUT_PATH = Path("data/processed/tagged_chunks.parquet")


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    formulas = config.get("tagging", {}).get("formulas", {})
    if not formulas:
        print("Error: no tagging.formulas in config. Run scripts/11_fit_tag_harness.py first.")
        return

    chunks = pd.read_parquet(Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet")
    atoms_path = Path(config["paths"]["candidate_chunks"]) / "keyword_atom_features.parquet"
    if not atoms_path.exists():
        print(f"Error: {atoms_path} not found. Run scripts/09_extract_keyword_atoms.py first.")
        return
    atoms = pd.read_parquet(atoms_path).drop_duplicates(subset="chunk_id")

    merged = chunks.merge(atoms, on="chunk_id", how="inner")
    dropped = len(chunks) - len(merged)
    if dropped:
        print(f"Warning: {dropped} chunk(s) missing atom features — re-run 09 after any re-chunk.")

    missing = [d for d in defs.DIMENSIONS if d not in formulas]
    if missing:
        print(f"Note: no frozen formula for {missing} (not frozen by 11) — those columns are omitted.")

    for dimension, frozen in formulas.items():
        merged[dimension] = defs.apply_formula_spec(frozen["spec"], merged)
        print(f"  {dimension}: {merged[dimension].mean()*100:.1f}% of chunks "
              f"(formula: {frozen['name']}, holdout F1={frozen.get('holdout_f1', '?')})")

    keep_cols = [c for c in chunks.columns] + list(formulas.keys())
    tagged = merged[keep_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tagged.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nTagged {len(tagged)} chunks on {len(formulas)} dimensions -> {OUTPUT_PATH}")

    pipeline_logger.log_event(
        pipeline_step="tag_chunks",
        level="SUCCESS",
        message=f"Tagged {len(tagged)} chunks on {len(formulas)} dimensions.",
        details={d: float(tagged[d].mean()) for d in formulas},
    )


if __name__ == "__main__":
    main()
