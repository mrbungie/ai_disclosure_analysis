"""The firm dataset: spines/firm/firm left-joined with every covariate and target
family of the grain, one row per spine key (nulls where a family has no
information). Only joins existing gold tables.

Column names: a family's column name is kept when unique across the joined
tables; on a collision identical values keep one column and differing values
are prefixed with their family (`<family>__<column>`); see
scripts/gold/datasets/join_families.py.

Output: datasets/firm/firm.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "datasets"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
from join_families import inputs, join_families, report  # noqa: E402


def main() -> None:
    dataset, collisions = join_families("firm")
    report("firm", dataset, collisions)
    L.write_gold("datasets", "firm", "firm", dataset, builder="scripts/gold/firm/build_dataset.py",
                 inputs=inputs("firm"), extra={"collisions": collisions})


if __name__ == "__main__":
    main()
