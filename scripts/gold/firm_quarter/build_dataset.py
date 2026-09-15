"""The firm_quarter dataset: spines/firm_quarter/firm_quarter left-joined with every covariate and target
family of the grain, one row per spine key (nulls where a family has no
information). Only joins existing gold tables.

Column names: a family's column name is kept when unique across the joined
tables; on a collision identical values keep one column and differing values
are prefixed with their family (`<family>__<column>`); see
scripts/gold/datasets/join_families.py.

Output: datasets/firm_quarter/firm_quarter.
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
    dataset, collisions = join_families("firm_quarter")
    report("firm_quarter", dataset, collisions)
    L.write_gold("datasets", "firm_quarter", "firm_quarter", dataset, builder="scripts/gold/firm_quarter/build_dataset.py",
                 inputs=inputs("firm_quarter"), extra={"collisions": collisions})


if __name__ == "__main__":
    main()
