"""The call dataset: spines/call/call left-joined with every covariate and target
family of the grain, one row per spine key (nulls where a family has no
information). Only joins existing gold tables.

Column names: a family's column name is kept when unique across the joined
tables; on a collision identical values keep one column and differing values
are prefixed with their family (`<family>__<column>`); see
scripts/gold/datasets/join_families.py.

Cross-grain: the firm-quarter dataset (datasets/firm_quarter/firm_quarter) as
of the call date, every column prefixed `fq__`: pit.asof_join of `fecha`
on `as_of_date`, backward within ticker; the snapshot only contains
information published before its `as_of_date` (quarter end + 1 day), so it is
point in time at the call. `fq__quarter` and `fq__as_of_date` record the
attached snapshot; both are null for calls before the ticker's first closed
quarter.

Output: datasets/call/call.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "datasets"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
from join_families import inputs, join_families, report, with_asof  # noqa: E402


def main() -> None:
    dataset, collisions = join_families("call")
    dataset = with_asof(dataset, "firm_quarter", "fq__")
    report("call", dataset, collisions)
    L.write_gold("datasets", "call", "call", dataset, builder="scripts/gold/call/build_dataset.py",
                 inputs=inputs("call", "firm_quarter"), extra={"collisions": collisions})


if __name__ == "__main__":
    main()
