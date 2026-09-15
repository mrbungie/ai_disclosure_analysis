# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb"]
# ///
"""
scripts/common/build_data_views.py — a DuckDB file with every parquet under
data/ exposed as a view, for ad-hoc browsing (DuckDB CLI, DBeaver, notebooks).

Views only: no data is copied, so the file is small and always reads the
current parquets. Rebuild it when datasets are added or removed.

One schema per top-level directory of data/ (raw, interim, bronze, silver,
gold, results, deprecated, archive); the view name is the rest of the path
joined with "__", e.g.

    silver.ai_frames                                  data/silver/ai_frames.parquet
    gold.covariates__firm_quarter__activities_all_quarter
    bronze.paragraphs                                 data/bronze/paragraphs/form=*/part-*.parquet
    interim.ai_classify__ai_frames                    data/interim/ai_classify/ai_frames__session=*.parquet
    raw.market__prices                                data/raw/market/prices/<TICKER>.parquet (+ filename)

Files are grouped into one dataset when they are parts of the same table:
  1. hive-partitioned directories (`key=value/`) -> the directory above the
     first partition level, with hive_partitioning;
  2. `<name>__part=|session=|run=...` file names   -> all files sharing <name>;
  3. `part-NNNNN.parquet`                          -> the directory;
  4. directories with >= COLLECTION_MIN plain files (one file per ticker /
     accession) -> the directory, with a `filename` column.
Everything else is one view per file. All multi-file views use union_by_name.

Usage:
    uv run --with duckdb python scripts/common/build_data_views.py [--out data_views.duckdb]
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
COLLECTION_MIN = 200
PART_SUFFIX = re.compile(r"__(?:part|session|run)=")
PART_FILE = re.compile(r"^part-\d+\.parquet$")


def ident(parts: list[str]) -> str:
    name = "__".join(parts)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    return name or "root"


def group_datasets() -> dict[tuple[str, str], tuple[str, bool, bool]]:
    """(schema, view) -> (glob, hive_partitioning, filename_column)."""
    files = sorted(p for p in DATA.rglob("*.parquet") if not p.name.startswith("."))
    plain_by_dir: dict[Path, list[Path]] = defaultdict(list)
    datasets: dict[str, tuple[Path, str, bool, bool]] = {}  # glob -> (root, stem-parts, hive, filename)

    for f in files:
        rel = f.relative_to(DATA)
        hive_idx = next((i for i, c in enumerate(rel.parts[:-1]) if "=" in c), None)
        if hive_idx is not None:
            root = DATA.joinpath(*rel.parts[:hive_idx])
            datasets.setdefault(f"{root}/**/*.parquet", (root, "", True, False))
        elif m := PART_SUFFIX.search(f.name):
            # glob on "<prefix>__run=" etc., so "x__run=*" never also picks up "x__v=2__run=*"
            prefix = f.name[: m.start()]
            datasets.setdefault(f"{f.parent}/{f.name[: m.end()]}*.parquet", (f.parent, prefix, False, False))
        elif PART_FILE.match(f.name):
            datasets.setdefault(f"{f.parent}/part-*.parquet", (f.parent, "", False, False))
        else:
            plain_by_dir[f.parent].append(f)

    for d, fs in plain_by_dir.items():
        if len(fs) >= COLLECTION_MIN:
            datasets[f"{d}/*.parquet"] = (d, "", False, True)
        else:
            for f in fs:
                datasets[str(f)] = (f.parent, f.stem, False, False)

    views: dict[tuple[str, str], tuple[str, bool, bool]] = {}
    for glob, (root, stem, hive, filename) in sorted(datasets.items()):
        rel_parts = list(root.relative_to(DATA).parts) + ([stem] if stem else [])
        if len(rel_parts) == 1 and stem:  # file directly under data/
            schema, name_parts = "main", rel_parts
        else:
            schema, name_parts = ident(rel_parts[:1]), rel_parts[1:]
        name, n = ident(name_parts), 2
        while (schema, name) in views:
            name = f"{ident(name_parts)}_{n}"
            n += 1
        views[(schema, name)] = (glob, hive, filename)
    return views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data_views.duckdb")
    args = ap.parse_args()

    views = group_datasets()
    tmp = args.out.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    failed = []
    t0 = time.time()
    for (schema, name), (glob, hive, filename) in views.items():
        opts = ["union_by_name = true"]
        if hive:
            opts.append("hive_partitioning = true")
        if filename:
            opts.append("filename = true")
        path = glob.replace("'", "''")
        try:
            con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            con.execute(f"""CREATE OR REPLACE VIEW "{schema}"."{name}" AS
                            SELECT * FROM read_parquet('{path}', {", ".join(opts)})""")
        except duckdb.Error as e:
            failed.append((schema, name, glob, str(e).splitlines()[0]))
    con.close()
    tmp.replace(args.out)

    print(f"{len(views) - len(failed)} views in {args.out} ({time.time() - t0:.0f}s)")
    for schema, name, glob, err in failed:
        print(f"  FAILED {schema}.{name} <- {glob}: {err}")


if __name__ == "__main__":
    main()
