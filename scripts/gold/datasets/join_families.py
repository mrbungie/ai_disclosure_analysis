"""
scripts/gold/datasets/join_families.py — a grain's dataset: its spine left-joined
with every covariate and target family of the grain (shared by the
scripts/gold/<grain>/build_dataset.py builders; no gold writes).

Rows and spine columns are the spine's (id, keys, date and spine attributes),
one row per spine key; a key without a family's information has nulls. Nothing
is recomputed or imputed.

Column names (collision policy): a family column keeps its name when no other
family of the grain (or spine attribute) has it. On a name collision the values
are compared over the spine rows (nulls equal, dtype ignored):
  - identical in every table -> one column under the plain name (the spine's
    attribute when the spine carries it, else the first family in
    covariates-then-targets, alphabetical order);
  - otherwise -> every family's column is prefixed with its family,
    `<family>__<column>` (a spine attribute keeps its plain name).
A prefixed name that is still ambiguous (same family name in covariates and
targets) is an error.

Cross-grain columns (`with_asof`, `with_static`) are prefixed with the source
grain's short prefix (`fq__` for firm_quarter, `firm__` for firm) and read the
source grain's dataset.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
from pit import asof_join  # noqa: E402


def _identical(a: pd.Series, b: pd.Series) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    both_null = a.isna() & b.isna()
    try:
        equal = (a == b).fillna(False).astype(bool)
    except TypeError:
        equal = a.astype(str) == b.astype(str)
    return bool((equal | both_null).all())


def family_paths(grain: str) -> list[tuple[str, Path]]:
    return [(kind, p) for kind in L.GOLD_FAMILY_KINDS for p in sorted((L.GOLD / kind / grain).glob("*.parquet"))]


def join_families(grain: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """(dataset, collisions): collisions maps each shared column name to the
    names it has in the dataset."""
    spine = pd.read_parquet(L.GOLD / "spines" / grain / f"{grain}.parquet")
    spine_cols = L.GOLD_SPINE_COLUMNS[grain]
    families = []
    for kind, path in family_paths(grain):
        values = pd.read_parquet(path)
        values = values.drop(columns=[c for c in spine_cols if c != "id"])
        values = spine[["id"]].merge(values, on="id", how="left", validate="one_to_one")
        families.append((kind, path.stem, values.drop(columns="id")))

    owners: dict[str, list[tuple[str, pd.Series]]] = defaultdict(list)
    for c in spine.columns:
        if c not in spine_cols:
            owners[c].append(("spine", spine[c]))
    for _, family, values in families:
        for c in values.columns:
            owners[c].append((family, values[c]))

    renames: dict[str, dict[str, str]] = defaultdict(dict)  # family -> {column: dataset name}
    collisions: dict[str, list[str]] = {}
    for c, tables in owners.items():
        if len(tables) == 1:
            continue
        first = tables[0][1]
        if all(_identical(first, s) for _, s in tables[1:]):
            for family, _ in tables[1:]:
                renames[family][c] = None
            collisions[c] = [c]
            continue
        names = []
        for family, _ in tables:
            name = c if family == "spine" else f"{family}__{c}"
            if family != "spine":
                renames[family][c] = name
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(f"{grain}: column {c!r} collides within families of the same name {names}")
        collisions[c] = names

    out = [spine.reset_index(drop=True)]
    seen = set(spine.columns)
    for _, family, values in families:
        keep = [c for c in values.columns if renames[family].get(c, c) is not None]
        values = values[keep].rename(columns={c: n for c, n in renames[family].items() if n is not None})
        dup = seen.intersection(values.columns)
        if dup:
            raise ValueError(f"{grain}: {family} would repeat columns {sorted(dup)}")
        seen.update(values.columns)
        out.append(values.reset_index(drop=True))
    return pd.concat(out, axis=1), collisions


def with_asof(dataset: pd.DataFrame, grain: str, prefix: str, *, event_date: str = "fecha") -> pd.DataFrame:
    """Attach `grain`'s dataset (a snapshot grain dated as_of_date) as of each
    row's `event_date`: the latest snapshot with as_of_date <= event_date for
    the same ticker (pit.asof_join). Every snapshot column except id and ticker
    is prefixed, so `<prefix>as_of_date` and the snapshot keys record which
    snapshot was attached."""
    snapshot = L.read_dataset(grain).drop(columns="id")
    snapshot = snapshot.rename(columns={c: f"{prefix}{c}" for c in snapshot.columns if c != "ticker"})
    events = dataset[["id", "ticker", event_date]]
    attached = asof_join(events, snapshot, event_date=event_date, snapshot_date=f"{prefix}{L.GOLD_DATES[grain]}")
    attached = attached.drop(columns=["ticker", event_date])
    return dataset.merge(attached, on="id", how="left", validate="one_to_one")


def with_static(dataset: pd.DataFrame, grain: str, prefix: str) -> pd.DataFrame:
    """Attach `grain`'s dataset (one row per ticker, no date) on ticker, every
    column except id and ticker prefixed."""
    static = L.read_dataset(grain).drop(columns="id")
    static = static.rename(columns={c: f"{prefix}{c}" for c in static.columns if c != "ticker"})
    return dataset.merge(static, on="ticker", how="left", validate="many_to_one")


def inputs(grain: str, *cross_grains: str) -> list[Path]:
    return ([L.GOLD / "spines" / grain / f"{grain}.parquet"] + [p for _, p in family_paths(grain)]
            + [L.GOLD / "datasets" / g / f"{g}.parquet" for g in cross_grains])


def report(grain: str, dataset: pd.DataFrame, collisions: dict[str, list[str]]) -> None:
    print(f"datasets/{grain}: {len(dataset):,} rows x {dataset.shape[1]} columns")
    for c, names in collisions.items():
        print(f"  collision {c}: {'identical, one column' if names == [c] else ', '.join(names)}")
