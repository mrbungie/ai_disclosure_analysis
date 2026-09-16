"""
scripts/gold/check_spine_alignment.py — every gold table is a left join on its spine.

Checks every file under data/gold:
  - layout: only <kind>/<grain>/<family>.parquet and its <family>._manifest.json,
    kind in layers.GOLD_KINDS, grain in layers.GOLD_GRAINS, one spine and at
    most one dataset per grain (spines/<grain>/<grain>, datasets/<grain>/<grain>);
  - spines: the grain's spine columns (id, keys, date) first, unique keys and
    ids;
  - covariates/targets/datasets: the spine columns first, then values; exactly
    one row per spine key (missing = outside = duplicates = 0) and the same id
    and date as the spine for every key;
  - spines/call: fiscal periods that contradict the previous call of the same
    ticker (`sequence_violations`) are reported, not treated as violations:
    transcript vendors disagree on some labels and the pipeline keeps what each
    transcript says rather than rewriting a date or a period.

Exits 1 on any violation.

Usage:
    .venv/bin/python scripts/gold/check_spine_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "call"))
import layers as L  # noqa: E402
from build_spine import sequence_violations  # noqa: E402


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Common string form so values from different builders compare equal."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        out[c] = out[c].astype(str)
    return out


def check_table(path: Path, grain: str, spine: pd.DataFrame | None) -> tuple[dict, list[str]]:
    spine_cols, keys = L.GOLD_SPINE_COLUMNS[grain], L.gold_keys(grain)
    names = pq.read_schema(path).names
    if names[:len(spine_cols)] != spine_cols:
        return {}, [f"first columns {names[:len(spine_cols)]} != {spine_cols}"]
    df = normalize(pd.read_parquet(path, columns=spine_cols))
    problems = []
    dups = int(df.duplicated(subset=keys).sum())
    if spine is None:
        if dups or df["id"].duplicated().any():
            problems.append("duplicated spine keys or ids")
        return {"rows": len(df), "missing": 0, "outside": 0, "duplicates": dups, "mismatched": 0}, problems
    merged = df.drop_duplicates(subset=keys).merge(spine, on=keys, how="outer", indicator=True, suffixes=("", "_spine"))
    both = merged[merged["_merge"] == "both"]
    mismatched = int(sum((both[c] != both[f"{c}_spine"]).sum() for c in spine_cols if c not in keys))
    r = {"rows": len(df), "missing": int((merged["_merge"] == "right_only").sum()),
         "outside": int((merged["_merge"] == "left_only").sum()), "duplicates": dups, "mismatched": mismatched}
    if r["missing"] or r["outside"] or r["duplicates"] or r["mismatched"]:
        problems.append("not aligned with the spine")
    return r, problems


def main() -> int:
    violations = 0
    notes: list[str] = []
    print(f"{'table':50s} {'rows':>7s} {'missing':>8s} {'outside':>8s} {'dups':>6s} {'id/date':>8s}  status")
    tables = sorted(p for p in L.GOLD.rglob("*") if p.is_file())
    for path in tables:
        rel = path.relative_to(L.GOLD)
        parts = rel.parts
        if path.name.endswith("._manifest.json"):
            if not path.with_name(path.name.removesuffix("._manifest.json") + ".parquet").exists():
                print(f"{str(rel):50s} manifest without table  VIOLATION")
                violations += 1
            continue
        if (len(parts) != 3 or parts[0] not in L.GOLD_KINDS or parts[1] not in L.GOLD_GRAINS
                or path.suffix != ".parquet" or (parts[0] in ("spines", "datasets") and path.stem != parts[1])):
            print(f"{str(rel):50s} not a <kind>/<grain>/<family>.parquet table  VIOLATION")
            violations += 1
    for grain in L.GOLD_GRAINS:
        spine_path = L.GOLD / "spines" / grain / f"{grain}.parquet"
        if not spine_path.exists():
            print(f"spines/{grain}/{grain}: missing  VIOLATION")
            violations += 1
            continue
        r, problems = check_table(spine_path, grain, None)
        if grain == "call":
            # reported, not a violation: vendors disagree on some fiscal labels
            # and the pipeline keeps what each transcript says
            bad = sequence_violations(pd.read_parquet(spine_path, columns=["ticker", "fecha", "fiscal_period", "call_sequence"]))
            if len(bad):
                notes.append(f"spines/call: {len(bad)} calls whose fiscal period contradicts the previous call "
                             f"({bad['ticker'].nunique()} tickers), labelled `conflict`")
        spine = normalize(pd.read_parquet(spine_path, columns=L.GOLD_SPINE_COLUMNS[grain]))
        rows = [(f"spines/{grain}/{grain}", r, problems)]
        for kind in L.GOLD_KINDS[1:]:
            for path in sorted((L.GOLD / kind / grain).glob("*.parquet")):
                rows.append((f"{kind}/{grain}/{path.stem}",) + check_table(path, grain, spine))
        for name, r, problems in rows:
            status = "ok" if not problems else "VIOLATION: " + "; ".join(problems)
            violations += bool(problems)
            if not r:
                print(f"{name:50s} {status}")
                continue
            print(f"{name:50s} {r['rows']:>7,} {r['missing']:>8,} {r['outside']:>8,} {r['duplicates']:>6,} "
                  f"{r['mismatched']:>8,}  {status}")
    for n in notes:
        print(n)
    print(f"\n{violations} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
