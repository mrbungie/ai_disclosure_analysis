#!/usr/bin/env python3
"""Compare two data/results-style snapshot directories.

Usage:
    uv run python scripts/analytics/compare_results_snapshots.py \
        data/deprecated/results_pre_fs_20260917T183741 \
        data/deprecated/results_fs_run1_20260917T185730 \
        --out-md data/results_compare/xbrl_vs_fs_run1.md \
        --out-csv data/results_compare/xbrl_vs_fs_run1.csv \
        --old-label OLD --new-label RUN1

Read-only: never writes into either snapshot directory. Designed to be rerun
for RUN1 vs RUN2 (or any two result trees with the same relative-path layout)
by passing different arguments.

For every file present in BOTH trees (csv/parquet/json), it diffs numeric
content:
  - csv/parquet: rows are matched on their non-numeric ("key") columns; for
    each numeric column present, records old/new/abs diff/pct diff and flags
      * sign flip (value crosses zero)
      * significance change, for columns that look like p-values (name is
        "p", starts/ends with "p_"/"_p", or contains "pvalue"): flags if the
        pair crosses 0.10, 0.05, or 0.01
      * N change > 5%, for columns that look like sample sizes (name is "n"
        or starts with "n_")
      * effect-size change > 25%, for columns that look like coefficients /
        R^2 (name contains "beta", "estimate", "coef", "r2", "delta_r2")
  - json: flattened to leaf key-paths and diffed the same way, using the
    leaf's final path segment as the "column name" for the heuristics above.

Output:
  - a long-format CSV with one row per (file, key, metric) comparison
  - a markdown report, headline changes first (flags), then per-topic detail
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

P_THRESHOLDS = (0.10, 0.05, 0.01)


def is_p_col(name: str) -> bool:
    n = name.lower()
    return n == "p" or n.startswith("p_") or n.endswith("_p") or "pvalue" in n or n == "p_value"


def is_n_col(name: str) -> bool:
    n = name.lower()
    return n == "n" or n.startswith("n_")


def is_effect_col(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in ("beta", "estimate", "coef", "r2", "delta_r2", "f", "spearman"))


def sig_bucket(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NA"
    for t in P_THRESHOLDS:
        if p < t:
            return f"<{t}"
    return ">=0.10"


def flag_for(colname: str, old_v, new_v) -> list[str]:
    flags = []
    if old_v is None or new_v is None:
        return flags
    try:
        old_f, new_f = float(old_v), float(new_v)
    except (TypeError, ValueError):
        return flags
    if math.isnan(old_f) or math.isnan(new_f):
        return flags

    if is_p_col(colname):
        b_old, b_new = sig_bucket(old_f), sig_bucket(new_f)
        if b_old != b_new:
            flags.append(f"sig_change({b_old}->{b_new})")
    if is_n_col(colname):
        if old_f != 0:
            pct = abs(new_f - old_f) / abs(old_f)
            if pct > 0.05:
                flags.append(f"N_change({pct*100:.1f}%)")
    if is_effect_col(colname):
        if (old_f > 0 and new_f < 0) or (old_f < 0 and new_f > 0):
            flags.append("sign_flip")
        if old_f != 0:
            pct = abs(new_f - old_f) / abs(old_f)
            if pct > 0.25:
                flags.append(f"effect_change({pct*100:.1f}%)")
    return flags


def load_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".csv":
            try:
                return pd.read_csv(path)
            except Exception:
                # fall back for free-text/notes csvs with stray commas in
                # unquoted numeric-looking fields (e.g. "8,642")
                return pd.read_csv(path, engine="python", on_bad_lines="skip")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed to read {path}: {e}", file=sys.stderr)
    return None


def flatten_json(obj, prefix="") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(flatten_json(v, key))
    else:
        out[prefix] = obj
    return out


def compare_json(old_path: Path, new_path: Path, rel: str) -> list[dict]:
    rows = []
    try:
        old_j = json.loads(old_path.read_text())
        new_j = json.loads(new_path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"  ! failed to parse json {rel}: {e}", file=sys.stderr)
        return rows
    old_flat = flatten_json(old_j)
    new_flat = flatten_json(new_j)
    common_keys = sorted(set(old_flat) & set(new_flat))
    for k in common_keys:
        old_v, new_v = old_flat[k], new_flat[k]
        colname = k.rsplit(".", 1)[-1]
        row = {
            "file": rel,
            "key": k,
            "metric": colname,
            "old_value": old_v,
            "new_value": new_v,
        }
        try:
            old_f, new_f = float(old_v), float(new_v)
            row["abs_diff"] = new_f - old_f
            row["pct_diff"] = (new_f - old_f) / abs(old_f) if old_f else float("nan")
        except (TypeError, ValueError):
            row["abs_diff"] = None
            row["pct_diff"] = None
        flags = flag_for(colname, old_v, new_v)
        row["flags"] = ";".join(flags)
        if row["flags"] or (isinstance(old_v, (int, float)) and old_v != new_v) or old_v != new_v:
            rows.append(row)
    return rows


KEY_HINT_RE = None


def _is_key_hint(name: str) -> bool:
    n = name.lower()
    return n in {
        "year", "quarter", "fy", "fiscal_year", "qtr", "period", "rank",
        "block", "target", "variable", "model", "contrast", "archetype",
        "ticker", "sector",
    } or n.endswith("_year") or n.endswith("_id")


def compare_table(old_df: pd.DataFrame, new_df: pd.DataFrame, rel: str) -> list[dict]:
    rows = []
    numeric_cols = [
        c for c in old_df.columns
        if c in new_df.columns and pd.api.types.is_numeric_dtype(old_df[c]) and not _is_key_hint(c)
    ]
    key_cols = [c for c in old_df.columns if c in new_df.columns and c not in numeric_cols]
    # value-like numeric cols we still want to key on if nothing else distinguishes rows
    if not key_cols:
        key_cols = [c for c in old_df.columns if c in new_df.columns][:1]

    # If the chosen keys don't uniquely identify rows in either frame (many-to-many
    # join would silently fan out and manufacture spurious diffs), fall back to
    # adding a stable row-order key instead of trusting an ambiguous join.
    if key_cols:
        old_dupe = old_df.duplicated(subset=key_cols).any()
        new_dupe = new_df.duplicated(subset=key_cols).any()
        if old_dupe or new_dupe:
            old_df = old_df.copy()
            new_df = new_df.copy()
            old_df["_row_ord"] = old_df.groupby(key_cols).cumcount()
            new_df["_row_ord"] = new_df.groupby(key_cols).cumcount()
            key_cols = key_cols + ["_row_ord"]

    old_df = old_df.copy()
    new_df = new_df.copy()
    for c in key_cols:
        old_df[c] = old_df[c].astype(str)
        new_df[c] = new_df[c].astype(str)

    merged = old_df.merge(new_df, on=key_cols, how="outer", suffixes=("_old", "_new"), indicator=True)

    n_only_old = int((merged["_merge"] == "left_only").sum())
    n_only_new = int((merged["_merge"] == "right_only").sum())
    if n_only_old or n_only_new:
        rows.append({
            "file": rel, "key": "|".join(key_cols), "metric": "row_membership",
            "old_value": n_only_old, "new_value": n_only_new,
            "abs_diff": None, "pct_diff": None,
            "flags": f"rows_only_in_old={n_only_old};rows_only_in_new={n_only_new}" if (n_only_old or n_only_new) else "",
        })

    both = merged[merged["_merge"] == "both"]
    for _, r in both.iterrows():
        key_str = "|".join(str(r[c]) for c in key_cols)
        for col in numeric_cols:
            oc, nc = f"{col}_old", f"{col}_new"
            if oc not in r or nc not in r:
                continue
            old_v, new_v = r[oc], r[nc]
            if pd.isna(old_v) and pd.isna(new_v):
                continue
            row = {"file": rel, "key": key_str, "metric": col, "old_value": old_v, "new_value": new_v}
            try:
                old_f, new_f = float(old_v), float(new_v)
                row["abs_diff"] = new_f - old_f
                row["pct_diff"] = (new_f - old_f) / abs(old_f) if old_f else float("nan")
                changed = not math.isclose(old_f, new_f, rel_tol=1e-9, abs_tol=1e-12)
            except (TypeError, ValueError):
                row["abs_diff"] = None
                row["pct_diff"] = None
                changed = old_v != new_v
            flags = flag_for(col, old_v, new_v)
            row["flags"] = ";".join(flags)
            if flags or changed:
                rows.append(row)
    return rows


def find_common_files(old_root: Path, new_root: Path) -> list[str]:
    exts = {".csv", ".parquet", ".json"}
    old_files = {str(p.relative_to(old_root)) for p in old_root.rglob("*") if p.is_file() and p.suffix in exts}
    new_files = {str(p.relative_to(new_root)) for p in new_root.rglob("*") if p.is_file() and p.suffix in exts}
    return sorted(old_files & new_files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--old-label", default="OLD")
    ap.add_argument("--new-label", default="NEW")
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    old_root, new_root = args.old_dir.resolve(), args.new_dir.resolve()
    common = find_common_files(old_root, new_root)
    print(f"Comparing {len(common)} common files: {old_root} vs {new_root}", file=sys.stderr)

    all_rows: list[dict] = []
    for rel in common:
        old_path, new_path = old_root / rel, new_root / rel
        if rel.endswith(".json"):
            all_rows.extend(compare_json(old_path, new_path, rel))
        else:
            old_df, new_df = load_table(old_path), load_table(new_path)
            if old_df is None or new_df is None:
                continue
            all_rows.extend(compare_table(old_df, new_df, rel))

    out_df = pd.DataFrame(all_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} diff rows to {args.out_csv}", file=sys.stderr)

    # ---- Markdown report ----
    lines = []
    lines.append(f"# Results snapshot comparison: {args.old_label} vs {args.new_label}\n")
    lines.append(f"- OLD: `{old_root}`")
    lines.append(f"- NEW: `{new_root}`")
    lines.append(f"- Files compared (present in both): {len(common)}")
    lines.append(f"- Total flagged/changed metric rows: {len(out_df)}\n")

    if not out_df.empty and "flags" in out_df.columns:
        flagged = out_df[out_df["flags"].fillna("") != ""]
    else:
        flagged = pd.DataFrame()

    lines.append("## Headline: flagged changes (sign flip, significance crossing, N>5%, effect>25%)\n")
    if flagged.empty:
        lines.append("None.\n")
    else:
        lines.append("| file | key | metric | old | new | flags |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in flagged.iterrows():
            lines.append(f"| {r['file']} | {r['key']} | {r['metric']} | {r['old_value']} | {r['new_value']} | {r['flags']} |")
        lines.append("")

    lines.append("## Per-topic detail\n")
    if not out_df.empty:
        out_df["topic"] = out_df["file"].apply(lambda f: f.split("/")[0])
        for topic, grp in out_df.groupby("topic"):
            lines.append(f"### {topic} ({len(grp)} changed metric rows across {grp['file'].nunique()} files)\n")
            for f, fg in grp.groupby("file"):
                n_flag = int((fg["flags"].fillna("") != "").sum())
                lines.append(f"- `{f}`: {len(fg)} changed metrics, {n_flag} flagged")
            lines.append("")
    else:
        lines.append("No numeric differences detected in any common file.\n")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote report to {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
