"""Illustrative peer comparisons across firm-years
(thesis.qmd `tbl-firm-comparisons`, ~3636-3711).

Reads gold:
  - covariates/firm_year/washing_score (scored firm-years: `w` not null)
    with frames_per_1k from covariates/firm_year/disclosure_volume.
  - covariates/firm_year/posture_archetype_static: the static pooled
    archetype projected onto each firm-year.
  - spines/activity/activity + covariates/activity/{extraction, taxonomy}:
    one row per disclosed AI activity.

Writes data/results/washing/firm_comparisons.parquet, one row per case
(ticker, year) with the same fields the qmd table renders: archetype,
pct_d, frames_per_1k, n_activities, n_activities_filing, grounding_index,
substance, pct_s, w. The firm's display name, ordering and hand-written
due-diligence diagnosis text stay in the qmd (they are prose, not data).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

CASES = [("JPM", 2024), ("GS", 2024), ("MSFT", 2024), ("AAPL", 2024), ("WELL", 2025)]


def main() -> None:
    ws = L.read_dataset("firm_year", ("covariates", "washing_score"),
                        ("covariates", "disclosure_volume", ["frames_per_1k"]))
    ws = ws[ws["w"].notna()].reset_index(drop=True)
    ws["pct_d"] = ws.groupby("year")["frames_per_1k"].rank(pct=True) * 100
    ws["pct_s"] = ws.groupby("year")["substance"].rank(pct=True) * 100

    dim = L.read_gold("firm_year", ("covariates", "posture_archetype_static", ["archetype"])) \
        .dropna(subset=["archetype"]).set_index(["ticker", "year"])["archetype"]

    act = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))

    rows = []
    for t, y in CASES:
        r = ws[(ws["ticker"] == t) & (ws["year"] == y)]
        a = act[act["ticker"] == t]
        arch_val = dim.get((t, y), "Defensive Disclosers" if t == "WELL" else "—")
        row = {"ticker": t, "year": y, "archetype": arch_val}
        if len(r):
            r = r.iloc[0]
            row.update({
                "pct_d": float(r["pct_d"]), "frames_per_1k": float(r["frames_per_1k"]),
                "n_activities": int(len(a)), "n_activities_filing": int((a["channel"] == "filing").sum()),
                "grounding_index": float(r["grounding_index"]), "substance": float(r["substance"]),
                "pct_s": float(r["pct_s"]), "w": float(r["w"]),
            })
        else:
            for k in ["pct_d", "frames_per_1k", "n_activities", "n_activities_filing", "grounding_index",
                     "substance", "pct_s", "w"]:
                row[k] = None
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out = L.results_path("washing", "firm_comparisons.parquet")
    out_df.to_parquet(out, index=False)
    print(out_df)
    print(f"-> {out} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
