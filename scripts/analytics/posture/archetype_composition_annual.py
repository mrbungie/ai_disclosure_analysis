"""Annual disclosure-posture composition and firm-level transition matrix
(thesis.qmd `fig-archetype-composition-annual`, ~1655-1800).

Reads data/gold/covariates/firm_year/posture_archetype_static.parquet
(`archetype`: the static full-sample posture archetype model,
models/posture_archetype_static/model.pkl, applied to each firm-year's
posture; descriptive, not point in time -- this script never refits),
restricted to the firm-years the model labels. No LLM/model call here: it
only reshapes the label panel into (1) each year's composition share and
(2) year-over-year firm-level persistence and transition counts.

Writes:
  - data/results/posture/archetype_composition_annual.parquet
    (year, archetype, n_firms, share_pct) -- one row per (year, archetype).
  - data/results/posture/archetype_transitions.parquet
    (from_year, to_year, origin, dest, n, pct_of_origin, persist_pct) --
    one row per (from_year, to_year, origin, dest); `persist_pct` (constant
    within a from_year/to_year pair) is the overall firm-level persistence
    rate mean(a == b) over firms common to both years, reproduced here as
    sum(diagonal n) / sum(all n) for that year pair so a consumer does not
    need to recompute it from the full matrix.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

ARCH_ORDER = ["No AI", "Defensive Disclosers", "Governance-Led Disclosers", "Vocal Substantives"]


def main() -> None:
    panel = L.read_gold("firm_year", ("covariates", "posture_archetype_static", ["archetype"]))
    panel = panel[panel["archetype"].notna()]
    years = sorted(panel["year"].unique())

    ct = pd.crosstab(panel["year"], panel["archetype"]).reindex(columns=ARCH_ORDER, fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0) * 100
    comp_rows = [{"year": int(y), "archetype": a, "n_firms": int(ct.loc[y, a]), "share_pct": float(share.loc[y, a])}
                for y in years for a in ARCH_ORDER]
    comp_df = pd.DataFrame(comp_rows)
    comp_out = L.results_path("posture", "archetype_composition_annual.parquet")
    comp_df.to_parquet(comp_out, index=False)

    trans_rows = []
    for y0, y1 in zip(years[:-1], years[1:]):
        a = panel[panel["year"] == y0].set_index("ticker")["archetype"]
        b = panel[panel["year"] == y1].set_index("ticker")["archetype"]
        common = a.index.intersection(b.index)
        persist_pct = float((a.loc[common].values == b.loc[common].values).mean()) * 100
        tm = pd.crosstab(a.loc[common], b.loc[common]).reindex(index=ARCH_ORDER, columns=ARCH_ORDER, fill_value=0)
        tm_pct = tm.div(tm.sum(axis=1), axis=0) * 100
        for origin, dest in product(ARCH_ORDER, ARCH_ORDER):
            trans_rows.append({"from_year": int(y0), "to_year": int(y1), "origin": origin, "dest": dest,
                              "n": int(tm.loc[origin, dest]), "pct_of_origin": float(tm_pct.loc[origin, dest]),
                              "persist_pct": persist_pct})
    trans_df = pd.DataFrame(trans_rows)
    trans_out = L.results_path("posture", "archetype_transitions.parquet")
    trans_df.to_parquet(trans_out, index=False)

    print(f"years: {years}")
    print(f"-> {comp_out} ({len(comp_df):,} rows)")
    print(f"-> {trans_out} ({len(trans_df):,} rows)")


if __name__ == "__main__":
    main()
