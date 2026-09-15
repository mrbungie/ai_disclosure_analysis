"""Word-normalized version of the "venue gap over time" trend (Chapter 5,
Section 5.3), consistent with the word-based denominator adopted for the
headline calls-vs-filings result (Section 5.1). Reuses `channel_gap_analysis`
and `channel_gap_words_robustness` without modifying either; recomputes the
firm-FE event-study trend on promo_per_1k_words instead of promo_per_1k
(paragraphs), on the same extensive-margin cells.

Determinístico, sin LLM. Requiere `duckdb/thesis.duckdb`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_gap_analysis import load_frames, DB, EVENT, _fe_ols  # noqa: E402
from channel_gap_words_robustness import load_documents_words  # noqa: E402


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    frames = load_frames(con)
    cell = load_documents_words(con, frames)
    con.close()

    outcomes = ["promo_per_1k_words", "quant_per_1k_words"]
    wide = cell.pivot(index=["ticker", "fy"], columns="channel", values=outcomes)
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=[f"{outcomes[0]}_call", f"{outcomes[0]}_filing"]).reset_index().rename(columns={"fy": "t"})
    for y in outcomes:
        wide[f"gap_{y}"] = wide[f"{y}_call"] - wide[f"{y}_filing"]
    wide["post"] = (wide["t"] >= EVENT["fy"]).astype(int)

    print(f"Extensive-margin cells: {len(wide):,} | firms {wide.ticker.nunique():,}")
    print("\nGap (call - filing) by fiscal year, promo_per_1k_words:")
    by_year = wide.groupby("t")["gap_promo_per_1k_words"].mean()
    print(by_year.round(3).to_string())

    d = wide[["ticker", "t", "post", "gap_promo_per_1k_words"]].dropna().rename(columns={"gap_promo_per_1k_words": "y"})
    both = d.groupby("ticker")["post"].nunique()
    d = d[d["ticker"].isin(both[both == 2].index)]
    res, _ = _fe_ols(d["y"], pd.DataFrame({"post": d["post"].astype(float)}), d["ticker"])
    print(f"\nFirm-FE DiD, promo_per_1k_words gap, post-{EVENT['fy']}: "
          f"b={res.params[0]:+.4f} (se {res.bse[0]:.4f}, p={res.pvalues[0]:.4f}), n={len(d)}, firms={d.ticker.nunique()}")

    periods = sorted(d["t"].unique()); base = periods[0]
    X = pd.DataFrame({f"t{p}": (d["t"] == p).astype(float) for p in periods[1:]})
    res2, _ = _fe_ols(d["y"], X, d["ticker"])
    print(f"Event study (base = {base}): " + " | ".join(
        f"{p}: {res2.params[i]:+.3f} (p={res2.pvalues[i]:.3f})" for i, p in enumerate(periods[1:])))


if __name__ == "__main__":
    main()
