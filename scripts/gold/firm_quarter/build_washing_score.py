"""
scripts/gold/firm_quarter/build_washing_score.py — layer 2: AI-washing index W
per (ticker, quarter), one file per channel family and window.

Same construct as scripts/gold/firm_year/build_washing_score.py, measured on layer-1
covariates (build_measures.py) instead of firm-years:

    Substance = log(1 + n_activities) * Grounding
    W         = pctrank(Disclosure) - pctrank(Substance)

Disclosure = `<family>_<window>_frames_per_1k` (disclosure_volume); n_activities
and the five grounding markers are the `<family>_<window>_*` columns of activities.
Everything population-level is estimated within each quarter's
cross-section, which only contains information published up to the quarter
end: the empirical-Bayes grounding priors, the percentile ranks and the
descriptive 5% tails. W is defined for firms with at least one AI frame in
the window (`any_ai`); every other spine row is kept with null W.

Windows: `expanding` = everything published up to the quarter end (the
firm's accumulated gap); `ttm` = the last four quarters; `quarter` = only the
quarter's documents (input for surprise measures against the expanding history).

Output: covariates/firm_quarter/washing_score, one column set
per family and window: `<family>_<window>_<metric>`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "firm_year"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from build_washing_score import GROUNDING_COMPONENTS, TAIL_Q, _apply_shrink, _fit_shrink_prior  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import layers as L  # noqa: E402

BUILDER = "scripts/gold/firm_quarter/build_washing_score.py"
FAMILIES = ("filings", "calls")
WINDOWS = ("quarter", "ttm", "expanding")
METRICS = ["grounding_index", "substance", "pct_disclosure", "pct_substance", "w", "washing", "callada"]


def score_quarter(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["any_ai"] > 0].copy()
    d["n_activities"] = d["n_activities"].fillna(0.0)
    priors = {c: _fit_shrink_prior(d[c], d["n_activities"]) for c in GROUNDING_COMPONENTS}
    shrunk = pd.concat([_apply_shrink(d[c], d["n_activities"], *priors[c]) for c in GROUNDING_COMPONENTS], axis=1)
    d["grounding_index"] = shrunk.mean(axis=1)
    d["substance"] = np.log1p(d["n_activities"]) * d["grounding_index"]
    d["pct_disclosure"] = d["frames_per_1k"].rank(pct=True)
    d["pct_substance"] = d["substance"].rank(pct=True)
    d["w"] = d["pct_disclosure"] - d["pct_substance"]
    d["washing"] = d["w"] >= d["w"].quantile(1 - TAIL_Q)
    d["callada"] = d["w"] <= d["w"].quantile(TAIL_Q)
    return d


def main() -> None:
    result = L.read_gold("firm_quarter")
    for family in FAMILIES:
        for window in WINDOWS:
            tag = f"{family}_{window}"
            vol = pd.read_parquet(L.gold_path("covariates", "firm_quarter", "disclosure_volume"),
                                  columns=["ticker", "quarter"] + [f"{tag}_{c}" for c in ("frames_per_1k", "any_ai")])
            act = pd.read_parquet(L.gold_path("covariates", "firm_quarter", "activities"),
                                  columns=["ticker", "quarter"] + [f"{tag}_{c}" for c in ["n_activities"] + GROUNDING_COMPONENTS])
            panel = vol.merge(act, on=["ticker", "quarter"], how="left").rename(columns=lambda c: c.removeprefix(f"{tag}_"))
            scored = pd.concat([score_quarter(g) for _, g in panel.groupby("quarter", sort=True)], ignore_index=True)
            scored = scored[["ticker", "quarter"] + METRICS].rename(columns={c: f"{tag}_{c}" for c in METRICS})
            result = result.merge(scored, on=["ticker", "quarter"], how="left", validate="one_to_one")
            print(f"{tag}: mean W by year {scored.groupby(scored['quarter'].str[:4])[f'{tag}_w'].mean().round(3).to_dict()}")
    L.write_gold("covariates", "firm_quarter", "washing_score", result, builder=BUILDER,
                 extra={"columns": "<family>_<window>_<metric>", "families": list(FAMILIES), "windows": list(WINDOWS),
                        "usable_from": "as_of_date = quarter end + 1 day"})


if __name__ == "__main__":
    main()
