"""Call disclosure: AI disclosure, disclosed substance and decoupling of each
call, and the firm's history over its earlier calls.

  disclosure      AI frames per 1,000 words of the call transcript
  n_activities    activities of the activity spine attributed to the call
                  (taxonomy channel == "call", representative document = the call)
  grounding       mean of the five grounding markers (named function,
                  deployed/scaled, named product or process, quantified
                  outcome, named third-party provider), each shrunk with the
                  empirical-Bayes prior of `build_washing_score` fitted on the
                  calls of all EARLIER calendar quarters; null without activities
  substance       log(1 + n_activities) * grounding (zero without activities)
  w               pctrank(disclosure) - pctrank(substance), each call ranked
                  against the calls of all earlier calendar quarters (the first
                  quarter against itself)
  n_prior_calls, hist_disclosure, hist_substance, hist_w
                  count and expanding mean over the firm's STRICTLY earlier
                  calls
  surprise_disclosure, surprise_substance
                  value of the call minus its history

Output: covariates/call/disclosure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "activity"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "firm_year"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from build_activity import flags  # noqa: E402
from build_washing_score import GROUNDING_COMPONENTS, _apply_shrink, _fit_shrink_prior  # noqa: E402

import layers as L  # noqa: E402

BUILDER = "scripts/gold/call/build_disclosure.py"
COLUMNS = ["n_words", "n_frames", "disclosure", "n_activities", "grounding", "substance", "w", "n_prior_calls",
           "hist_disclosure", "surprise_disclosure", "hist_substance", "surprise_substance", "hist_w"]


def _pit_rank(values: pd.Series, quarters: pd.Series) -> pd.Series:
    """Percentile of each value within the calls of all EARLIER calendar
    quarters (average rank for ties). The first quarter has no history and
    is ranked against itself."""
    out = pd.Series(np.nan, index=values.index)
    for q in sorted(quarters.unique()):
        cur = quarters == q
        ref = values[quarters < q]
        ref = np.sort((ref if len(ref) else values[cur]).dropna().values)
        v = values[cur].values
        out[cur] = (np.searchsorted(ref, v, side="left") + np.searchsorted(ref, v, side="right")) / (2 * len(ref))
    return out


def attach_activities(panel: pd.DataFrame) -> pd.DataFrame:
    act = L.read_gold("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))
    act_calls = act[act.channel == "call"].copy()
    fl = flags(act_calls)
    for c in GROUNDING_COMPONENTS:
        act_calls[c] = fl[c].astype(float)
    per_call = act_calls.groupby("accession_number").agg(
        n_activities=("text_hash", "size"), **{c: (c, "sum") for c in GROUNDING_COMPONENTS}
    ).reset_index().rename(columns={"accession_number": "call_accession_number"})
    per_call = per_call.merge(panel[["call_accession_number", "fecha"]], on="call_accession_number", how="inner")
    per_call["quarter"] = pd.to_datetime(per_call["fecha"]).dt.to_period("Q")
    parts = []
    for q, cur in per_call.groupby("quarter", sort=True):
        hist = per_call[per_call["quarter"] < q]
        hist = hist if len(hist) else cur
        shrunk = pd.concat([_apply_shrink(cur[c], cur["n_activities"], *_fit_shrink_prior(hist[c], hist["n_activities"]))
                            for c in GROUNDING_COMPONENTS], axis=1)
        parts.append(cur.assign(grounding=shrunk.mean(axis=1)))
    per_call = pd.concat(parts)
    panel = panel.merge(per_call[["call_accession_number", "n_activities", "grounding"]], on="call_accession_number", how="left")
    panel["n_activities"] = panel["n_activities"].fillna(0.0)
    # grounding is undefined (null) for a call without activities; its substance is zero
    panel["substance"] = np.log1p(panel["n_activities"]) * panel["grounding"].fillna(0.0)
    quarters = pd.to_datetime(panel["fecha"]).dt.to_period("Q")
    panel["w"] = _pit_rank(panel["disclosure"], quarters) - _pit_rank(panel["substance"], quarters)
    return panel


def attach_history(panel: pd.DataFrame) -> pd.DataFrame:
    # A ticker's calls are at least 30 days apart (the call-sequence invariant
    # of silver.filing_manifest); call_accession_number only keeps the order
    # deterministic.
    panel = panel.sort_values(["ticker", "fecha", "call_accession_number"]).reset_index(drop=True)
    panel["n_prior_calls"] = panel.groupby("ticker").cumcount()
    panel["hist_disclosure"] = panel.groupby("ticker")["disclosure"].transform(lambda s: s.expanding().mean().shift(1))
    panel["hist_substance"] = panel.groupby("ticker")["substance"].transform(lambda s: s.expanding().mean().shift(1))
    panel["surprise_disclosure"] = panel["disclosure"] - panel["hist_disclosure"]
    panel["surprise_substance"] = panel["substance"] - panel["hist_substance"]
    panel["hist_w"] = panel.groupby("ticker")["w"].transform(lambda s: s.expanding().mean().shift(1))
    return panel


def main() -> None:
    panel = L.read_gold("call", spine_columns=L.GOLD_SPINE_COLUMNS["call"]).merge(
        L.read_gold("document", ("covariates", "disclosure_volume", ["n_words", "n_frames"]))
        [["accession_number", "n_words", "n_frames"]].rename(columns={"accession_number": "call_accession_number"}),
        on="call_accession_number", how="left", validate="one_to_one")
    panel["disclosure"] = 1000.0 * panel["n_frames"] / panel["n_words"]
    panel = attach_history(attach_activities(panel))
    L.write_gold("covariates", "call", "disclosure", panel[L.GOLD_SPINE_COLUMNS["call"] + COLUMNS], builder=BUILDER)


if __name__ == "__main__":
    main()
