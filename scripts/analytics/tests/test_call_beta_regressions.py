"""Tests for the point-in-time leverage controls used in call-beta models."""

from __future__ import annotations

import pandas as pd


def test_select_instant_values_prefers_primary_tag_and_latest_context() -> None:
    from scripts.analytics.call_beta_regressions import select_instant_values

    facts = pd.DataFrame({
        "accession_number": ["a", "a", "a", "a"],
        "concept": ["us-gaap:Assets", "us-gaap:Assets", "us-gaap:Liabilities", "us-gaap:Liabilities"],
        "numeric_value": [100.0, 120.0, 60.0, 70.0],
        "period_end": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"]),
    })

    selected = select_instant_values(facts, {
        "assets": ["us-gaap:Assets"],
        "liabilities": ["us-gaap:Liabilities"],
    })

    assert selected.loc[0, "assets"] == 120.0
    assert selected.loc[0, "liabilities"] == 70.0
