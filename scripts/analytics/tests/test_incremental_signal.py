"""Regression checks for coefficient reporting in the M2 specification."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analytics.shock import incremental_signal as signal


class CoefficientReportingTests(unittest.TestCase):
    def test_m2_coefficients_include_clustered_confidence_intervals(self) -> None:
        rng = np.random.default_rng(7)
        rows = 80
        d = pd.DataFrame({
            "ticker": np.repeat([f"F{i}" for i in range(20)], 4),
            "sector_year": np.repeat(["10_2021", "10_2022", "20_2021", "20_2022"], 20),
        })
        for column in signal.FUNDAMENTALS + ["frames_per_1k"] + list(signal.SEMANTIC.values()):
            d[column] = rng.uniform(0.05, 2.0, size=rows)
        d["outcome"] = 0.5 * d["spec_per_1k"] + rng.normal(size=rows)

        reported = signal.coefficients(d, "outcome", "sector_year", "")

        specificity = reported["especificidad"]
        self.assertIn("ci95", specificity)
        self.assertEqual(len(specificity["ci95"]), 2)
        self.assertLess(specificity["ci95"][0], specificity["beta_std"])
        self.assertGreater(specificity["ci95"][1], specificity["beta_std"])


if __name__ == "__main__":
    unittest.main()
