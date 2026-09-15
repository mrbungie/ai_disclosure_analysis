"""Regression checks for the XBRL concept fallbacks used in firm financials."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


MODULE = Path(__file__).resolve().parents[2] / "gold" / "financials" / "build_firm_financials.py"


def duration_metrics() -> dict[str, list[str]]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "DURATION_METRICS":
            return ast.literal_eval(node.value)
    raise AssertionError("DURATION_METRICS was not found")


class ResearchAndDevelopmentConceptTests(unittest.TestCase):
    def test_rd_expense_includes_software_and_ifrs_equivalents(self) -> None:
        concepts = duration_metrics()["rd_expense"]

        self.assertIn(
            "us-gaap:ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
            concepts,
        )
        self.assertIn("ifrs-full:ResearchAndDevelopmentExpense", concepts)


if __name__ == "__main__":
    unittest.main()
