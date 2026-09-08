"""Tests for point-in-time facts parsed from a filing's own inline XBRL."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path


class InlineXbrlFactTests(unittest.TestCase):
    def test_parses_duration_and_instant_facts_with_filing_provenance(self) -> None:
        """Dropping the filing metadata or context dates would reintroduce leakage."""
        from scripts.analytics.filing_xbrl_facts import parse_inline_xbrl

        html = """
        <html><body>
          <xbrli:context id="duration"><xbrli:entity><xbrli:identifier>1</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-03-31</xbrli:endDate></xbrli:period>
          </xbrli:context>
          <xbrli:context id="instant"><xbrli:entity><xbrli:identifier>1</xbrli:identifier></xbrli:entity>
            <xbrli:period><xbrli:instant>2024-03-31</xbrli:instant></xbrli:period>
          </xbrli:context>
          <ix:nonFraction name="us-gaap:Revenues" contextRef="duration" scale="3">1,234</ix:nonFraction>
          <ix:nonFraction name="us-gaap:Assets" contextRef="instant">5,678</ix:nonFraction>
        </body></html>
        """

        facts = parse_inline_xbrl(
            html,
            ticker="ACME",
            accession_number="0000000000-24-000001",
            filing_date="2024-05-01",
        )

        self.assertEqual(
            facts,
            [
                {
                    "ticker": "ACME",
                    "accession_number": "0000000000-24-000001",
                    "filing_date": "2024-05-01",
                    "context_id": "duration",
                    "has_dimensions": False,
                    "concept": "us-gaap:Revenues",
                    "numeric_value": 1234000.0,
                    "period_type": "duration",
                    "period_start": "2024-01-01",
                    "period_end": "2024-03-31",
                },
                {
                    "ticker": "ACME",
                    "accession_number": "0000000000-24-000001",
                    "filing_date": "2024-05-01",
                    "context_id": "instant",
                    "has_dimensions": False,
                    "concept": "us-gaap:Assets",
                    "numeric_value": 5678.0,
                    "period_type": "instant",
                    "period_start": None,
                    "period_end": "2024-03-31",
                },
            ],
        )

    def test_can_limit_extraction_to_the_financial_concepts_needed_downstream(self) -> None:
        """A broad parser must not turn every presentational tag into raw data."""
        from scripts.analytics.filing_xbrl_facts import parse_inline_xbrl

        html = """
        <xbrli:context id="instant"><xbrli:period><xbrli:instant>2024-03-31</xbrli:instant></xbrli:period></xbrli:context>
        <ix:nonFraction name="us-gaap:Assets" contextRef="instant">50</ix:nonFraction>
        <ix:nonFraction name="dei:EntityCommonStockSharesOutstanding" contextRef="instant">10</ix:nonFraction>
        """

        facts = parse_inline_xbrl(
            html,
            ticker="ACME",
            accession_number="0000000000-24-000001",
            filing_date="2024-05-01",
            concepts={"us-gaap:Assets"},
        )

        self.assertEqual([fact["concept"] for fact in facts], ["us-gaap:Assets"])

    def test_filing_with_no_numeric_facts_writes_a_readable_empty_parquet(self) -> None:
        """An empty filing must not poison DuckDB's glob reader with a zero-column file."""
        import importlib
        import pandas as pd

        extract_one = importlib.import_module("scripts.us.04_extract_inline_xbrl_facts").extract_one

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.html.gz"
            destination = Path(directory) / "facts.parquet"
            with gzip.open(source, "wt", encoding="utf-8") as file:
                file.write("<html><body>No inline facts.</body></html>")
            _, count, error = extract_one({
                "ticker": "ACME",
                "accession_number": "0000000000-24-000001",
                "filing_date": "2024-05-01",
                "local_path": str(source),
                "destination": str(destination),
            })

            self.assertEqual((count, error), (0, None))
            self.assertIn("accession_number", pd.read_parquet(destination).columns)


if __name__ == "__main__":
    unittest.main()
