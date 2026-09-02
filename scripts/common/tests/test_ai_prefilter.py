"""Tests for the terminal lexical + semantic AI prefilter."""

import sys
import unittest
from pathlib import Path

import duckdb
import numpy as np

COMMON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON_DIR))

import ai_prefilter
import ai_prefilter_anchors


class AnchorRegistrationTests(unittest.TestCase):
    def test_register_anchors_matches_the_configured_anchors(self):
        """Counts come from configs/ai_prefilter.yaml, not from a literal here:
        anchors are meant to be retuned, and a test that hardcodes how many
        there are just breaks every time someone does the intended thing."""
        expected_positive = sum(len(texts) for texts in ai_prefilter_anchors.positive_anchors().values())
        expected_negative = len(ai_prefilter_anchors.negative_anchors())

        con = duckdb.connect()
        try:
            ai_prefilter_anchors.register_anchors(con)
            rows = con.sql(
                "SELECT polarity, count(*) FROM ai_prefilter_anchors GROUP BY polarity ORDER BY polarity"
            ).fetchall()
        finally:
            con.close()

        self.assertEqual(rows, [("negative", expected_negative), ("positive", expected_positive)])
        self.assertGreater(expected_positive, 0)
        self.assertGreater(expected_negative, 0)

    def test_anchor_ids_are_unique(self):
        ids = [row["anchor_id"] for row in ai_prefilter_anchors.anchor_rows()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_anchor_text_is_single_line(self):
        """YAML block scalars wrap; an anchor's embedding must not depend on
        where the line happened to break."""
        for row in ai_prefilter_anchors.anchor_rows():
            self.assertNotIn("\n", row["anchor_text"])
            self.assertEqual(row["anchor_text"], " ".join(row["anchor_text"].split()))

    def test_lexical_terms_are_lowercase_for_the_duckdb_contains_match(self):
        for term in ai_prefilter_anchors.strong_terms() + ai_prefilter_anchors.weak_terms():
            self.assertEqual(term, term.lower())


class LexicalScoringTests(unittest.TestCase):
    def test_duckdb_lexical_query_separates_strong_and_weak_matches(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE paragraphs (paragraph_index INTEGER, paragraph_text VARCHAR)")
        con.execute(
            "INSERT INTO paragraphs VALUES "
            "(1, 'We deploy generative AI in our products.'), "
            "(2, 'Our valuation model uses an algorithm.'), "
            "(3, 'No relevant term appears here.')"
        )

        rows = con.sql(ai_prefilter.lexical_scores_sql("paragraphs")).fetchall()
        con.close()

        self.assertEqual(rows[0], (1, ["generative ai"], [], True, False))
        self.assertEqual(rows[1], (2, [], ["algorithm", "model"], False, True))
        self.assertEqual(rows[2], (3, [], [], False, False))


class SemanticScoringTests(unittest.TestCase):
    def test_scores_use_maximum_anchor_per_category_and_negative_maximum(self):
        paragraph_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        anchor_embeddings = np.array(
            [[0.8, 0.0], [0.0, 0.7], [0.6, 0.0]], dtype=np.float32
        )
        anchors = [
            {"anchor_id": "ai_use_01", "category": "ai_use", "polarity": "positive"},
            {"anchor_id": "ai_use_02", "category": "ai_use", "polarity": "positive"},
            {"anchor_id": "negative_01", "category": "negative", "polarity": "negative"},
        ]

        result = ai_prefilter.score_embeddings(paragraph_embeddings, anchor_embeddings, anchors)

        self.assertAlmostEqual(result["score_ai_use"][0], 0.8)
        self.assertAlmostEqual(result["score_ai_use"][1], 0.7)
        self.assertAlmostEqual(result["negative_similarity"][0], 0.6)
        self.assertEqual(result["best_semantic_anchor"].tolist(), ["ai_use", "ai_use"])


if __name__ == "__main__":
    unittest.main()
