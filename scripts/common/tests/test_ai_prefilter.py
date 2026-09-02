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
    def test_register_anchors_exposes_positive_and_negative_anchor_rows(self):
        con = duckdb.connect()
        try:
            ai_prefilter_anchors.register_anchors(con)
            rows = con.sql(
                "SELECT polarity, count(*) FROM ai_prefilter_anchors GROUP BY polarity ORDER BY polarity"
            ).fetchall()
        finally:
            con.close()

        self.assertEqual(rows, [("negative", 4), ("positive", 16)])


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
