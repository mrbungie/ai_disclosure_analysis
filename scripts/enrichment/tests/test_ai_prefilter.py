"""Tests for the terminal lexical + semantic AI prefilter."""

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

COMMON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON_DIR))

import ai_prefilter
import ai_prefilter_anchors


class AnchorRegistrationTests(unittest.TestCase):
    def test_anchor_rows_match_the_configured_anchors(self):
        """Counts come from configs/ai_prefilter.yaml, not from a literal here:
        anchors are meant to be retuned, and a test that hardcodes how many
        there are just breaks every time someone does the intended thing."""
        expected_positive = sum(len(texts) for texts in ai_prefilter_anchors.positive_anchors().values())
        expected_negative = len(ai_prefilter_anchors.negative_anchors())

        rows = (pl.DataFrame(ai_prefilter_anchors.anchor_rows())
                .group_by("polarity").len().sort("polarity").rows())

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

    def test_lexical_terms_are_lowercase_for_the_lowercased_text_match(self):
        for term in ai_prefilter_anchors.strong_terms() + ai_prefilter_anchors.weak_terms():
            self.assertEqual(term, term.lower())


class LexicalScoringTests(unittest.TestCase):
    def test_lexical_scores_separate_strong_and_weak_matches(self):
        paragraphs = pl.DataFrame({
            "paragraph_index": [3, 1, 2],
            "paragraph_text": ["No relevant term appears here.",
                               "We deploy generative AI in our products.",
                               "Our valuation model uses an algorithm."],
        })

        rows = ai_prefilter.lexical_scores(paragraphs).rows()

        # "generative AI" matchea DOS términos: la sigla suelta y el compuesto.
        self.assertEqual(rows[0], (1, ["ai", "generative ai"], [], True, False))
        self.assertEqual(rows[1], (2, [], ["algorithm", "model"], False, True))
        self.assertEqual(rows[2], (3, [], [], False, False))


class LexicalBoundaryTests(unittest.TestCase):
    """El límite de palabra es lo único que separa la sigla del ruido: sin él,
    "ai" matchea "said", "certain", "chair" y "remain", que son de las palabras
    más comunes del corpus. Con él puesto, distinguir mayúsculas ya no aporta
    (medido: 29 párrafos de diferencia en 3,28M)."""

    CASOS = [
        ("We deploy AI across our operations.", True, "sigla suelta"),
        ("Our AI-powered platform scales.", True, "sigla con guion"),
        ("The board said that margins improved.", False, "'said' contiene 'ai'"),
        ("A certain chair remains available.", False, "'certain', 'chair', 'remains'"),
        ("Revenue in Thailand grew.", False, "'Thailand' contiene 'ai'"),
        ("We use artificial intelligence.", True, "término largo"),
    ]

    def test_word_boundary_separates_the_acronym_from_ordinary_words(self):
        paragraphs = pl.DataFrame({
            "paragraph_index": list(range(1, len(self.CASOS) + 1)),
            "paragraph_text": [text for text, _, _ in self.CASOS],
        })
        rows = ai_prefilter.lexical_scores(paragraphs).rows()

        for (text, esperado, motivo), row in zip(self.CASOS, rows):
            with self.subTest(texto=text, motivo=motivo):
                self.assertEqual(row[3], esperado, f"{text!r} ({motivo})")


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
