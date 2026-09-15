"""Unit tests for scripts/bronze/text_split.py.

The split rules define `paragraph_index` and `text_hash`, the keys every
additive output was written against, so each quirk the corpus depends on is
pinned here.
"""

import sys
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_split import split_paragraphs, split_sentences, text_hash8  # noqa: E402


def paragraphs(text: str) -> pl.DataFrame:
    sections = pl.LazyFrame({"country_code": ["us"], "form": ["10-K"], "accession_number": ["0001"],
                             "item_key": ["1A"], "section_text": [text]})
    return split_paragraphs(sections).collect().sort("paragraph_index")


class TextHashTest(unittest.TestCase):
    def test_blake2b8_big_endian(self):
        self.assertEqual(text_hash8(""), text_hash8(None))
        self.assertEqual(text_hash8("abc"), int.from_bytes(__import__("hashlib").blake2b(b"abc", digest_size=8).digest(), "big"))


class ParagraphSplitTest(unittest.TestCase):
    def test_prose_lines_stay_separate_tables_and_lists_merge(self):
        p = paragraphs("Intro line.\nSecond line.\n| a | b |\n| --- | --- |\n| 1 | 2 |\n• one\n• two")
        self.assertEqual(p["content_type"].to_list(), ["prose", "prose", "table", "list"])
        self.assertEqual(p["paragraph_index"].to_list(), [1, 2, 3, 6])
        self.assertEqual(p["paragraph_text"][2], "| a | b |\n| --- | --- |\n| 1 | 2 |")

    def test_links_become_control_char(self):
        p = paragraphs("See [Risk Factors](#a1) below.")
        self.assertEqual(p["paragraph_text"][0], "See \x01 below.")

    def test_images_removed_and_page_furniture_dropped(self):
        p = paragraphs("![img1.jpg](img1.jpg)\n12\nTable of Contents\nReal text here.\n---\nMore text.")
        self.assertEqual(p["paragraph_text"].to_list(), ["Real text here.", "More text."])
        self.assertEqual(p["paragraph_index"].to_list(), [4, 6])

    def test_trim_strips_unicode_spaces_not_tabs(self):
        p = paragraphs("  Text with nbsp  ")
        self.assertEqual(p["paragraph_text"][0], "Text with nbsp")
        p = paragraphs("\tTabbed")
        self.assertEqual(p["paragraph_text"][0], "\tTabbed")

    def test_is_scorable(self):
        p = paragraphs("•\nAI\nReal words")
        self.assertEqual(p["is_scorable"].to_list(), [False, False, True])


class SentenceSplitTest(unittest.TestCase):
    def test_abbreviations_are_not_boundaries(self):
        s = split_sentences(paragraphs("Mr. Smith joined the U.S. team. It grew.").lazy()).collect()
        self.assertEqual(s["sentence_text"].to_list(), ["Mr. Smith joined the U.S. team.", "It grew."])

    def test_list_items_are_sentences(self):
        s = split_sentences(paragraphs("• one\n• two").lazy()).collect().sort("sentence_index")
        self.assertEqual(s["sentence_text"].to_list(), ["• one", "• two"])


if __name__ == "__main__":
    unittest.main()
