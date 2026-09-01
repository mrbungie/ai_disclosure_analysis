import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import section_segmenter as ss  # noqa: E402


def toc_row(item, title, page):
    return f"| [{item}.]({page}) | | | [{title}]({page}) | | | [{page}]({page}) | | |"


class TestItemRegex(unittest.TestCase):
    """The 3 real letter-suffix formats filers actually use — a regex that
    only handles one silently merges the other two's content into a
    neighboring plain-numbered item (the CLX bug: an entire Item 1C
    section absorbed into "Item 1" because "1.C." parsed as just "1")."""

    def _key(self, text):
        m = ss.ITEM_RE.search(text)
        return ss._item_key(m) if m else None

    def test_suffixed_letter(self):
        self.assertEqual(self._key("ITEM 1A. Risk Factors"), "1A")

    def test_parenthesized_letter(self):
        self.assertEqual(self._key("ITEM 1 (B). Unresolved Staff Comments"), "1B")

    def test_dotted_letter(self):
        self.assertEqual(self._key("ITEM 1.C. Cybersecurity"), "1C")

    def test_plain_number_unaffected(self):
        self.assertEqual(self._key("ITEM 1. Business"), "1")

    def test_no_false_letter_from_following_word(self):
        # "Item 14. Certain Relationships..." must NOT read as "Item 14C" —
        # a bare letter is only valid right after the digit/period, never
        # after a plain space into an unrelated word.
        self.assertEqual(self._key("Item 14. Certain Relationships and Related Transactions"), "14")


class TestTocRowDetection(unittest.TestCase):
    def test_link_row_is_toc(self):
        self.assertTrue(ss.is_link_row("| [Item 1.](#anchor) | | | [Business](#anchor) | | |"))

    def test_plain_heading_is_not_toc(self):
        self.assertFalse(ss.is_link_row("| ITEM 1. | | | BUSINESS | | |"))


class TestTocPrefixDetection(unittest.TestCase):
    """A TOC is the maximal prefix of item candidates, from the very first
    one, that increases strictly with no repeat and no big gap. This
    replaced a proximity-window heuristic that swept up real headings
    sitting right after the TOC block ends (the AAPL regression: Item 1
    coverage dropped 53% before this fix)."""

    def test_toc_prefix_excludes_real_heading_immediately_after(self):
        raw = [("1", 0), ("1A", 1), ("1B", 2), ("2", 3), ("3", 4),
               ("1", 30)]  # real heading, right after the TOC ends
        end = ss._toc_prefix_end(raw)
        self.assertEqual(raw[end:], [("1", 30)])

    def test_sparse_non_toc_candidates_not_treated_as_toc(self):
        # Too few / too far apart to be a real TOC (e.g. a filer whose TOC
        # doesn't spell "Item N" at all — only its 1-2 real body headings
        # match anything). The prefix-min/gap guards must not eat these.
        raw = [("1", 122), ("6", 4577)]
        self.assertEqual(ss._toc_prefix_end(raw), 0)


class TestGeneralSegment(unittest.TestCase):
    def test_toc_and_body_both_present(self):
        body_text = "Real content. " * 100
        lines = [
            toc_row("Item 1", "Business", 5),
            toc_row("Item 1A", "Risk Factors", 15),
            toc_row("Item 1B", "Unresolved Staff Comments", 20),
            toc_row("Item 2", "Properties", 22),
            toc_row("Item 3", "Legal Proceedings", 24),
            "ITEM 1. BUSINESS",
            body_text,
            "ITEM 1A. RISK FACTORS",
            "Different content here.",
        ]
        segments = ss.general_segment(lines)
        self.assertIn("1", segments)
        self.assertIn(body_text.strip(), segments["1"])
        self.assertNotIn("Different content", segments["1"])

    def test_alias_heading_recognized(self):
        lines = ["ABOUT HONEYWELL", "Honeywell is a diversified company."]
        segments = ss.general_segment(lines)
        self.assertIn("1", segments)
        self.assertIn("diversified company", segments["1"])


if __name__ == "__main__":
    unittest.main()
