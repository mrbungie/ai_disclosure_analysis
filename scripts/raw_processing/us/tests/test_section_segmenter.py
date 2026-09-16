import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    one, that is table-row-formatted (a markdown table row and/or a link)
    and tightly packed. This replaced a proximity-window heuristic that
    swept up real headings sitting right after the TOC block ends (the
    AAPL regression: Item 1 coverage dropped 53% before that fix), and
    later a rank-based (strictly-increasing, no-repeat) version that broke
    on a 10-Q's combined TOC, which lists Part I's items then RESTARTS at
    Part II's item 1 — a rank decrease inside the TOC itself,
    indistinguishable by rank alone from real content starting."""

    def test_toc_prefix_excludes_real_heading_immediately_after(self):
        lines = [
            toc_row("Item 1", "Business", 5),
            toc_row("Item 1A", "Risk Factors", 15),
            toc_row("Item 1B", "Unresolved Staff Comments", 20),
            toc_row("Item 2", "Properties", 22),
            toc_row("Item 3", "Legal Proceedings", 24),
            "ITEM 1. BUSINESS",  # real heading, right after the TOC ends
        ]
        raw = ss._raw_candidates(lines)
        end = ss._toc_prefix_end(raw, lines)
        self.assertEqual(raw[end:], [("1", 5)])

    def test_dual_part_toc_restart_not_treated_as_body(self):
        # A 10-Q's single combined TOC lists Part I (items 1-4) then
        # restarts at Part II's item 1 — still all table rows, still one
        # contiguous TOC block, even though the item numbers decrease.
        lines = [
            toc_row("Item 1", "Financial Statements", 3),
            toc_row("Item 2", "MD&A", 10),
            toc_row("Item 3", "Market Risk", 20),
            toc_row("Item 4", "Controls and Procedures", 21),
            toc_row("Item 1", "Legal Proceedings", 22),
            toc_row("Item 1A", "Risk Factors", 22),
            "ITEM 1. FINANCIAL STATEMENTS",  # real heading, after the TOC
        ]
        raw = ss._raw_candidates(lines)
        end = ss._toc_prefix_end(raw, lines)
        self.assertEqual(raw[end:], [("1", 6)])

    def test_sparse_non_toc_candidates_not_treated_as_toc(self):
        # Too few / too far apart to be a real TOC (e.g. a filer whose TOC
        # doesn't spell "Item N" at all — only its 1-2 real body headings
        # match anything). The prefix-min/gap guards must not eat these.
        lines = ["ITEM 1. BUSINESS"] + ["filler"] * 120 + ["ITEM 6. EXHIBITS"]
        raw = ss._raw_candidates(lines)
        self.assertEqual(ss._toc_prefix_end(raw, lines), 0)


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

    def test_10q_mda_alias_heading_recognized(self):
        # Illumina's 10-Q Item 2 heading is bare "MANAGEMENT'S DISCUSSION
        # & ANALYSIS" — the TOC entry says "Item 2" as usual, but the real
        # body heading never restates it. Only applies under form="10-Q";
        # the same alias text means Item 7 on a 10-K, not Item 2.
        lines = [
            "| MANAGEMENT’S DISCUSSION & ANALYSIS | | |",
            "Revenue increased 5% in Q1 primarily due to demand.",
        ]
        segments = ss.general_segment(lines, form="10-Q")
        self.assertIn("2", segments)
        self.assertIn("Revenue increased", segments["2"])

    def test_10q_mda_alias_not_applied_under_10k_form(self):
        lines = [
            "| MANAGEMENT’S DISCUSSION & ANALYSIS | | |",
            "Revenue increased 5% in Q1 primarily due to demand.",
        ]
        segments = ss.general_segment(lines, form="10-K")
        self.assertNotIn("2", segments)


class TestItem8Recovery(unittest.TestCase):
    """Item 8 (Financial Statements) frequently has no real heading to find
    at all — a stub ("see our Consolidated Financial Statements...") or a
    reused page-header stamp, with the actual content starting elsewhere
    under its own natural headers. Found on real filings (NVDA/INTC/SYF/AAL
    had a 0-206 char "Item 8" before this fix; 335K+/148K+/315K+/194K+
    after). The recovery anchors on the audit report's opening phrase,
    which every audited 10-K's financial statements begin with."""

    def test_recovers_stub_pointing_elsewhere(self):
        lines = (
            ["Item 8. Financial Statements and Supplementary Data",
             "The information required by this Item is set forth in our Consolidated Financial Statements."]
            + ["padding line " + str(i) for i in range(20)]
            + ["Report of Independent Registered Public Accounting Firm",
               "Opinion on the Financial Statements"]
            + ["financial statement content line " + str(i) for i in range(200)]
            + ["Item 9. Changes in and Disagreements with Accountants"]
        )
        segments = ss.general_segment(lines)
        self.assertIn("8", segments)
        self.assertIn("Opinion on the Financial Statements", segments["8"])
        self.assertGreater(len(segments["8"]), ss._ITEM_8_MIN_PLAUSIBLE_CHARS)

    def test_no_recovery_when_segment_already_plausible(self):
        real_content = "real financial statement content. " * 200  # > 3000 chars
        lines = ["Item 8. Financial Statements and Supplementary Data", real_content,
                 "Item 9. Changes in and Disagreements with Accountants"]
        segments = ss.general_segment(lines)
        self.assertIn("real financial statement content", segments["8"])
        self.assertNotIn("Report of Independent", segments["8"])

    def test_no_recovery_when_no_anchor_present(self):
        lines = ["Item 8. Financial Statements and Supplementary Data",
                 "The information required by this Item is set forth elsewhere.",
                 "Item 9. Changes in and Disagreements with Accountants"]
        segments = ss.general_segment(lines)
        # No audit-report anchor anywhere in the doc -> left as the short stub.
        self.assertLess(len(segments.get("8", "")), ss._ITEM_8_MIN_PLAUSIBLE_CHARS)


class TestTocSubRowAnchor(unittest.TestCase):
    """A cross-reference-index TOC (WFC's 10-Qs) gives the item row itself
    no page number, and therefore no anchor — the page links sit on the
    sub-section rows listed under it. The item's start is its first
    sub-row's anchor."""

    def _toc(self):
        return [
            "| Item 1. | | | Financial Statements | | | Page | | |",
            "|  | | | Consolidated Balance Sheet | | | [55](#stmts) | | |",
            "| Item 2. | | | Management's Discussion and Analysis | | |  | | |",
            "|  | | | Summary Financial Data | | | [2](#mda) | | |",
            "|  | | | Overview | | | [3](#overview) | | |",
            "| PART II | | | Other Information | | |  | | |",
            "| Item 1. | | | Legal Proceedings | | | [128](#legal) | | |",
        ]

    def test_item_inherits_first_subrow_anchor(self):
        lines = self._toc()
        rows = ss._raw_candidates(lines, form="10-Q")
        entries = dict(ss._toc_anchor_entries(lines, rows))
        self.assertEqual(entries["1"], "stmts")
        self.assertEqual(entries["2"], "mda")
        self.assertEqual(entries["1__partII"], "legal")

    def test_inheritance_stops_at_the_next_part_row(self):
        # Item 2's own row and its sub-rows carry no link at all here, so it
        # must resolve to nothing rather than borrow Part II's first anchor.
        lines = [l for l in self._toc() if "#mda" not in l and "#overview" not in l]
        rows = ss._raw_candidates(lines, form="10-Q")
        entries = dict(ss._toc_anchor_entries(lines, rows))
        self.assertNotIn("2", entries)


if __name__ == "__main__":
    unittest.main()
