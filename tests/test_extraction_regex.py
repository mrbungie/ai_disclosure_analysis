import unittest
import re
import sys
import importlib.util
from pathlib import Path

# Load scripts/04_extract_sections.py dynamically because its name starts with numbers
spec = importlib.util.spec_from_file_location(
    "extract_sections", 
    str(Path(__file__).parent.parent / "scripts" / "04_extract_sections.py")
)
extract_sections = importlib.util.module_from_spec(spec)
sys.modules["extract_sections"] = extract_sections
spec.loader.exec_module(extract_sections)

PATTERNS = extract_sections.PATTERNS
extract_section = extract_sections.extract_section

class TestExtractionRegex(unittest.TestCase):
    def test_item_1_start_patterns(self):
        pattern = PATTERNS["Item 1"]["start"]
        
        # Standard format
        self.assertTrue(re.search(pattern, "Item 1. Business", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "  Item 1 Business  ", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "ITEM 1. BUSINESS", re.IGNORECASE))
        
        # Markdown headers
        self.assertTrue(re.search(pattern, "# Item 1. Business", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "## Item 1. BUSINESS", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "**Item 1. Business**", re.IGNORECASE))
        
        # Markdown table row/pipe formatted (failed cases previously)
        self.assertTrue(re.search(pattern, "| Item 1. | Business |", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "|Item 1.|Business|", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "  |  Item 1.  |  Business  |", re.IGNORECASE))
        
        # Honeywell fallback style (failed cases previously)
        self.assertTrue(re.search(pattern, "ABOUT HONEYWELL", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "  ABOUT HONEYWELL  ", re.IGNORECASE))
        
        # False positives should not match
        self.assertFalse(re.search(pattern, "This is about Honeywell's business", re.IGNORECASE))
        self.assertFalse(re.search(pattern, "Item 10. Directors and Officers", re.IGNORECASE))

    def test_item_1a_start_patterns(self):
        pattern = PATTERNS["Item 1A"]["start"]
        
        # Standard format
        self.assertTrue(re.search(pattern, "Item 1A. Risk Factors", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "ITEM 1A RISK FACTORS", re.IGNORECASE))
        
        # Markdown / Pipe formatting
        self.assertTrue(re.search(pattern, "# Item 1A. Risk Factors", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "| Item 1A. | Risk Factors |", re.IGNORECASE))
        
        # Honeywell fallback style
        self.assertTrue(re.search(pattern, "RISK FACTORS", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "  RISK FACTORS  ", re.IGNORECASE))

    def test_item_7_start_patterns(self):
        pattern = PATTERNS["Item 7"]["start"]
        
        # Standard format
        self.assertTrue(re.search(pattern, "Item 7. Management's Discussion and Analysis", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "ITEM 7 MANAGEMENT'S DISCUSSION", re.IGNORECASE))
        
        # Markdown / Pipe formatting
        self.assertTrue(re.search(pattern, "| Item 7. | Management's Discussion |", re.IGNORECASE))
        
        # Honeywell fallback style / Typographic variations (curly quotes)
        self.assertTrue(re.search(pattern, "MANAGEMENT'S DISCUSSION AND ANALYSIS", re.IGNORECASE))
        self.assertTrue(re.search(pattern, "MANAGEMENT’S DISCUSSION AND ANALYSIS", re.IGNORECASE))

    def test_extraction_boundary_logic(self):
        # We need mock lines of text to test extract_section
        # The section_text must be > 1000 characters to be successfully returned by extract_section.
        content_item_1 = "This is the business content. " * 50  # ~1500 chars
        content_item_1a = "This is the risk factors content. " * 50  # ~1700 chars
        content_item_7 = "This is the management discussion. " * 50  # ~1800 chars
        
        mock_filing_lines = [
            "TABLE OF CONTENTS",
            "Item 1. Business",
            "Item 1A. Risk Factors",
            "Item 1B. Unresolved Staff Comments",
            "Item 7. Management's Discussion",
            "Item 8. Financial Statements",
            "--- ACTUAL DOCUMENT BODY ---",
            "Item 1. Business",
            content_item_1,
            "Item 1A. Risk Factors",
            content_item_1a,
            "Item 1B. Unresolved Staff Comments",
            "Item 7. Management's Discussion",
            content_item_7,
            "Item 8. Financial Statements"
        ]
        
        # 1. Test Item 1 Extraction (should skip TOC and extract from second match to Item 1A)
        item_1_text = extract_section(
            mock_filing_lines, 
            PATTERNS["Item 1"]["start"], 
            PATTERNS["Item 1"]["end"]
        )
        self.assertIn("This is the business content", item_1_text)
        self.assertNotIn("This is the risk factors content", item_1_text)
        
        # 2. Test Item 1A Extraction (should stop at Item 1B)
        item_1a_text = extract_section(
            mock_filing_lines,
            PATTERNS["Item 1A"]["start"],
            PATTERNS["Item 1A"]["end"]
        )
        self.assertIn("This is the risk factors content", item_1a_text)
        self.assertNotIn("This is the management discussion", item_1a_text)
        
        # 3. Test Item 7 Extraction (should stop at Item 8)
        item_7_text = extract_section(
            mock_filing_lines,
            PATTERNS["Item 7"]["start"],
            PATTERNS["Item 7"]["end"]
        )
        self.assertIn("This is the management discussion", item_7_text)
        self.assertNotIn("Financial Statements", item_7_text)

    def test_extract_section_fallback(self):
        # Test fallback when no end tag is found
        content_item_7_long = "This is long fallback discussion. " * 60  # ~2000 chars
        mock_filing_lines = [
            "Item 7. Management's Discussion and Analysis",
            content_item_7_long
        ]
        
        item_7_text = extract_section(
            mock_filing_lines,
            PATTERNS["Item 7"]["start"],
            PATTERNS["Item 7"]["end"]
        )
        self.assertIn("This is long fallback discussion", item_7_text)
        
    def test_extract_section_length_filter(self):
        # If the extracted section text is too short (<= 1000 chars), it should not be returned (returns empty string)
        short_content = "Too short."
        mock_filing_lines = [
            "Item 1. Business",
            short_content,
            "Item 1A. Risk Factors"
        ]
        
        extracted_text = extract_section(
            mock_filing_lines,
            PATTERNS["Item 1"]["start"],
            PATTERNS["Item 1"]["end"]
        )
        self.assertEqual(extracted_text, "")

if __name__ == "__main__":
    unittest.main()
