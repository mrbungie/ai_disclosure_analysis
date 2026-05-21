import unittest
import importlib.util
import sys
from pathlib import Path

# Load scripts/07_score_with_rules.py dynamically because its name starts with numbers
spec = importlib.util.spec_from_file_location(
    "score_with_rules", 
    str(Path(__file__).parent.parent / "scripts" / "07_score_with_rules.py")
)
score_with_rules = importlib.util.module_from_spec(spec)
sys.modules["score_with_rules"] = score_with_rules
spec.loader.exec_module(score_with_rules)

run_rules = score_with_rules.run_rules

class TestScoreWithRules(unittest.TestCase):
    def test_board_oversight_pattern(self):
        # Direct phrase
        self.assertTrue(run_rules("Our Board oversight extends to AI.")["has_board_oversight"])
        self.assertTrue(run_rules("oversight by the board of directors")["has_board_oversight"])
        self.assertTrue(run_rules("board's active oversight of technology")["has_board_oversight"])
        
        # Too far apart should fail
        self.assertFalse(run_rules("The board met yesterday. Also, the oversight team was present.")["has_board_oversight"])

    def test_audit_committee_pattern(self):
        self.assertTrue(run_rules("The audit committee reviewed the report.")["has_audit_committee"])
        self.assertTrue(run_rules("discussion by our audit committees")["has_audit_committee"])
        self.assertFalse(run_rules("The audit team met with the committee.")["has_audit_committee"])

    def test_vendors_patterns(self):
        self.assertTrue(run_rules("We deploy Nvidia H100 GPUs.")["has_vendor_nvidia"])
        self.assertTrue(run_rules("Partnered with OpenAI.")["has_vendor_openai"])
        self.assertTrue(run_rules("Running on Microsoft Azure.")["has_vendor_microsoft"])
        self.assertTrue(run_rules("Using Google Gemini.")["has_vendor_google"])
        self.assertTrue(run_rules("Integrating DeepSeek models.")["has_vendor_deepseek"])

    def test_metrics_patterns(self):
        # Percentage
        self.assertTrue(run_rules("AI improves performance by 15.5%.")["has_metric_percentage"])
        self.assertTrue(run_rules("a 20 percent increase")["has_metric_percentage"])
        
        # Dollars
        self.assertTrue(run_rules("We spent $10 million on AI compute.")["has_metric_dollar"])
        self.assertTrue(run_rules("invested 500 million dollars")["has_metric_dollar"])
        self.assertTrue(run_rules("costs of 50000 USD")["has_metric_dollar"])

if __name__ == "__main__":
    unittest.main()
