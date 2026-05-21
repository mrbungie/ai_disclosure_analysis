import unittest
import importlib.util
import sys
from pathlib import Path

# Load scripts/07_extract_bow_features.py dynamically because its name starts with numbers
spec = importlib.util.spec_from_file_location(
    "extract_bow_features", 
    str(Path(__file__).parent.parent / "scripts" / "07_extract_bow_features.py")
)
extract_bow_features = importlib.util.module_from_spec(spec)
sys.modules["extract_bow_features"] = extract_bow_features
spec.loader.exec_module(extract_bow_features)

run_bow_extraction = extract_bow_features.run_bow_extraction

class TestBowFeatures(unittest.TestCase):
    def test_board_oversight_pattern(self):
        # Direct phrase
        self.assertTrue(run_bow_extraction("Our Board oversight extends to AI.")["has_board_oversight"])
        self.assertTrue(run_bow_extraction("oversight by the board of directors")["has_board_oversight"])
        self.assertTrue(run_bow_extraction("board's active oversight of technology")["has_board_oversight"])
        
        # Too far apart should fail
        self.assertFalse(run_bow_extraction("The board met yesterday. Also, the oversight team was present.")["has_board_oversight"])

    def test_audit_committee_pattern(self):
        self.assertTrue(run_bow_extraction("The audit committee reviewed the report.")["has_audit_committee"])
        self.assertTrue(run_bow_extraction("discussion by our audit committees")["has_audit_committee"])
        self.assertFalse(run_bow_extraction("The audit team met with the committee.")["has_audit_committee"])

    def test_ethics_policy_pattern(self):
        self.assertTrue(run_bow_extraction("Our responsible AI policies ensure safety.")["has_ethics_policy"])
        self.assertTrue(run_bow_extraction("reviewed by our ethics committee")["has_ethics_policy"])
        self.assertTrue(run_bow_extraction("established ethical standards for AI")["has_ethics_policy"])
        self.assertFalse(run_bow_extraction("We ethics.")["has_ethics_policy"])

    def test_compliance_pattern(self):
        self.assertTrue(run_bow_extraction("We comply with data governance regulations.")["has_compliance"])
        self.assertTrue(run_bow_extraction("meeting regulatory compliance standards")["has_compliance"])
        self.assertTrue(run_bow_extraction("assessing internal controls over AI systems")["has_compliance"])

    def test_risk_factor_pattern(self):
        self.assertTrue(run_bow_extraction("This presents a significant risk factor.")["has_risk_factor"])
        self.assertTrue(run_bow_extraction("could adversely affect our margins")["has_risk_factor"])
        self.assertTrue(run_bow_extraction("risks and uncertainties discussed below")["has_risk_factor"])

    def test_regulatory_risk_pattern(self):
        self.assertTrue(run_bow_extraction("Subject to increasing regulatory scrutiny.")["has_regulatory_risk"])
        self.assertTrue(run_bow_extraction("in compliance with the EU AI Act")["has_regulatory_risk"])

    def test_cyber_privacy_risk_pattern(self):
        self.assertTrue(run_bow_extraction("risks of data breach or unauthorized access")["has_cyber_privacy_risk"])
        self.assertTrue(run_bow_extraction("compliance with GDPR and CCPA rules")["has_cyber_privacy_risk"])

    def test_ethics_bias_risk_pattern(self):
        self.assertTrue(run_bow_extraction("potential model bias and hallucinate issues")["has_ethics_bias_risk"])
        self.assertTrue(run_bow_extraction("reputational issues with inaccuracies in output")["has_ethics_bias_risk"])

    def test_ip_copyright_risk_pattern(self):
        self.assertTrue(run_bow_extraction("subject to intellectual property and copyright infringement claims")["has_ip_copyright_risk"])
        self.assertTrue(run_bow_extraction("licensing proprietary data or patents")["has_ip_copyright_risk"])

    def test_supply_infra_risk_pattern(self):
        self.assertTrue(run_bow_extraction("mitigating chip shortages and compute constraints")["has_supply_infra_risk"])
        self.assertTrue(run_bow_extraction("elevated energy consumption and power constraints")["has_supply_infra_risk"])

    def test_vendors_patterns(self):
        self.assertTrue(run_bow_extraction("We deploy Nvidia H100 GPUs.")["has_vendor_nvidia"])
        self.assertTrue(run_bow_extraction("Partnered with OpenAI.")["has_vendor_openai"])
        self.assertTrue(run_bow_extraction("open ai models in Azure.")["has_vendor_openai"])
        self.assertTrue(run_bow_extraction("Running on Microsoft Azure.")["has_vendor_microsoft"])
        self.assertTrue(run_bow_extraction("Using Google Gemini.")["has_vendor_google"])
        self.assertTrue(run_bow_extraction("Integrating DeepSeek models.")["has_vendor_deepseek"])
        self.assertTrue(run_bow_extraction("Deployed on Amazon AWS Bedrock.")["has_vendor_amazon"])
        self.assertTrue(run_bow_extraction("Training Meta Llama models.")["has_vendor_meta"])
        self.assertTrue(run_bow_extraction("Using Anthropic Claude.")["has_vendor_anthropic"])
        self.assertTrue(run_bow_extraction("Using AMD Instinct chips.")["has_vendor_amd"])

    def test_metrics_patterns(self):
        # Percentage
        self.assertTrue(run_bow_extraction("AI improves performance by 15.5%.")["has_metric_percentage"])
        self.assertTrue(run_bow_extraction("a 20 percent increase")["has_metric_percentage"])
        
        # Dollars
        self.assertTrue(run_bow_extraction("We spent $10 million on AI compute.")["has_metric_dollar"])
        self.assertTrue(run_bow_extraction("invested 500 million dollars")["has_metric_dollar"])
        self.assertTrue(run_bow_extraction("costs of 50000 USD")["has_metric_dollar"])

    def test_specific_product_pattern(self):
        self.assertTrue(run_bow_extraction("Using Microsoft Copilot for productivity.")["has_specific_product"])
        self.assertTrue(run_bow_extraction("Adobe Firefly generative AI.")["has_specific_product"])
        self.assertTrue(run_bow_extraction("NVIDIA RTX graphics cards.")["has_specific_product"])

    def test_deployment_verb_pattern(self):
        self.assertTrue(run_bow_extraction("We launched a new generative tool.")["has_deployment_verb"])
        self.assertTrue(run_bow_extraction("integrated AI into our product suite")["has_deployment_verb"])
        self.assertTrue(run_bow_extraction("monetize through a subscription model")["has_deployment_verb"])

    def test_operational_training_pattern(self):
        self.assertTrue(run_bow_extraction("We focus on model training and dataset curation.")["has_model_training"])
        self.assertTrue(run_bow_extraction("scale compute infrastructure using GPUs and TPUs")["has_compute_infra"])
        self.assertTrue(run_bow_extraction("training on proprietary datasets")["has_proprietary_data"])

    def test_new_dimensions_patterns(self):
        self.assertTrue(run_bow_extraction("recruiting AI talent and hiring software engineers")["has_workforce_talent"])
        self.assertTrue(run_bow_extraction("strategic alliance and partnership with leading startups")["has_partnership"])
        self.assertTrue(run_bow_extraction("integrated a customer service chatbot virtual assistant")["has_customer_facing"])
        self.assertTrue(run_bow_extraction("developing self-driving technology for autonomous vehicles")["has_safety_critical"])
        self.assertTrue(run_bow_extraction("securing content licensing agreements with publishers")["has_data_licensing"])

    def test_competitor_academic_opensource(self):
        self.assertTrue(run_bow_extraction("Our competitors are investing heavily in competing technologies.")["has_competitor_mention"])
        self.assertTrue(run_bow_extraction("We are collaborating with academic research institutes and university scientists.")["has_academic_research"])
        self.assertTrue(run_bow_extraction("We publish our weights under open source licenses on GitHub.")["has_open_source"])

    def test_sentiment_features(self):
        # High positive sentiment
        pos_features = run_bow_extraction("AI adoption brings huge productivity benefits, improved quality, and growth opportunities.")
        self.assertGreater(pos_features["count_positive_words"], 0)
        self.assertEqual(pos_features["count_negative_words"], 0)
        self.assertGreater(pos_features["ratio_positive_words"], 0.0)
        self.assertEqual(pos_features["ratio_negative_words"], 0.0)
        self.assertEqual(pos_features["bow_sentiment_score"], 1.0)

        # High negative sentiment
        neg_features = run_bow_extraction("There are risks of system failure, data breaches, and litigation costs.")
        self.assertEqual(neg_features["count_positive_words"], 0)
        self.assertGreater(neg_features["count_negative_words"], 0)
        self.assertEqual(neg_features["ratio_positive_words"], 0.0)
        self.assertGreater(neg_features["ratio_negative_words"], 0.0)
        self.assertEqual(neg_features["bow_sentiment_score"], -1.0)

        # Mixed sentiment
        mixed_features = run_bow_extraction("While AI offers opportunities, it also introduces substantial risks.")
        self.assertGreater(mixed_features["count_positive_words"], 0)
        self.assertGreater(mixed_features["count_negative_words"], 0)
        self.assertTrue(-1.0 < mixed_features["bow_sentiment_score"] < 1.0)

        # Neutral sentiment
        neutral_features = run_bow_extraction("We use technology in our operations.")
        self.assertEqual(neutral_features["count_positive_words"], 0)
        self.assertEqual(neutral_features["count_negative_words"], 0)
        self.assertEqual(neutral_features["bow_sentiment_score"], 0.0)

    def test_counts_ratios_formulas(self):
        text = "This revolutionary next-generation AI system uses NVIDIA GPUs for training on datasets."
        features = run_bow_extraction(text)
        
        self.assertGreaterEqual(features["count_total_words"], 10)
        self.assertGreaterEqual(features["count_vague_words"], 1)
        self.assertGreaterEqual(features["count_specific_words"], 1)
        self.assertGreaterEqual(features["count_ai_mentions"], 1)
        
        # Ratios should be floats between 0 and 1
        self.assertTrue(0.0 <= features["ratio_vague_words"] <= 1.0)
        self.assertTrue(0.0 <= features["ratio_specific_words"] <= 1.0)
        self.assertTrue(0.0 <= features["ratio_ai_mentions"] <= 1.0)
        
        # Formulas
        self.assertTrue(features["vagueness_ratio"] >= 0.0)
        self.assertTrue(-1.0 <= features["specificity_score"] <= 1.0)
        self.assertTrue(features["total_risk_indicators_count"] >= 0)
        self.assertTrue(features["total_governance_indicators_count"] >= 0)

if __name__ == "__main__":
    sys.exit(unittest.main())
