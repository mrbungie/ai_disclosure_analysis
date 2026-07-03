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

    def test_enhanced_categories_and_words(self):
        # Generative AI vs Classical ML
        self.assertTrue(run_bow_extraction("We are leveraging ChatGPT and generative AI.")["has_gen_ai_mention"])
        self.assertTrue(run_bow_extraction("Using machine learning for classification.")["has_classical_ml_mention"])
        
        # Internal Productivity vs Product Integration
        self.assertTrue(run_bow_extraction("Automating tasks to achieve productivity gains.")["has_internal_productivity"])
        self.assertTrue(run_bow_extraction("Monetize the product offering via pricing tiers.")["has_product_integration"])
        
        # Financial specificity
        self.assertTrue(run_bow_extraction("Capex datacenter investments.")["has_capex_mention"])
        self.assertTrue(run_bow_extraction("Running costs are operating expenses.")["has_opex_mention"])
        self.assertTrue(run_bow_extraction("Research and development r&d expenses.")["has_rd_mention"])
        self.assertTrue(run_bow_extraction("Top-line growth and new revenue streams.")["has_revenue_impact"])
        
        # Regulations
        self.assertTrue(run_bow_extraction("complying with the FTC and SEC.")["has_us_regulation"])
        self.assertTrue(run_bow_extraction("adhering to the EU AI Act and GDPR.")["has_eu_regulation"])
        
        # Labor displacement
        self.assertTrue(run_bow_extraction("risks of workforce reduction and layoffs.")["has_labor_displacement_risk"])
        
        # Individual word presence features
        features = run_bow_extraction("We purchased a gpu and many GPUs for model training.")
        self.assertTrue(features["has_word_gpu"])
        self.assertTrue(features["has_word_gpus"])
        self.assertFalse(features["has_word_tpu"])
        
        # Check normalization collisions (e.g. "r&d" and "r & d" -> "has_word_r_d")
        features_rd1 = run_bow_extraction("our r&d budget")
        features_rd2 = run_bow_extraction("our r & d budget")
        self.assertTrue(features_rd1["has_word_r_d"])
        self.assertTrue(features_rd2["has_word_r_d"])

    def test_new_thesis_features(self):
        # 1. has_financial_quantification
        self.assertTrue(run_bow_extraction("AI improves yield by 12.5%.")["has_financial_quantification"])
        self.assertTrue(run_bow_extraction("We spent $50 million on servers.")["has_financial_quantification"])
        self.assertFalse(run_bow_extraction("We use AI in our products.")["has_financial_quantification"])

        # 2. has_ai_use_case_specific
        self.assertTrue(run_bow_extraction("We use AI for fraud detection.")["has_ai_use_case_specific"])
        self.assertTrue(run_bow_extraction("underwriting loans using machine learning")["has_ai_use_case_specific"])
        self.assertTrue(run_bow_extraction("predictive maintenance of aircraft engines")["has_ai_use_case_specific"])
        self.assertFalse(run_bow_extraction("We use general artificial intelligence.")["has_ai_use_case_specific"])

        # 3. has_competitor_ai_mention
        self.assertTrue(run_bow_extraction("We compete with other firms in building large AI models.")["has_competitor_ai_mention"])
        self.assertTrue(run_bow_extraction("AI models developed by our rivals.")["has_competitor_ai_mention"])
        # Too far apart should be False (or in different sentences)
        self.assertFalse(run_bow_extraction("The competition is fierce. We also use AI.")["has_competitor_ai_mention"])

        # 4. has_deepseek_impact
        self.assertTrue(run_bow_extraction("the rise of DeepSeek models")["has_deepseek_impact"])
        self.assertTrue(run_bow_extraction("disruption from open-source models")["has_deepseek_impact"])
        self.assertTrue(run_bow_extraction("threat of cost-efficient models")["has_deepseek_impact"])
        self.assertFalse(run_bow_extraction("we have open source software")["has_deepseek_impact"])

        # 5. has_ai_model_name
        self.assertTrue(run_bow_extraction("We use GPT-4 for code generation.")["has_ai_model_name"])
        self.assertTrue(run_bow_extraction("Running Llama-3 models.")["has_ai_model_name"])
        self.assertTrue(run_bow_extraction("Google Gemini is integrated.")["has_ai_model_name"])
        self.assertFalse(run_bow_extraction("Generic model details.")["has_ai_model_name"])

        # 6. count_ai_use_case_types
        # Matches: fraud_underwriting ("underwriting"), customer_support ("chatbot"), and code_generation ("developer productivity")
        features_multi = run_bow_extraction("We use a customer support chatbot, automated underwriting, and improve developer productivity.")
        self.assertEqual(features_multi["count_ai_use_case_types"], 3)
        self.assertEqual(run_bow_extraction("Normal text.")["count_ai_use_case_types"], 0)

    def test_synonyms_and_inflections(self):
        # 1. Test get_wordnet_synonyms
        transform_syns = extract_bow_features.get_wordnet_synonyms("transform")
        # WordNet should return synonyms like transmute, metamorphose, etc.
        self.assertTrue(len(transform_syns) > 0)
        # Ensure it has some expected synonyms
        self.assertTrue("transmute" in transform_syns or "metamorphose" in transform_syns)
        
        # Test technical/acronym exclusions
        self.assertEqual(extract_bow_features.get_wordnet_synonyms("gpu"), set())
        self.assertEqual(extract_bow_features.get_wordnet_synonyms("GPU"), set())
        self.assertEqual(extract_bow_features.get_wordnet_synonyms("GDPR"), set())
        self.assertEqual(extract_bow_features.get_wordnet_synonyms("SEC"), set())
        
        # 2. Test get_inflections
        run_infls = extract_bow_features.get_inflections("run")
        self.assertIn("running", run_infls)
        self.assertIn("runs", run_infls)
        
        # 3. Test synonym mapping to parent column
        # "transmute" is a synonym of "transform" (vague word).
        # "transmuting" is an inflection of "transmute".
        # It should trigger 'has_word_transform' and increment 'count_vague_words'.
        features = run_bow_extraction("We are transmuting our operations.")
        self.assertTrue(features["has_word_transform"], "Should fold 'transmuting' back to 'has_word_transform'")
        self.assertGreaterEqual(features["count_vague_words"], 1, "Should count 'transmuting' as a vague word match")

if __name__ == "__main__":
    sys.exit(unittest.main())
