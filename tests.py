from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch

from .authenticity_engine import classify_review
from .quality_engine import analyze_product_quality
from .trust_engine import analyze_review_trustworthiness, jaccard_similarity
from .product_intelligence import analyze_online_genuineness, compute_recommendation, analyze_product
from .runtime_config import get_kimi_deployment_issue
from .models import CheckedReview

class AuthenticityEngineTests(TestCase):
    def test_obviously_genuine_review(self):
        text = "I purchased this laptop two weeks ago. The battery lasts for about 6 hours of web browsing. The keyboard feels comfortable and keys have good travel. It is a bit heavy at 4.2 lbs, but overall it matches my expectations for a daily office device."
        result = classify_review(text)
        self.assertEqual(result["classification"], "Genuine")
        self.assertLess(result["authenticity_score"], 0.35)
        self.assertTrue(result["confidence"] > 0.6)

    def test_obviously_promotional_review(self):
        text = "OMG this is the BEST product ever!!! It completely changed my life!!! You must buy this right now before it sells out!!! ABSOLUTELY PERFECT AND AMAZING!!!"
        result = classify_review(text)
        self.assertIn(result["classification"], ["Likely Promotional", "Likely Fake"])
        self.assertTrue(result["authenticity_score"] > 0.50)

    def test_suspicious_repetitive_review(self):
        text = "perfect perfect perfect best best best love love love amazing amazing amazing super super super"
        result = classify_review(text)
        self.assertIn(result["classification"], ["Likely Fake", "Likely Promotional", "Suspicious"])


class QualityEngineTests(TestCase):
    def test_excellent_quality_reviews(self):
        reviews = [
            "This phone is amazing, the battery lasts forever and the screen is beautiful.",
            "Great value for money. Highly recommend this sleek design and powerful performance.",
            "Extremely comfortable fits perfectly. Excellent durability after a month of heavy use."
        ]
        result = analyze_product_quality(reviews)
        self.assertIn(result["rating"], ["Good", "Excellent"])
        self.assertTrue(result["quality_score"] > 0.5)
        self.assertTrue(len(result["strengths"]) > 0)

    def test_poor_quality_reviews(self):
        reviews = [
            "This was the worst purchase ever, the battery stopped working on day 1. Total waste of money.",
            "Terrible build quality. Cheap plastic cracked instantly. Do not buy.",
            "Highly defective performance. Malfunctions constantly and overheats."
        ]
        result = analyze_product_quality(reviews)
        self.assertEqual(result["rating"], "Poor")
        self.assertTrue(result["quality_score"] < 0.35)
        self.assertTrue(result["severe_issues_count"] >= 1)

    def test_empty_reviews_graceful_handling(self):
        result = analyze_product_quality([])
        self.assertEqual(result["rating"], "Average")
        self.assertEqual(result["quality_score"], 0.5)

    def test_negation_awareness(self):
        # Even though "durable" and "easy to use" are strength keywords, they are negated here
        reviews = [
            "This product is not durable at all, and it is definitely not easy to use.",
            "Do not buy. The screen is cheap and broke within two days.",
            "I had a terrible experience and regret this purchase."
        ]
        result = analyze_product_quality(reviews)
        self.assertNotIn("Durability Reliability", result["strengths"])
        self.assertNotIn("Usability Design", result["strengths"])
        self.assertEqual(result["rating"], "Poor")

    def test_bad_product_detection(self):
        # A mixed review with complaints should not be biased to "Good"
        reviews = [
            "The design is sleek, but it stops working constantly.",
            "It is lightweight but a total waste of money.",
            "Flimsy construction, do not buy."
        ]
        result = analyze_product_quality(reviews)
        self.assertIn(result["rating"], ["Poor", "Average"])
        self.assertTrue(result["quality_score"] < 0.55)


class TrustEngineTests(TestCase):
    def test_jaccard_similarity(self):
        s1 = {"apple", "banana", "orange"}
        s2 = {"banana", "orange", "grapes"}
        self.assertEqual(jaccard_similarity(s1, s2), 0.5)
        self.assertEqual(jaccard_similarity(s1, set()), 0.0)

    def test_high_trust_corpus(self):
        reviews = [
            "I bought this vacuum for my apartment. It picks up pet hair really well and fits in the closet.",
            "Decent sound bar. Setup was about 15 minutes. Bass is deep enough but the remote control feels cheap.",
            "Love the fabric of these sheets. Soft and cool for summer. Fits my queen mattress nicely."
        ]
        result = analyze_review_trustworthiness(reviews)
        self.assertEqual(result["trust_level"], "High Trust")
        self.assertTrue(result["trust_score"] > 0.7)
        self.assertEqual(result["duplicate_count"], 0)

    def test_low_trust_duplicate_corpus(self):
        reviews = [
            "This is a fantastic product! Best purchase I've ever made. 5 stars!",
            "This is a fantastic product! Best purchase I've ever made. 5 stars!",
            "I bought this phone and it works great. Simple assembly."
        ]
        result = analyze_review_trustworthiness(reviews)
        self.assertEqual(result["trust_level"], "Low Trust")
        self.assertTrue(result["duplicate_count"] >= 1)


class RecommendationEngineTests(TestCase):
    def test_recommendation_matrix(self):
        # Above 70: Recommended
        self.assertEqual(compute_recommendation(75), "Recommended")
        self.assertEqual(compute_recommendation(0.80), "Recommended")
        # 50 to 70: Buy With Caution
        self.assertEqual(compute_recommendation(65), "Buy With Caution")
        self.assertEqual(compute_recommendation(0.55), "Buy With Caution")
        self.assertEqual(compute_recommendation(50), "Buy With Caution")
        # Below 50: Not Recommended
        self.assertEqual(compute_recommendation(45), "Not Recommended")
        self.assertEqual(compute_recommendation(0.30), "Not Recommended")

    def test_trust_engine_rating_extraction(self):
        reviews = [
            "This is a 5 star product! The screen is beautiful.",
            "I'd rate it 1/5 stars. It broke instantly.",
            "Two stars because it takes forever to charge."
        ]
        result = analyze_review_trustworthiness(reviews)
        self.assertEqual(result["rating_distribution"].get("5"), 1)
        self.assertEqual(result["rating_distribution"].get("1"), 1)
        self.assertEqual(result["rating_distribution"].get("2"), 1)

    def test_polarized_rating_anomaly(self):
        reviews = [
            "Amazing! 5-star product.",
            "Five stars all the way.",
            "Worst ever. 1 star.",
            "Complete waste of money! One star.",
            "Perfect. 5 stars."
        ]
        result = analyze_review_trustworthiness(reviews)
        self.assertTrue(any("polarized" in w.lower() for w in result["warnings"]))

    def test_rating_sentiment_mismatch_anomaly(self):
        reviews = [
            "This is garbage, broke on day one! 5 stars",
            "This is garbage, broke on day one! 5 stars",
            "Terrible construction, completely cheap. 5-star",
            "I love it, works perfectly. 5 stars",
            "Works ok. 5 stars"
        ]
        result = analyze_review_trustworthiness(reviews)
        self.assertTrue(any("mismatch" in w.lower() for w in result["warnings"]))

    def test_online_genuineness_report_flags_counterfeit_risk(self):
        sources = [
            {
                "title": "Fake product warning",
                "url": "https://example.com/warning",
                "snippet": "Customers report counterfeit and duplicate units with no warranty.",
                "platform": "example.com",
                "evidence_type": "genuineness",
            },
            {
                "title": "How to verify serial number",
                "url": "https://brand.example/verify",
                "snippet": "Use the official serial number verification page for original products.",
                "platform": "brand.example",
                "evidence_type": "genuineness",
            },
        ]
        report = analyze_online_genuineness("Sample Headphones", "https://shop.example/item", sources)
        self.assertEqual(report["sources_checked"], 2)
        self.assertEqual(report["assessment"], "Mixed Authenticity Signals")
        self.assertTrue(report["risk_signals"])
        self.assertTrue(report["trust_signals"])

    def test_public_nvidia_kimi_endpoint_is_reported_as_unavailable(self):
        issue = get_kimi_deployment_issue("nvidia", "moonshotai/kimi-k2.6", "https://integrate.api.nvidia.com/v1")
        self.assertIn("free endpoint as unavailable", issue)

    def test_partner_kimi_endpoint_is_allowed(self):
        issue = get_kimi_deployment_issue("nvidia", "moonshotai/kimi-k2.6", "https://partner.example.com/v1")
        self.assertEqual(issue, "")


class APIIntegrationTests(APITestCase):
    @patch('reviews.views.generate_llm_review_report')
    @patch('reviews.views.analyze_platform_reviews')
    def test_post_predict_review_valid(self, mock_analyze, mock_llm):
        mock_analyze.return_value = {"available": False, "summary": "Scraping bypassed for test."}
        mock_llm.return_value = {"available": False, "summary": "LLM bypassed for test."}
        url = reverse('predict')
        data = {'review': 'This is a genuine review about a standard coffee machine. Works okay.'}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('prediction', response.data)
        self.assertIn('confidence', response.data)
        self.assertIn('explanation', response.data)

    def test_post_predict_review_invalid(self):
        url = reverse('predict')
        data = {'review': ''}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('reviews.product_intelligence.generate_kimi_advice')
    @patch('reviews.product_intelligence.search_duckduckgo')
    @patch('reviews.product_intelligence.analyze_platform_reviews')
    def test_post_product_trust_valid(self, mock_platform, mock_search, mock_kimi):
        mock_search.return_value = [
            {"title": "Good coffee maker", "url": "http://coffee.com", "snippet": "Decent value for money, but takes time.", "query": "test", "platform": "coffee.com"},
            {"title": "Coffee machine reviews", "url": "http://coffee.com", "snippet": "Strong performance and easy cleaning.", "query": "test", "platform": "coffee.com"}
        ]
        mock_platform.return_value = {"available": False, "summary": "Scraping bypassed for test."}
        mock_kimi.return_value = {"available": False, "provider": "nvidia", "model": "moonshotai/kimi-k2.6"}
        
        url = reverse('product-trust')
        data = {'product_name': 'Sample Coffee Machine'}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('recommendation', response.data)
        self.assertIn('trust_score', response.data)
        self.assertIn('fake_review_risk', response.data)


class HybridDecisionPipelineTests(APITestCase):
    @patch('reviews.views.generate_llm_review_report')
    @patch('reviews.views.analyze_platform_reviews')
    def test_hybrid_single_review_prediction_success(self, mock_platform, mock_llm):
        mock_platform.return_value = {
            "available": True,
            "samples": [{"text": "Works well, but the lid feels fragile."}],
            "summary": "One extracted platform review.",
        }
        mock_llm.return_value = {
            "available": True,
            "provider": "nvidia",
            "model": "moonshotai/kimi-k2.6",
            "analysis": {
                "ai_risk_score": 0.8,
                "authenticity_assessment": "Likely Fake",
                "summary": "Mock summary saying it's highly suspicious.",
                "severity": "High",
                "recommended_action": "Reject review",
                "evidence_strength": "Strong",
                "risk_factors": ["High repetition"],
                "trust_factors": [],
                "limitations": []
            }
        }
        
        url = reverse('predict')
        # This review text triggers high/low ML scores, but we want to verify the combination logic
        data = {
            'review': 'This is a genuine review about a standard coffee machine. Works okay.',
            'product_name': 'Super Coffee'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify the structure contains hybrid-specific keys
        self.assertIn('ml_prediction', response.data)
        self.assertIn('ml_confidence', response.data)
        self.assertEqual(response.data['ai_prediction'], 'Likely Fake')
        self.assertAlmostEqual(response.data['ai_confidence'], 0.77, places=2)
        self.assertIn('hybrid_confidence', response.data)
        self.assertEqual(response.data['decision_mode'], 'hybrid')
        self.assertEqual(response.data['engine_status']['ml']['status'], 'completed')
        self.assertEqual(response.data['engine_status']['kimi']['status'], 'completed')
        
        # Verify mathematical weight application: final risk score is 60% ML risk score + 40% AI risk score (0.8)
        self.assertIn('risk_score', response.data)
        
    @patch('reviews.views.generate_llm_review_report')
    @patch('reviews.views.analyze_platform_reviews')
    def test_hybrid_single_review_kimi_failure(self, mock_platform, mock_llm):
        mock_platform.return_value = {
            "available": True,
            "samples": [{"text": "Works well, but the lid feels fragile."}],
            "summary": "One extracted platform review.",
        }
        mock_llm.return_value = {
            "available": False,
            "provider": "nvidia",
            "model": "moonshotai/kimi-k2.6",
            "status": "request_failed",
            "error": "Connection timed out",
            "analysis": None
        }
        
        url = reverse('predict')
        data = {'review': 'This is a genuine review about a standard coffee machine. Works okay.'}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify failure graceful fallbacks
        self.assertFalse(response.data['llm']['available'])
        self.assertEqual(response.data['llm']['status'], 'request_failed')
        self.assertEqual(response.data['ai_prediction'], 'Unavailable')
        self.assertEqual(response.data['ai_confidence'], 0.0)
        self.assertEqual(response.data['prediction'], response.data['ml_prediction'])
        self.assertEqual(response.data['confidence'], response.data['ml_confidence'])
        self.assertEqual(response.data['decision_mode'], 'ml_only')
        self.assertTrue(response.data['warnings'])

    @patch('reviews.views.predict')
    @patch('reviews.views.generate_llm_review_report')
    @patch('reviews.views.analyze_platform_reviews')
    def test_hybrid_single_review_ml_failure_uses_kimi(self, mock_platform, mock_llm, mock_predict):
        mock_platform.return_value = {"available": False, "samples": [], "summary": "Bypassed"}
        mock_predict.side_effect = RuntimeError("Model unavailable")
        mock_llm.return_value = {
            "available": True,
            "provider": "nvidia",
            "model": "moonshotai/kimi-k2.6",
            "analysis": {
                "ai_risk_score": 0.8,
                "authenticity_assessment": "Likely Fake",
                "summary": "Kimi detected manipulation.",
                "severity": "High",
                "recommended_action": "Reject review",
                "evidence_strength": "Strong",
                "risk_factors": ["Manipulated wording"],
                "trust_factors": [],
                "limitations": [],
                "reasoning": "The review is promotional and lacks product detail.",
            },
        }
        response = self.client.post(reverse('predict'), {'review': 'BEST PRODUCT EVER buy now!!!'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['decision_mode'], 'kimi_only')
        self.assertEqual(response.data['engine_status']['ml']['status'], 'failed')
        self.assertEqual(response.data['engine_status']['kimi']['status'], 'completed')
        self.assertEqual(response.data['ai_prediction'], 'Likely Fake')
        self.assertTrue(response.data['warnings'])

    @patch('reviews.product_intelligence.generate_kimi_advice')
    @patch('reviews.product_intelligence.search_duckduckgo')
    @patch('reviews.product_intelligence.analyze_platform_reviews')
    def test_hybrid_product_trust_success(self, mock_platform, mock_search, mock_kimi):
        mock_search.return_value = [
            {"title": "Good coffee maker", "url": "http://coffee.com", "snippet": "Decent value for money, but takes time.", "query": "test", "platform": "coffee.com"},
            {"title": "Coffee machine reviews", "url": "http://coffee.com", "snippet": "Strong performance and easy cleaning.", "query": "test", "platform": "coffee.com"}
        ]
        mock_platform.return_value = {
            "available": True,
            "samples": [{"text": "Works well, but the lid feels fragile."}],
            "summary": "One extracted platform review.",
        }
        mock_kimi.return_value = {
            "available": True,
            "provider": "nvidia",
            "model": "moonshotai/kimi-k2.6",
            "recommendation": "Avoid for now",
            "summary": "Mock summary of why to avoid.",
            "ai_trust_score": 30,
            "authenticity_assessment": "Likely Fake",
            "trust_assessment": "Low Trust",
            "reasoning": "A lot of duplicates detected."
        }
        
        url = reverse('product-trust')
        data = {'product_name': 'Sample Coffee Machine', 'product_url': 'https://example.com/coffee'}
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify hybrid trust score calculation
        ml_score = response.data['ml_trust_score']
        ai_score = response.data['ai_trust_score']
        expected_hybrid = int(round((ml_score * 0.60) + (ai_score * 0.40)))
        self.assertEqual(response.data['hybrid_trust_score'], expected_hybrid)
        self.assertEqual(response.data['trust_score'], expected_hybrid)
        self.assertEqual(response.data['llm']['available'], True)
        self.assertEqual(response.data['decision_mode'], 'hybrid')
        self.assertEqual(mock_kimi.call_args.args[4]['samples'][0]['text'], 'Works well, but the lid feels fragile.')

    @patch('reviews.product_intelligence.generate_kimi_advice')
    @patch('reviews.product_intelligence.search_duckduckgo')
    @patch('reviews.product_intelligence.analyze_platform_reviews')
    @patch('reviews.product_intelligence.analyze_patterns')
    def test_product_url_ml_failure_uses_kimi(self, mock_patterns, mock_platform, mock_search, mock_kimi):
        mock_patterns.side_effect = RuntimeError("ML trust engine unavailable")
        mock_search.return_value = []
        mock_platform.return_value = {
            "available": True,
            "samples": [{"text": "Battery failed after one week."}],
            "summary": "One extracted review.",
        }
        mock_kimi.return_value = {
            "available": True,
            "provider": "nvidia",
            "model": "moonshotai/kimi-k2.6",
            "ai_trust_score": 25,
            "summary": "Kimi found low trust.",
            "reasoning": "Extracted review signals indicate quality risk.",
        }
        response = self.client.post(
            reverse('product-trust'),
            {'product_name': 'Sample Headphones', 'product_url': 'https://example.com/product'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['decision_mode'], 'kimi_only')
        self.assertEqual(response.data['engine_status']['ml']['status'], 'failed')
        self.assertEqual(response.data['engine_status']['kimi']['status'], 'completed')
        self.assertEqual(response.data['trust_score'], 25)
        self.assertTrue(response.data['warnings'])
