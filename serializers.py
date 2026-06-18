"""
reviews/serializers.py
======================
DRF serializers for input validation and output formatting.
"""
from rest_framework import serializers


class ReviewInputSerializer(serializers.Serializer):
    """Validates incoming POST /api/predict/ payload."""
    product_name = serializers.CharField(required=False, allow_blank=True, max_length=200)
    product_url = serializers.URLField(required=False, allow_blank=True, max_length=1000)
    review = serializers.CharField(
        min_length=1,
        max_length=5000,
        error_messages={
            'min_length': 'Review text cannot be empty.',
            'max_length': 'Review is too long (max 5000 characters).',
        }
    )


class PredictionOutputSerializer(serializers.Serializer):
    """Shapes the response returned to the frontend."""
    prediction  = serializers.CharField()
    confidence  = serializers.FloatField(min_value=0.0, max_value=1.0)
    explanation = serializers.ListField(child=serializers.CharField())
    risk_score = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)
    text_prediction = serializers.CharField(required=False)
    text_confidence = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)
    media_analysis = serializers.DictField(required=False)
    ai_analysis = serializers.DictField(required=False)
    llm = serializers.DictField(required=False)
    product_context = serializers.DictField(required=False)
    platform_review_analysis = serializers.DictField(required=False)
    # Extra analysis details
    sentiment_score  = serializers.FloatField()
    word_count       = serializers.IntegerField()
    exclamation_count = serializers.IntegerField()
    # Hybrid intelligence fields
    ml_prediction = serializers.CharField(required=False)
    ml_confidence = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)
    ai_prediction = serializers.CharField(required=False)
    ai_confidence = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)
    hybrid_confidence = serializers.FloatField(required=False, min_value=0.0, max_value=1.0)
    engine_status = serializers.DictField(required=False)
    ml_analysis = serializers.DictField(required=False)
    kimi_analysis = serializers.DictField(required=False, allow_null=True)
    warnings = serializers.ListField(child=serializers.CharField(), required=False)
    decision_mode = serializers.CharField(required=False)


class ProductTrustInputSerializer(serializers.Serializer):
    product_name = serializers.CharField(min_length=1, max_length=200)
    product_url = serializers.URLField(required=False, allow_blank=True, max_length=1000)


class ProductTrustOutputSerializer(serializers.Serializer):
    product_name = serializers.CharField()
    product_url = serializers.CharField(allow_blank=True)
    recommendation = serializers.CharField()
    trust_score = serializers.IntegerField(min_value=0, max_value=100)
    fake_review_risk = serializers.CharField()
    advisor_summary = serializers.CharField()
    review_patterns = serializers.ListField(child=serializers.CharField())
    suspicious_signals = serializers.ListField(child=serializers.CharField())
    genuine_signals = serializers.ListField(child=serializers.CharField())
    common_complaints = serializers.ListField(child=serializers.CharField())
    score_basis = serializers.ListField(child=serializers.CharField(), required=False)
    platform_review_analysis = serializers.DictField(required=False, allow_null=True)
    quality_analysis = serializers.DictField(required=False)
    trust_analysis = serializers.DictField(required=False)
    product_genuineness_report = serializers.DictField(required=False)
    sources = serializers.ListField(child=serializers.DictField())
    alternatives = serializers.ListField(child=serializers.DictField())
    llm = serializers.DictField()
    limitations = serializers.ListField(child=serializers.CharField())
    ml_trust_score = serializers.IntegerField(required=False, allow_null=True, min_value=0, max_value=100)
    ai_trust_score = serializers.IntegerField(required=False, allow_null=True, min_value=0, max_value=100)
    hybrid_trust_score = serializers.IntegerField(required=False, min_value=0, max_value=100)
    engine_status = serializers.DictField(required=False)
    ml_analysis = serializers.DictField(required=False, allow_null=True)
    kimi_analysis = serializers.DictField(required=False, allow_null=True)
    warnings = serializers.ListField(child=serializers.CharField(), required=False)
    decision_mode = serializers.CharField(required=False)
