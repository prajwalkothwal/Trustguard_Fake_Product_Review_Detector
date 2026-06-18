import logging
import re
import time
from urllib.parse import quote_plus

from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .hybrid_decision import build_review_ml_analysis, failed_review_ml_analysis, merge_review_analyses
from .llm_review_analyzer import generate_llm_review_report
from .media_analyzer import analyze_media
from .platform_review_analyzer import analyze_platform_reviews
from .predictor import predict
from .product_intelligence import analyze_product
from .serializers import (
    PredictionOutputSerializer,
    ProductTrustInputSerializer,
    ProductTrustOutputSerializer,
    ReviewInputSerializer,
)

logger = logging.getLogger("reviews.pipeline")


def clean_audit_review_text(value):
    value = re.sub(r"\bRead more\b", " ", value or "", flags=re.IGNORECASE)
    value = re.sub(r"\bBrief content visible, double tap to read full content\.?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bFull content visible, double tap to read brief content\.?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r'\s*"\s*"\s*', " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \"'")


class PredictView(APIView):
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def post(self, request):
        input_ser = ReviewInputSerializer(data=request.data)
        if not input_ser.is_valid():
            return Response({"error": input_ser.errors}, status=status.HTTP_400_BAD_REQUEST)
        review_text = input_ser.validated_data["review"]
        product_name = input_ser.validated_data.get("product_name", "").strip()
        product_url = input_ser.validated_data.get("product_url", "").strip()
        media_files = request.FILES.getlist("media")

        logger.info("[MEDIA ANALYSIS STARTED] files=%d", len(media_files))
        media_result = analyze_media(media_files)
        logger.info("[MEDIA ANALYSIS COMPLETE] provided=%s", media_result.get("provided", False))

        try:
            logger.info("[ML ANALYSIS STARTED] review_len=%d product=%s", len(review_text), product_name or "(none)")
            started = time.monotonic()
            result = predict(review_text, media_result, product_name=product_name)
            ml_analysis = build_review_ml_analysis(result)
            logger.info(
                "[ML ANALYSIS COMPLETE] prediction=%s confidence=%.4f risk_score=%.4f elapsed_ms=%d",
                result.get("prediction"), result.get("confidence", 0), result.get("risk_score", 0),
                round((time.monotonic() - started) * 1000),
            )
        except Exception as exc:
            logger.exception("[ML ANALYSIS FAILED] error=%s", exc)
            ml_analysis = failed_review_ml_analysis(exc)
            result = {
                "prediction": "Suspicious",
                "confidence": 0.5,
                "risk_score": 0.5,
                "explanation": ["ML analysis failed. Kimi analysis was still requested."],
                "sentiment_score": 0.0,
                "word_count": len(review_text.split()),
                "exclamation_count": review_text.count("!"),
                "media_analysis": media_result,
                "signal_breakdown": {},
            }

        result["ml_analysis"] = ml_analysis
        result["product_context"] = build_product_context(product_name, product_url)
        logger.info("[PLATFORM ANALYSIS STARTED] url=%s", product_url or "(none)")
        try:
            result["platform_review_analysis"] = analyze_platform_reviews(product_url)
        except Exception as exc:
            logger.warning("[PLATFORM ANALYSIS FAILED] error=%s", str(exc)[:200])
            result["platform_review_analysis"] = {"available": False, "samples": [], "summary": "Platform review extraction failed."}
        logger.info("[PLATFORM ANALYSIS COMPLETE] available=%s", result["platform_review_analysis"].get("available", False))

        llm_report = generate_llm_review_report(review_text, product_name, product_url, result)
        merged = merge_review_analyses(result if ml_analysis["available"] else None, ml_analysis, llm_report)
        result.update(merged)
        result["decision_mode"] = merged["mode"]
        result["engine_status"] = {
            "ml": {"status": ml_analysis["status"], "available": ml_analysis["available"], "error": ml_analysis.get("error")},
            "kimi": {"status": "completed" if llm_report.get("available") else "failed", "available": bool(llm_report.get("available")), "error": llm_report.get("error")},
        }
        result["llm"] = {
            "provider": llm_report.get("provider", "nvidia"),
            "model": llm_report.get("model", "moonshotai/kimi-k2.6"),
            "available": bool(llm_report.get("available")),
            "status": llm_report.get("status", "completed" if llm_report.get("available") else "unavailable"),
        }
        if llm_report.get("error"):
            result["llm"]["message"] = llm_report["error"]
        if merged["kimi_analysis"]:
            result["ai_analysis"] = {**merged["kimi_analysis"], "generated_by": "kimi"}
        else:
            ml_report = result.get("ai_analysis", {})
            result["ai_analysis"] = {
                **ml_report,
                "summary": ml_report.get("summary", "Kimi analysis was unavailable."),
                "generated_by": "ml_only",
                "limitations": list(ml_report.get("limitations", [])) + merged["warnings"],
            }
        result["ai_analysis"]["alternatives"] = result["product_context"]["alternatives"]
        if product_name and result["ai_analysis"].get("summary"):
            summary = result["ai_analysis"]["summary"]
            result["ai_analysis"]["summary"] = f"For {product_name}, {summary[0].lower()}{summary[1:]}"

        output_ser = PredictionOutputSerializer(data=result)
        output_ser.is_valid(raise_exception=True)
        return Response(output_ser.validated_data, status=status.HTTP_200_OK)


class ProductTrustView(APIView):
    parser_classes = [JSONParser, FormParser]

    def post(self, request):
        input_ser = ProductTrustInputSerializer(data=request.data)
        if not input_ser.is_valid():
            return Response({"error": input_ser.errors}, status=status.HTTP_400_BAD_REQUEST)
        product_name = input_ser.validated_data["product_name"]
        product_url = input_ser.validated_data.get("product_url", "")
        try:
            logger.info("[PRODUCT TRUST ANALYSIS STARTED] product=%s url=%s", product_name, product_url or "(none)")
            started = time.monotonic()
            result = analyze_product(product_name, product_url)
            logger.info(
                "[PRODUCT TRUST ANALYSIS COMPLETE] recommendation=%s trust_score=%s elapsed_ms=%d",
                result.get("recommendation"), result.get("trust_score"), round((time.monotonic() - started) * 1000),
            )
        except Exception as exc:
            logger.exception("[PRODUCT TRUST ANALYSIS FAILED] product=%s error=%s", product_name, exc)
            return Response({"error": f"Product analysis failed: {str(exc)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        output_ser = ProductTrustOutputSerializer(data=result)
        output_ser.is_valid(raise_exception=True)
        return Response(output_ser.validated_data, status=status.HTTP_200_OK)


def build_product_context(product_name, product_url):
    query = quote_plus(product_name or "similar product")
    alternatives_query = quote_plus(f"best alternatives to {product_name}" if product_name else "best product alternatives")
    review_query = quote_plus(f"{product_name} reviews complaints" if product_name else "product reviews complaints")
    return {
        "product_name": product_name,
        "product_url": product_url,
        "comparison_links": [
            {"label": "Google product reviews", "url": f"https://www.google.com/search?q={review_query}"},
            {"label": "Amazon search", "url": f"https://www.amazon.in/s?k={query}"},
            {"label": "Flipkart search", "url": f"https://www.flipkart.com/search?q={query}"},
        ],
        "alternatives": [
            {"name": f"Compare alternatives to {product_name}" if product_name else "Compare product alternatives", "reason": "Compare prices, ratings, and complaints before trusting one review.", "url": f"https://www.google.com/search?q={alternatives_query}"},
            {"name": "Amazon similar options", "reason": "Check competing products and verified buyer review patterns.", "url": f"https://www.amazon.in/s?k={alternatives_query}"},
            {"name": "Flipkart similar options", "reason": "Compare ratings, recent reviews, and seller consistency.", "url": f"https://www.flipkart.com/search?q={alternatives_query}"},
        ],
    }


class AnalyticsView(APIView):
    def get(self, request):
        from .models import CheckedReview
        recent = CheckedReview.objects.all().order_by("-created_at")[:100]
        return Response({
            "total_count": CheckedReview.objects.count(),
            "fake_count": CheckedReview.objects.filter(prediction="Fake").count(),
            "genuine_count": CheckedReview.objects.filter(prediction="Genuine").count(),
            "recent_reviews": [{
                "id": item.id, "review_text": clean_audit_review_text(item.review_text),
                "prediction": item.prediction, "confidence": round(item.confidence, 4),
                "media_prediction": item.media_prediction, "media_confidence": round(item.media_confidence, 4),
                "media_summary": item.media_summary, "product_name": item.product_name,
                "created_at": item.created_at.isoformat(),
            } for item in recent],
        }, status=status.HTTP_200_OK)

    def delete(self, request):
        from .models import CheckedReview
        CheckedReview.objects.all().delete()
        return Response({"message": "Analytics history cleared successfully."}, status=status.HTTP_200_OK)
