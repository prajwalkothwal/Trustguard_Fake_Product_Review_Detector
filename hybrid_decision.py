"""Shared scoring helpers for the ML + Kimi hybrid analysis pipeline."""
from __future__ import annotations

import logging

logger = logging.getLogger("reviews.pipeline")

ML_WEIGHT = 0.60
KIMI_WEIGHT = 0.40


def score_confidence(score: float) -> float:
    """Return confidence from distance to the ambiguous midpoint."""
    return round(0.50 + (abs(float(score) - 0.5) * 2 * 0.45), 4)


def classify_review_risk(risk_score: float, marketing_score: float = 0.0, kimi_prediction: str = "") -> str:
    if risk_score < 0.30:
        return "Genuine"
    if risk_score >= 0.65:
        return "Likely Promotional" if marketing_score > 0.5 or kimi_prediction == "Likely Promotional" else "Likely Fake"
    if risk_score >= 0.50:
        return "Likely Promotional" if marketing_score > 0.35 or kimi_prediction == "Likely Promotional" else "Suspicious"
    return "Suspicious" if risk_score >= 0.40 else "Genuine"


def build_review_ml_analysis(result: dict) -> dict:
    risk_score = float(result.get("risk_score", result.get("authenticity_score", 0.5)))
    sentiment_score = float(result.get("sentiment_score", 0.0))
    signal_breakdown = result.get("signal_breakdown", {})
    suspicious_patterns = list(result.get("negative_factors", []))
    trust_indicators = list(result.get("positive_factors", []))
    return {
        "status": "completed",
        "available": True,
        "prediction": result.get("prediction", "Suspicious"),
        "authenticity_score": round(1.0 - risk_score, 4),
        "fake_review_probability": round(risk_score, 4),
        "trust_score": round((1.0 - risk_score) * 100, 2),
        "confidence": float(result.get("confidence", score_confidence(risk_score))),
        "sentiment_breakdown": {
            "compound": sentiment_score,
            "indicator": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral",
        },
        "suspicious_patterns": suspicious_patterns,
        "trust_indicators": trust_indicators,
        "signal_breakdown": signal_breakdown,
        "reasoning": " ".join(suspicious_patterns[:3] + trust_indicators[:2]),
    }


def failed_review_ml_analysis(error: Exception) -> dict:
    return {
        "status": "failed",
        "available": False,
        "prediction": "Unavailable",
        "authenticity_score": None,
        "fake_review_probability": None,
        "trust_score": None,
        "confidence": 0.0,
        "sentiment_breakdown": {},
        "suspicious_patterns": [],
        "trust_indicators": [],
        "signal_breakdown": {},
        "reasoning": "ML analysis failed before a score could be generated.",
        "error": str(error)[:240],
    }


def merge_review_analyses(ml_result: dict | None, ml_analysis: dict, kimi_report: dict) -> dict:
    kimi_available = bool(kimi_report.get("available") and kimi_report.get("analysis"))
    ml_available = bool(ml_analysis.get("available"))
    warnings = []

    if kimi_available:
        kimi_analysis = dict(kimi_report["analysis"])
        kimi_risk_score = float(kimi_analysis.get("ai_risk_score", 0.5))
        kimi_confidence = score_confidence(kimi_risk_score)
        kimi_prediction = kimi_analysis.get("authenticity_assessment", "Suspicious")
    else:
        kimi_analysis = None
        kimi_risk_score = None
        kimi_confidence = 0.0
        kimi_prediction = "Unavailable"
        warnings.append("Kimi analysis failed or is unavailable. Displaying an ML-only recommendation.")

    if ml_available:
        ml_risk_score = float(ml_analysis["fake_review_probability"])
        ml_confidence = float(ml_analysis["confidence"])
        ml_prediction = ml_analysis["prediction"]
    else:
        ml_risk_score = None
        ml_confidence = 0.0
        ml_prediction = "Unavailable"
        warnings.append("ML analysis failed. Displaying a Kimi-only recommendation.")

    if ml_available and kimi_available:
        risk_score = round((ml_risk_score * ML_WEIGHT) + (kimi_risk_score * KIMI_WEIGHT), 4)
        confidence = score_confidence(risk_score)
        mode = "hybrid"
    elif ml_available:
        risk_score = round(ml_risk_score, 4)
        confidence = ml_confidence
        mode = "ml_only"
    elif kimi_available:
        risk_score = round(kimi_risk_score, 4)
        confidence = kimi_confidence
        mode = "kimi_only"
    else:
        risk_score = 0.5
        confidence = 0.5
        mode = "unavailable"
        warnings.append("Neither analysis engine completed. Manual review is required.")

    marketing_score = (ml_result or {}).get("signal_breakdown", {}).get("marketing_language", {}).get("score", 0.0)
    prediction = classify_review_risk(risk_score, marketing_score, kimi_prediction)
    logger.info(
        "[HYBRID SCORE GENERATED] mode=%s ml_risk=%s kimi_risk=%s final_risk=%.4f prediction=%s",
        mode, ml_risk_score, kimi_risk_score, risk_score, prediction,
    )
    return {
        "mode": mode,
        "prediction": prediction,
        "risk_score": risk_score,
        "confidence": confidence,
        "hybrid_confidence": confidence,
        "ml_prediction": ml_prediction,
        "ml_confidence": ml_confidence,
        "ai_prediction": kimi_prediction,
        "ai_confidence": kimi_confidence,
        "ml_analysis": ml_analysis,
        "kimi_analysis": kimi_analysis,
        "warnings": warnings,
    }


def merge_product_scores(ml_trust_score: int | None, kimi_report: dict) -> dict:
    kimi_available = bool(kimi_report.get("available") and kimi_report.get("ai_trust_score") is not None)
    ml_available = ml_trust_score is not None
    warnings = []
    ai_trust_score = int(kimi_report.get("ai_trust_score", 0)) if kimi_available else None

    if ml_available and kimi_available:
        hybrid_score = int(round((ml_trust_score * ML_WEIGHT) + (ai_trust_score * KIMI_WEIGHT)))
        mode = "hybrid"
    elif ml_available:
        hybrid_score = int(ml_trust_score)
        mode = "ml_only"
        warnings.append("Kimi analysis failed or is unavailable. Displaying an ML-only recommendation.")
    elif kimi_available:
        hybrid_score = ai_trust_score
        mode = "kimi_only"
        warnings.append("ML analysis failed. Displaying a Kimi-only recommendation.")
    else:
        hybrid_score = 50
        mode = "unavailable"
        warnings.append("Neither analysis engine completed. Manual review is required.")

    logger.info(
        "[HYBRID SCORE GENERATED] mode=%s ml_trust=%s kimi_trust=%s final_trust=%d",
        mode, ml_trust_score, ai_trust_score, hybrid_score,
    )
    return {
        "mode": mode,
        "ml_trust_score": ml_trust_score,
        "ai_trust_score": ai_trust_score,
        "hybrid_trust_score": hybrid_score,
        "warnings": warnings,
    }
