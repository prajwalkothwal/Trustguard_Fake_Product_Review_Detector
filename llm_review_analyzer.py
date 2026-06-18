"""
LLM-backed review analysis.

The scikit-learn model remains the source of the numeric risk score. This
module asks NVIDIA-hosted Kimi to turn those signals into a concise analyst
report when an API key is configured.
"""
from __future__ import annotations

import json
import logging
import re

import requests

from .runtime_config import describe_nvidia_error, get_kimi_deployment_issue, get_kimi_runtime_config

logger = logging.getLogger('reviews.llm')


def generate_simulated_llm_review_report(review_text: str, product_name: str, product_url: str, prediction_result: dict, error_message: str = "") -> dict:
    ml_analysis = prediction_result.get("ml_analysis", prediction_result)
    ml_prediction = ml_analysis.get("prediction", "Suspicious")
    ml_risk_score = float(ml_analysis.get("fake_review_probability", 0.5))
    ml_confidence = float(ml_analysis.get("confidence", 0.5))
    sentiment_score = float(prediction_result.get("sentiment_score", 0.0))
    word_count = int(prediction_result.get("word_count", len(review_text.split())))
    exclamations = int(prediction_result.get("exclamation_count", review_text.count("!")))

    # Define dynamic properties based on the ML prediction and review text features
    if ml_prediction == "Genuine":
        ai_risk_score = max(0.0, ml_risk_score - 0.05)
        severity = "Low" if ai_risk_score < 0.2 else "Clear"
        evidence_strength = "Strong" if ml_confidence > 0.75 else "Moderate"
        authenticity_assessment = "Genuine"
        trust_assessment = "High Trust"
        recommended_action = "Approve review for public display."
        summary_desc = "The review exhibits organic variations in length and punctuation, matching typical customer feedback."
    elif ml_prediction in ["Likely Promotional", "Promotional"]:
        ai_risk_score = min(0.99, ml_risk_score + 0.05)
        severity = "Medium"
        evidence_strength = "Moderate"
        authenticity_assessment = "Likely Promotional"
        trust_assessment = "Medium Trust"
        recommended_action = "Flag for marketing-intent moderation."
        summary_desc = "Highly enthusiastic phrasing and repetitive positive tokens suggest promotional intent."
    else: # Suspicious, Likely Fake, Fake
        ai_risk_score = min(0.99, ml_risk_score + 0.03)
        severity = "High" if ai_risk_score > 0.7 else "Medium"
        evidence_strength = "Strong" if ml_confidence > 0.75 else "Moderate"
        authenticity_assessment = "Likely Fake" if ai_risk_score > 0.7 else "Suspicious"
        trust_assessment = "Low Trust"
        recommended_action = "Reject review or hold for secondary verification."
        summary_desc = "Abnormal linguistic indicators and potential text duplication suggest elevated authenticity risk."

    risk_factors = []
    trust_factors = []
    
    if word_count < 10:
        risk_factors.append("Very short review text provides little usage detail.")
    elif word_count > 100:
        trust_factors.append("Comprehensive review details suggest actual user experience.")

    if exclamations > 2:
        risk_factors.append(f"Excessive punctuation usage ({exclamations} exclamations).")
    else:
        trust_factors.append("Standard, measured punctuation style.")

    caps_count = sum(1 for c in review_text if c.isupper())
    caps_ratio = caps_count / max(1, len(review_text))
    if caps_ratio > 0.3:
        risk_factors.append(f"Unusually high capital letter ratio ({int(caps_ratio*100)}%).")

    if abs(sentiment_score) > 0.8:
        risk_factors.append(f"Extreme sentiment score ({sentiment_score:.2f}) indicates bias.")
    elif abs(sentiment_score) < 0.3:
        trust_factors.append("Balanced and objective tone.")

    lower_text = review_text.lower()
    promotional_keywords = ["best product", "must buy", "amazing", "perfect", "life changing", "recommend"]
    found_keywords = [kw for kw in promotional_keywords if kw in lower_text]
    if len(found_keywords) > 1:
        risk_factors.append(f"Contains promotional phrases: {', '.join(found_keywords[:2])}.")
    else:
        trust_factors.append("Absence of obvious promotional templates.")

    if not risk_factors:
        risk_factors.append("No critical risk signals flagged in text syntax.")
    if not trust_factors:
        trust_factors.append("Stylistic markers fall within normal thresholds.")

    quality_keywords = ["quality", "durable", "broken", "cheap", "premium", "material", "plastic", "sturdy"]
    found_quality = [q for q in quality_keywords if q in lower_text]
    if "broken" in lower_text or "cheap" in lower_text:
        quality_assessment = "Criticisms regarding product build quality and material durability."
    elif "premium" in lower_text or "sturdy" in lower_text or "quality" in lower_text:
        quality_assessment = "Positive evaluation of material sturdiness and premium feel."
    else:
        quality_assessment = "General product performance described without specific material issues."

    simulated_analysis = {
        "summary": f"Hybrid analyst review completed. {summary_desc}",
        "severity": severity,
        "recommended_action": recommended_action,
        "evidence_strength": evidence_strength,
        "risk_factors": risk_factors[:4],
        "trust_factors": trust_factors[:4],
        "limitations": [
            f"Kimi API request failed or was offline ({error_message or 'missing key'}), so local simulated LLM advice was generated to maintain hybrid pipeline.",
            "Visual verification is unavailable without review media files."
        ],
        "ai_risk_score": round(ai_risk_score, 4),
        "authenticity_assessment": authenticity_assessment,
        "trust_assessment": trust_assessment,
        "product_quality_assessment": quality_assessment,
        "reasoning": f"Local simulated analyst evaluated the ML signals and text features. The review displays a {authenticity_assessment.lower()} profile with {evidence_strength.lower()} evidence. {summary_desc}"
    }

    return {
        "available": True,
        "provider": "nvidia",
        "model": "moonshotai/kimi-k2.6 (simulated)",
        "analysis": simulated_analysis
    }


def generate_llm_review_report(review_text: str, product_name: str, product_url: str, prediction_result: dict) -> dict:
    config = get_kimi_runtime_config()
    api_key = config["api_key"]
    base_url = config["base_url"]
    model = config["model"]
    provider = config["provider"]

    if not api_key:
        logger.warning('[KIMI ADVICE] NVIDIA_API_KEY not configured — using simulated Kimi fallback')
        return generate_simulated_llm_review_report(
            review_text, product_name, product_url, prediction_result, "NVIDIA_API_KEY is not configured"
        )

    deployment_issue = get_kimi_deployment_issue(provider, model, base_url)
    if deployment_issue:
        logger.warning('[KIMI ADVICE] deployment unavailable — using simulated Kimi fallback: %s', deployment_issue)
        return generate_simulated_llm_review_report(
            review_text, product_name, product_url, prediction_result, deployment_issue
        )

    logger.info('[KIMI INITIALIZED] model=%s base_url=%s', model, base_url)

    ml_analysis = prediction_result.get("ml_analysis", prediction_result)
    platform_analysis = prediction_result.get("platform_review_analysis") or {}
    extracted_reviews = [
        sample.get("text", "")
        for sample in platform_analysis.get("samples", [])
        if sample.get("text")
    ][:10]
    prompt = {
        "task": "Evaluate the review after the ML stage. Validate or challenge its findings and return an independent structured assessment for hybrid scoring.",
        "instructions": [
            "Act as an expert review-authenticity and product-quality evaluator.",
            "Validate or challenge the ML findings. Do not merely repeat them.",
            "Detect promotional, manipulated, templated, or suspicious review patterns.",
            "Assess product quality and trustworthiness using the original and extracted reviews.",
            "Provide a final risk score (ai_risk_score) between 0.0 (genuine) and 1.0 (fake).",
            "Provide human-readable reasoning.",
            "Return strict JSON only matching the output_schema.",
        ],
        "ml_findings": ml_analysis,
        "review_text": review_text[:2500],
        "extracted_reviews": extracted_reviews,
        "media_analysis": prediction_result.get("media_analysis"),
        "trust_indicators": ml_analysis.get("trust_indicators", []),
        "suspicious_patterns": ml_analysis.get("suspicious_patterns", []),
        "output_schema": {
            "summary": "2 concise sentences",
            "severity": "Clear | Low | Medium | High",
            "recommended_action": "one short operational recommendation",
            "evidence_strength": "Limited | Moderate | Strong",
            "risk_factors": ["up to 4 short bullets"],
            "trust_factors": ["up to 4 short bullets"],
            "limitations": ["up to 3 short bullets"],
            "ai_risk_score": "float value between 0.0 and 1.0",
            "authenticity_assessment": "Genuine | Suspicious | Likely Promotional | Likely Fake",
            "trust_assessment": "High Trust | Medium Trust | Low Trust",
            "product_quality_assessment": "short quality assessment",
            "reasoning": "concise explanation validating or challenging the ML findings"
        },
    }

    try:
        logger.info('[KIMI REQUEST SENT] model=%s prompt_len=%d', model, len(json.dumps(prompt)))
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a review-risk analyst. Return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt),
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 700,
            },
            timeout=(8, 60),
        )
        response.raise_for_status()
        payload = response.json()
        logger.info('[KIMI RESPONSE RECEIVED]')
        content = extract_message_content(payload)
        generated = parse_llm_analysis(content)
        normalized = _normalize_analysis(generated)
        logger.info('[KIMI ANALYSIS COMPLETE] ai_risk_score=%.4f', normalized.get('ai_risk_score', 0.5))
        return {
            "available": True,
            "provider": provider,
            "model": model,
            "analysis": normalized,
        }
    except Exception as exc:
        err_msg = describe_nvidia_error(exc, model, base_url) if provider == "nvidia" else str(exc)[:240]
        logger.error('[KIMI REQUEST FAILED] error=%s — falling back to simulated Kimi', err_msg)
        return generate_simulated_llm_review_report(
            review_text, product_name, product_url, prediction_result, err_msg
        )


def _normalize_analysis(data: dict) -> dict:
    severity = data.get("severity") if data.get("severity") in {"Clear", "Low", "Medium", "High"} else "Medium"
    evidence = data.get("evidence_strength")
    if evidence not in {"Limited", "Moderate", "Strong"}:
        evidence = "Moderate"

    try:
        ai_risk_score = float(data.get("ai_risk_score", 0.5))
    except (ValueError, TypeError):
        ai_risk_score = 0.5

    return {
        "summary": str(data.get("summary") or "The LLM reviewed the model signals and generated an advisory assessment."),
        "severity": severity,
        "recommended_action": str(data.get("recommended_action") or "Send to a human reviewer if this decision affects publication or purchase."),
        "evidence_strength": evidence,
        "risk_factors": _string_list(data.get("risk_factors"), 4),
        "trust_factors": _string_list(data.get("trust_factors"), 4),
        "limitations": _string_list(data.get("limitations"), 3),
        "ai_risk_score": round(min(max(ai_risk_score, 0.0), 1.0), 4),
        "authenticity_assessment": str(data.get("authenticity_assessment") or "Suspicious"),
        "trust_assessment": str(data.get("trust_assessment") or "Medium Trust"),
        "product_quality_assessment": str(data.get("product_quality_assessment") or "Product quality could not be determined from this review alone."),
        "reasoning": str(data.get("reasoning") or data.get("summary") or "Kimi evaluated the ML findings."),
    }


def _string_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()][:limit]


def extract_message_content(payload: dict) -> str:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or message.get("reasoning_content") or choice.get("text") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part.strip())
    return str(content or "")


def parse_llm_analysis(content: str) -> dict:
    content = (content or "").strip()
    if not content:
        return {
            "summary": "NVIDIA Kimi returned an empty response, so the structured fallback report was used.",
            "limitations": ["The LLM response was empty for this request."],
        }

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {
        "summary": content[:700],
        "limitations": ["Kimi returned plain text instead of strict JSON, so the app used it as the analyst summary."],
    }
