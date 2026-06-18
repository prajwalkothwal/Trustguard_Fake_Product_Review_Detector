"""
reviews/predictor.py
====================
Loads the trained model and exposes a single predict(review_text) function.
This keeps all ML logic separate from the Django view.
Uses the unified feature extraction and authenticity engine.
"""

from __future__ import annotations
import os
import re
from typing import Optional

from .authenticity_engine import classify_review, _load_model
from .feature_extraction import get_sentiment

def _fake_probability(prediction: str, confidence: float) -> float:
    return confidence if prediction in ['Fake', 'Likely Fake', 'Likely Promotional'] else 1.0 - confidence

def _build_ai_analysis(result: dict) -> dict:
    risk_score = result.get('risk_score', result.get('authenticity_score', 0.5))
    media = result.get('media_analysis') or {}
    media_provided = media.get('provided', False)
    media_score = media.get('score', 0.0)
    sentiment = result.get('sentiment_score', 0.0)
    word_count = result.get('word_count', 0)
    exclamation_count = result.get('exclamation_count', 0)

    if risk_score >= 0.75:
        severity = 'High'
        recommended_action = 'Escalate for manual review before publishing.'
    elif risk_score >= 0.5:
        severity = 'Medium'
        recommended_action = 'Hold for reviewer confirmation.'
    elif risk_score >= 0.3:
        severity = 'Low'
        recommended_action = 'Allow with light monitoring.'
    else:
        severity = 'Clear'
        recommended_action = 'No action required unless other policy signals appear.'

    risk_factors = []
    trust_factors = []

    text_pred = result.get('text_prediction', 'Genuine')
    if text_pred in ['Fake', 'Likely Fake', 'Likely Promotional', 'Suspicious']:
        risk_factors.append('Text model found linguistic patterns associated with fake reviews.')
    else:
        trust_factors.append('Text model found patterns closer to genuine reviews.')

    if media_provided:
        if media.get('prediction') == 'Suspicious':
            risk_factors.append('Media model found suspicious image/video authenticity signals.')
        elif media.get('prediction') == 'Needs Review':
            risk_factors.append('Media model found weak signals that need human review.')
        else:
            trust_factors.append('Attached media supports the review context.')
    else:
        risk_factors.append('No media evidence was provided, so the decision relies on text only.')

    if abs(sentiment) > 0.7:
        risk_factors.append('Sentiment is unusually intense for ordinary product feedback.')
    elif -0.55 < sentiment < 0.55:
        trust_factors.append('Sentiment is balanced rather than overly promotional or hostile.')

    if exclamation_count > 3:
        risk_factors.append('High exclamation usage suggests exaggerated emphasis.')
    elif exclamation_count <= 1:
        trust_factors.append('Punctuation is measured and consistent with natural writing.')

    if word_count < 8:
        risk_factors.append('Review is very short and has limited product-specific detail.')
    elif 10 <= word_count <= 120:
        trust_factors.append('Review length is within a normal range for product feedback.')

    if media_provided and media_score >= 0.5 and text_pred in ['Fake', 'Likely Fake', 'Likely Promotional', 'Suspicious']:
        evidence_strength = 'Strong'
    elif risk_score >= 0.5 or media_score >= 0.35:
        evidence_strength = 'Moderate'
    else:
        evidence_strength = 'Limited'

    summary = (
        f"The AI analysis classifies this case as {result['prediction']} with "
        f"{round(result['confidence'] * 100)}% confidence. Severity is {severity.lower()} "
        f"because the combined risk score is {risk_score:.2f}."
    )

    if not media_provided:
        summary += ' Media was not attached, so the final judgment is primarily text-driven.'

    limitations = [
        'This is an ML-assisted signal, not proof of fraud.',
        'Small or biased training datasets can cause false positives and false negatives.',
    ]
    training_sources = {
        file.get('details', {}).get('model', {}).get('training_source')
        for file in media.get('files', [])
    }
    if 'synthetic_baseline' in training_sources:
        limitations.append('Media model is baseline-trained; use custom real/fake media data for stronger reliability.')

    return {
        'summary': summary,
        'severity': severity,
        'recommended_action': recommended_action,
        'evidence_strength': evidence_strength,
        'risk_factors': risk_factors[:5],
        'trust_factors': trust_factors[:5],
        'limitations': limitations,
    }

def _combine_with_media(text_result: dict, media_result: dict) -> dict:
    text_fake_score = text_result['authenticity_score']
    media_score = media_result.get('score', 0.0)

    if media_result.get('provided'):
        risk_score = min(1.0, (text_fake_score * 0.72) + (media_score * 0.28))
        if media_result.get('prediction') == 'Suspicious' and text_fake_score > 0.42:
            risk_score = min(1.0, risk_score + 0.1)
    else:
        risk_score = text_fake_score

    # Reclassify combined result into 4-tier
    marketing_score = text_result.get('signal_breakdown', {}).get('marketing_language', {}).get('score', 0.0)
    
    if risk_score < 0.30:
        prediction = 'Genuine'
    elif risk_score >= 0.65:
        prediction = 'Likely Promotional' if marketing_score > 0.5 else 'Likely Fake'
    elif risk_score >= 0.50:
        prediction = 'Likely Promotional' if marketing_score > 0.35 else 'Suspicious'
    else:
        prediction = 'Suspicious' if risk_score >= 0.40 else 'Genuine'

    distance_from_center = abs(risk_score - 0.5) * 2  # 0-1
    confidence = 0.50 + (distance_from_center * 0.45)  # 0.50-0.95

    explanation = list(text_result['positive_factors'] + text_result['negative_factors'])
    if media_result.get('provided'):
        training_sources = {
            file.get('details', {}).get('model', {}).get('training_source')
            for file in media_result.get('files', [])
        }
        uses_baseline_media_model = 'synthetic_baseline' in training_sources

        if uses_baseline_media_model:
            explanation.append('Attached media was checked with a baseline media model; train it with your own real/fake examples before treating media signals as decisive.')
        elif media_result.get('prediction') == 'Suspicious':
            explanation.append('Attached media contains suspicious format, metadata, or quality signals.')
        elif media_result.get('prediction') == 'Needs Review':
            explanation.append('Attached media has minor authenticity signals that should be manually reviewed.')
        else:
            explanation.append('Attached media passed basic authenticity checks and supports the review context.')

    combined = {
        **text_result,
        'prediction': prediction,
        'confidence': round(confidence, 4),
        'risk_score': round(risk_score, 4),
        'text_prediction': text_result['classification'],
        'text_confidence': text_result['confidence'],
        'media_analysis': media_result,
        'explanation': explanation,
    }
    combined['ai_analysis'] = _build_ai_analysis(combined)
    return combined

def _audit_media_status(media_result: dict) -> tuple[str, float, str]:
    if not media_result or not media_result.get('provided'):
        return 'No image', 0.0, 'No review image was attached.'

    prediction = media_result.get('prediction', 'Needs Review')
    if prediction == 'Looks Authentic':
        label = 'Real'
    elif prediction == 'Suspicious':
        label = 'Fake'
    elif prediction == 'Needs Review':
        label = 'Needs review'
    else:
        label = prediction

    return (
        label,
        float(media_result.get('confidence', 0.0) or 0.0),
        str(media_result.get('summary', ''))[:300],
    )

def _clean_audit_review_text(value: str) -> str:
    value = re.sub(r'\bRead more\b', ' ', value or '', flags=re.IGNORECASE)
    value = re.sub(r'\bBrief content visible, double tap to read full content\.?', ' ', value, flags=re.IGNORECASE)
    value = re.sub(r'\bFull content visible, double tap to read brief content\.?', ' ', value, flags=re.IGNORECASE)
    value = re.sub(r'\s*"\s*"\s*', ' ', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip(' "\'')

def predict(review_text: str, media_result: Optional[dict] = None, product_name: str = "") -> dict:
    """
    Analyse a review and return a prediction dictionary.
    """
    # 1. Run authenticity engine classification
    text_result = classify_review(review_text, product_name)

    if media_result is not None:
        final_result = _combine_with_media(text_result, media_result)
    else:
        text_result['prediction'] = text_result['classification']
        text_result['risk_score'] = text_result['authenticity_score']
        text_result['text_prediction'] = text_result['classification']
        text_result['text_confidence'] = text_result['confidence']
        text_result['media_analysis'] = {
            'provided': False,
            'prediction': 'Not Provided',
            'confidence': 0.0,
            'score': 0.0,
            'summary': 'No images or videos were attached.',
            'evidence': [],
            'files': [],
        }
        # Build list explanation for output compatibility
        text_result['explanation'] = text_result['positive_factors'] + text_result['negative_factors']
        text_result['ai_analysis'] = _build_ai_analysis(text_result)
        final_result = text_result

    # 2. Save to database
    try:
        from .models import CheckedReview
        media_prediction, media_confidence, media_summary = _audit_media_status(final_result.get('media_analysis'))
        CheckedReview.objects.create(
            review_text=_clean_audit_review_text(review_text),
            prediction=final_result['prediction'],
            confidence=final_result['confidence'],
            media_prediction=media_prediction,
            media_confidence=round(media_confidence, 4),
            media_summary=media_summary,
            product_name=product_name
        )
    except Exception as e:
        print(f"Error saving checked review: {e}")

    return final_result
