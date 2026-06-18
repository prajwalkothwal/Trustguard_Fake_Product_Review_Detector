"""
reviews/authenticity_engine.py
==============================
ENGINE 1: Review Authenticity — "Is this review likely written by a genuine customer?"

Outputs one of four classifications:
  • Genuine          — Strong evidence of authentic customer experience
  • Suspicious       — Some red flags but not conclusive
  • Likely Promotional — Marketing language patterns, excessive praise, no usage evidence
  • Likely Fake      — Strong fake indicators (duplication, impossible claims, spam)

This engine does NOT rely primarily on sentiment. Positive reviews are not
automatically fake. Negative reviews are not automatically genuine.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Optional

from .feature_extraction import (
    preprocess_text,
    extract_features_single,
    get_sentiment,
    compute_specificity_score,
    compute_usage_evidence_score,
    compute_balance_score,
    compute_marketing_score,
    compute_naturalness_score,
    compute_repetition_score,
    HYPE_WORDS,
)

# ── Model path ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'ml', 'model.pkl'
)

_bundle = None


def _load_model():
    global _bundle
    if _bundle is None:
        path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            _bundle = pickle.load(f)
    return _bundle


# ── Classification thresholds ──────────────────────────────────────────────
#
# The authenticity_score ranges from 0.0 (definitely genuine) to 1.0 (definitely fake).
# These thresholds are calibrated so that:
#   - Genuine reviews from the training set score < 0.30
#   - Obvious fakes score > 0.65
#   - The ambiguous middle (0.30-0.65) produces Suspicious or Likely Promotional
#
THRESHOLD_GENUINE = 0.30
THRESHOLD_SUSPICIOUS = 0.50
THRESHOLD_PROMOTIONAL = 0.65


def classify_review(review_text: str, product_name: str = "") -> dict:
    """
    Analyse a single review and return a structured authenticity result.

    Returns:
        {
            classification: str,       # Genuine | Suspicious | Likely Promotional | Likely Fake
            confidence: float,         # 0.0 to 1.0
            authenticity_score: float,  # 0.0 (genuine) to 1.0 (fake)
            positive_factors: list,
            negative_factors: list,
            signal_breakdown: dict,
            explanation: str,
            ml_prediction: dict,       # Raw ML model output
        }
    """
    # ── 1. Run ML model ────────────────────────────────────────────────
    ml_result = _run_ml_model(review_text)

    # ── 2. Compute heuristic signals ───────────────────────────────────
    specificity_score, specificity_evidence = compute_specificity_score(review_text)
    usage_score, usage_evidence = compute_usage_evidence_score(review_text)
    balance_score, balance_evidence = compute_balance_score(review_text)
    marketing_score, marketing_evidence = compute_marketing_score(review_text)
    naturalness_score, naturalness_evidence = compute_naturalness_score(review_text)
    repetition_score, repetition_evidence = compute_repetition_score(review_text)

    # ── 3. Combine into authenticity score ─────────────────────────────
    # ML fake probability (0 = genuine, 1 = fake)
    ml_fake_prob = ml_result['fake_probability']

    # Heuristic signals (inverted where needed so higher = more suspicious)
    signals = {
        'ml_model': ml_fake_prob,
        'specificity': 1.0 - specificity_score,      # Low specificity = suspicious
        'usage_evidence': 1.0 - usage_score,          # No usage = suspicious
        'balance': 1.0 - balance_score,               # Unbalanced = suspicious
        'marketing_language': marketing_score,         # High marketing = suspicious
        'naturalness': 1.0 - naturalness_score,        # Unnatural = suspicious
        'repetition': repetition_score,                # High repetition = suspicious
    }

    # Weighted combination
    weights = {
        'ml_model': 0.25,
        'specificity': 0.15,
        'usage_evidence': 0.15,
        'balance': 0.10,
        'marketing_language': 0.12,
        'naturalness': 0.13,
        'repetition': 0.10,
    }

    authenticity_score = sum(signals[k] * weights[k] for k in signals)
    authenticity_score = round(min(max(authenticity_score, 0.0), 1.0), 4)

    # ── 4. Determine classification ────────────────────────────────────
    classification = _classify(authenticity_score, marketing_score, signals)

    # ── 5. Build positive / negative factors ───────────────────────────
    positive_factors = []
    negative_factors = []

    # Specificity
    if specificity_score > 0.4:
        positive_factors.extend(specificity_evidence[:2])
    elif specificity_score < 0.15:
        negative_factors.extend(specificity_evidence[:2])

    # Usage evidence
    if usage_score > 0.4:
        positive_factors.extend(usage_evidence[:2])
    elif usage_score < 0.15:
        negative_factors.extend(usage_evidence[:2])

    # Balance
    if balance_score > 0.5:
        positive_factors.extend(balance_evidence[:2])
    elif balance_score < 0.2:
        negative_factors.extend(balance_evidence[:2])

    # Marketing
    if marketing_score > 0.4:
        negative_factors.extend(marketing_evidence[:2])
    elif marketing_score < 0.1:
        positive_factors.append("No significant promotional language detected")

    # Naturalness
    if naturalness_score > 0.6:
        positive_factors.extend([e for e in naturalness_evidence if 'natural' in e.lower() or 'diversity' in e.lower()][:1])
    if naturalness_score < 0.3:
        negative_factors.extend(naturalness_evidence[:2])

    # Repetition
    if repetition_score > 0.3:
        negative_factors.extend(repetition_evidence[:2])

    # ML model signal
    if ml_result['prediction'] == 'Fake' and ml_result['confidence'] > 0.7:
        negative_factors.append(
            f"ML model detected fake patterns (confidence: {ml_result['confidence']*100:.0f}%)"
        )
    elif ml_result['prediction'] == 'Genuine' and ml_result['confidence'] > 0.7:
        positive_factors.append(
            f"ML model found genuine patterns (confidence: {ml_result['confidence']*100:.0f}%)"
        )

    # Ensure at least one factor in each list
    if not positive_factors:
        positive_factors.append("No strong authenticity signals found")
    if not negative_factors:
        negative_factors.append("No significant red flags detected")

    # Deduplicate
    positive_factors = list(dict.fromkeys(positive_factors))[:5]
    negative_factors = list(dict.fromkeys(negative_factors))[:5]

    # ── 6. Build explanation ───────────────────────────────────────────
    confidence = _compute_confidence(authenticity_score)
    explanation = _build_explanation(classification, confidence, positive_factors, negative_factors)

    return {
        'classification': classification,
        'confidence': round(confidence, 4),
        'authenticity_score': authenticity_score,
        'positive_factors': positive_factors,
        'negative_factors': negative_factors,
        'signal_breakdown': {
            'ml_model': {'score': round(ml_fake_prob, 4), 'weight': weights['ml_model']},
            'specificity': {'score': round(specificity_score, 4), 'weight': weights['specificity']},
            'usage_evidence': {'score': round(usage_score, 4), 'weight': weights['usage_evidence']},
            'balance': {'score': round(balance_score, 4), 'weight': weights['balance']},
            'marketing_language': {'score': round(marketing_score, 4), 'weight': weights['marketing_language']},
            'naturalness': {'score': round(naturalness_score, 4), 'weight': weights['naturalness']},
            'repetition': {'score': round(repetition_score, 4), 'weight': weights['repetition']},
        },
        'explanation': explanation,
        'ml_prediction': ml_result,
        'sentiment_score': round(get_sentiment(review_text), 4),
        'word_count': len(review_text.split()),
        'exclamation_count': review_text.count('!'),
    }


def classify_reviews_batch(review_texts: list[str], product_name: str = "") -> list[dict]:
    """Classify a batch of reviews. More efficient than calling classify_review() individually."""
    return [classify_review(text, product_name) for text in review_texts]


# ── Internal helpers ───────────────────────────────────────────────────────

def _run_ml_model(text: str) -> dict:
    """Run the ML model and return prediction with probabilities."""
    bundle = _load_model()
    if bundle is None:
        return {
            'prediction': 'Unknown',
            'confidence': 0.5,
            'fake_probability': 0.5,
            'available': False,
        }

    clf = bundle['classifier']
    tfidf = bundle['tfidf']
    labels = bundle['labels']

    processed = preprocess_text(text)
    tfidf_vec = tfidf.transform([processed]).toarray()
    feat_vec = extract_features_single(text)
    X = np.hstack([tfidf_vec, feat_vec])

    pred_idx = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    confidence = float(proba[pred_idx])
    prediction = labels[pred_idx]

    fake_idx = labels.index('Fake') if 'Fake' in labels else 1
    fake_probability = float(proba[fake_idx])

    return {
        'prediction': prediction,
        'confidence': round(confidence, 4),
        'fake_probability': round(fake_probability, 4),
        'available': True,
        'dataset_size': bundle.get('dataset_size', 'unknown'),
    }


def _classify(authenticity_score: float, marketing_score: float, signals: dict) -> str:
    """Determine the 4-tier classification."""
    if authenticity_score < THRESHOLD_GENUINE:
        return 'Genuine'

    if authenticity_score >= THRESHOLD_PROMOTIONAL:
        # Distinguish between Likely Promotional and Likely Fake
        if marketing_score > 0.5:
            return 'Likely Promotional'
        return 'Likely Fake'

    if authenticity_score >= THRESHOLD_SUSPICIOUS:
        if marketing_score > 0.35:
            return 'Likely Promotional'
        return 'Suspicious'

    # Between GENUINE and SUSPICIOUS thresholds — lean toward Suspicious
    # if multiple signals are elevated
    elevated_signals = sum(1 for v in signals.values() if v > 0.5)
    if elevated_signals >= 3:
        return 'Suspicious'

    return 'Genuine'


def _compute_confidence(authenticity_score: float) -> float:
    """
    Confidence is highest when the score is clearly near 0 or 1,
    and lowest in the ambiguous middle zone.
    """
    distance_from_center = abs(authenticity_score - 0.5) * 2  # 0-1
    return 0.50 + (distance_from_center * 0.45)  # 0.50-0.95


def _build_explanation(classification: str, confidence: float,
                       positive_factors: list, negative_factors: list) -> str:
    """Build a concise explanation sentence."""
    conf_pct = round(confidence * 100)
    if classification == 'Genuine':
        return (
            f"This review appears genuine with {conf_pct}% confidence. "
            f"Key indicators: {positive_factors[0].lower() if positive_factors else 'natural writing patterns'}."
        )
    elif classification == 'Suspicious':
        return (
            f"This review has suspicious indicators ({conf_pct}% confidence). "
            f"Concerns: {negative_factors[0].lower() if negative_factors else 'mixed signals detected'}."
        )
    elif classification == 'Likely Promotional':
        return (
            f"This review appears to be promotional content ({conf_pct}% confidence). "
            f"Concerns: {negative_factors[0].lower() if negative_factors else 'marketing language detected'}."
        )
    else:  # Likely Fake
        return (
            f"This review is likely fake or fabricated ({conf_pct}% confidence). "
            f"Concerns: {negative_factors[0].lower() if negative_factors else 'multiple fake indicators detected'}."
        )
