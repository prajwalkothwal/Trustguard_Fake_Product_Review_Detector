"""
reviews/quality_engine.py
=========================
ENGINE 2: Product Quality — "Is this product actually good based on reviews?"

Analyzes review text regardless of authenticity to extract:
  • Complaint categories & frequency
  • Product strengths & features praised
  • Defect concentrations (e.g., >30% mention battery issues)
  • Severe quality issues (e.g., "stopped working", "refund")
  • An overall quality rating: Excellent | Good | Average | Poor
"""

from __future__ import annotations
import re
from typing import TypedDict, List, Dict
from .feature_extraction import get_sentiment

# ── Negation-aware matching ────────────────────────────────────────────────
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
    'nor', 'cannot', "can't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", 'barely', 'hardly',
    'scarcely', 'lack', 'lacking', 'lacks', 'without', 'zero',
    # Negative adjectives that negate a following positive word
    'poor', 'bad', 'terrible', 'awful', 'horrible', 'worst',
    'pathetic', 'disappointing', 'inferior', 'subpar', 'mediocre',
    'fake', 'cheap',
}

NEGATION_WINDOW = 4  # how many words before a keyword to check for negation


def _is_negated(text_lower: str, keyword: str) -> bool:
    """Check if a keyword appears in a negated context within the text."""
    idx = text_lower.find(keyword)
    while idx != -1:
        # Extract words in a window before the keyword
        prefix = text_lower[max(0, idx - 80):idx]
        prefix_words = prefix.split()
        window = prefix_words[-NEGATION_WINDOW:] if prefix_words else []
        if any(neg in window for neg in NEGATION_WORDS):
            # This occurrence is negated — check if there's a non-negated one
            idx = text_lower.find(keyword, idx + len(keyword))
            continue
        # Found a non-negated occurrence
        return False
        # next occurrence
        idx = text_lower.find(keyword, idx + len(keyword))
    # All occurrences are negated (or keyword not found)
    return True


# Define categories and keyword matches
COMPLAINT_CATEGORIES = {
    "battery_power": ["heating", "dies", "drain", "heat", "hot", "overheating", "discharges", "low battery", "battery issue", "battery problem", "charges slowly", "doesn't charge"],
    "build_materials": ["flimsy", "cheap", "broken", "fragile", "broke", "cracked", "torn", "scratched", "ripped", "poor construction", "poor build", "fell apart", "falls apart", "peeling", "chipped"],
    "performance_function": ["slow", "lag", "freeze", "crashed", "stopped working", "useless", "defective", "malfunction", "fails", "fail", "broken", "glitch", "doesn't work", "not working", "does not work", "hang", "hangs", "buggy", "unresponsive"],
    "comfort_sizing": ["tight", "loose", "hurt", "painful", "pain", "tightness", "uncomfortable", "blisters", "digs in", "irritation", "itchy", "wrong size", "size issue"],
    "audio_sound": ["noise", "static", "distorted", "muffled", "buzzing", "crackling", "silent", "quiet", "no sound", "sound issue"],
    "general_dissatisfaction": [
        "disappointed", "disappointing", "not worth", "waste of money", "waste",
        "regret", "terrible", "horrible", "awful", "worst", "pathetic",
        "don't buy", "do not buy", "avoid", "stay away", "not recommended",
        "bad quality", "poor quality", "low quality", "inferior",
        "misleading", "false advertising", "not as described", "not as shown",
        "overpriced", "not satisfied", "unsatisfied", "unhappy",
        "rubbish", "junk", "trash", "crap",
    ],
}

STRENGTH_CATEGORIES = {
    "value_money": ["value for money", "worth the price", "affordable", "good deal", "great deal", "bang for the buck", "worth every penny"],
    "usability_design": ["beautiful design", "sleek", "easy to use", "intuitive", "stylish", "user friendly", "well designed"],
    "durability_reliability": ["durable", "sturdy", "long-lasting", "reliable", "premium quality", "well built", "well made", "solid build"],
    "comfort_ergonomics": ["comfortable", "cozy", "perfect fit", "ergonomic", "lightweight", "feels great", "feels good"],
    "performance_efficiency": ["fast performance", "powerful", "works great", "works perfectly", "works well", "performs well", "excellent performance", "runs smoothly"]
}

SEVERE_COMPLAINTS = [
    "stopped working", "broken", "defective", "dead on arrival", "crashed",
    "scam", "refund", "return", "useless", "worst", "garbage", "waste of money",
    "fraud", "fake product", "counterfeit", "dangerous", "safety hazard",
    "health risk", "don't buy", "do not buy", "stay away", "total waste",
    "threw it away", "threw away", "rip off", "ripoff",
]

SATISFACTION_PHRASES = [
    "love it", "highly recommend", "exceeded expectations",
    "great purchase", "would buy again", "worth every penny",
    "very satisfied", "extremely happy", "best purchase",
]

def analyze_product_quality(review_texts: list[str]) -> dict:
    """
    Analyze a collection of reviews for product quality.
    
    Returns:
        {
            rating: str,                   # Excellent | Good | Average | Poor
            quality_score: float,          # 0.0 (poor) to 1.0 (excellent)
            strengths: list[str],          # Highlighted praised themes
            complaint_frequency: dict,     # Category -> count / percent
            defect_concentrations: list,   # Warnings where defects are > 30%
            severe_issues_count: int,
            evidence: list[str],           # Specific sentences supporting the score
            sentiment_summary: dict,
            explanation: str,
        }
    """
    if not review_texts:
        return {
            "rating": "Average",
            "quality_score": 0.5,
            "strengths": ["No data available"],
            "complaint_frequency": {},
            "defect_concentrations": [],
            "severe_issues_count": 0,
            "evidence": ["No review text available to assess quality."],
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
            "explanation": "No reviews were available to evaluate product quality. Assigned neutral rating."
        }

    total_reviews = len(review_texts)
    complaint_counts = {cat: 0 for cat in COMPLAINT_CATEGORIES}
    strength_counts = {cat: 0 for cat in STRENGTH_CATEGORIES}
    
    severe_count = 0
    satisfaction_count = 0
    
    positive_sent_count = 0
    negative_sent_count = 0
    neutral_sent_count = 0
    sentiment_sum = 0.0
    
    # Store evidence candidates
    strength_sentences = []
    complaint_sentences = []
    severe_sentences = []

    for text in review_texts:
        lower_text = text.lower()
        
        # Sentiment tracking
        sentiment = get_sentiment(text)
        sentiment_sum += sentiment
        if sentiment > 0.15:
            positive_sent_count += 1
        elif sentiment < -0.15:
            negative_sent_count += 1
        else:
            neutral_sent_count += 1

        # Check complaints
        has_any_complaint = False
        for cat, keywords in COMPLAINT_CATEGORIES.items():
            matched_keywords = [kw for kw in keywords if kw in lower_text]
            if matched_keywords:
                complaint_counts[cat] += 1
                has_any_complaint = True
                if len(text) < 150:
                    complaint_sentences.append(text)
                else:
                    # Find a sentence with the keyword
                    for sentence in re.split(r'[.!?]+', text):
                        if any(kw in sentence.lower() for kw in matched_keywords):
                            complaint_sentences.append(sentence.strip())
                            break

        # Check strengths — with negation awareness
        for cat, keywords in STRENGTH_CATEGORIES.items():
            # Only count a keyword as a strength if it is NOT negated
            matched_keywords = [
                kw for kw in keywords
                if kw in lower_text and not _is_negated(lower_text, kw)
            ]
            # Additionally, skip this strength if the same review already
            # triggered a complaint — a review saying "not durable, broke
            # in a week" should not count "durable" as a strength.
            if matched_keywords and not has_any_complaint:
                strength_counts[cat] += 1
                if len(text) < 150:
                    strength_sentences.append(text)
                else:
                    for sentence in re.split(r'[.!?]+', text):
                        if any(kw in sentence.lower() for kw in matched_keywords):
                            strength_sentences.append(sentence.strip())
                            break

        # Check severe issues
        matched_severe = [sc for sc in SEVERE_COMPLAINTS if sc in lower_text]
        if matched_severe:
            severe_count += 1
            for sentence in re.split(r'[.!?]+', text):
                if any(sc in sentence.lower() for sc in matched_severe):
                    severe_sentences.append(sentence.strip())
                    break

        # Check satisfaction signals
        if any(sp in lower_text for sp in SATISFACTION_PHRASES):
            satisfaction_count += 1

    # Normalize frequencies (percentage of reviews mentioning)
    complaint_pct = {cat: round(count / total_reviews, 4) for cat, count in complaint_counts.items()}
    strength_pct = {cat: round(count / total_reviews, 4) for cat, count in strength_counts.items()}

    # Identify defect concentrations (>30% of reviews complaining about something)
    defect_concentrations = []
    for cat, pct in complaint_pct.items():
        if pct >= 0.30:
            friendly_name = cat.replace("_", " ").title()
            defect_concentrations.append({
                "category": cat,
                "percentage": round(pct * 100),
                "warning": f"High defect concentration: {round(pct * 100)}% of reviews complain about {friendly_name}."
            })

    # Strengths identification (>20% positive and has notable count)
    strengths = []
    for cat, pct in strength_pct.items():
        if pct >= 0.25:
            strengths.append(cat.replace("_", " ").title())
    
    if not strengths:
        # Fallback to the top strength
        top_strength = max(strength_pct.items(), key=lambda x: x[1])
        if top_strength[1] > 0.0:
            strengths.append(top_strength[0].replace("_", " ").title())
        else:
            strengths.append("General Functionality")

    # Score calculation
    # Base starts at 0.5 (neutral)
    # Plus factors: positive sentiment, strengths, satisfaction phrases
    # Minus factors: negative sentiment, complaints, severe complaints, defect concentrations
    avg_sentiment = sentiment_sum / total_reviews
    
    # Calculate score weights — rebalanced to reduce positive bias
    complaint_penalty = sum(complaint_pct.values()) * 0.25       # Increased from 0.15
    severe_penalty = (severe_count / total_reviews) * 0.45       # Increased from 0.35
    defect_penalty = len(defect_concentrations) * 0.12           # Slightly increased from 0.10
    
    # Penalty for high proportion of negative reviews
    negative_ratio = negative_sent_count / total_reviews
    negative_sentiment_penalty = negative_ratio * 0.30           # NEW: penalize high negative ratio

    satisfaction_bonus = (satisfaction_count / total_reviews) * 0.15   # Reduced from 0.25
    sentiment_adjustment = avg_sentiment * 0.20                       # Reduced from 0.35
    
    # Strength discount: if complaint-heavy reviews also match strengths, discount the bonus
    total_complaints = sum(complaint_counts.values())
    total_strengths = sum(strength_counts.values())
    if total_complaints > 0 and total_strengths > 0:
        # When complaints exist, strengths get discounted proportionally
        overlap_ratio = min(total_complaints / max(total_strengths, 1), 1.0)
        sentiment_adjustment *= (1.0 - overlap_ratio * 0.5)
        satisfaction_bonus *= (1.0 - overlap_ratio * 0.5)

    quality_score = 0.5 + sentiment_adjustment + satisfaction_bonus - complaint_penalty - severe_penalty - defect_penalty - negative_sentiment_penalty
    quality_score = round(min(max(quality_score, 0.0), 1.0), 4)

    # Determine Rating — thresholds raised to make "Good" harder to reach
    if quality_score >= 0.78:
        rating = "Excellent"
    elif quality_score >= 0.55:
        rating = "Good"
    elif quality_score >= 0.32:
        rating = "Average"
    else:
        rating = "Poor"

    # Assemble specific evidence list
    evidence = []
    if severe_sentences and rating in ["Poor", "Average"]:
        evidence.extend([f"Severe issue: \"{s}\"" for s in severe_sentences[:2]])
    if complaint_sentences and rating in ["Poor", "Average", "Good"]:
        evidence.extend([f"Complaint: \"{s}\"" for s in complaint_sentences[:2]])
    if strength_sentences and rating in ["Good", "Excellent"]:
        evidence.extend([f"Praise: \"{s}\"" for s in strength_sentences[:2]])
        
    # Remove duplicates and clean
    evidence = list(dict.fromkeys([e for e in evidence if len(e) < 200]))[:3]
    if not evidence:
        evidence.append("Reviews exhibit standard user feedback patterns with mixed sentiment.")

    # Explanation text
    explanation = _build_explanation(rating, strengths, defect_concentrations, total_reviews)

    return {
        "rating": rating,
        "quality_score": quality_score,
        "strengths": strengths,
        "complaint_frequency": complaint_pct,
        "defect_concentrations": defect_concentrations,
        "severe_issues_count": severe_count,
        "evidence": evidence,
        "sentiment_summary": {
            "positive": positive_sent_count,
            "negative": negative_sent_count,
            "neutral": neutral_sent_count,
            "average_sentiment": round(avg_sentiment, 4)
        },
        "explanation": explanation
    }

def _build_explanation(rating: str, strengths: list[str], defect_concentrations: list, total_reviews: int) -> str:
    strength_list = ", ".join(strengths)
    if rating == "Excellent":
        return f"Based on {total_reviews} reviews, the product shows excellent quality, with strong praise for {strength_list} and virtually no severe complaints."
    elif rating == "Good":
        return f"The product is generally good, with notable strengths in {strength_list}. There are minor complaints but no critical defect concentrations."
    elif rating == "Average":
        if defect_concentrations:
            defects = ", ".join([d["category"].replace("_", " ") for d in defect_concentrations])
            return f"This product is rated average. While it has some strengths, there is a significant concentration of complaints regarding {defects}."
        return f"The product has mixed feedback and is rated average. Praise is balanced by standard product complaints."
    else:  # Poor
        if defect_concentrations:
            defects = ", ".join([d["category"].replace("_", " ") for d in defect_concentrations])
            return f"Poor product quality detected. A significant portion of buyers complain about {defects}, along with multiple reports of severe defects."
        return f"Poor quality rating based on high volume of complaints, negative sentiment, and reports of defective performance."
