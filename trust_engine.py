"""
reviews/trust_engine.py
=======================
ENGINE 3: Review Ecosystem Trustworthiness — "Can we trust this product's reviews as a whole?"

Analyzes a corpus of reviews for patterns of manipulation:
  • Duplicate & near-duplicate review text (using Jaccard word similarity)
  • Review diversity (vocabulary variation, standard deviation of lengths)
  • Rating concentration (extreme imbalance of 5-star vs moderate ratings)
  • Verified purchase ratio (if available)
  • Suspicious similarity warnings
  • Overall trust rating: High Trust | Medium Trust | Low Trust
"""

from __future__ import annotations
import re
from typing import List, Dict, Set
from collections import Counter
import numpy as np

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate the Jaccard similarity between two word sets."""
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def extract_rating_from_text(text: str) -> int | None:
    """Extract a star rating (1-5) from review text using common patterns."""
    text_lower = text.lower()
    
    # Pattern 1: "X.0 out of 5 stars" or "X out of 5"
    match = re.search(r"\b([1-5])(?:\.0)?\s*(?:out of|/)\s*5\s*(?:stars?)?\b", text_lower)
    if match:
        return int(match.group(1))
        
    # Pattern 2: "X star review", "X-star", "X stars"
    match = re.search(r"\b([1-5])\s*-?\s*stars?\b", text_lower)
    if match:
        return int(match.group(1))
        
    # Pattern 3: Written out numbers e.g. "five stars", "one star"
    words_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, val in words_map.items():
        if f"{word} star" in text_lower or f"{word}-star" in text_lower:
            return val
            
    # Pattern 4: "X/5"
    match = re.search(r"\b([1-5])/5\b", text_lower)
    if match:
        return int(match.group(1))
        
    return None

def analyze_review_trustworthiness(review_texts: list[str], ratings: list[int] | None = None, verified_statuses: list[bool] | None = None) -> dict:
    """
    Analyze the trustworthiness of a collection of reviews.
    
    Returns:
        {
            trust_level: str,              # High Trust | Medium Trust | Low Trust
            trust_score: float,            # 0.0 (low trust) to 1.0 (high trust)
            similarity_score: float,       # 0.0 (diverse) to 1.0 (identical)
            diversity_score: float,        # 0.0 (monolithic) to 1.0 (highly varied)
            duplicate_count: int,
            near_duplicates: list[dict],   # List of duplicate review pairs
            rating_distribution: dict,     # Clustered rating signals
            verified_purchase_ratio: float,
            warnings: list[str],
            explanation: str
        }
    """
    if not review_texts or len(review_texts) < 3:
        # Not enough data to assess ecosystem trust
        return {
            "trust_level": "Medium Trust",
            "trust_score": 0.5,
            "similarity_score": 0.0,
            "diversity_score": 0.7,
            "duplicate_count": 0,
            "near_duplicates": [],
            "rating_distribution": {"unknown": len(review_texts)},
            "verified_purchase_ratio": 1.0,
            "warnings": ["Insufficient review corpus size (need at least 3 reviews) to run pattern analysis."],
            "explanation": "Insufficient reviews were available to establish review ecosystem trust patterns."
        }

    total_reviews = len(review_texts)
    warnings = []
    
    # ── 1. Near-Duplicate Detection (Jaccard similarity of word tokens) ──
    # Clean and tokenize each review text into sets of words (length > 2)
    word_sets: list[set[str]] = []
    for text in review_texts:
        words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        word_sets.append(words)

    duplicate_pairs = []
    duplicate_indices = set()
    high_similarity_count = 0
    similarity_sum = 0.0
    comparison_count = 0

    for i in range(total_reviews):
        for j in range(i + 1, total_reviews):
            comparison_count += 1
            sim = jaccard_similarity(word_sets[i], word_sets[j])
            similarity_sum += sim
            
            if sim >= 0.65:
                high_similarity_count += 1
                if sim >= 0.80:
                    duplicate_indices.add(i)
                    duplicate_indices.add(j)
                    duplicate_pairs.append({
                        "review_a": review_texts[i][:120] + "...",
                        "review_b": review_texts[j][:120] + "...",
                        "similarity": round(sim, 4),
                        "type": "exact_duplicate" if sim >= 0.95 else "near_duplicate"
                    })

    avg_similarity = similarity_sum / comparison_count if comparison_count > 0 else 0.0
    duplicate_ratio = high_similarity_count / total_reviews

    # ── 2. Review Diversity Analysis (Vocabulary & Lengths) ──
    # Concatenate all reviews to measure global vocabulary
    all_words = re.findall(r"\b[a-z]{3,}\b", " ".join(review_texts).lower())
    global_vocab = len(set(all_words))
    total_words_count = len(all_words)
    
    # Vocabulary diversity = unique words / total words
    vocab_diversity = global_vocab / max(total_words_count, 1)
    
    # Length standard deviation (suspicious if all reviews have exactly the same length)
    lengths = [len(text.split()) for text in review_texts]
    length_std = float(np.std(lengths)) if len(lengths) > 1 else 0.0
    # Normalize length std: std of 0-5 = highly suspicious (0.0), std > 25 = natural variety (1.0)
    length_variety_score = min(length_std / 25.0, 1.0)
    
    # Combined diversity score
    diversity_score = (vocab_diversity * 0.5) + (length_variety_score * 0.5)
    diversity_score = round(min(max(diversity_score, 0.0), 1.0), 4)

    # ── 3. Rating & Sentiment Processing (with explicit/inferred star ratings) ──
    inferred_ratings = []
    if ratings:
        inferred_ratings = list(ratings)
    else:
        for text in review_texts:
            rating = extract_rating_from_text(text)
            if rating is not None:
                inferred_ratings.append(rating)

    # If we couldn't extract enough ratings, infer the rest from sentiment
    from .feature_extraction import get_sentiment
    if len(inferred_ratings) < total_reviews:
        missing_count = total_reviews - len(inferred_ratings)
        # We start filling from the end/sentiment of all reviews
        for text in review_texts[len(inferred_ratings):]:
            sent = get_sentiment(text)
            if sent > 0.6:
                inferred_ratings.append(5)
            elif sent > 0.2:
                inferred_ratings.append(4)
            elif sent > -0.2:
                inferred_ratings.append(3)
            elif sent > -0.6:
                inferred_ratings.append(2)
            else:
                inferred_ratings.append(1)

    rating_counts = Counter(inferred_ratings)
    five_star_signals = rating_counts.get(5, 0)
    five_star_ratio = five_star_signals / total_reviews
    rating_dist = {str(k): v for k, v in sorted(rating_counts.items())}

    # ── 4. Verified Purchase Signal ──
    verified_ratio = 1.0
    if verified_statuses:
        verified_count = sum(1 for status in verified_statuses if status)
        verified_ratio = verified_count / total_reviews

    # ── 5. Trust Score Calculation ──
    duplicate_penalty = min(duplicate_ratio * 3.0, 0.60)
    similarity_penalty = max(0, (avg_similarity - 0.12) * 2.0)
    similarity_penalty = min(similarity_penalty, 0.30)
    diversity_penalty = (1.0 - diversity_score) * 0.40
    verified_penalty = (1.0 - verified_ratio) * 0.20

    # Conditionally applied concentration penalty:
    # A high 5-star ratio (mostly positive feedback) is normal for genuinely excellent products.
    # We only penalize positive skew if other suspicious metrics (low diversity, high similarity, duplicates) are present.
    concentration_penalty = 0.0
    has_suspicious_patterns = (diversity_score < 0.60) or (duplicate_ratio > 0.02) or (avg_similarity > 0.25)
    if has_suspicious_patterns:
        if five_star_ratio > 0.85:
            concentration_penalty = 0.20
        elif five_star_ratio > 0.70:
            concentration_penalty = 0.10

    # Polarized anomaly check: large concentrations of 5 and 1 stars, with very few moderates (2-4 stars)
    polarized_penalty = 0.0
    total_rated = len(inferred_ratings)
    if total_rated >= 5:
        ones = rating_counts.get(1, 0)
        fives = rating_counts.get(5, 0)
        moderates = rating_counts.get(2, 0) + rating_counts.get(3, 0) + rating_counts.get(4, 0)
        moderate_ratio = moderates / total_rated
        if ones / total_rated >= 0.10 and fives / total_rated >= 0.60 and moderate_ratio < 0.12:
            polarized_penalty = 0.15
            warnings.append("Polarized review distribution: high concentrations of both 1-star and 5-star reviews with very few moderate ratings.")

    # Rating-sentiment mismatch anomaly check
    mismatch_count = 0
    for text, rating in zip(review_texts, inferred_ratings):
        sent = get_sentiment(text)
        if rating == 5 and sent < -0.4:
            mismatch_count += 1
        elif rating == 1 and sent > 0.6:
            mismatch_count += 1
    
    mismatch_ratio = mismatch_count / total_reviews
    mismatch_penalty = 0.0
    if mismatch_ratio > 0.12:
        mismatch_penalty = min(mismatch_ratio * 1.5, 0.25)
        warnings.append(f"Rating-sentiment mismatch: {round(mismatch_ratio*100)}% of reviews have conflicting ratings and text sentiment.")

    # Final overall trust score sum
    trust_score = 1.0 - duplicate_penalty - similarity_penalty - diversity_penalty - concentration_penalty - verified_penalty - polarized_penalty - mismatch_penalty
    trust_score = round(min(max(trust_score, 0.0), 1.0), 4)

    # Determine Trust Level (aligned with 50 and 70 thresholds on 0-100 scale)
    if trust_score > 0.70:
        trust_level = "High Trust"
    elif trust_score >= 0.50:
        trust_level = "Medium Trust"
    else:
        trust_level = "Low Trust"

    # Compile warnings
    if duplicate_pairs:
        warnings.append(f"Detected {len(duplicate_pairs)} duplicate or near-duplicate reviews.")
    if avg_similarity > 0.35:
        warnings.append(f"Extremely high review similarity ({round(avg_similarity*100)}% average overlap) indicating templates.")
    if diversity_score < 0.35:
        warnings.append("Low review diversity: review lengths and vocabulary are highly uniform.")
    if five_star_ratio > 0.85 and has_suspicious_patterns:
        warnings.append(f"Highly concentrated rating profile: {round(five_star_ratio*100)}% of reviews are 5-star/extremely positive.")
    if verified_ratio < 0.50:
        warnings.append(f"Low verified purchase ratio: only {round(verified_ratio*100)}% of reviews are verified buyers.")

    if not warnings:
        warnings.append("Review ecosystem displays normal organic variance in length, wording, and ratings.")

    explanation = _build_explanation(trust_level, len(duplicate_pairs), avg_similarity, diversity_score, five_star_ratio)

    return {
        "trust_level": trust_level,
        "trust_score": trust_score,
        "similarity_score": round(avg_similarity, 4),
        "diversity_score": diversity_score,
        "duplicate_count": len(duplicate_pairs),
        "duplicate_indices": duplicate_indices,
        "near_duplicates": duplicate_pairs[:4],
        "rating_distribution": rating_dist,
        "verified_purchase_ratio": round(verified_ratio, 4),
        "warnings": warnings,
        "explanation": explanation
    }

def _build_explanation(trust_level: str, dup_count: int, avg_sim: float, div_score: float, five_star_ratio: float) -> str:
    if trust_level == "High Trust":
        return "The review ecosystem is highly trustworthy. Reviews exhibit normal organic distribution of text, varied wording, and low duplicate signals."
    elif trust_level == "Medium Trust":
        if dup_count > 0:
            return f"Review ecosystem has moderate risk. Found {dup_count} near-duplicate reviews, which suggests some promotional activity."
        if five_star_ratio > 0.80:
            return "Review trust is moderate due to rating concentration; reviews are heavily positive with few critical perspectives."
        return "Review ecosystem trust is moderate. Some repetitive patterns or language overlap were detected but not dominant."
    else:  # Low Trust
        if dup_count > 0 or avg_sim > 0.40:
            return f"Ecosystem trust is LOW. Clear signs of review manipulation: {dup_count} duplicates found, with highly coordinated and template-like reviews."
        return "Review trust is LOW. Highly unnatural review profiles detected, featuring low linguistic diversity and extreme sentiment clustering."
