"""
reviews/feature_extraction.py
=============================
Single source of truth for text preprocessing, feature extraction, and
sentiment analysis.  Used by both the ML training script and the runtime
predictor so features are guaranteed to match.
"""

import re
import math
import numpy as np
from typing import Optional

# ── Stopwords ──────────────────────────────────────────────────────────────
STOP_WORDS = {
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
    'who', 'this', 'that', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'as', 'of', 'at', 'by', 'for', 'with', 'in',
    'out', 'on', 'to', 'from', 'up', 'down', 'not', 'so', 'than', 'can',
    'will', 'just', 'should', 'now', 'very', 'also',
}

# ── Hype / exaggeration words ─────────────────────────────────────────────
HYPE_WORDS = {
    'best', 'worst', 'amazing', 'terrible', 'perfect', 'awful',
    'incredible', 'horrible', 'fantastic', 'dreadful', 'love', 'hate',
    'outstanding', 'atrocious', 'phenomenal', 'garbage', 'revolutionary',
    'miracle', 'life-changing', 'flawless', 'scam', 'fraud', 'never',
    'always', 'every', 'absolutely', 'unbelievable', 'extraordinary',
    'exceptional', 'greatest',
}

# ── Marketing / promotional phrases ───────────────────────────────────────
MARKETING_PHRASES = [
    'must buy', 'must have', 'buy now', 'order now', 'highly recommend',
    'best product', 'best purchase', 'changed my life', 'life changing',
    'game changer', 'you won\'t regret', 'tell everyone', 'share this',
    'doctors don\'t want', 'before it gets taken down', 'buy with confidence',
    'five stars', '5 stars', 'worth every penny', 'value for money',
    'exceeded all expectations', 'exceeded my expectations',
    'no complaints', 'zero complaints', 'not a single issue',
    'perfect in every way', 'flawless', 'perfection',
]

# ── Specificity indicators ────────────────────────────────────────────────
SPECIFICITY_PATTERNS = [
    r'\b\d+\s*(?:hour|hr|minute|min|day|week|month|year)s?\b',   # duration
    r'\b\d+\s*(?:inch|cm|mm|ft|lb|kg|oz|gram|ml|liter)s?\b',    # measurement
    r'\b\d+(?:\.\d+)?\s*%\b',                                     # percentage
    r'\b(?:size|model|version)\s+\w+\b',                           # model/size
    r'\b\d+(?:\.\d+)?\s*(?:star|rating)\b',                       # rating
    r'\b\d+(?:st|nd|rd|th)\b',                                     # ordinal
    r'\b\d+\s*(?:times?|x)\b',                                    # frequency
    r'\$\d+|\₹\d+|\d+\s*(?:dollar|rupee|buck)s?\b',              # price
]

# ── Usage evidence patterns ───────────────────────────────────────────────
USAGE_PATTERNS = [
    r'\bi\s+(?:bought|purchased|ordered|received|got|tried|used|tested|wore|ran|walked)\b',
    r'\b(?:after|for)\s+\d+\s*(?:day|week|month|year|hour|mile|km)s?\b',
    r'\bi(?:\'ve|\'m|\s+have|\s+am)\s+(?:been\s+)?(?:using|wearing|running|testing)\b',
    r'\b(?:my|mine)\s+(?:first|second|third)\b',
    r'\b(?:compared to|better than|worse than|similar to)\b',
    r'\b(?:fits|fit)\s+(?:true|perfectly|well|snug|tight|loose)\b',
]

# ── Hedging / balanced language ───────────────────────────────────────────
HEDGING_WORDS = {
    'somewhat', 'a bit', 'slightly', 'mostly', 'fairly', 'reasonably',
    'adequate', 'decent', 'okay', 'acceptable', 'average', 'moderate',
    'minor', 'however', 'although', 'though', 'except', 'but',
    'on the other hand', 'the only issue', 'my only complaint',
    'could be better', 'room for improvement', 'not perfect',
}


# ── Sentiment helper ───────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()

    def get_sentiment(text: str) -> float:
        return _vader.polarity_scores(text)['compound']
except Exception:
    try:
        from textblob import TextBlob

        def get_sentiment(text: str) -> float:
            return TextBlob(text).sentiment.polarity
    except Exception:
        def get_sentiment(text: str) -> float:
            return 0.0


# ── Text preprocessing ────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Lowercase → remove non-alpha → tokenize → remove stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = re.findall(r'\b[a-z]+\b', text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


def clean_review_text(value: str) -> str:
    """Remove Amazon/Flipkart boilerplate from scraped review text."""
    value = re.sub(r'\bRead more\b', ' ', value or '', flags=re.IGNORECASE)
    value = re.sub(
        r'\bBrief content visible, double tap to read full content\.?',
        ' ', value, flags=re.IGNORECASE,
    )
    value = re.sub(
        r'\bFull content visible, double tap to read brief content\.?',
        ' ', value, flags=re.IGNORECASE,
    )
    value = re.sub(r'\s*"\s*"\s*', ' ', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip(' "\'')


# ── Core feature extraction (12 features for ML model) ────────────────────
FEATURE_NAMES = [
    'sentiment', 'sentiment_extreme', 'char_len', 'word_count',
    'avg_word_len', 'excl_count', 'caps_ratio', 'unique_ratio',
    'hype_ratio', 'is_short', 'is_long', 'repeat_ratio',
]


def extract_features_single(text: str) -> np.ndarray:
    """Extract the 12-feature vector for a single review text."""
    words = text.lower().split()
    wc = len(words)

    sentiment = get_sentiment(text)
    sentiment_extreme = abs(sentiment)
    char_len = len(text)
    avg_word_len = sum(len(w) for w in words) / max(wc, 1)
    excl_count = text.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    unique_ratio = len(set(words)) / max(wc, 1)
    hype_count = sum(1 for w in words if w in HYPE_WORDS)
    hype_ratio = hype_count / max(wc, 1)
    is_short = 1 if wc < 10 else 0
    is_long = 1 if wc > 120 else 0

    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    max_repeat = max(word_freq.values()) if word_freq else 0
    repeat_ratio = max_repeat / max(wc, 1)

    return np.array([[
        sentiment, sentiment_extreme, char_len, wc,
        avg_word_len, excl_count, caps_ratio, unique_ratio,
        hype_ratio, is_short, is_long, repeat_ratio,
    ]], dtype=float)


def extract_features_batch(texts: list[str]) -> np.ndarray:
    """Extract the 12-feature matrix for a list of review texts."""
    rows = []
    for text in texts:
        rows.append(extract_features_single(text)[0])
    return np.array(rows, dtype=float)


# ── Extended analysis signals (for authenticity engine) ────────────────────
def compute_specificity_score(text: str) -> tuple[float, list[str]]:
    """
    Score 0-1 indicating how specific and detailed the review is.
    Returns (score, evidence_list).
    """
    evidence = []
    matches = 0
    for pattern in SPECIFICITY_PATTERNS:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            matches += len(found)
            evidence.append(f"Contains specific detail: '{found[0]}'")

    words = text.split()
    wc = max(len(words), 1)
    # Normalize: 0 matches = 0.0, 4+ matches in a moderate review = 1.0
    raw = min(matches / max(wc / 25, 1), 1.0)
    score = min(raw * 1.2, 1.0)

    if not evidence:
        evidence.append("No specific measurements, durations, or details found")

    return round(score, 4), evidence[:4]


def compute_usage_evidence_score(text: str) -> tuple[float, list[str]]:
    """
    Score 0-1 indicating how much real usage evidence the review contains.
    Returns (score, evidence_list).
    """
    evidence = []
    matches = 0
    for pattern in USAGE_PATTERNS:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            matches += len(found)
            evidence.append(f"Usage indicator: '{found[0]}'")

    # Check for first-person pronouns + past tense
    lower = text.lower()
    first_person = sum(1 for p in ['i ', 'my ', 'me ', 'i\'ve', 'i\'m'] if p in lower)
    past_tense = len(re.findall(r'\b\w+ed\b', lower))

    personal_experience = min((first_person + past_tense) / 8, 1.0)
    pattern_score = min(matches / 3, 1.0)
    score = (pattern_score * 0.6) + (personal_experience * 0.4)

    if first_person > 0 and past_tense > 0:
        evidence.append("First-person narrative with past-tense usage")
    elif first_person == 0:
        evidence.append("No first-person experience described")

    if not evidence:
        evidence.append("No product usage evidence detected")

    return round(min(score, 1.0), 4), evidence[:4]


def compute_balance_score(text: str) -> tuple[float, list[str]]:
    """
    Score 0-1 indicating how balanced (containing both pros and cons) the review is.
    Returns (score, evidence_list).
    """
    evidence = []
    lower = text.lower()
    sentiment = get_sentiment(text)

    # Check for hedging language
    hedging_found = [h for h in HEDGING_WORDS if h in lower]
    hedging_score = min(len(hedging_found) / 3, 1.0)

    # Check for contrasting conjunctions
    contrast_words = ['but', 'however', 'although', 'though', 'except', 'unfortunately']
    contrast_found = sum(1 for w in contrast_words if w in lower)
    contrast_score = min(contrast_found / 2, 1.0)

    # Sentiment balance: moderate sentiment = high balance
    sentiment_balance = max(0, 1.0 - abs(sentiment) * 1.5)

    score = (hedging_score * 0.35) + (contrast_score * 0.35) + (sentiment_balance * 0.30)

    if hedging_found:
        evidence.append(f"Uses balanced language: {', '.join(hedging_found[:3])}")
    if contrast_found > 0:
        evidence.append("Contains contrasting opinions (pros and cons)")
    if abs(sentiment) < 0.3:
        evidence.append(f"Moderate sentiment ({sentiment:.2f}) suggests balanced view")
    elif abs(sentiment) > 0.7:
        evidence.append(f"Extreme sentiment ({sentiment:.2f}) suggests one-sided view")

    if not evidence:
        evidence.append("Review does not show balanced perspective")

    return round(min(score, 1.0), 4), evidence[:4]


def compute_marketing_score(text: str) -> tuple[float, list[str]]:
    """
    Score 0-1 indicating how much promotional/marketing language is present.
    HIGH score = MORE marketing language = MORE suspicious.
    Returns (score, evidence_list).
    """
    evidence = []
    lower = text.lower()

    # Check marketing phrases
    phrase_matches = [p for p in MARKETING_PHRASES if p in lower]
    phrase_score = min(len(phrase_matches) / 3, 1.0)

    # Check hype word density
    words = lower.split()
    wc = max(len(words), 1)
    hype_count = sum(1 for w in words if w in HYPE_WORDS)
    hype_density = min(hype_count / max(wc / 5, 1), 1.0)

    # Check exclamation density
    excl = text.count('!')
    excl_density = min(excl / max(wc / 10, 1), 1.0)

    # Check for urgency/call-to-action
    urgency_patterns = ['buy now', 'order now', 'right now', 'immediately',
                        'don\'t miss', 'limited time', 'hurry', 'act fast']
    urgency_found = sum(1 for p in urgency_patterns if p in lower)
    urgency_score = min(urgency_found / 2, 1.0)

    score = (phrase_score * 0.35) + (hype_density * 0.25) + (excl_density * 0.20) + (urgency_score * 0.20)

    if phrase_matches:
        evidence.append(f"Marketing phrases: {', '.join(phrase_matches[:3])}")
    if hype_count >= 3:
        evidence.append(f"High density of hyperbolic words ({hype_count} found)")
    if excl > 3:
        evidence.append(f"Excessive exclamation marks ({excl})")
    if urgency_found > 0:
        evidence.append("Contains urgency/call-to-action language")

    if not evidence:
        evidence.append("No significant promotional language detected")

    return round(min(score, 1.0), 4), evidence[:4]


def compute_naturalness_score(text: str) -> tuple[float, list[str]]:
    """
    Score 0-1 indicating how natural/human the writing style is.
    Returns (score, evidence_list).
    """
    evidence = []
    words = text.split()
    wc = max(len(words), 1)
    lower_words = text.lower().split()

    # Sentence length variety
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 1:
        sent_lengths = [len(s.split()) for s in sentences]
        sent_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0
        variety_score = min(sent_std / 5, 1.0)
        if variety_score > 0.3:
            evidence.append("Varied sentence lengths suggest natural writing")
    else:
        variety_score = 0.2  # Single sentence

    # Vocabulary diversity
    unique_ratio = len(set(lower_words)) / max(wc, 1)
    diversity_score = min(unique_ratio / 0.7, 1.0) if unique_ratio < 0.95 else 0.6
    if unique_ratio > 0.65:
        evidence.append(f"Good vocabulary diversity ({unique_ratio:.2f})")
    elif unique_ratio < 0.5:
        evidence.append(f"Low vocabulary diversity ({unique_ratio:.2f}) — repetitive")

    # CAPS abuse
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    caps_penalty = max(0, 1.0 - (caps_ratio * 5)) if caps_ratio > 0.15 else 1.0
    if caps_ratio > 0.15:
        evidence.append(f"High caps ratio ({caps_ratio*100:.0f}%) suggests artificial emphasis")

    # Natural length
    length_score = 1.0
    if wc < 5:
        length_score = 0.2
        evidence.append(f"Very short ({wc} words) — insufficient for analysis")
    elif wc > 150:
        length_score = 0.7
    elif 15 <= wc <= 80:
        evidence.append(f"Natural review length ({wc} words)")

    score = (variety_score * 0.25) + (diversity_score * 0.30) + (caps_penalty * 0.25) + (length_score * 0.20)

    if not evidence:
        evidence.append("Writing style appears natural")

    return round(min(score, 1.0), 4), evidence[:4]


def compute_repetition_score(text: str) -> tuple[float, list[str]]:
    """
    Score 0-1 indicating suspicious repetition. HIGH score = MORE repetition = MORE suspicious.
    Returns (score, evidence_list).
    """
    evidence = []
    words = text.lower().split()
    wc = max(len(words), 1)

    # Word-level repetition
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    non_stop_freq = {w: c for w, c in word_freq.items() if w not in STOP_WORDS and len(w) > 2}
    max_repeat = max(non_stop_freq.values()) if non_stop_freq else 0
    word_repeat_score = min(max_repeat / max(wc / 8, 1), 1.0) if max_repeat > 2 else 0.0

    # Bigram repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bigram_freq = {}
    for bg in bigrams:
        bigram_freq[bg] = bigram_freq.get(bg, 0) + 1
    repeated_bigrams = {bg: c for bg, c in bigram_freq.items() if c > 1 and len(bg) > 5}
    bigram_score = min(len(repeated_bigrams) / 3, 1.0)

    score = (word_repeat_score * 0.5) + (bigram_score * 0.5)

    if repeated_bigrams:
        top = sorted(repeated_bigrams.items(), key=lambda x: -x[1])[:3]
        evidence.append(f"Repeated phrases: {', '.join(f'\"{bg}\" ({c}x)' for bg, c in top)}")
    if max_repeat > 3:
        top_word = max(non_stop_freq.items(), key=lambda x: x[1])
        evidence.append(f"Word '{top_word[0]}' repeated {top_word[1]} times")

    if not evidence:
        evidence.append("No suspicious repetition patterns")

    return round(min(score, 1.0), 4), evidence[:4]
