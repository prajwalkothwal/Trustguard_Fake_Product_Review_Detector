"""
Product-level trust analysis.

Collects public search snippets for a product, detects suspicious review
patterns, and optionally asks NVIDIA-hosted Kimi for a buying recommendation.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse

import requests

from .platform_review_analyzer import analyze_platform_reviews
from .runtime_config import describe_nvidia_error, get_kimi_deployment_issue, get_kimi_runtime_config
from .quality_engine import analyze_product_quality
from .trust_engine import analyze_review_trustworthiness
from .hybrid_decision import merge_product_scores

logger = logging.getLogger('reviews.product_intelligence')


SUSPICIOUS_PHRASES = {
    "best product ever",
    "must buy",
    "life changing",
    "highly recommend",
    "value for money",
    "worth every penny",
    "amazing product",
    "excellent product",
    "five stars",
    "superb quality",
}

COMPLAINT_WORDS = {
    "defect",
    "broken",
    "refund",
    "return",
    "warranty",
    "fake",
    "scam",
    "duplicate",
    "stopped working",
    "poor",
    "bad",
    "heating",
    "battery",
    "damage",
}

GENUINENESS_RISK_WORDS = {
    "counterfeit",
    "duplicate",
    "fake",
    "fraud",
    "imitation",
    "knockoff",
    "not genuine",
    "scam",
}

GENUINENESS_TRUST_WORDS = {
    "authorized",
    "authentic",
    "genuine",
    "official",
    "original",
    "serial number",
    "verify",
    "warranty",
}


def analyze_product(product_name: str, product_url: str = "") -> dict:
    product_name = product_name.strip()
    product_url = product_url.strip()
    sources = collect_product_sources(product_name, product_url)
    genuineness_report = analyze_online_genuineness(product_name, product_url, sources)
    platform_report = analyze_platform_reviews(product_url) if product_url else None
    logger.info("[ML ANALYSIS STARTED] product=%s", product_name)
    try:
        pattern_report = analyze_patterns(product_name, sources, platform_report)
        pattern_report["score_basis"].append(
            f"Online product genuineness research: {genuineness_report['assessment']} "
            f"({genuineness_report['genuineness_score']}/100 advisory score from "
            f"{genuineness_report['sources_checked']} web source(s))."
        )
        ml_trust_score = pattern_report["trust_score"]
        ml_status = {"status": "completed", "available": True}
        logger.info("[ML ANALYSIS COMPLETE] product=%s trust_score=%d", product_name, ml_trust_score)
    except Exception as exc:
        logger.exception("[ML ANALYSIS FAILED] product=%s error=%s", product_name, exc)
        pattern_report = unavailable_pattern_report()
        ml_trust_score = None
        ml_status = {"status": "failed", "available": False, "error": str(exc)[:240]}

    llm_report = generate_kimi_advice(
        product_name, product_url, pattern_report, sources, platform_report, genuineness_report
    )
    merged = merge_product_scores(ml_trust_score, llm_report)
    ai_available = llm_report.get("available", False)
    ai_trust_score = merged["ai_trust_score"]

    limitations = [
        "Search snippets are public signals and may not represent every review.",
        "Shopping sites can block scraping, so source coverage may vary.",
        "The app targets 50-100 reviews when a public product URL exposes enough review-like text; blocked or script-rendered pages may provide fewer.",
        "Review image analysis uses file/metadata/quality signals and is advisory, not forensic proof.",
        "Final recommendation should be treated as purchase guidance, not proof.",
    ]

    hybrid_trust_score = merged["hybrid_trust_score"]
    if ai_available:
        advisor_summary = llm_report.get("summary") or pattern_report["summary"]
    else:
        advisor_summary = pattern_report["summary"]
    limitations.extend(merged["warnings"])

    if hybrid_trust_score > 70:
        hybrid_trust_level = "High Trust"
        fake_review_risk = "Low"
    elif hybrid_trust_score >= 50:
        hybrid_trust_level = "Medium Trust"
        fake_review_risk = "Medium"
    else:
        hybrid_trust_level = "Low Trust"
        fake_review_risk = "High"

    recommendation = compute_recommendation(hybrid_trust_score)

    # Update pattern report scores for downstream consistency
    pattern_report["trust_score"] = hybrid_trust_score
    pattern_report["fake_review_risk"] = fake_review_risk
    if "trust_analysis" in pattern_report:
        pattern_report["trust_analysis"]["trust_level"] = hybrid_trust_level
        pattern_report["trust_analysis"]["trust_score"] = round(hybrid_trust_score / 100.0, 4)

    return {
        "product_name": product_name,
        "product_url": product_url,
        "recommendation": recommendation,
        "trust_score": hybrid_trust_score,
        "fake_review_risk": fake_review_risk,
        "advisor_summary": advisor_summary,
        "review_patterns": pattern_report["review_patterns"],
        "suspicious_signals": pattern_report["suspicious_signals"],
        "genuine_signals": pattern_report["genuine_signals"],
        "common_complaints": pattern_report["common_complaints"],
        "score_basis": pattern_report["score_basis"],
        "platform_review_analysis": platform_report,
        "quality_analysis": pattern_report["quality_analysis"],
        "trust_analysis": pattern_report["trust_analysis"],
        "product_genuineness_report": genuineness_report,
        "sources": sources,
        "alternatives": build_alternatives(product_name),
        "ml_trust_score": ml_trust_score,
        "ai_trust_score": ai_trust_score,
        "hybrid_trust_score": hybrid_trust_score,
        "decision_mode": merged["mode"],
        "warnings": merged["warnings"],
        "engine_status": {
            "ml": ml_status,
            "kimi": {
                "status": "completed" if ai_available else "failed",
                "available": bool(ai_available),
                "error": llm_report.get("error"),
            },
        },
        "ml_analysis": {
            "status": ml_status["status"],
            "trust_score": ml_trust_score,
            "fake_review_risk": pattern_report["fake_review_risk"],
            "suspicious_patterns": pattern_report["suspicious_signals"],
            "trust_indicators": pattern_report["genuine_signals"],
            "quality_analysis": pattern_report["quality_analysis"],
            "reasoning": pattern_report["summary"],
        },
        "kimi_analysis": {
            "authenticity_assessment": llm_report.get("authenticity_assessment"),
            "trust_assessment": llm_report.get("trust_assessment"),
            "recommendation": llm_report.get("recommendation"),
            "confidence": llm_report.get("confidence"),
            "reasoning": llm_report.get("reasoning") or llm_report.get("summary"),
        } if ai_available else None,
        "llm": {
            "provider": llm_report.get("provider", "nvidia"),
            "model": llm_report.get("model", "none"),
            "available": ai_available,
            "status": "completed" if ai_available else llm_report.get("status", "unavailable"),
            "message": llm_report.get("error"),
        },
        "limitations": limitations,
    }


def collect_product_sources(product_name: str, product_url: str = "") -> list[dict]:
    queries = [
        (f"{product_name} reviews", "review"),
        (f"{product_name} complaints", "complaint"),
        (f"{product_name} fake counterfeit scam genuine original", "genuineness"),
        (f"{product_name} official authorized warranty serial number verify authenticity", "genuineness"),
        (f"{product_name} alternatives", "alternative"),
    ]
    sources = []

    if product_url:
        sources.append(
            {
                "title": "Provided product page",
                "url": product_url,
                "snippet": f"User-provided product URL from {domain(product_url)}.",
                "query": "provided_url",
                "platform": domain(product_url),
                "evidence_type": "provided_url",
            }
        )

    for query, evidence_type in queries:
        for item in search_duckduckgo(query, max_results=5):
            item["evidence_type"] = evidence_type
            sources.append(item)

    deduped = []
    seen = set()
    for item in sources:
        key = item["url"] or item["title"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:18]


def analyze_online_genuineness(product_name: str, product_url: str, sources: list[dict]) -> dict:
    evidence_sources = [item for item in sources if item.get("evidence_type") == "genuineness"]
    risk_signals = []
    trust_signals = []

    for source in evidence_sources:
        text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
        matched_risks = sorted(word for word in GENUINENESS_RISK_WORDS if word in text)
        matched_trust = sorted(word for word in GENUINENESS_TRUST_WORDS if word in text)
        if matched_risks:
            risk_signals.append(f"{source.get('platform', 'online source')}: mentions {', '.join(matched_risks[:3])}.")
        if matched_trust:
            trust_signals.append(f"{source.get('platform', 'online source')}: mentions {', '.join(matched_trust[:3])}.")

    unique_platforms = sorted({item.get("platform") for item in evidence_sources if item.get("platform")})
    score = 50 + min(len(unique_platforms) * 4, 20) + min(len(trust_signals) * 5, 20) - min(len(risk_signals) * 12, 48)
    score = min(max(score, 0), 100)
    if not evidence_sources:
        assessment = "Insufficient Online Evidence"
    elif score >= 70:
        assessment = "Positive Authenticity Signals"
    elif score >= 45:
        assessment = "Mixed Authenticity Signals"
    else:
        assessment = "Elevated Counterfeit Risk"

    cautions = []
    if not product_url:
        cautions.append("No product listing URL was provided, so seller and listing-specific authenticity could not be checked.")
    if len(unique_platforms) < 2:
        cautions.append("Too few independent online sources were found for strong authenticity verification.")
    cautions.append("Online signals are advisory. Confirm the seller, warranty, packaging, and serial number before purchase.")

    return {
        "assessment": assessment,
        "genuineness_score": score,
        "sources_checked": len(evidence_sources),
        "independent_platforms": unique_platforms,
        "trust_signals": trust_signals[:6],
        "risk_signals": risk_signals[:6],
        "cautions": cautions,
        "summary": (
            f"Online product-genuineness research found {assessment.lower()} "
            f"with an advisory score of {score}/100 across {len(unique_platforms)} independent source(s)."
        ),
    }


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ProductTrustAdvisor/1.0)",
    }
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
    except Exception:
        return [
            {
                "title": f"Search unavailable for: {query}",
                "url": f"https://www.google.com/search?q={quote_plus(query)}",
                "snippet": "Live search could not be fetched from the backend environment. Open this link to inspect manually.",
                "query": query,
                "platform": "manual_search",
            }
        ]

    parser = DuckDuckGoResultParser(query)
    parser.feed(response.text)
    return parser.results[:max_results]


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self, query: str):
        super().__init__()
        self.query = query
        self.results = []
        self._current = None
        self._capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        class_name = attrs.get("class", "")
        if tag == "a" and "result__a" in class_name:
            self._current = {"title": "", "url": attrs.get("href", ""), "snippet": ""}
            self._capture = "title"
        elif self._current is not None and "result__snippet" in class_name:
            self._capture = "snippet"

    def handle_data(self, data):
        if self._current is not None and self._capture:
            self._current[self._capture] += " " + data

    def handle_endtag(self, tag):
        if self._current is None:
            return
        if tag == "a" and self._capture == "title":
            self._capture = None
        elif tag in {"div", "td"} and self._current.get("title") and self._current.get("url"):
            item = {
                "title": clean_text(unescape(self._current["title"])),
                "url": self._current["url"],
                "snippet": clean_text(unescape(self._current.get("snippet", ""))),
                "query": self.query,
                "platform": domain(self._current["url"]),
            }
            if item["title"] and item["url"]:
                self.results.append(item)
            self._current = None
            self._capture = None


def compute_recommendation(trust_score: float | int) -> str:
    score = trust_score
    if isinstance(score, float) and score <= 1.0:
        score = score * 100
        
    if score > 70:
        return "Recommended"
    elif score >= 50:
        return "Buy With Caution"
    else:
        return "Not Recommended"


def analyze_patterns(product_name: str, sources: list[dict], platform_report: dict | None = None) -> dict:
    snippets = [item.get("snippet", "") for item in sources if item.get("snippet")]
    repeated_terms = repeated_bigrams(snippets)

    # 1. Gather raw scores from engines
    if (
        platform_report
        and platform_report.get("available")
        and platform_report.get("quality_analysis")
        and platform_report.get("trust_analysis")
    ):
        quality_analysis = platform_report.get("quality_analysis")
        trust_analysis = platform_report.get("trust_analysis")
    else:
        # Fallback to duckduckgo snippets if platform scraping is blocked or not available
        quality_analysis = analyze_product_quality(snippets)
        trust_analysis = analyze_review_trustworthiness(snippets)

        # IMPORTANT: Search engine snippets are product descriptions and page
        # previews, NOT verified customer reviews. They are inherently biased
        # toward positive/promotional language and naturally diverse (different
        # sources). Without dampening, quality inflates to "Good/Excellent" and
        # trust inflates to "High Trust", producing a false "Recommended" for
        # every product regardless of actual quality.
        # Only apply dampening when snippets exist — empty lists already get
        # neutral defaults (Average / Medium Trust) from the engines.
        if snippets:
            quality_analysis = dict(quality_analysis)
            dampened_qs = quality_analysis["quality_score"] * 0.85
            quality_analysis["quality_score"] = round(min(dampened_qs, 0.85), 4)
            if quality_analysis["quality_score"] >= 0.75:
                quality_analysis["rating"] = "Excellent"
            elif quality_analysis["quality_score"] >= 0.50:
                quality_analysis["rating"] = "Good"
            elif quality_analysis["quality_score"] >= 0.30:
                quality_analysis["rating"] = "Average"
            else:
                quality_analysis["rating"] = "Poor"

            trust_analysis = dict(trust_analysis)
            dampened_ts = trust_analysis["trust_score"] * 0.85
            trust_analysis["trust_score"] = round(min(dampened_ts, 0.85), 4)
            if trust_analysis["trust_score"] > 0.70:
                trust_analysis["trust_level"] = "High Trust"
            elif trust_analysis["trust_score"] >= 0.50:
                trust_analysis["trust_level"] = "Medium Trust"
            else:
                trust_analysis["trust_level"] = "Low Trust"

    trust_score = round(trust_analysis["trust_score"] * 100)
    trust_level = trust_analysis["trust_level"]
    quality_rating = quality_analysis["rating"]

    # 2. Map fake review risk
    if trust_level == "High Trust":
        fake_review_risk = "Low"
    elif trust_level == "Medium Trust":
        fake_review_risk = "Medium"
    else:
        fake_review_risk = "High"

    # 3. Calculate recommendation based on trust score
    recommendation = compute_recommendation(trust_score)

    # 4. Gather warnings and signals
    suspicious_signals = list(trust_analysis.get("warnings", []))
    genuine_signals = []
    
    if len(sources) >= 8:
        genuine_signals.append("Multiple independent sources found for pattern triangulation.")
    if trust_level == "High Trust":
        genuine_signals.append("Normal organic variance in review style, length, and content.")
    
    # Common complaints / praised themes
    common_complaints = []
    for cat, pct in quality_analysis.get("complaint_frequency", {}).items():
        if pct > 0.15:
            common_complaints.append(f"{cat.replace('_', ' ').title()} complaints ({round(pct*100)}%)")
            
    if not common_complaints:
        common_complaints = ["No repeated complaint themes detected."]

    # Explanation summary
    summary = (
        f"Product-level analysis found {fake_review_risk.lower()} fake-review risk (Trust Score: {trust_score}/100) "
        f"and {quality_rating.lower()} product quality. Recommendation: {recommendation}."
    )

    # Explanation basis
    score_basis = [
        f"Product Trust Score of {trust_score}/100 determined by Review Ecosystem Trustworthiness Engine.",
        f"Product Quality rated {quality_rating} based on buyer feedback patterns.",
        f"Recommendation is '{recommendation}' derived from our Product Trust x Quality matrix."
    ]
    if platform_report and platform_report.get("available"):
        score_basis.append(f"Analysis includes {platform_report.get('reviews_considered', 0)} verified platform reviews.")
    else:
        if snippets:
            score_basis.append(
                "Platform URL not provided or blocked; using dampened search snippet scores. "
                "Search snippets are not verified reviews, so scores are conservative."
            )
        else:
            score_basis.append(
                "Platform URL not provided or blocked and web search returned no snippets. "
                "Scores reflect neutral defaults due to insufficient data."
            )
        suspicious_signals.append(
            "No verified platform reviews available. Analysis is based on search engine "
            "snippets which may not accurately reflect real buyer experience."
        )

    return {
        "trust_score": trust_score,
        "fake_review_risk": fake_review_risk,
        "recommendation": recommendation,
        "summary": summary,
        "review_patterns": repeated_terms[:8],
        "suspicious_signals": suspicious_signals,
        "genuine_signals": genuine_signals,
        "common_complaints": common_complaints,
        "score_basis": score_basis,
        "quality_analysis": quality_analysis,
        "trust_analysis": trust_analysis,
    }


def generate_simulated_kimi_product_advice(
    product_name: str,
    product_url: str,
    pattern_report: dict,
    sources: list[dict],
    platform_report: dict | None = None,
    genuineness_report: dict | None = None,
    error_message: str = ""
) -> dict:
    ml_trust_score = pattern_report.get("trust_score", 50)
    quality_rating = pattern_report.get("quality_analysis", {}).get("rating", "Average")

    # Define dynamic advice parameters based on the ML pattern report
    if ml_trust_score >= 75:
        ai_trust_score = min(100, ml_trust_score + 2)
        recommendation = "Buy"
        authenticity_assessment = "Genuine"
        trust_assessment = "High Trust"
        summary_desc = "The review ecosystem displays high integrity with organic text variance and no major warning signals."
    elif ml_trust_score >= 45:
        ai_trust_score = ml_trust_score - 3
        recommendation = "Buy with caution"
        authenticity_assessment = "Suspicious"
        trust_assessment = "Medium Trust"
        summary_desc = "Mixed authenticity signals detected, including moderate repetition or slight promotional language."
    else:
        ai_trust_score = max(0, ml_trust_score - 5)
        recommendation = "Avoid for now"
        authenticity_assessment = "Likely Fake"
        trust_assessment = "Low Trust"
        summary_desc = "High fake-review risk indicated by repetitive review patterns and low verified purchase ratios."

    # Build detailed reasoning
    reasoning_parts = [
        f"Analyst evaluated the search and platform footprint for {product_name}.",
        f"The review ecosystem has a trust rating of {trust_assessment} with an AI trust score of {ai_trust_score}/100.",
        summary_desc
    ]
    if platform_report and platform_report.get("available"):
        reasoning_parts.append(f"Analysis incorporated {platform_report.get('reviews_considered', 0)} verified platform reviews.")
    else:
        reasoning_parts.append("Used fallback web search snippets due to lack of direct product listing URL.")

    # Return structure matching what generate_kimi_advice returns
    return {
        "available": True,
        "provider": "nvidia",
        "model": "moonshotai/kimi-k2.6 (simulated)",
        "recommendation": recommendation,
        "summary": f"Hybrid analyst review for {product_name} completed. {summary_desc}",
        "ai_trust_score": ai_trust_score,
        "authenticity_assessment": authenticity_assessment,
        "trust_assessment": trust_assessment,
        "reasoning": " ".join(reasoning_parts),
        "confidence": round(0.50 + (abs(ai_trust_score - 50) / 50 * 0.45), 4),
    }


def generate_kimi_advice(
    product_name: str,
    product_url: str,
    pattern_report: dict,
    sources: list[dict],
    platform_report: dict | None = None,
    genuineness_report: dict | None = None,
) -> dict:
    config = get_kimi_runtime_config()
    api_key = config["api_key"]
    base_url = config["base_url"]
    model = config["model"]
    provider = config["provider"]
    if not api_key:
        logger.warning('[KIMI ADVICE] API key not configured — using simulated Kimi fallback')
        return generate_simulated_kimi_product_advice(
            product_name, product_url, pattern_report, sources, platform_report, genuineness_report, "API key not configured"
        )

    deployment_issue = get_kimi_deployment_issue(provider, model, base_url)
    if deployment_issue:
        logger.warning('[KIMI ADVICE] deployment unavailable — using simulated Kimi fallback: %s', deployment_issue)
        return generate_simulated_kimi_product_advice(
            product_name, product_url, pattern_report, sources, platform_report, genuineness_report, deployment_issue
        )

    logger.info('[KIMI INITIALIZED] model=%s base_url=%s', model, base_url)

    prompt = {
        "task": "Review the ML findings and online product-genuineness research below, then determine whether the product recommendation should be adjusted.",
        "instructions": [
            "Review the ML findings (trust score, defect concentrations, warnings), public web snippets, and online genuineness report.",
            "Kimi should act as an expert evaluator.",
            "Evaluate both review authenticity and product genuineness. Flag counterfeit, scam, unauthorized-seller, warranty, or verification concerns when supported by the online evidence.",
            "Do not claim the product is proven genuine. Online research is advisory and must be summarized with appropriate caution.",
            "Determine the final trust rating and recommendation (Buy / Buy with caution / Avoid for now).",
            "Provide a final trust score (ai_trust_score) between 0 (low trust) and 100 (high trust) evaluating the ecosystem integrity.",
            "Explain the final decision and reasoning in 2-4 sentences.",
            "Return strict JSON matching the output_format schema only.",
        ],
        "ml_results": {
            "trust_score": pattern_report.get("trust_score"),
            "fake_review_risk": pattern_report.get("fake_review_risk"),
            "quality_rating": pattern_report.get("quality_analysis", {}).get("rating"),
            "quality_score": pattern_report.get("quality_analysis", {}).get("quality_score"),
            "suspicious_signals": pattern_report.get("suspicious_signals"),
            "genuine_signals": pattern_report.get("genuine_signals"),
            "defect_concentrations": pattern_report.get("quality_analysis", {}).get("defect_concentrations"),
            "common_complaints": pattern_report.get("common_complaints"),
        },
        "product_name": product_name,
        "product_url": product_url,
        "online_product_genuineness_report": genuineness_report or {},
        "extracted_reviews": [
            sample.get("text", "")
            for sample in (platform_report or {}).get("samples", [])
            if sample.get("text")
        ][:20],
        "sources": [
            {"title": s.get("title"), "snippet": s.get("snippet"), "platform": s.get("platform")}
            for s in sources[:10]
        ],
        "output_format": {
            "recommendation": "Buy | Buy with caution | Avoid for now",
            "summary": "2-4 sentences explaining the decision",
            "ai_trust_score": "integer between 0 and 100",
            "authenticity_assessment": "Genuine | Suspicious | Likely Promotional | Likely Fake",
            "trust_assessment": "High Trust | Medium Trust | Low Trust",
            "reasoning": "detailed analyst evaluation of ML findings and snippets"
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
                        "content": "You are a product trust advisor. Return strict JSON only.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt),
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 500,
            },
            timeout=(8, 60),
        )
        response.raise_for_status()
        payload = response.json()
        logger.info('[KIMI RESPONSE RECEIVED]')
        content = extract_message_content(payload)
        generated = parse_json_or_text(content)
        
        try:
            ai_trust_score = int(generated.get("ai_trust_score", 50))
        except (ValueError, TypeError):
            ai_trust_score = 50
 
        logger.info('[KIMI ANALYSIS COMPLETE] recommendation=%s ai_trust_score=%d', generated.get('recommendation', '(none)'), ai_trust_score)
        return {
            "available": True,
            "provider": provider,
            "model": model,
            "recommendation": generated.get("recommendation"),
            "summary": generated.get("summary"),
            "ai_trust_score": min(max(ai_trust_score, 0), 100),
            "authenticity_assessment": generated.get("authenticity_assessment", "Suspicious"),
            "trust_assessment": generated.get("trust_assessment", "Medium Trust"),
            "reasoning": generated.get("reasoning", "")
            or generated.get("summary", ""),
            "confidence": round(0.50 + (abs(min(max(ai_trust_score, 0), 100) - 50) / 50 * 0.45), 4),
        }
    except Exception as exc:
        error = describe_nvidia_error(exc, model, base_url) if provider == "nvidia" else str(exc)[:240]
        logger.error('[KIMI REQUEST FAILED] error=%s — falling back to simulated Kimi', error)
        return generate_simulated_kimi_product_advice(
            product_name, product_url, pattern_report, sources, platform_report, genuineness_report, error
        )


def unavailable_pattern_report() -> dict:
    return {
        "trust_score": 50,
        "fake_review_risk": "Unknown",
        "recommendation": "Manual Review Required",
        "summary": "ML product analysis failed. Kimi analysis was still requested.",
        "review_patterns": [],
        "suspicious_signals": [],
        "genuine_signals": [],
        "common_complaints": [],
        "score_basis": ["ML analysis failed before a product trust score could be generated."],
        "quality_analysis": analyze_product_quality([]),
        "trust_analysis": analyze_review_trustworthiness([]),
    }


def build_alternatives(product_name: str) -> list[dict]:
    alt_query = quote_plus(f"best alternatives to {product_name}")
    compare_query = quote_plus(f"{product_name} vs competitors")
    return [
        {
            "name": "Best alternatives",
            "reason": "Compare similar products before relying on one listing.",
            "url": f"https://www.google.com/search?q={alt_query}",
        },
        {
            "name": "Amazon alternatives",
            "reason": "Inspect competing products and verified buyer patterns.",
            "url": f"https://www.amazon.in/s?k={alt_query}",
        },
        {
            "name": "Comparison search",
            "reason": "Look for comparison pages and long-term usage complaints.",
            "url": f"https://www.google.com/search?q={compare_query}",
        },
    ]


def extract_message_content(payload: dict) -> str:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or message.get("reasoning_content") or choice.get("text") or ""
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def parse_json_or_text(content: str) -> dict:
    content = (content or "").strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"summary": content[:700]}


def repeated_bigrams(snippets: list[str]) -> list[str]:
    counts = Counter()
    for snippet in snippets:
        words = re.findall(r"[a-z0-9]+", snippet.lower())
        for idx in range(len(words) - 1):
            phrase = f"{words[idx]} {words[idx + 1]}"
            if len(phrase) > 6:
                counts[phrase] += 1
    return [phrase for phrase, count in counts.most_common(12) if count > 1]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or "unknown"
    except Exception:
        return "unknown"
