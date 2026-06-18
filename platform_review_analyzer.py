"""
Platform review mix analysis.

Given a product URL, this module fetches the public page, extracts review-like
text snippets and media URLs, scores them with the existing fake-review and
media models, and returns a fake/genuine percentage summary plus a purchase
suggestion.
"""
from __future__ import annotations

from io import BytesIO
import json
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

from .media_analyzer import analyze_media
from .predictor import predict


MAX_PAGE_BYTES = 8_000_000
MIN_REVIEW_TARGET = 50
MAX_REVIEW_TARGET = 100
MEDIA_DOWNLOAD_BYTES = 2 * 1024 * 1024


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            cleaned = clean_text(data)
            if cleaned:
                self.parts.append(cleaned)


class PageArtifactExtractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.text_parts = []
        self.image_items = []
        self._seen_image_urls = set()
        self.review_depth = 0
        self.in_reviews_with_images = False
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return
        context = attrs_to_text(attrs)
        starts_review_context = tag in {"div", "section", "li", "ul", "a"} and is_review_image_context(context)
        if starts_review_context:
            self.review_depth += 1
        if tag != "img":
            return

        for src in image_sources(attrs):
            image_url = urljoin(self.base_url, src)
            if image_url not in self._seen_image_urls and is_user_review_image(
                image_url,
                attrs,
                self.review_depth > 0,
                self.in_reviews_with_images,
            ):
                self._seen_image_urls.add(image_url)
                self.image_items.append(
                    {
                        "url": image_url,
                        "alt": attrs.get("alt", ""),
                        "context": context,
                    }
                )

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if self.review_depth and tag in {"div", "section", "li", "ul"}:
            self.review_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            cleaned = clean_text(data)
            if cleaned:
                lower = cleaned.lower()
                if "reviews with images" in lower or "customer images" in lower:
                    self.in_reviews_with_images = True
                elif self.in_reviews_with_images and (
                    "top reviews" in lower
                    or "review this product" in lower
                    or "customers who viewed" in lower
                    or "product information" in lower
                ):
                    self.in_reviews_with_images = False
                self.text_parts.append(cleaned)


class RemoteMediaFile(BytesIO):
    def __init__(self, data: bytes, name: str, content_type: str, source_url: str, alt: str = ""):
        super().__init__(data)
        self.name = name
        self.content_type = content_type
        self.source_url = source_url
        self.alt = alt
        self.size = len(data)


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


def analyze_platform_reviews(product_url: str, max_reviews: int = MAX_REVIEW_TARGET) -> dict:
    product_url = (product_url or "").strip()
    if not product_url:
        return unavailable("Product URL was not provided.")

    max_reviews = max(1, min(max_reviews, MAX_REVIEW_TARGET))
    platform = domain(product_url)
    try:
        artifacts = fetch_page_artifacts(product_url)
        snippets = extract_review_snippets(artifacts["text"], max_reviews=max_reviews)
        media_result = analyze_remote_media(artifacts["image_items"], product_url)
    except Exception:
        return unavailable("Could not fetch public review text from the product URL.", platform, product_url)

    if not snippets:
        result = unavailable("No review-like text could be extracted from the product page.", platform, product_url)
        result["media_analysis"] = media_result
        return result

    from .authenticity_engine import classify_review
    from .quality_engine import analyze_product_quality
    from .trust_engine import analyze_review_trustworthiness

    # Run Quality and Trust engines on the full snippet set first to identify duplicates
    quality_analysis = analyze_product_quality(snippets)
    trust_analysis = analyze_review_trustworthiness(snippets)
    duplicate_indices = trust_analysis.get("duplicate_indices", set())

    scored = []
    fake_count = 0
    genuine_count = 0
    fake_risk_total = 0.0

    for i, snippet in enumerate(snippets[:max_reviews]):
        result = classify_review(snippet, product_name=platform)
        classification = result["classification"]
        
        # Override classification if the review is part of a duplicate cluster
        if i in duplicate_indices:
            classification = "Likely Fake"
            result["authenticity_score"] = max(result["authenticity_score"], 0.85)
            result["confidence"] = max(result["confidence"], 0.95)

        is_fake = classification in ["Likely Fake", "Likely Promotional"]
        fake_count += 1 if is_fake else 0
        genuine_count += 0 if is_fake else 1
        fake_risk_total += result["authenticity_score"]

        scored.append(
            {
                "text": snippet[:220],
                "prediction": classification,
                "confidence": result["confidence"],
                "fake_probability": round(result["authenticity_score"], 4),
                "sentiment_score": result.get("sentiment_score", 0.0),
            }
        )

    total = max(len(scored), 1)
    fake_percentage = round((fake_count / total) * 100)
    genuine_percentage = 100 - fake_percentage
    average_fake_risk = fake_risk_total / total

    # Compute Recommendation
    suggestion_verdict = compute_recommendation(trust_analysis["trust_score"])
    
    if suggestion_verdict == "Recommended":
        suggestion_message = "This product has excellent quality and the reviews are highly trustworthy. Recommended to purchase!"
    elif suggestion_verdict == "Buy With Caution":
        suggestion_message = "Buyer discretion advised: reviews show either elevated trust anomalies or mixed product quality."
    else:  # Not Recommended
        suggestion_message = "Avoid purchasing: high volume of critical quality issues or strong review manipulation indicators."

    suggestion = {
        "verdict": suggestion_verdict,
        "message": suggestion_message
    }

    coverage = "strong" if total >= MIN_REVIEW_TARGET else "limited"

    return {
        "available": True,
        "platform": platform,
        "source_url": product_url,
        "sample_size": total,
        "reviews_considered": total,
        "fake_count": fake_count,
        "genuine_count": genuine_count,
        "fake_percentage": fake_percentage,
        "genuine_percentage": genuine_percentage,
        "average_fake_risk": round(average_fake_risk, 4),
        "review_sample_target_min": MIN_REVIEW_TARGET,
        "review_sample_target_max": MAX_REVIEW_TARGET,
        "coverage": coverage,
        "media_analysis": media_result,
        "suggestion": suggestion,
        "quality_analysis": quality_analysis,
        "trust_analysis": trust_analysis,
        "positive_count": quality_analysis["sentiment_summary"]["positive"],
        "negative_count": quality_analysis["sentiment_summary"]["negative"],
        "neutral_count": quality_analysis["sentiment_summary"]["neutral"],
        "summary": (
            f"Sampled {total} reviews from {platform}. "
            f"Authenticity mix: {genuine_percentage}% genuine/trustworthy, {fake_percentage}% fake/suspicious. "
            f"Product quality is rated {quality_analysis['rating']} with ecosystem trust at {trust_analysis['trust_level']}."
        ),
        "samples": scored[:5],
    }


def fetch_page_text(url: str) -> str:
    return fetch_page_artifacts(url)["text"]


def fetch_page_artifacts(url: str) -> dict:
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    ]
    best_text = ""
    best_image_items = []
    best_text_parts = []

    for ua in user_agents:
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": ua,
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=12,
                stream=True,
            )
            response.raise_for_status()
            content = response.raw.read(MAX_PAGE_BYTES, decode_content=True)
            encoding = response.encoding or "utf-8"
            html = content.decode(encoding, errors="ignore")
            parser = PageArtifactExtractor(url)
            parser.feed(html)
            candidate_text = clean_text(" ".join(parser.text_parts))
            if len(candidate_text) > len(best_text):
                best_text = candidate_text
                best_text_parts = list(parser.text_parts)
            for item in parser.image_items:
                if item["url"] not in {existing["url"] for existing in best_image_items}:
                    best_image_items.append(item)
        except Exception:
            continue

    if not best_text:
        raise RuntimeError("Could not fetch any page content.")

    for item in fetch_amazon_review_media_items(url):
        if item["url"] not in {existing["url"] for existing in best_image_items}:
            best_image_items.append(item)
    return {
        "text": best_text,
        "image_items": best_image_items,
    }


def image_sources(attrs: dict) -> list[str]:
    sources = []
    for key in ("src", "data-src", "data-original", "data-old-hires"):
        value = attrs.get(key)
        if value:
            sources.append(value)

    for key in ("srcset", "data-srcset"):
        value = attrs.get(key) or ""
        if value:
            sources.extend(part.strip().split(" ")[0] for part in value.split(",") if part.strip())

    dynamic = attrs.get("data-a-dynamic-image")
    if dynamic:
        try:
            decoded = json.loads(unescape(dynamic))
            sources.extend(decoded.keys())
        except Exception:
            sources.extend(re.findall(r'https?://[^"\\]+', unescape(dynamic)))

    return [source for source in sources if source]


def fetch_amazon_review_media_items(product_url: str) -> list[dict]:
    parsed = urlparse(product_url)
    if "amazon." not in parsed.netloc:
        return []

    asin = extract_amazon_asin(product_url)
    if not asin:
        return []

    review_urls = [
        f"{parsed.scheme or 'https'}://{parsed.netloc}/product-reviews/{asin}/?reviewerType=all_reviews&mediaType=media_reviews",
        f"{parsed.scheme or 'https'}://{parsed.netloc}/hz/reviews-render/ajax/medley-filtered-reviews/get/ref=cm_cr_dp_d_fltrs_srt?asin={asin}&reviewerType=all_reviews&mediaType=media_reviews",
    ]

    found = []
    seen = set()
    for review_url in review_urls:
        try:
            response = requests.get(
                review_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": product_url,
                },
                timeout=10,
                stream=True,
            )
            response.raise_for_status()
            content = response.raw.read(MAX_PAGE_BYTES, decode_content=True)
            html = content.decode(response.encoding or "utf-8", errors="ignore")
        except Exception:
            continue

        parser = PageArtifactExtractor(review_url)
        parser.in_reviews_with_images = True
        parser.feed(html)
        for item in parser.image_items:
            if item["url"] not in seen:
                seen.add(item["url"])
                found.append(item)

        for url in extract_review_photo_urls_from_html(html, review_url):
            if url not in seen:
                seen.add(url)
                found.append({"url": url, "alt": "Customer uploaded review image", "context": "amazon-review-media-page"})

    return found


def extract_amazon_asin(url: str) -> str:
    patterns = (
        r"/dp/([A-Z0-9]{10})",
        r"/gp/product/([A-Z0-9]{10})",
        r"/product/([A-Z0-9]{10})",
        r"[?&]ASIN=([A-Z0-9]{10})",
        r"[?&]asin=([A-Z0-9]{10})",
    )
    for pattern in patterns:
        match = re.search(pattern, url, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def extract_review_photo_urls_from_html(html: str, base_url: str) -> list[str]:
    urls = []
    for match in re.finditer(r'https?://[^"\\\s<>]+', html):
        url = unescape(match.group(0))
        context = html[max(0, match.start() - 180): match.end() + 180].lower()
        if "media-amazon.com" not in url.lower() and "images-amazon.com" not in url.lower():
            continue
        if not re.search(r"\.(jpg|jpeg|png|webp)(?:$|[._?])", urlparse(url).path.lower()):
            continue
        if any(token in context for token in ("review-image", "customer-image", "review photo", "customer photo", "media_reviews", "image-tile")):
            urls.append(urljoin(base_url, url))
    return urls


def attrs_to_text(attrs: dict) -> str:
    keys = (
        "id",
        "class",
        "alt",
        "title",
        "data-hook",
        "data-action",
        "aria-label",
        "data-testid",
    )
    return " ".join(str(attrs.get(key, "")) for key in keys).lower()


def is_review_image_context(context: str) -> bool:
    positive = (
        "review",
        "customer",
        "cr-",
        "photo",
        "image-tile",
        "review-image",
        "reviews-medley",
        "customer_image",
        "customer-image",
        "see all photos",
    )
    return any(token in context for token in positive)


def is_user_review_image(url: str, attrs: dict, in_review_context: bool, in_reviews_with_images: bool = False) -> bool:
    path = urlparse(url).path.lower()
    filename = path.rsplit("/", 1)[-1]
    context = attrs_to_text(attrs)
    combined = f"{url.lower()} {context}"

    blocked = (
        "sprite",
        "nav-",
        "logo",
        "megamenu",
        "prime",
        "fresh",
        "smile",
        "favicon",
        "icon",
        "stars",
        "rating",
        "badge",
        "packard",
        "desktop-grid",
        "gateway",
        "advertising",
        "aplus",
        "brand",
        "video",
        "main-image",
        "landing-image",
        "avatar",
        "profile",
        "histogram",
        "swatch",
        "button",
        "carousel-arrow",
        "hero-image",
    )
    if any(token in combined for token in blocked):
        return False

    if not re.search(r"\.(jpg|jpeg|png|webp)(?:$|[._?])", path):
        return False

    explicit_review_photo_tokens = (
        "image-tile",
        "review-image",
        "customer_image",
        "customer-image",
        "reviews-medley",
        "review-photo",
        "customer-photo",
        "reviews-image-gallery",
        "cr-media",
    )
    if in_reviews_with_images:
        return True

    alt = (attrs.get("alt") or "").lower()
    if in_review_context and ("customer image" in alt or "review image" in alt):
        return True

    if in_review_context and any(token in combined for token in explicit_review_photo_tokens):
        return True

    amazon_customer_photo = ("media-amazon.com" in url.lower() or "images-amazon.com" in url.lower()) and re.search(r"_[a-z]{2,3}\d+", filename)
    if amazon_customer_photo and in_review_context:
        return True
    return bool(amazon_customer_photo and ("customer image" in alt or "review image" in alt))


def extract_review_snippets(text: str, max_reviews: int = MAX_REVIEW_TARGET) -> list[str]:
    if not text:
        return []

    chunks = re.split(r"(?<=[.!?])\s+|\s{2,}", text)
    candidates = []
    review_terms = {
        "rating",
        "stars",
        "verified",
        "bought",
        "purchase",
        "quality",
        "delivery",
        "return",
        "refund",
        "worth",
        "product",
        "size",
        "color",
        "colour",
        "comfort",
        "fit",
        "shoe",
        "recommend",
        "price",
        "material",
        "comfortable",
        "durable",
        "shipping",
        "received",
        "arrived",
        "original",
        "genuine",
        "review",
        "customer",
        "ordered",
        "amazon",
        "flipkart",
        "wear",
        "wearing",
        "running",
        "walking",
        "pair",
        "brand",
        "value",
        "money",
        "cheap",
        "expensive",
        "warranty",
        "lightweight",
        "heavy",
        "soft",
        "sole",
        "grip",
        "cushion",
    }

    # Pass 1: strict review-term + review-signal match
    for chunk in chunks:
        chunk = normalize_review_text(chunk)
        words = chunk.split()
        if len(words) < 5 or len(words) > 150:
            continue
        lower = chunk.lower()
        if not any(term in lower for term in review_terms):
            continue
        if not looks_like_customer_review(chunk):
            continue
        if 30 <= len(chunk) <= 800 and chunk not in candidates:
            candidates.append(chunk)
        if len(candidates) >= max_reviews:
            break

    if len(candidates) >= max_reviews:
        return candidates

    # Pass 2: relaxed — only needs review signals, no keyword requirement
    seen = set(candidates)
    for chunk in chunks:
        chunk = normalize_review_text(chunk)
        if chunk in seen:
            continue
        words = chunk.split()
        if len(words) < 5 or len(words) > 150:
            continue
        if not looks_like_customer_review(chunk):
            continue
        if 30 <= len(chunk) <= 800:
            candidates.append(chunk)
            seen.add(chunk)
        if len(candidates) >= max_reviews:
            break

    if len(candidates) >= max_reviews:
        return candidates

    # Pass 3: most lenient — any text that looks like a sentence of reasonable length
    for chunk in chunks:
        chunk = normalize_review_text(chunk)
        if chunk in seen:
            continue
        words = chunk.split()
        if len(words) < 8 or len(words) > 150:
            continue
        if 40 <= len(chunk) <= 600 and _has_sentence_structure(chunk):
            candidates.append(chunk)
            seen.add(chunk)
        if len(candidates) >= max_reviews:
            break

    return candidates


def _has_sentence_structure(text: str) -> bool:
    """Check if text has basic sentence structure (not just navigation/UI text)."""
    lower = text.lower()
    blocked_phrases = {
        "add to cart",
        "buy now",
        "subtotal",
        "free delivery",
        "payment method",
        "ships from",
        "sold by",
        "sponsored",
        "previous set of slides",
        "next set of slides",
        "your transaction is secure",
        "gift options",
        "wish list",
        "customers who viewed",
        "frequently bought together",
        "report an issue",
    }
    if any(phrase in lower for phrase in blocked_phrases):
        return False
    # Must contain at least some lowercase letters (not all caps UI labels)
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    lower_ratio = sum(1 for c in alpha_chars if c.islower()) / len(alpha_chars)
    return lower_ratio > 0.5


def normalize_review_text(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r'\bRead more\b', '', value, flags=re.IGNORECASE)
    value = re.sub(r'\bBrief content visible, double tap to read full content\.?', '', value, flags=re.IGNORECASE)
    value = re.sub(r'\bFull content visible, double tap to read brief content\.?', '', value, flags=re.IGNORECASE)
    return clean_text(value.strip(' "\''))


def looks_like_customer_review(value: str) -> bool:
    lower = value.lower()
    blocked_phrases = {
        "add to cart",
        "buy now",
        "subtotal",
        "free delivery",
        "return policy",
        "payment method",
        "customers who viewed",
        "frequently bought together",
        "to calculate the overall star rating",
        "ai generated from the text",
        "report an issue with this product",
        "gift options",
        "wish list",
        "ships from",
        "sold by",
        "sponsored",
        "previous set of slides",
        "next set of slides",
        "top brand indicates",
        "your transaction is secure",
    }
    if any(phrase in lower for phrase in blocked_phrases):
        return False

    review_signals = {
        "i ",
        " my ",
        "we ",
        " bought",
        "ordered",
        "used",
        "using",
        "quality",
        "battery",
        "sound",
        "comfortable",
        "worth",
        "bad",
        "good",
        "excellent",
        "poor",
        "verified purchase",
        "stars",
        "received",
        "arrived",
        "works",
        "fits",
        "size",
        "color",
        "colour",
        "original",
        "cheap",
        "waste",
        "nice",
        "great",
        "love",
        "hate",
        "happy",
        "disappointed",
        "recommend",
        "pair",
        "shoe",
        "wear",
        "wearing",
        "looking",
        "looks",
        "perfect",
        "awful",
        "terrible",
        "fantastic",
        "horrible",
        "sturdy",
        "flimsy",
        "durable",
        "lightweight",
        "heavy",
        "smooth",
        "rough",
        "value",
        "money",
        "price",
        "running",
        "walking",
        "grip",
        "sole",
        "cushion",
        "soft",
        "tight",
        "loose",
        "snug",
    }
    return any(signal in lower for signal in review_signals)


def analyze_remote_media(image_items: list[dict], page_url: str) -> dict:
    files = []
    attempted = 0
    image_records = []
    selected = list(image_items)

    for item in selected:
        url = item.get("url", "")
        if not url:
            continue
        attempted += 1
        name = urlparse(url).path.rsplit("/", 1)[-1] or f"remote-image-{attempted}.jpg"
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": page_url,
                },
                timeout=8,
                stream=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                image_records.append(
                    {
                        "name": name,
                        "url": url,
                        "alt": item.get("alt", ""),
                        "verdict": "Not analyzed",
                        "score": 0.0,
                        "reason": "The URL did not return image content.",
                    }
                )
                continue
            data = response.raw.read(MEDIA_DOWNLOAD_BYTES, decode_content=True)
            if not data:
                image_records.append(
                    {
                        "name": name,
                        "url": url,
                        "alt": item.get("alt", ""),
                        "verdict": "Not analyzed",
                        "score": 0.0,
                        "reason": "The image could not be downloaded.",
                    }
                )
                continue
            plausible, reason = is_plausible_customer_photo(data)
            if not plausible:
                image_records.append(
                    {
                        "name": name,
                        "url": url,
                        "alt": item.get("alt", ""),
                        "verdict": "Not analyzed",
                        "score": 0.0,
                        "reason": reason,
                    }
                )
                continue
            files.append(RemoteMediaFile(data, name, content_type, url, item.get("alt", "")))
        except Exception:
            image_records.append(
                {
                    "name": name,
                    "url": url,
                    "alt": item.get("alt", ""),
                    "verdict": "Not analyzed",
                    "score": 0.0,
                    "reason": "The image request failed or timed out.",
                }
            )
            continue

    media_result = analyze_media(files)
    analyzed_records = []
    for upload, item in zip(files, media_result.get("files", [])):
        item = calibrate_customer_photo_result(item)
        verdict = media_verdict_label(item.get("verdict", "Needs Review"))
        analyzed_records.append(
            {
                "name": item.get("name") or upload.name,
                "url": upload.source_url,
                "alt": upload.alt,
                "verdict": verdict,
                "raw_verdict": item.get("verdict", ""),
                "score": item.get("score", 0.0),
                "confidence": round(max(item.get("score", 0.0), 1 - item.get("score", 0.0)), 4),
                "signals": item.get("signals", [])[:3],
                "size_kb": item.get("size_kb", 0),
            }
        )

    image_records = analyzed_records + image_records
    genuine_count = sum(1 for item in image_records if item.get("verdict") == "Genuine")
    fake_count = sum(1 for item in image_records if item.get("verdict") == "Fake")
    needs_review_count = sum(1 for item in image_records if item.get("verdict") == "Needs review")
    not_analyzed_count = sum(1 for item in image_records if item.get("verdict") == "Not analyzed")
    analyzed_scores = [item.get("score", 0.0) for item in analyzed_records]
    if analyzed_scores:
        calibrated_avg = sum(analyzed_scores) / len(analyzed_scores)
        calibrated_max = max(analyzed_scores)
        calibrated_combined = min(1.0, (calibrated_avg * 0.65) + (calibrated_max * 0.35))
        media_result["score"] = round(calibrated_combined, 4)
        media_result["confidence"] = round(max(calibrated_combined, 1 - calibrated_combined), 4)
        media_result["prediction"] = media_verdict_label_to_raw(
            "Fake" if calibrated_combined >= 0.55 else "Needs review" if calibrated_combined >= 0.25 else "Genuine"
        )

    media_result["source"] = "product_page_images"
    media_result["images_found"] = len(image_items)
    media_result["images_attempted"] = attempted
    media_result["target_max"] = len(image_items)
    media_result["image_records"] = image_records
    media_result["image_breakdown"] = {
        "genuine": genuine_count,
        "fake": fake_count,
        "needs_review": needs_review_count,
        "not_analyzed": not_analyzed_count,
    }
    if not files and image_items:
        media_result["summary"] = "Review image URLs were found in the page HTML, but the images could not be downloaded for analysis (the platform may block direct image requests)."
    if not image_items:
        media_result["summary"] = "No customer review images were found in the static page HTML. Most e-commerce platforms load review images dynamically via JavaScript, which limits what can be extracted from a static page fetch."
    return media_result


def is_plausible_customer_photo(data: bytes) -> tuple[bool, str]:
    try:
        from PIL import Image
    except Exception:
        return True, ""

    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except Exception:
        return False, "The file could not be decoded as a customer photo."

    if width < 80 or height < 80:
        return False, "Ignored because it is too small and likely an icon or page asset."

    ratio = max(width / max(height, 1), height / max(width, 1))
    if ratio > 3.2:
        return False, "Ignored because its shape looks like a banner or page asset, not a customer photo."

    return True, ""


def calibrate_customer_photo_result(item: dict) -> dict:
    details = item.get("details", {}) or {}
    model = details.get("model", {}) or {}
    heuristic_score = float(details.get("heuristic_score", item.get("score", 0.0)) or 0.0)
    signals = list(item.get("signals", []))
    strong_problem_terms = (
        "file header does not match",
        "extension",
        "could not be decoded",
        "dimensions could not be verified",
        "low complexity",
        "extreme aspect ratio",
    )
    has_strong_problem = any(
        any(term in signal.lower() for term in strong_problem_terms)
        for signal in signals
    )

    # Clean e-commerce review thumbnails should be calibrated as genuine if they lack structural anomalies
    if not has_strong_problem:
        calibrated = dict(item)
        calibrated["score"] = round(min(item.get("score", 0.0), 0.18), 4)
        calibrated["verdict"] = "Looks Authentic"
        calibrated["signals"] = [
            "E-commerce review thumbnail loaded cleanly with no structural anomalies."
        ]
        return calibrated

    if model.get("training_source") != "synthetic_baseline":
        return item

    calibrated = dict(item)
    calibrated["details"] = details
    if has_strong_problem or heuristic_score >= 0.55:
        calibrated["score"] = round(max(heuristic_score, 0.56), 4)
        calibrated["verdict"] = "Suspicious"
        calibrated["signals"] = ["Customer photo has concrete file or quality issues."] + signals[:2]
    elif heuristic_score >= 0.25:
        calibrated["score"] = round(heuristic_score, 4)
        calibrated["verdict"] = "Needs Review"
        calibrated["signals"] = ["Customer photo has minor quality signals; baseline model is advisory only."] + signals[:2]
    else:
        calibrated["score"] = round(heuristic_score, 4)
        calibrated["verdict"] = "Looks Authentic"
        calibrated["signals"] = ["Customer-uploaded review photo looks plausible; baseline model is advisory only."]
    return calibrated


def media_verdict_label(verdict: str) -> str:
    if verdict == "Looks Authentic":
        return "Genuine"
    if verdict == "Suspicious":
        return "Fake"
    if verdict == "Needs Review":
        return "Needs review"
    return verdict or "Not analyzed"


def media_verdict_label_to_raw(verdict: str) -> str:
    if verdict == "Genuine":
        return "Looks Authentic"
    if verdict == "Fake":
        return "Suspicious"
    if verdict == "Needs review":
        return "Needs Review"
    return verdict


def build_purchase_suggestion(fake_percentage: int, average_fake_risk: float, sample_size: int, media_result: dict | None = None) -> dict:
    media_score = (media_result or {}).get("score", 0.0)
    if sample_size < MIN_REVIEW_TARGET:
        verdict = "Not enough platform review data"
        message = f"Only {sample_size} review-like entries were extractable. Open the platform reviews manually before making a purchase decision."
    elif fake_percentage >= 60 or average_fake_risk >= 0.65 or media_score >= 0.55:
        verdict = "Avoid for now"
        message = "The sampled platform reviews or review images show high authenticity risk, so compare alternatives first."
    elif fake_percentage >= 35 or average_fake_risk >= 0.45 or media_score >= 0.25:
        verdict = "Buy with caution"
        message = "The product may be worth considering, but review text or media quality is mixed. Check recent verified reviews and return policy."
    else:
        verdict = "Reasonable to consider"
        message = "The sampled reviews and images look mostly genuine, so the product appears worth considering if price and specifications fit."

    return {
        "verdict": verdict,
        "message": message,
    }


def unavailable(reason: str, platform: str = "", source_url: str = "") -> dict:
    return {
        "available": False,
        "platform": platform,
        "source_url": source_url,
        "sample_size": 0,
        "reviews_considered": 0,
        "fake_count": 0,
        "genuine_count": 0,
        "fake_percentage": 0,
        "genuine_percentage": 0,
        "average_fake_risk": 0.0,
        "review_sample_target_min": MIN_REVIEW_TARGET,
        "review_sample_target_max": MAX_REVIEW_TARGET,
        "coverage": "none",
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "media_analysis": {
            "provided": False,
            "prediction": "Not Provided",
            "confidence": 0.0,
            "score": 0.0,
            "summary": "No review images were analyzed.",
            "evidence": [],
            "files": [],
            "source": "product_page_images",
            "images_found": 0,
            "images_attempted": 0,
            "target_max": 0,
            "image_records": [],
            "image_breakdown": {
                "genuine": 0,
                "fake": 0,
                "needs_review": 0,
                "not_analyzed": 0,
            },
        },
        "suggestion": {
            "verdict": "Manual review needed",
            "message": reason,
        },
        "summary": reason,
        "samples": [],
    }


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value or "")).strip()


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or "provided platform"
    except Exception:
        return "provided platform"
