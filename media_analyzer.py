"""
Lightweight media analysis for review attachments.

This is intentionally heuristic. It looks for media quality, format, metadata,
and consistency signals that can support the text classifier, but it should not
be treated as forensic proof that an image or video is synthetic.
"""
from __future__ import annotations

import math
import os
import pickle
from collections import Counter
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
MAX_ANALYSIS_BYTES = 512 * 1024
MEDIA_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml", "media_model.pkl")
)
MEDIA_FEATURE_NAMES = [
    "is_image",
    "is_video",
    "size_kb",
    "width",
    "height",
    "pixel_count",
    "has_exif",
    "entropy",
    "header_known",
    "extension_matches",
    "low_resolution",
    "extreme_aspect_ratio",
]
_media_model_bundle = None


def _safe_name(upload) -> str:
    return os.path.basename(getattr(upload, "name", "") or "uploaded-media")


def _extension(name: str) -> str:
    return os.path.splitext(name.lower())[1]


def _media_kind(name: str, content_type: str) -> str:
    ext = _extension(name)
    if content_type.startswith("image/") or ext in IMAGE_EXTENSIONS:
        return "image"
    if content_type.startswith("video/") or ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def _read_sample(upload, limit: int = MAX_ANALYSIS_BYTES) -> bytes:
    position = None
    try:
        position = upload.tell()
    except Exception:
        position = None

    try:
        upload.seek(0)
    except Exception:
        pass

    data = upload.read(limit)

    try:
        upload.seek(position or 0)
    except Exception:
        pass

    return data or b""


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _format_from_header(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "webp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if b"ftyp" in data[:32]:
        return "mp4/mov"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm/mkv"
    if data.startswith(b"RIFF") and b"AVI " in data[:16]:
        return "avi"
    return "unknown"


def _load_media_model():
    global _media_model_bundle
    if _media_model_bundle is None:
        if not os.path.exists(MEDIA_MODEL_PATH):
            return None
        with open(MEDIA_MODEL_PATH, "rb") as handle:
            _media_model_bundle = pickle.load(handle)
    return _media_model_bundle


def _feature_vector(kind: str, size: int, details: dict, header_format: str, ext_matches: bool, entropy: float):
    width = float(details.get("width") or 0)
    height = float(details.get("height") or 0)
    pixel_count = width * height
    low_resolution = 1.0 if kind == "image" and pixel_count and pixel_count < 120_000 else 0.0
    extreme_aspect = 0.0
    if width and height and (width / max(height, 1) > 4 or height / max(width, 1) > 4):
        extreme_aspect = 1.0

    return [
        1.0 if kind == "image" else 0.0,
        1.0 if kind == "video" else 0.0,
        float(size) / 1024,
        width,
        height,
        pixel_count,
        1.0 if details.get("has_exif") else 0.0,
        float(entropy),
        1.0 if header_format != "unknown" else 0.0,
        1.0 if ext_matches else 0.0,
        low_resolution,
        extreme_aspect,
    ]


def _model_score(features):
    bundle = _load_media_model()
    if not bundle:
        return None

    classifier = bundle["classifier"]
    labels = bundle.get("labels", ["Looks Authentic", "Suspicious"])
    proba = classifier.predict_proba([features])[0]
    suspicious_index = labels.index("Suspicious") if "Suspicious" in labels else 1
    suspicious_probability = float(proba[suspicious_index])
    return {
        "score": suspicious_probability,
        "verdict": _verdict(suspicious_probability),
        "model_version": bundle.get("version", "unknown"),
        "training_source": bundle.get("training_source", "unknown"),
    }


def _calibrated_media_score(ml_result, heuristic_score):
    if not ml_result:
        return min(heuristic_score, 1.0)

    score = ml_result["score"]
    if ml_result.get("training_source") == "synthetic_baseline":
        # Synthetic baseline models are useful for wiring the pipeline, but they
        # are not reliable enough to make a strong media-authenticity claim.
        score = (score * 0.4) + (min(heuristic_score, 1.0) * 0.6)
        return min(score, 0.49)

    return score


def _extension_matches(ext: str, header_format: str, expected_map: dict) -> bool:
    if header_format == "unknown":
        return False
    expected = expected_map.get(header_format)
    if not expected:
        return True
    return ext in expected


def _image_details(upload):
    try:
        from PIL import Image, ExifTags
    except Exception:
        return {}, ["Image dimensions and EXIF metadata unavailable because Pillow is not installed."]

    notes = []
    details = {}
    position = None
    try:
        position = upload.tell()
    except Exception:
        position = None

    try:
        upload.seek(0)
        with Image.open(upload) as image:
            width, height = image.size
            details.update(
                {
                    "width": width,
                    "height": height,
                    "format": image.format,
                    "mode": image.mode,
                }
            )
            try:
                exif = image.getexif()
                details["has_exif"] = bool(exif)
                if exif:
                    readable = {
                        ExifTags.TAGS.get(tag, tag): value
                        for tag, value in exif.items()
                        if ExifTags.TAGS.get(tag, tag) in {"Make", "Model", "DateTime", "Software"}
                    }
                    details["exif_summary"] = {k: str(v)[:80] for k, v in readable.items()}
            except Exception:
                details["has_exif"] = False
                notes.append("Could not read image EXIF metadata.")
    except Exception:
        notes.append("Image could not be decoded cleanly.")
    finally:
        try:
            upload.seek(position or 0)
        except Exception:
            pass

    return details, notes


def _score_image(upload, name: str, content_type: str, size: int) -> dict:
    sample = _read_sample(upload)
    header_format = _format_from_header(sample)
    entropy = _entropy(sample)
    details, notes = _image_details(upload)
    signals = []
    heuristic_score = 0.0

    ext = _extension(name)
    ext_matches = _extension_matches(
        ext,
        header_format,
        {"jpeg": {".jpg", ".jpeg"}, "png": {".png"}, "webp": {".webp"}, "gif": {".gif"}},
    )
    if header_format == "unknown":
        heuristic_score += 0.22
        signals.append("File header does not match common image signatures.")
    elif ext and header_format not in {"mp4/mov", "webm/mkv", "avi"}:
        if not ext_matches:
            heuristic_score += 0.12
            signals.append(f"File extension ({ext}) does not match detected format ({header_format}).")

    if size < 20 * 1024:
        heuristic_score += 0.18
        signals.append("Image file is very small, which may indicate a thumbnail, repost, or compressed asset.")
    if size > 8 * 1024 * 1024:
        heuristic_score += 0.06
        signals.append("Image file is unusually large for a product review attachment.")

    width = details.get("width")
    height = details.get("height")
    if width and height:
        pixels = width * height
        if pixels < 120_000:
            heuristic_score += 0.16
            signals.append("Image resolution is low for credible product evidence.")
        if width / max(height, 1) > 4 or height / max(width, 1) > 4:
            heuristic_score += 0.12
            signals.append("Image has an extreme aspect ratio that may not be natural review media.")
    elif details == {}:
        heuristic_score += 0.2
        signals.append("Image dimensions could not be verified.")

    if details.get("has_exif") is False and header_format in {"jpeg", "webp"}:
        heuristic_score += 0.08
        signals.append("No camera metadata was found; this is common online, but weakens authenticity evidence.")

    if entropy < 2.5 and sample:
        heuristic_score += 0.14
        signals.append("Image byte pattern has low complexity, suggesting a simple graphic or placeholder.")
    elif entropy > 7.95:
        heuristic_score += 0.08
        signals.append("Image byte pattern is extremely compressed or noisy.")

    signals.extend(notes)
    features = _feature_vector("image", size, details, header_format, ext_matches, entropy)
    ml_result = _model_score(features)
    score = _calibrated_media_score(ml_result, heuristic_score)

    if ml_result:
        if ml_result.get("training_source") == "synthetic_baseline":
            signals.insert(0, "Media model is using baseline training data; treat this media result as advisory until you train it with your own examples.")
        else:
            signals.insert(0, f"Media ML model suspicious probability: {score * 100:.0f}%.")
    else:
        signals.insert(0, "Media ML model not trained yet; using fallback media heuristics.")

    if not signals:
        signals.append("Image format, size, and basic metadata look consistent with normal review media.")

    return {
        "name": name,
        "kind": "image",
        "content_type": content_type or "unknown",
        "size_kb": round(size / 1024, 1),
        "score": round(min(score, 1.0), 4),
        "verdict": _verdict(score),
        "signals": signals,
        "details": {
            **details,
            "detected_format": header_format,
            "entropy": round(entropy, 3),
            "features": dict(zip(MEDIA_FEATURE_NAMES, features)),
            "model": ml_result or {"training_source": "heuristic_fallback"},
            "heuristic_score": round(min(heuristic_score, 1.0), 4),
        },
    }


def _score_video(upload, name: str, content_type: str, size: int) -> dict:
    sample = _read_sample(upload)
    header_format = _format_from_header(sample)
    entropy = _entropy(sample)
    signals = []
    heuristic_score = 0.0
    ext = _extension(name)
    ext_matches = _extension_matches(
        ext,
        header_format,
        {"mp4/mov": {".mp4", ".mov"}, "webm/mkv": {".webm", ".mkv"}, "avi": {".avi"}},
    )

    if header_format == "unknown":
        heuristic_score += 0.24
        signals.append("File header does not match common video container signatures.")
    elif ext:
        if not ext_matches:
            heuristic_score += 0.12
            signals.append(f"File extension ({ext}) does not match detected container ({header_format}).")

    if size < 150 * 1024:
        heuristic_score += 0.2
        signals.append("Video file is very small, which may indicate a clipped, placeholder, or invalid upload.")
    if size > 60 * 1024 * 1024:
        heuristic_score += 0.08
        signals.append("Video file is very large for a review attachment and should be manually checked.")

    if entropy < 2.5 and sample:
        heuristic_score += 0.12
        signals.append("Video byte pattern has low complexity, suggesting a simple placeholder or corrupt file.")

    features = _feature_vector("video", size, {}, header_format, ext_matches, entropy)
    ml_result = _model_score(features)
    score = _calibrated_media_score(ml_result, heuristic_score)

    if ml_result:
        if ml_result.get("training_source") == "synthetic_baseline":
            signals.insert(0, "Media model is using baseline training data; treat this media result as advisory until you train it with your own examples.")
        else:
            signals.insert(0, f"Media ML model suspicious probability: {score * 100:.0f}%.")
    else:
        signals.insert(0, "Media ML model not trained yet; using fallback media heuristics.")

    if not signals:
        signals.append("Video container and file size look plausible for review media.")

    return {
        "name": name,
        "kind": "video",
        "content_type": content_type or "unknown",
        "size_kb": round(size / 1024, 1),
        "score": round(min(score, 1.0), 4),
        "verdict": _verdict(score),
        "signals": signals,
        "details": {
            "detected_container": header_format,
            "entropy": round(entropy, 3),
            "features": dict(zip(MEDIA_FEATURE_NAMES, features)),
            "model": ml_result or {"training_source": "heuristic_fallback"},
            "heuristic_score": round(min(heuristic_score, 1.0), 4),
        },
    }


def _verdict(score: float) -> str:
    if score >= 0.55:
        return "Suspicious"
    if score >= 0.25:
        return "Needs Review"
    return "Looks Authentic"


def analyze_media(files: Iterable) -> dict:
    files = list(files or [])
    if not files:
        return {
            "provided": False,
            "prediction": "Not Provided",
            "confidence": 0.0,
            "score": 0.0,
            "summary": "No images or videos were attached.",
            "evidence": [],
            "files": [],
        }

    analyzed = []
    unsupported = 0
    for upload in files:
        name = _safe_name(upload)
        content_type = getattr(upload, "content_type", "") or ""
        size = getattr(upload, "size", 0) or 0
        kind = _media_kind(name, content_type)
        if kind == "image":
            analyzed.append(_score_image(upload, name, content_type, size))
        elif kind == "video":
            analyzed.append(_score_video(upload, name, content_type, size))
        else:
            unsupported += 1
            analyzed.append(
                {
                    "name": name,
                    "kind": "unknown",
                    "content_type": content_type or "unknown",
                    "size_kb": round(size / 1024, 1),
                    "score": 0.5,
                    "verdict": "Needs Review",
                    "signals": ["Unsupported media type; upload images or videos for better analysis."],
                    "details": {},
                }
            )

    avg_score = sum(item["score"] for item in analyzed) / max(len(analyzed), 1)
    max_score = max((item["score"] for item in analyzed), default=0.0)
    combined_score = min(1.0, (avg_score * 0.65) + (max_score * 0.35) + (0.05 if unsupported else 0.0))
    prediction = _verdict(combined_score)

    evidence = []
    for item in analyzed:
        if item["signals"]:
            evidence.append(f"{item['name']}: {item['signals'][0]}")

    return {
        "provided": True,
        "prediction": prediction,
        "confidence": round(max(combined_score, 1 - combined_score), 4),
        "score": round(combined_score, 4),
        "summary": _summary(prediction, len(analyzed)),
        "evidence": evidence,
        "files": analyzed,
    }


def _summary(prediction: str, count: int) -> str:
    noun = "file" if count == 1 else "files"
    if prediction == "Suspicious":
        return f"{count} media {noun} include signals that raise authenticity concerns."
    if prediction == "Needs Review":
        return f"{count} media {noun} look partly plausible but have signals worth checking."
    return f"{count} media {noun} passed basic format, size, and metadata checks."
