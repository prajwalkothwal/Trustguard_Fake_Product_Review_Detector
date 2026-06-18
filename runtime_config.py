"""
Runtime configuration helpers.

Reads normal environment variables first, then falls back to a local .env file
so Windows dev runs do not depend on which terminal launched Django.
"""
from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
ENV_FILES = (BACKEND_DIR / ".env", PROJECT_DIR / ".env")


def get_config(name: str, default: str = "", aliases: tuple[str, ...] = ()) -> str:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value:
            return value.strip()

    file_values = load_env_files()
    for key in (name, *aliases):
        value = file_values.get(key)
        if value:
            return value.strip()

    return default


def get_nvidia_kimi_model() -> str:
    model = get_config("NVIDIA_KIMI_MODEL", "moonshotai/kimi-k2.6")
    if model == "moonshotai/kimi-k2-instruct":
        return "moonshotai/kimi-k2.6"
    return model


def get_kimi_runtime_config() -> dict[str, str]:
    moonshot_key = get_config("MOONSHOT_API_KEY")
    if moonshot_key:
        return {
            "provider": "moonshot",
            "api_key": moonshot_key,
            "base_url": get_config("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/"),
            "model": get_config("MOONSHOT_KIMI_MODEL", "kimi-k2.5"),
        }
    return {
        "provider": "nvidia",
        "api_key": get_config("NVIDIA_API_KEY", aliases=("NVIDIA_NIM_API_KEY", "NGC_API_KEY")),
        "base_url": get_config("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
        "model": get_nvidia_kimi_model(),
    }


def get_kimi_deployment_issue(provider: str, model: str, base_url: str) -> str:
    public_nvidia_url = "https://integrate.api.nvidia.com/v1"
    allow_public_attempt = get_config("NVIDIA_ALLOW_UNAVAILABLE_PUBLIC_ENDPOINT").lower() in {"1", "true", "yes"}
    if provider == "nvidia" and model == "moonshotai/kimi-k2.6" and base_url.rstrip("/") == public_nvidia_url and not allow_public_attempt:
        return (
            "NVIDIA Build lists the moonshotai/kimi-k2.6 free endpoint as unavailable. "
            "Configure NVIDIA_BASE_URL with a partner or self-hosted Kimi deployment /v1 URL."
        )
    return ""


def describe_nvidia_error(exc: Exception, model: str, base_url: str) -> str:
    message = str(exc)
    if "Read timed out" in message and base_url == "https://integrate.api.nvidia.com/v1":
        return (
            f"NVIDIA hosted inference timed out for {model}. NVIDIA Build currently lists this Kimi model "
            "as partner/self-hosted only, so configure NVIDIA_BASE_URL for an available deployment."
        )
    if "Bearer " in message:
        return "NVIDIA request failed. Check that the API key, model, and NVIDIA_BASE_URL are valid."
    return message[:240]


def load_env_files() -> dict[str, str]:
    values = {}
    for path in ENV_FILES:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                parsed = parse_env_line(line)
                if parsed:
                    key, value = parsed
                    values.setdefault(key, value)
        except OSError:
            continue
    return values


def parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip().lstrip("\ufeff")
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value
