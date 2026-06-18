from django.apps import AppConfig
import logging


class ReviewsConfig(AppConfig):
    name = 'reviews'

    def ready(self):
        from .runtime_config import get_kimi_runtime_config
        api_key = get_kimi_runtime_config()["api_key"]
        if not api_key:
            logging.getLogger("reviews.pipeline").warning(
                "[KIMI UNAVAILABLE] MOONSHOT_API_KEY or NVIDIA_API_KEY is not configured; requests will return ML-only results."
            )
