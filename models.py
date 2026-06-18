from django.db import models

class CheckedReview(models.Model):
    review_text = models.TextField()
    prediction = models.CharField(max_length=30)  # 'Genuine', 'Suspicious', 'Likely Promotional', 'Likely Fake'
    confidence = models.FloatField()
    media_prediction = models.CharField(max_length=30, blank=True, default='No image')
    media_confidence = models.FloatField(default=0.0)
    media_summary = models.CharField(max_length=300, blank=True, default='')
    product_name = models.CharField(max_length=200, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction} ({self.confidence:.2f}) - {self.review_text[:30]}"
