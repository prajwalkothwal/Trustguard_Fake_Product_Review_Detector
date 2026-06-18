"""reviews/urls.py"""
from django.urls import path
from .views import PredictView, ProductTrustView, AnalyticsView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('product-trust/', ProductTrustView.as_view(), name='product-trust'),
    path('analytics/', AnalyticsView.as_view(), name='analytics'),
]
