"""snn_web URL configuration."""

from django.urls import path, include

urlpatterns = [
    path("", include("dashboard.urls")),
]
