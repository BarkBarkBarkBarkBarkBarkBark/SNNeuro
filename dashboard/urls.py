"""dashboard URL patterns."""

from django.urls import path
from . import views, api

urlpatterns = [
    # Pages
    path("", views.input_page, name="input"),
    path("monitor/", views.monitor_page, name="monitor"),
    # JSON API
    path("api/launch/", api.launch_source, name="api_launch"),
    path("api/files/", api.list_files, name="api_files"),
    path("api/status/", api.pipeline_status, name="api_status"),
]
