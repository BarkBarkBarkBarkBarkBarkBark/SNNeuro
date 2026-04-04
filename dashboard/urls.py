"""dashboard URL patterns."""

from django.urls import path
from . import views, api, docs

urlpatterns = [
    # Pages
    path("", views.input_page, name="input"),
    path("monitor/", views.monitor_page, name="monitor"),
    path("docs/", docs.docs_index, name="docs"),
    path("docs/<slug:slug>/", docs.docs_page, name="docs_page"),
    # JSON API
    path("api/launch/", api.launch_source, name="api_launch"),
    path("api/files/", api.list_files, name="api_files"),
    path("api/status/", api.pipeline_status, name="api_status"),
    path("api/config/", docs.api_config, name="api_config"),
]
