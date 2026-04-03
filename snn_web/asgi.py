"""
snn_web ASGI configuration.

Handles both HTTP (Django) and WebSocket (Channels) routing.
"""

import os
import django
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "snn_web.settings")
django.setup()

from django.conf import settings  # noqa: E402
from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from channels.security.websocket import AllowedHostsOriginValidator  # noqa: E402
from dashboard.routing import websocket_urlpatterns  # noqa: E402

_http = get_asgi_application()

# In DEBUG mode wrap with the staticfiles handler so daphne serves /static/
# automatically — same behaviour as `runserver`.
if settings.DEBUG:
    from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
    _http = ASGIStaticFilesHandler(_http)

application = ProtocolTypeRouter(
    {
        "http": _http,
        "websocket": AllowedHostsOriginValidator(
            URLRouter(websocket_urlpatterns)
        ),
    }
)
