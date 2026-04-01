"""
snn_web.settings — Django settings for the SNN Agent web dashboard.

Quick-start / dev configuration. Not for production deployment.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security (dev only) ────────────────────────────────────────────────────────
SECRET_KEY = "snn-dev-secret-key-change-in-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]

# ── Applications ──────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "daphne",               # must be first so it handles ASGI
    "django.contrib.staticfiles",
    "channels",
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "snn_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

# ── ASGI / Channels ────────────────────────────────────────────────────────────
ASGI_APPLICATION = "snn_web.asgi.application"

CHANNEL_LAYERS = {
    "default": {
        # In-memory layer — suitable for single-process dev. Swap for Redis in prod.
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

# ── Static files ───────────────────────────────────────────────────────────────
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# ── Pipeline WebSocket ─────────────────────────────────────────────────────────
# The asyncio pipeline server (snn-serve) exposes its WebSocket here.
PIPELINE_WS_URL = "ws://localhost:8765"

# ── Misc ───────────────────────────────────────────────────────────────────────
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
