"""
snn_web.management — Entry point for the ``snn-web`` CLI command.

Usage:
    snn-web runserver              # dev server (HTTP only)
    snn-web runserver 0.0.0.0:8000
    snn-web collectstatic --noinput
    daphne snn_web.asgi:application -p 8000   # production ASGI
"""

import os
import sys


def main(args=None):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "snn_web.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Activate the virtualenv: "
            "source .venv/bin/activate"
        ) from exc
    argv = args if args is not None else sys.argv
    execute_from_command_line(argv)


if __name__ == "__main__":
    main()
