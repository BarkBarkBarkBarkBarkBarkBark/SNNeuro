"""
run_all — start both the SNN pipeline server and the Django web server.

Usage:
    python manage.py run_all
    python manage.py run_all --mode synthetic
    python manage.py run_all --mode synthetic --channels 4 --web-port 8000
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Start snn-serve (pipeline) and daphne (web) in parallel."

    def add_arguments(self, parser):
        parser.add_argument(
            "--mode",
            default="synthetic",
            choices=["synthetic", "electrode", "lsl"],
            help="Pipeline input mode (default: synthetic)",
        )
        parser.add_argument(
            "--channels",
            type=int,
            default=1,
            help="Number of channels for synthetic mode (default: 1)",
        )
        parser.add_argument(
            "--config",
            default=None,
            help="Path to best_config.json (optional)",
        )
        parser.add_argument(
            "--web-port",
            default="8000",
            help="Django web server port (default: 8000)",
        )
        parser.add_argument(
            "--pipeline-port",
            default="8080",
            help="Pipeline HTTP port (default: 8080)",
        )

    def handle(self, *args, **options):
        # Build snn-serve command
        pipeline_cmd = [
            sys.executable, "-m", "snn_agent",
            "--mode", options["mode"],
        ]
        if options["channels"] > 1:
            pipeline_cmd += ["--channels", str(options["channels"])]
        if options["config"]:
            pipeline_cmd += ["--config", options["config"]]

        # Build daphne command
        daphne_cmd = [
            sys.executable, "-m", "daphne",
            "-p", options["web_port"],
            "-b", "0.0.0.0",
            "snn_web.asgi:application",
        ]

        env = os.environ.copy()
        env["DJANGO_SETTINGS_MODULE"] = "snn_web.settings"
        # Ensure both the repo root (snn_web, dashboard) and src/ (snn_agent)
        # are importable regardless of how the command was invoked.
        repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        src_dir = str(Path(repo_root) / "src")
        env["PYTHONPATH"] = f"{repo_root}:{src_dir}:{env.get('PYTHONPATH', '')}"

        self.stdout.write(self.style.SUCCESS("⚡ Starting SNN pipeline…"))
        self.stdout.write(f"   Pipeline: {' '.join(pipeline_cmd)}")
        self.stdout.write(f"   Web:      {' '.join(daphne_cmd)}")
        self.stdout.write(
            f"\n   Browser → http://localhost:{options['web_port']}/\n"
        )

        procs = []
        try:
            procs.append(subprocess.Popen(pipeline_cmd, env=env))
            time.sleep(1.5)  # give pipeline a head start
            procs.append(subprocess.Popen(daphne_cmd, env=env))

            # Wait until interrupted
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            self.stdout.write("\nShutting down…")
        finally:
            for p in procs:
                try:
                    p.send_signal(signal.SIGTERM)
                except Exception:
                    pass
