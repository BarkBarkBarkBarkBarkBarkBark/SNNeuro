"""
dashboard.api — JSON API endpoints.

These endpoints are thin wrappers that forward commands to the asyncio
pipeline server via HTTP-to-WebSocket bridging.  For simple commands we
just open a short-lived WS connection, send the command, read one reply,
and close.
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

PIPELINE_WS = getattr(settings, "PIPELINE_WS_URL", "ws://localhost:8765")
DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


async def _send_and_recv(payload: dict) -> dict:
    """Open a WS connection to the pipeline, send payload, return one reply."""
    import websockets

    try:
        async with websockets.connect(PIPELINE_WS, open_timeout=3) as ws:
            await ws.send(json.dumps(payload))
            reply = await asyncio.wait_for(ws.recv(), timeout=10)
            return json.loads(reply)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _run(coro):
    """Run an async coroutine from a sync Django view."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@csrf_exempt
@require_POST
def launch_source(request) -> JsonResponse:
    """
    POST /api/launch/
    Body: JSON matching WebSocket launch commands understood by app.py
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)

    source_type = body.get("source_type", "synthetic")
    num_channels = int(body.get("num_channels", 1))

    if source_type == "synthetic":
        cmd = {
            "launch_synthetic": {
                "duration_s": float(body.get("synth_duration_s", 20.0)),
                "num_units": int(body.get("synth_num_units", 2)),
                "noise_level": float(body.get("synth_noise_level", 8.0)),
                "num_channels": num_channels,
            }
        }
    elif source_type == "file":
        file_path = body.get("file_path", "")
        if not file_path:
            return JsonResponse({"status": "error", "message": "No file path provided"}, status=400)
        cmd = {"launch_file": file_path}
    else:
        return JsonResponse(
            {"status": "error", "message": f"Source '{source_type}' must be started via CLI (snn-serve --mode {source_type})"},
            status=400,
        )

    result = _run(_send_and_recv(cmd))
    return JsonResponse(result)


@require_GET
def list_files(request) -> JsonResponse:
    """GET /api/files/ — return .ncs files from data/raw/"""
    files: list[str] = []
    if DATA_RAW_DIR.is_dir():
        files = sorted(str(f) for f in DATA_RAW_DIR.glob("*.ncs"))
    return JsonResponse({"files": files, "directory": str(DATA_RAW_DIR)})


@require_GET
def pipeline_status(request) -> JsonResponse:
    """GET /api/status/ — query pipeline mode."""
    result = _run(_send_and_recv({"get_status": True}))
    return JsonResponse(result)
