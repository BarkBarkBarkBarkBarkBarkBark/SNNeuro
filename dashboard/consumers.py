"""
dashboard.consumers — Django Channels WebSocket consumer.

Acts as a transparent proxy between the browser and the asyncio pipeline
WebSocket server (snn-serve, ws://localhost:8765).  Every browser tab gets
its own connection; messages are forwarded in both directions.
"""

from __future__ import annotations

import asyncio
import logging

import websockets
import websockets.exceptions
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

logger = logging.getLogger(__name__)

PIPELINE_WS = getattr(settings, "PIPELINE_WS_URL", "ws://localhost:8765")


class StreamConsumer(AsyncWebsocketConsumer):
    """Proxy between a single browser tab and the pipeline WS server."""

    RETRY_DELAY = 2.0   # seconds between reconnect attempts
    MAX_RETRIES = 30     # give up after ~60 s

    async def connect(self) -> None:
        await self.accept()
        self._pipeline_ws: websockets.WebSocketClientProtocol | None = None
        self._reader: asyncio.Task | None = None
        self._closing = False
        await self._open_pipeline_connection()

    async def _open_pipeline_connection(self) -> None:
        """Connect (or reconnect) to the pipeline WS server, with retries."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            if self._closing:
                return
            try:
                self._pipeline_ws = await websockets.connect(
                    PIPELINE_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=5,
                )
                self._reader = asyncio.create_task(self._read_from_pipeline())
                logger.debug("Proxy connected to pipeline at %s", PIPELINE_WS)
                await self.send(
                    text_data='{"status":"ok","message":"Pipeline connected."}'
                )
                return
            except Exception as exc:
                logger.info(
                    "Pipeline WS attempt %d/%d (%s): %s",
                    attempt, self.MAX_RETRIES, PIPELINE_WS, exc,
                )
                if attempt == 1:
                    await self.send(
                        text_data='{"status":"error","message":"Waiting for pipeline server…"}'
                    )
                await asyncio.sleep(self.RETRY_DELAY)

        # Exhausted retries
        await self.send(
            text_data='{"status":"error","message":"Pipeline server not running. Start snn-serve first."}'
        )

    async def _read_from_pipeline(self) -> None:
        """Forward every pipeline message to the browser."""
        try:
            async for msg in self._pipeline_ws:
                await self.send(text_data=msg)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as exc:
            logger.debug("Pipeline reader stopped: %s", exc)
        finally:
            # Notify browser that pipeline disconnected
            try:
                await self.send(
                    text_data='{"status":"error","message":"Pipeline disconnected."}'
                )
            except Exception:
                pass

    async def receive(self, text_data: str | None = None, bytes_data=None) -> None:
        """Forward browser commands to the pipeline."""
        if text_data and self._pipeline_ws:
            try:
                await self._pipeline_ws.send(text_data)
            except websockets.exceptions.ConnectionClosed:
                # Pipeline went away — try reconnecting
                await self._open_pipeline_connection()

    async def disconnect(self, code: int) -> None:
        self._closing = True
        if self._reader:
            self._reader.cancel()
        if self._pipeline_ws:
            try:
                await self._pipeline_ws.close()
            except Exception:
                pass
