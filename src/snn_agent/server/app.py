"""
snn_agent.server.app — Asyncio event loop for the SNN agent.

Supports three modes (set in Config):

  ``electrode`` — UDP raw electrode samples in, control signal out.
  ``lsl``       — Lab Streaming Layer input (e.g. NCS replay).
  ``synthetic`` — SpikeInterface ground-truth for benchmarking.

Services:
  1. UDP  electrode input
  2. WS   spike stream to browser
  3. HTTP serves static viz (index.html)
  4. UDP  control signal output
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import struct
import subprocess
import threading
import time
import http.server
from pathlib import Path

import numpy as np
import websockets

from snn_agent.config import Config, DEFAULT_CONFIG
from snn_agent.core.pipeline import build_pipeline, complete_pipeline

# ── Shared state ──────────────────────────────────────────────────────────────
ws_clients: set = set()
pipeline_refs: dict = {}

# Electrode frame format: magic(uint16) + sample(float32)
UDP_MAGIC = 0xABCD
ELEC_FMT = "!Hf"
ELEC_FRAME_SIZE = struct.calcsize(ELEC_FMT)

STATIC_DIR = Path(__file__).parent / "static"


# ── Port cleanup ──────────────────────────────────────────────────────────────
def _free_port(port: int) -> None:
    """Best-effort release of a port (macOS / Linux)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        for pid_str in out.splitlines():
            pid = int(pid_str)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
        time.sleep(0.3)
    except (subprocess.CalledProcessError, OSError, ValueError):
        pass


# ── HTTP static server ───────────────────────────────────────────────────────
class _StaticHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC_DIR), **kw)

    def log_message(self, *_):
        pass


class _ReusableHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True


def _run_http(port: int) -> None:
    _ReusableHTTPServer(("", port), _StaticHandler).serve_forever()


# ── UDP receiver ──────────────────────────────────────────────────────────────
class UDPReceiver(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue, fmt: str, frame_size: int):
        self.queue = queue
        self.fmt = fmt
        self.frame_size = frame_size

    def datagram_received(self, data: bytes, _addr) -> None:
        if len(data) < self.frame_size:
            return
        fields = struct.unpack_from(self.fmt, data)
        if fields[0] != UDP_MAGIC:
            return
        try:
            self.queue.put_nowait(fields[1])
        except asyncio.QueueFull:
            pass


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def ws_handler(websocket) -> None:
    ws_clients.add(websocket)
    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)
                if "dn_threshold" in data:
                    dn = pipeline_refs.get("dn")
                    if dn is not None:
                        dn.threshold = float(data["dn_threshold"])
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
    except Exception:
        pass
    finally:
        ws_clients.discard(websocket)


async def _broadcast(msg: str) -> None:
    if ws_clients:
        await asyncio.gather(
            *(c.send(msg) for c in ws_clients),
            return_exceptions=True,
        )


# ── Common pipeline step (used by all three modes) ───────────────────────────
async def _process_stream(
    cfg: Config,
    sample_source,
    *,
    pace_realtime: bool = False,
    pace_fs: float | None = None,
):
    """
    Core streaming pipeline — shared by electrode, LSL, and synthetic modes.

    ``sample_source`` is an async iterable yielding ``(frame_idx, raw_sample)``
    tuples.
    """
    broadcast_every = cfg.broadcast_every

    preproc, encoder, effective_cfg = build_pipeline(cfg)
    pipeline_obj = None

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = (cfg.control_target_host, cfg.udp_control_port)

    step_count = 0
    last_control = 0.0
    last_confidence = 0.0
    sample_buf: list[float] = []
    dn_buf: list[int] = []
    l1_spike_set: set[int] = set()

    dec_info = (
        f"  decimation {cfg.sampling_rate_hz}→{preproc.effective_fs} Hz"
        if preproc.do_decimate
        else ""
    )
    print(f"   ⏳ Calibrating encoder (collecting noise statistics)…{dec_info}")

    t0 = time.perf_counter()

    async for frame_idx, raw_sample in sample_source:
        processed = preproc.step(float(raw_sample))
        if not processed:
            # Pace even for decimated samples
            if pace_realtime and pace_fs and frame_idx % 500 == 0:
                expected = frame_idx / pace_fs
                elapsed = time.perf_counter() - t0
                if expected > elapsed:
                    await asyncio.sleep(expected - elapsed)
            continue

        for pp_sample in processed:
            step_count += 1
            sample_buf.append(round(pp_sample, 6))

            afferents = encoder.step(pp_sample)

            if not encoder.is_calibrated:
                dn_buf.append(0)
                if step_count % broadcast_every == 0 and sample_buf:
                    await _broadcast(json.dumps({
                        "t": step_count, "samples": list(sample_buf),
                        "dn_flags": list(dn_buf), "spikes": [],
                        "control": 0, "confidence": 0,
                    }))
                    sample_buf.clear()
                    dn_buf.clear()
                if step_count % 2000 == 0:
                    print(
                        f"   ⏳ Calibrating… "
                        f"{step_count}/{effective_cfg.encoder.noise_init_samples}"
                    )
                continue

            # First step after calibration — build downstream stages
            if pipeline_obj is None:
                pipeline_obj = complete_pipeline(cfg, effective_cfg, preproc, encoder)
                pipeline_refs["dn"] = pipeline_obj.attention
                print(
                    f"   ✅ Encoder calibrated: {encoder.n_centers} centres × "
                    f"{encoder.twindow} delays = {encoder.n_afferents} afferents"
                )
                print(
                    f"   ✅ Pipeline ready — processing at "
                    f"{preproc.effective_fs} Hz"
                )

            dn_spike = pipeline_obj.attention.step(afferents)
            dn_buf.append(int(dn_spike))

            l1_spikes = pipeline_obj.template.step(afferents, dn_spike)
            for idx in np.flatnonzero(l1_spikes):
                l1_spike_set.add(int(idx))

            result = pipeline_obj.decoder.step(l1_spikes, dn_spike)
            if result is not None:
                ctrl_val, confidence = result
                last_control = ctrl_val
                last_confidence = confidence
                ctrl_sock.sendto(
                    struct.pack("!ff", ctrl_val, confidence), ctrl_addr
                )

            if step_count % broadcast_every == 0 and sample_buf:
                await _broadcast(json.dumps({
                    "t": step_count,
                    "samples": list(sample_buf),
                    "dn_flags": list(dn_buf),
                    "spikes": sorted(l1_spike_set),
                    "control": round(last_control, 4),
                    "confidence": round(last_confidence, 4),
                    "dn_th": (
                        round(pipeline_obj.attention.threshold, 2)
                        if pipeline_obj
                        else 0
                    ),
                }))
                sample_buf.clear()
                dn_buf.clear()
                l1_spike_set.clear()

        # Yield periodically
        if step_count % 100 == 0:
            await asyncio.sleep(0)

    elapsed_total = time.perf_counter() - t0
    print(
        f"\n   🏁 Finished: {step_count} processed samples "
        f"in {elapsed_total:.1f}s"
    )


# ── Sample sources (async iterables) ─────────────────────────────────────────
async def _electrode_source(queue: asyncio.Queue):
    """Yield samples from the UDP queue."""
    frame_idx = 0
    while True:
        sample = None
        while not queue.empty():
            sample = queue.get_nowait()
        if sample is None:
            await asyncio.sleep(0.0001)
            continue
        yield frame_idx, sample
        frame_idx += 1


async def _lsl_source(cfg: Config):
    """Yield samples from an LSL stream."""
    from mne_lsl.stream import StreamLSL

    lsl_cfg = cfg.lsl
    print(f"   🔍 Searching for LSL stream '{lsl_cfg.stream_name}' …")

    stream = StreamLSL(bufsize=lsl_cfg.bufsize_sec, name=lsl_cfg.stream_name)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: stream.connect(acquisition_delay=0.001, timeout=60)
    )

    sfreq = stream.info["sfreq"]
    ch_names = list(stream.ch_names)

    if lsl_cfg.pick_channel and lsl_cfg.pick_channel in ch_names:
        ch_idx = ch_names.index(lsl_cfg.pick_channel)
    else:
        ch_idx = 0

    print(
        f"   ✅ Connected: '{lsl_cfg.stream_name}'  |"
        f"  {sfreq:.0f} Hz  |  ch: {ch_names[ch_idx]}"
    )

    frame_idx = 0
    while True:
        n_new = stream.n_new_samples
        if n_new == 0:
            await asyncio.sleep(lsl_cfg.poll_interval_s)
            continue

        data, _ts = stream.get_data(winsize=n_new / sfreq)
        for i in range(data.shape[1]):
            yield frame_idx, float(data[ch_idx, i])
            frame_idx += 1
        await asyncio.sleep(0)


async def _synthetic_source(cfg: Config):
    """Yield samples from a SpikeInterface synthetic recording."""
    import spikeinterface.full as si

    syn = cfg.synthetic
    print(f"   🧪 Generating synthetic recording …")
    print(f"      Duration : {syn.duration_s} s  |  Fs : {syn.fs} Hz")
    print(f"      Units    : {syn.num_units}  |  Noise : {syn.noise_level}")

    recording, sorting = si.generate_ground_truth_recording(
        durations=[syn.duration_s],
        sampling_frequency=float(syn.fs),
        num_channels=1,
        num_units=syn.num_units,
        seed=syn.seed,
    )

    for uid in sorting.unit_ids:
        train = sorting.get_unit_spike_train(unit_id=uid)
        rate = len(train) / syn.duration_s
        print(f"      Unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")

    traces = recording.get_traces(segment_index=0)[:, 0]
    n_total = len(traces)
    t0 = time.perf_counter()

    for frame_idx in range(n_total):
        yield frame_idx, float(traces[frame_idx])

        if syn.realtime and frame_idx % 500 == 0:
            expected = frame_idx / syn.fs
            elapsed = time.perf_counter() - t0
            if expected > elapsed:
                await asyncio.sleep(expected - elapsed)
            else:
                await asyncio.sleep(0)
        elif frame_idx % 500 == 0:
            await asyncio.sleep(0)

    # Keep WS alive
    while True:
        await asyncio.sleep(1)


# ── Simulation loops ──────────────────────────────────────────────────────────
async def sim_loop_electrode(cfg: Config, queue: asyncio.Queue) -> None:
    source = _electrode_source(queue)
    await _process_stream(cfg, source)


async def sim_loop_lsl(cfg: Config) -> None:
    lsl_cfg = cfg.lsl

    # Read actual sample rate from LSL stream, override config
    from mne_lsl.stream import StreamLSL

    stream = StreamLSL(bufsize=lsl_cfg.bufsize_sec, name=lsl_cfg.stream_name)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: stream.connect(acquisition_delay=0.001, timeout=60)
    )
    sfreq = int(stream.info["sfreq"])
    stream.disconnect()

    cfg = cfg.with_overrides(sampling_rate_hz=sfreq)
    source = _lsl_source(cfg)
    await _process_stream(cfg, source)


async def sim_loop_synthetic(cfg: Config) -> None:
    cfg = cfg.with_overrides(sampling_rate_hz=cfg.synthetic.fs)
    source = _synthetic_source(cfg)
    await _process_stream(
        cfg, source,
        pace_realtime=cfg.synthetic.realtime,
        pace_fs=float(cfg.synthetic.fs),
    )


# ── Entry point ───────────────────────────────────────────────────────────────
async def _async_main(cfg: Config) -> None:
    mode = cfg.mode
    queue = asyncio.Queue(maxsize=4096)

    _free_port(cfg.ws_port)
    _free_port(cfg.http_port)

    # HTTP static server (daemon thread)
    threading.Thread(target=_run_http, args=(cfg.http_port,), daemon=True).start()

    # WebSocket server
    async def run_ws() -> None:
        async with websockets.serve(
            ws_handler,
            "0.0.0.0",
            cfg.ws_port,
            reuse_address=True,
            reuse_port=True,
        ):
            await asyncio.Future()

    loop = asyncio.get_running_loop()

    n_l1 = cfg.l1.n_neurons
    print(f"\n⚡ SNN Agent  [{mode} mode]")
    print(f"   L1 neurons   →  {n_l1}")
    print(f"   Strategy     →  {cfg.decoder.strategy}")
    print(f"   Browser      →  http://localhost:{cfg.http_port}")
    print(f"   WebSocket    →  ws://localhost:{cfg.ws_port}\n")

    if mode == "synthetic":
        await asyncio.gather(sim_loop_synthetic(cfg), run_ws())
    elif mode == "lsl":
        print(f"   LSL stream   →  '{cfg.lsl.stream_name}'")
        print(f"   Control out  →  UDP port {cfg.udp_control_port}\n")
        await asyncio.gather(sim_loop_lsl(cfg), run_ws())
    else:
        await loop.create_datagram_endpoint(
            lambda: UDPReceiver(queue, ELEC_FMT, ELEC_FRAME_SIZE),
            local_addr=("0.0.0.0", cfg.udp_electrode_port),
        )
        print(f"   Electrode in →  UDP port {cfg.udp_electrode_port}")
        print(f"   Control out  →  UDP port {cfg.udp_control_port}\n")
        await asyncio.gather(sim_loop_electrode(cfg, queue), run_ws())


def main() -> None:
    """CLI entry point (``snn-serve``)."""
    cfg = DEFAULT_CONFIG
    try:
        asyncio.run(_async_main(cfg))
    except KeyboardInterrupt:
        print("\nAgent stopped.")


if __name__ == "__main__":
    main()
