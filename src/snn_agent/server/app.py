# AGENT-HINT: Asyncio event loop — the runtime heart of the SNN agent.
# PURPOSE: Three input modes (electrode/lsl/synthetic), WebSocket broadcast,
#          HTTP static file serving, UDP control output.
# PIPELINE LOOP: _process_stream() runs build_pipeline() → complete_pipeline()
#   then loops: preprocess → encode → DN + noise_gate → inhibit → L1 → L2 → decode.
# WEBSOCKET COMMANDS: JSON {"key": value} from browser → ws_handler() dispatches.
#   Supported: dn_threshold, l1_stdp_ltp, l1_stdp_ltd, inh_duration_ms,
#   inh_strength_threshold, ng_inhibit_below_sd, decoder_strategy
# CONFIG: Config in config.py (all params). Ports, broadcast_every.
# SEE ALSO: pipeline.py (factory), index.html (GUI), config.py (all params)
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

# ── Runtime mode switching ────────────────────────────────────────────────────
_stream_task: asyncio.Task | None = None
_current_mode: str = "idle"
_base_cfg: Config = DEFAULT_CONFIG

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
                # DN threshold
                if "dn_threshold" in data:
                    dn = pipeline_refs.get("dn")
                    if dn is not None:
                        dn.threshold = float(data["dn_threshold"])
                # L1 STDP params (live tuning)
                if "l1_stdp_ltp" in data:
                    tpl = pipeline_refs.get("template")
                    if tpl is not None:
                        tpl.ltp = float(data["l1_stdp_ltp"])
                if "l1_stdp_ltd" in data:
                    tpl = pipeline_refs.get("template")
                    if tpl is not None:
                        tpl.ltd = float(data["l1_stdp_ltd"])
                # Inhibition params
                if "inh_duration_ms" in data:
                    inh = pipeline_refs.get("inhibitor")
                    if inh is not None:
                        fs = pipeline_refs.get("effective_fs", 20000)
                        inh.blanking_samples = max(1, int(float(data["inh_duration_ms"]) * 1e-3 * fs))
                if "inh_strength_threshold" in data:
                    inh = pipeline_refs.get("inhibitor")
                    if inh is not None:
                        inh.strength_threshold = float(data["inh_strength_threshold"])
                # Noise gate params
                if "ng_inhibit_below_sd" in data:
                    ng = pipeline_refs.get("noise_gate")
                    if ng is not None:
                        ng.inhibit_below_sd = float(data["ng_inhibit_below_sd"])
                # Decoder strategy switch
                if "decoder_strategy" in data:
                    dec = pipeline_refs.get("decoder")
                    if dec is not None:
                        dec.strategy = str(data["decoder_strategy"])
                # TTL pulse width (ms)
                if "ttl_width_ms" in data:
                    dec = pipeline_refs.get("decoder")
                    if dec is not None:
                        fs = pipeline_refs.get("effective_fs", 20000)
                        dec._ttl_width_samples = max(1, int(float(data["ttl_width_ms"]) * 1e-3 * fs))
                # TTL high level (0–1)
                if "ttl_high" in data:
                    dec = pipeline_refs.get("decoder")
                    if dec is not None:
                        dec._ttl_high = float(data["ttl_high"])
                # ── Source launch commands ─────────────────────────
                if "launch_synthetic" in data:
                    params = data["launch_synthetic"]
                    if not isinstance(params, dict):
                        params = {}
                    result = await _launch_mode("synthetic", **params)
                    await websocket.send(json.dumps(result))
                if "launch_file" in data:
                    result = await _launch_mode(
                        "file", file_path=str(data["launch_file"])
                    )
                    await websocket.send(json.dumps(result))
                if "get_status" in data:
                    await websocket.send(json.dumps({
                        "status": "ok", "mode": _current_mode,
                    }))
                if "list_files" in data:
                    raw_dir = (
                        Path(__file__).resolve().parent.parent.parent.parent
                        / "data" / "raw"
                    )
                    files = []
                    if raw_dir.is_dir():
                        files = sorted(
                            str(f) for f in raw_dir.glob("*.ncs")
                        )
                    await websocket.send(json.dumps({
                        "files": files, "directory": str(raw_dir),
                    }))
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
    dec_spike_set: set[int] = set()
    last_noise_gate: float = 1.0
    last_inhibition: bool = False
    l1_membrane_snapshot: list[float] = []
    any_l1_fired_prev: bool = False

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
                pipeline_refs["template"] = pipeline_obj.template
                pipeline_refs["decoder"] = pipeline_obj.decoder
                pipeline_refs["inhibitor"] = pipeline_obj.inhibitor
                pipeline_refs["noise_gate"] = pipeline_obj.noise_gate
                pipeline_refs["effective_fs"] = preproc.effective_fs
                print(
                    f"   ✅ Encoder calibrated: {encoder.n_centers} centres × "
                    f"{encoder.twindow} delays = {encoder.n_afferents} afferents"
                )
                components = ["DN", "L1"]
                if pipeline_obj.inhibitor:
                    components.append("Inhibitor")
                if pipeline_obj.noise_gate:
                    components.append("NoiseGate")
                if pipeline_obj.dec_layer:
                    components.append("DEC(16)")
                print(
                    f"   ✅ Pipeline ready [{' → '.join(components)}] — "
                    f"processing at {preproc.effective_fs} Hz"
                )
                # Threshold reachability diagnostic
                _w_max = float(pipeline_obj.template.W.max())
                _n_est = min(
                    encoder.n_afferents,
                    int(effective_cfg.encoder.overlap * effective_cfg.encoder.window_depth),
                )
                _max_I = _n_est * _w_max + pipeline_obj.template.dn_weight
                _beta = float(np.exp(-1.0 / effective_cfg.l1.tm_samples))
                _v_ss = _max_I / (1.0 - _beta)
                _thr = pipeline_obj.template.threshold
                if _v_ss < _thr * 0.8:
                    print(
                        f"   ⚠ L1 threshold {_thr:.0f} may be unreachable "
                        f"(V_ss≈{_v_ss:.0f}) — consider raising dn_weight"
                    )
                else:
                    print(
                        f"   📊 L1 threshold={_thr:.0f}  V_ss≈{_v_ss:.0f}  "
                        f"headroom={_v_ss/_thr:.1%}"
                    )

            dn_spike = pipeline_obj.attention.step(afferents)
            dn_buf.append(int(dn_spike))

            # Noise gate: Kalman-filter suppression (parallel to DN)
            suppression = 1.0
            if pipeline_obj.noise_gate is not None:
                suppression = pipeline_obj.noise_gate.step(pp_sample)
                last_noise_gate = suppression

            # Global inhibition: post-spike blanking
            if pipeline_obj.inhibitor is not None:
                max_current = pipeline_obj.template.last_current_magnitude
                inh_factor = pipeline_obj.inhibitor.gate(max_current, any_l1_fired_prev)
                suppression *= inh_factor
                last_inhibition = pipeline_obj.inhibitor.active

            l1_spikes = pipeline_obj.template.step(afferents, dn_spike, suppression)
            any_l1_fired_prev = bool(np.any(l1_spikes))
            for idx in np.flatnonzero(l1_spikes):
                l1_spike_set.add(int(idx))

            # Capture membrane state for visualization
            l1_membrane_snapshot = pipeline_obj.template.mem.tolist()

            # Optional DEC spiking decoder layer (16 neurons)
            decoder_input = l1_spikes
            if pipeline_obj.dec_layer is not None:
                dec_spikes = pipeline_obj.dec_layer.step(l1_spikes, dn_spike)
                for idx in np.flatnonzero(dec_spikes):
                    dec_spike_set.add(int(idx))
                decoder_input = dec_spikes

            ctrl_val, confidence = pipeline_obj.decoder.step(
                decoder_input, dn_spike
            )
            last_control = ctrl_val
            last_confidence = confidence
            # Send UDP: hex bitmask if DEC active, else (ctrl, conf) floats
            if pipeline_obj.dec_layer is not None:
                hex_word = pipeline_obj.dec_layer.hex_output
                if hex_word > 0:
                    ctrl_sock.sendto(
                        struct.pack("!H", hex_word), ctrl_addr
                    )
            elif abs(ctrl_val) > 0.05:
                ctrl_sock.sendto(
                    struct.pack("!ff", ctrl_val, confidence), ctrl_addr
                )

            if step_count % broadcast_every == 0 and sample_buf:
                broadcast_msg = {
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
                    "noise_gate": round(last_noise_gate, 4),
                    "inhibition_active": last_inhibition,
                    "l1_membrane": [round(v, 2) for v in l1_membrane_snapshot],
                }
                if pipeline_obj and pipeline_obj.dec_layer is not None:
                    broadcast_msg["dec_spikes"] = sorted(dec_spike_set)
                    broadcast_msg["dec_hex"] = f"0x{pipeline_obj.dec_layer.hex_output:04X}"
                await _broadcast(json.dumps(broadcast_msg))
                sample_buf.clear()
                dn_buf.clear()
                l1_spike_set.clear()
                dec_spike_set.clear()

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


# ── Array source (for file playback) ──────────────────────────────────────────
async def _array_source(traces: np.ndarray, fs: float, name: str = "array"):
    """Yield samples from a pre-loaded numpy array at real-time pace."""
    n = len(traces)
    print(f"   ▶  Streaming {name}: {n} samples at {fs:.0f} Hz ({n/fs:.1f}s)")
    t0 = time.perf_counter()
    for i in range(n):
        yield i, float(traces[i])
        if i % 500 == 0:
            expected = i / fs
            elapsed = time.perf_counter() - t0
            if expected > elapsed:
                await asyncio.sleep(expected - elapsed)
            else:
                await asyncio.sleep(0)
    print(f"   ⏹  Playback complete: {name}")
    await _broadcast(json.dumps(
        {"mode_change": {"mode": "idle", "state": "finished"}}
    ))
    while True:
        await asyncio.sleep(1)


# ── Runtime mode launcher ─────────────────────────────────────────────────────
async def _launch_mode(mode: str, **kwargs) -> dict:
    """Cancel current stream and start a new source. Returns status dict."""
    global _stream_task, _current_mode, pipeline_refs

    # Cancel existing stream
    if _stream_task is not None and not _stream_task.done():
        _stream_task.cancel()
        try:
            await _stream_task
        except (asyncio.CancelledError, Exception):
            pass

    pipeline_refs.clear()
    _current_mode = mode
    await _broadcast(json.dumps(
        {"mode_change": {"mode": mode, "state": "starting"}}
    ))

    cfg = _base_cfg

    try:
        if mode == "synthetic":
            try:
                import spikeinterface  # noqa: F401
            except ImportError:
                _current_mode = "error"
                return {
                    "status": "error",
                    "message": "SpikeInterface not installed. "
                    "Run: uv pip install -e '.[eval]'",
                }
            from dataclasses import replace as _repl

            syn = cfg.synthetic
            if kwargs:
                syn = _repl(
                    syn,
                    duration_s=float(kwargs.get("duration_s", syn.duration_s)),
                    num_units=int(kwargs.get("num_units", syn.num_units)),
                    noise_level=float(
                        kwargs.get("noise_level", syn.noise_level)
                    ),
                )
            cfg = _repl(cfg, synthetic=syn, sampling_rate_hz=syn.fs)
            source = _synthetic_source(cfg)
            _stream_task = asyncio.create_task(
                _process_stream(
                    cfg,
                    source,
                    pace_realtime=syn.realtime,
                    pace_fs=float(syn.fs),
                )
            )
            return {"status": "ok", "mode": "synthetic"}

        elif mode == "file":
            file_path = kwargs.get("file_path", "")
            p = Path(file_path)
            if not p.exists():
                _current_mode = "error"
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}",
                }
            if p.suffix.lower() not in (".ncs",):
                _current_mode = "error"
                return {
                    "status": "error",
                    "message": f"Unsupported format: {p.suffix} (need .ncs)",
                }
            try:
                import mne

                mne.set_log_level("ERROR")
                raw = mne.io.read_raw_neuralynx(
                    str(p.parent), preload=True
                )
                ch = p.stem
                if ch in raw.ch_names:
                    raw.pick([ch])
                elif len(raw.ch_names) > 1:
                    raw.pick([raw.ch_names[0]])
                sfreq = int(raw.info["sfreq"])
                traces = raw.get_data()[0]
            except ImportError:
                _current_mode = "error"
                return {
                    "status": "error",
                    "message": "MNE not installed. "
                    "Run: uv pip install -e '.[lsl]'",
                }
            except Exception as e:
                _current_mode = "error"
                return {
                    "status": "error",
                    "message": f"Read error: {e}",
                }
            cfg = cfg.with_overrides(sampling_rate_hz=sfreq)
            source = _array_source(traces, float(sfreq), p.name)
            _stream_task = asyncio.create_task(
                _process_stream(
                    cfg, source, pace_realtime=True, pace_fs=float(sfreq)
                )
            )
            return {
                "status": "ok",
                "mode": "file",
                "file": p.name,
                "sfreq": sfreq,
                "duration_s": round(len(traces) / sfreq, 1),
            }

        elif mode == "lsl":
            _stream_task = asyncio.create_task(sim_loop_lsl(cfg))
            return {"status": "ok", "mode": "lsl"}

        elif mode == "electrode":
            queue = kwargs.get("queue", asyncio.Queue(maxsize=4096))
            _stream_task = asyncio.create_task(
                sim_loop_electrode(cfg, queue)
            )
            return {"status": "ok", "mode": "electrode"}

        else:
            return {"status": "error", "message": f"Unknown mode: {mode}"}

    except Exception as e:
        _current_mode = "error"
        return {"status": "error", "message": str(e)}


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
    global _base_cfg
    _base_cfg = cfg
    mode = cfg.mode

    _free_port(cfg.ws_port)
    _free_port(cfg.http_port)

    # HTTP static server (daemon thread)
    threading.Thread(target=_run_http, args=(cfg.http_port,), daemon=True).start()

    n_l1 = cfg.l1.n_neurons
    print(f"\n⚡ SNN Agent")
    print(f"   L1 neurons   →  {n_l1}")
    print(f"   Strategy     →  {cfg.decoder.strategy}")
    print(f"   Browser      →  http://localhost:{cfg.http_port}")
    print(f"   WebSocket    →  ws://localhost:{cfg.ws_port}")
    print(f"   Mode         →  {mode}\n")

    # Start initial stream based on configured mode
    if mode == "electrode":
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=4096)
        await loop.create_datagram_endpoint(
            lambda: UDPReceiver(queue, ELEC_FMT, ELEC_FRAME_SIZE),
            local_addr=("0.0.0.0", cfg.udp_electrode_port),
        )
        print(f"   Electrode in →  UDP port {cfg.udp_electrode_port}")
        print(f"   Control out  →  UDP port {cfg.udp_control_port}\n")
        await _launch_mode("electrode", queue=queue)
    elif mode == "lsl":
        print(f"   LSL stream   →  '{cfg.lsl.stream_name}'")
        print(f"   Control out  →  UDP port {cfg.udp_control_port}\n")
        await _launch_mode("lsl")
    else:
        await _launch_mode("synthetic")

    # WebSocket server — runs until shutdown
    async with websockets.serve(
        ws_handler,
        "0.0.0.0",
        cfg.ws_port,
        reuse_address=True,
        reuse_port=True,
    ):
        await asyncio.Future()  # run forever


_BEST_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "best_config.json"


def _load_optimized_config() -> Config | None:
    """Load best_config.json and return a Config built from its parameters.

    Returns ``None`` if the file doesn't exist or can't be parsed.
    """
    if not _BEST_CONFIG_PATH.exists():
        return None
    try:
        with open(_BEST_CONFIG_PATH) as f:
            data = json.load(f)
        params = data.get("parameters", {})
        if not params:
            return None
        # Strip non-config keys that the optimizer stores inside "parameters"
        params = {k: v for k, v in params.items()
                  if k not in ("f_half", "accuracy", "precision", "recall")}
        return Config.from_flat(params)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠  Could not load {_BEST_CONFIG_PATH}: {exc}")
        return None


def main() -> None:
    """CLI entry point (``snn-serve``)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SNN Agent — live spike sorting server"
    )
    parser.add_argument(
        "--mode",
        choices=["electrode", "lsl", "synthetic"],
        default=None,
        help="Input source mode (default: from config)",
    )
    parser.add_argument(
        "--no-optimized",
        action="store_true",
        default=False,
        help="Ignore data/best_config.json and use built-in defaults",
    )
    args = parser.parse_args()

    # ── Build config: optimized params → CLI overrides ────────────────────
    if not args.no_optimized:
        optimized = _load_optimized_config()
    else:
        optimized = None

    if optimized is not None:
        cfg = optimized
        print(f"✓  Loaded optimized hyperparameters from {_BEST_CONFIG_PATH.name}")
    else:
        cfg = DEFAULT_CONFIG
        if not args.no_optimized:
            print("ℹ  No optimized config found — using built-in defaults")

    if args.mode:
        cfg = cfg.with_overrides(mode=args.mode)

    try:
        asyncio.run(_async_main(cfg))
    except KeyboardInterrupt:
        print("\nAgent stopped.")


if __name__ == "__main__":
    main()
