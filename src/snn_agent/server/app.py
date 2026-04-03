# AGENT-HINT: Asyncio event loop — the runtime heart of the SNN agent.
# PURPOSE: Three input modes (electrode/lsl/synthetic), WebSocket broadcast,
#          UDP control output.  Django on port 8000 is the browser UI.
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
  2. WS   spike stream to browser  (Django dashboard on port 8000 proxies this)
  3. UDP  control signal output
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import signal
import socket
import struct
import subprocess
import time
from pathlib import Path

# Prevent PyTorch/OpenMP from spawning threads for tiny matmuls.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import websockets

# Fast JSON serialisation (orjson is ~5× faster than stdlib json)
try:
    import orjson
    def _json_dumps(obj: dict) -> str:
        return orjson.dumps(obj).decode("utf-8")
except ImportError:
    def _json_dumps(obj: dict) -> str:  # type: ignore[misc]
        return json.dumps(obj)

from snn_agent.config import Config, DEFAULT_CONFIG
from snn_agent.core.pipeline import build_pipeline, complete_pipeline, build_multichannel

# ── Shared state ──────────────────────────────────────────────────────────────
ws_clients: set = set()
pipeline_refs: dict = {}

# ── Async broadcast queue (decouples serialisation from pipeline hot path) ────
_broadcast_queue: asyncio.Queue | None = None
_broadcast_task: asyncio.Task | None = None


async def _broadcast_sender() -> None:
    """Dedicated coroutine that drains the broadcast queue and sends to clients.

    Runs in the background so the pipeline loop never blocks on WebSocket I/O.
    """
    while True:
        msg = await _broadcast_queue.get()  # type: ignore[union-attr]
        if ws_clients:
            await asyncio.gather(
                *(c.send(msg) for c in ws_clients),
                return_exceptions=True,
            )


def _ensure_broadcast_queue() -> None:
    """Create the queue and sender task if they don't exist yet."""
    global _broadcast_queue, _broadcast_task
    if _broadcast_queue is None:
        _broadcast_queue = asyncio.Queue(maxsize=256)
        _broadcast_task = asyncio.create_task(_broadcast_sender())

# ── Runtime mode switching ────────────────────────────────────────────────────
_stream_tasks: list[asyncio.Task] = []
_current_mode: str = "idle"
_num_channels: int = 1
_base_cfg: Config = DEFAULT_CONFIG

# Electrode frame format: magic(uint16) + sample(float32)
UDP_MAGIC = 0xABCD
ELEC_FMT = "!Hf"
ELEC_FRAME_SIZE = struct.calcsize(ELEC_FMT)

STATIC_DIR = Path(__file__).parent / "static"


# ── Port cleanup ──────────────────────────────────────────────────────────────
def _pids_listening_tcp(port: int) -> set[int]:
    """PIDs that have this TCP port in LISTEN state (exclude current process)."""
    me = os.getpid()
    found: set[int] = set()
    for cmd in (
        ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        ["lsof", "-ti", f":{port}"],
    ):
        try:
            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL, timeout=5
            )
            for s in out.split():
                found.add(int(s))
        except (
            subprocess.CalledProcessError,
            ValueError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pass
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        out = ""
    port_re = re.compile(rf":{port}\b")
    for line in out.splitlines():
        if not port_re.search(line):
            continue
        for m in re.finditer(r"\bpid=(\d+)", line):
            found.add(int(m.group(1)))
    return {p for p in found if p != me}


def _free_port(port: int) -> None:
    """SIGTERM/SIGKILL listeners; on Linux fall back to ``fuser -k``."""
    pids = _pids_listening_tcp(port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(0.45)
    for pid in _pids_listening_tcp(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.12)
    if _pids_listening_tcp(port) and sys.platform.startswith("linux"):
        try:
            subprocess.run(
                ["fuser", "-k", "-SIGKILL", f"{port}/tcp"],
                capture_output=True,
                timeout=8,
                check=False,
            )
        except FileNotFoundError:
            pass
        time.sleep(0.15)



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
                # DN threshold (single-ch or multichannel)
                if "dn_threshold" in data:
                    v = float(data["dn_threshold"])
                    dn = pipeline_refs.get("dn")
                    if dn is not None:
                        dn.threshold = v
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.batched_attention is not None:
                        bank.batched_attention.threshold = v
                # L1 STDP params (single-ch or multichannel)
                if "l1_stdp_ltp" in data:
                    v = float(data["l1_stdp_ltp"])
                    tpl = pipeline_refs.get("template")
                    if tpl is not None:
                        tpl.ltp = v
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.template is not None:
                        bank.template.ltp = v
                if "l1_stdp_ltd" in data:
                    v = float(data["l1_stdp_ltd"])
                    tpl = pipeline_refs.get("template")
                    if tpl is not None:
                        tpl.ltd = v
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.template is not None:
                        bank.template.ltd = v
                # Inhibition params
                if "inh_duration_ms" in data:
                    inh = pipeline_refs.get("inhibitor")
                    if inh is not None:
                        fs = pipeline_refs.get("effective_fs", 20000)
                        inh.blanking_samples = max(1, int(float(data["inh_duration_ms"]) * 1e-3 * fs))
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.inhibitors:
                        fs = pipeline_refs.get("effective_fs", 20000)
                        bs = max(1, int(float(data["inh_duration_ms"]) * 1e-3 * fs))
                        for _inh in bank.inhibitors:
                            if _inh is not None:
                                _inh.blanking_samples = bs
                if "inh_strength_threshold" in data:
                    v = float(data["inh_strength_threshold"])
                    inh = pipeline_refs.get("inhibitor")
                    if inh is not None:
                        inh.strength_threshold = v
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.inhibitors:
                        for _inh in bank.inhibitors:
                            if _inh is not None:
                                _inh.strength_threshold = v
                # Noise gate (single-ch or per-channel multichannel)
                if "ng_inhibit_below_sd" in data:
                    v = float(data["ng_inhibit_below_sd"])
                    ng = pipeline_refs.get("noise_gate")
                    if ng is not None:
                        ng.inhibit_below_sd = v
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.noise_gates:
                        for _ng in bank.noise_gates:
                            if _ng is not None:
                                _ng.inhibit_below_sd = v
                # DEC unit threshold (multichannel: neurons 1-15)
                if "dec_unit_threshold" in data:
                    v = float(data["dec_unit_threshold"])
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.dec_layer is not None:
                        bank.dec_layer.unit_threshold = v
                # DEC DN integration window (ms) — widens gate after each DN spike
                if "dec_dn_window_ms" in data:
                    v = float(data["dec_dn_window_ms"])
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.dec_layer is not None:
                        fs = pipeline_refs.get("effective_fs", 20000)
                        bank.dec_layer._dn_window_samples = max(1, int(v * 1e-3 * fs))
                # Decoder strategy switch
                if "decoder_strategy" in data:
                    dec = pipeline_refs.get("decoder")
                    if dec is not None:
                        dec.strategy = str(data["decoder_strategy"])
                    bank = pipeline_refs.get("bank")
                    if bank is not None and bank.decoders:
                        for _d in bank.decoders:
                            _d.strategy = str(data["decoder_strategy"])
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
                # ── Save / snapshot config ──────────────────────────────
                if "get_config" in data:
                    bank = pipeline_refs.get("bank")
                    snap: dict = {}
                    if bank is not None:
                        if bank.batched_attention is not None:
                            snap["dn_threshold"] = round(bank.batched_attention.threshold, 4)
                        if bank.template is not None:
                            snap["l1_stdp_ltp"] = round(bank.template.ltp, 6)
                            snap["l1_stdp_ltd"] = round(bank.template.ltd, 6)
                        if bank.dec_layer is not None:
                            snap["dec_unit_threshold"] = round(bank.dec_layer.unit_threshold, 4)
                            snap["dec_any_fire_threshold"] = round(bank.dec_layer.any_fire_threshold, 4)
                            fs = pipeline_refs.get("effective_fs", 20000)
                            snap["dec_dn_window_ms"] = round(bank.dec_layer._dn_window_samples / fs * 1000, 3)
                        if bank.inhibitors:
                            _inh0 = next((x for x in bank.inhibitors if x is not None), None)
                            if _inh0:
                                fs = pipeline_refs.get("effective_fs", 20000)
                                snap["inh_duration_ms"] = round(_inh0.blanking_samples / fs * 1000, 2)
                                snap["inh_strength_threshold"] = round(_inh0.strength_threshold, 2)
                        if bank.noise_gates:
                            _ng0 = next((x for x in bank.noise_gates if x is not None), None)
                            if _ng0:
                                snap["ng_inhibit_below_sd"] = round(_ng0.inhibit_below_sd, 3)
                    else:
                        # single-channel path
                        dn = pipeline_refs.get("dn")
                        if dn:
                            snap["dn_threshold"] = round(dn.threshold, 4)
                        tpl = pipeline_refs.get("template")
                        if tpl:
                            snap["l1_stdp_ltp"] = round(tpl.ltp, 6)
                            snap["l1_stdp_ltd"] = round(tpl.ltd, 6)
                        ng = pipeline_refs.get("noise_gate")
                        if ng:
                            snap["ng_inhibit_below_sd"] = round(ng.inhibit_below_sd, 3)
                    await websocket.send(json.dumps({"config_snapshot": snap}))
                # ── Channel selection (multi-channel) ────────────
                if "select_channel" in data:
                    pipeline_refs["selected_ch"] = int(data["select_channel"])
                if "viz_detail" in data:
                    pipeline_refs["viz_detail"] = bool(data["viz_detail"])
                if "network_visible" in data:
                    pipeline_refs["network_visible"] = bool(data["network_visible"])
                # ── Source launch commands ─────────────────────────
                if "launch_synthetic" in data:
                    params = data["launch_synthetic"]
                    if not isinstance(params, dict):
                        params = {}
                    result = await _launch_mode("synthetic", **params)
                    await websocket.send(_json_dumps(result))
                if "launch_file" in data:
                    result = await _launch_mode(
                        "file", file_path=str(data["launch_file"])
                    )
                    await websocket.send(_json_dumps(result))
                if "get_status" in data:
                    await websocket.send(_json_dumps({
                        "status": "ok", "mode": _current_mode,
                        "num_channels": _num_channels,
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
                    await websocket.send(_json_dumps({
                        "files": files, "directory": str(raw_dir),
                    }))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
    except Exception:
        pass
    finally:
        ws_clients.discard(websocket)


async def _broadcast(msg: str) -> None:
    """Enqueue a message for broadcast to all connected WebSocket clients.

    If the queue is full (clients slower than pipeline), the oldest
    message is silently dropped to prevent back-pressure stalling the
    pipeline hot loop.
    """
    _ensure_broadcast_queue()
    q = _broadcast_queue
    assert q is not None
    if q.full():
        try:
            q.get_nowait()  # drop oldest
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:
        pass


# ── Common pipeline step (used by all three modes) ───────────────────────────
async def _process_stream(
    cfg: Config,
    sample_source,
    *,
    channel_idx: int = 0,
    pace_realtime: bool = False,
    pace_fs: float | None = None,
    gt_spike_trains: dict | None = None,
):
    # Always use the GPU-batched multi-channel path (works for C=1 too)
    await _process_stream_multi(
        cfg, sample_source,
        pace_realtime=pace_realtime,
        pace_fs=pace_fs,
        gt_spike_trains=gt_spike_trains,
    )


async def _process_stream_single(
    cfg: Config,
    sample_source,
    *,
    pace_realtime: bool = False,
    pace_fs: float | None = None,
    gt_spike_trains: dict | None = None,
):
    """
    Core streaming pipeline — shared by electrode, LSL, and synthetic modes.

    ``sample_source`` is an async iterable yielding ``(frame_idx, raw_sample)``
    tuples.
    """
    _eff_fs = float(cfg.effective_fs())
    _hz_cap = max(1.0, float(cfg.broadcast_max_hz_mc))
    broadcast_every = max(cfg.broadcast_every, int(_eff_fs / _hz_cap + 0.999))

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

    # ── Live accuracy tracking (synthetic mode with GT) ────────────────
    _has_gt = gt_spike_trains is not None
    if _has_gt:
        _all_gt = np.sort(np.concatenate(list(gt_spike_trains.values())))
        _gt_matched: set[int] = set()
        _gt_ptr = 0
        _native_fs = pace_fs or float(cfg.sampling_rate_hz)
        _delta_samp = int(2.0 * 1e-3 * _native_fs)  # 2 ms tolerance
        _tp = 0
        _fp = 0
        _fn = 0
        _latencies: list[int] = []
        _last_det_frame = -99999
        _det_debounce = int(1.0 * 1e-3 * _native_fs)  # 1 ms refractory

    dec_info = (
        f"  decimation {cfg.sampling_rate_hz}→{preproc.effective_fs} Hz"
        if preproc.do_decimate
        else ""
    )
    print(f"   ⏳ Calibrating encoder (collecting noise statistics)…{dec_info}")

    t0 = time.perf_counter()

    async for frame_idx, raw_sample in sample_source:
        # Yield on every iteration so parallel channel tasks get fair scheduling.
        # Without this, channel 0 hogs the single-threaded event loop and
        # starves channels 1..N.
        await asyncio.sleep(0)

        processed = preproc.step(float(raw_sample))
        if not processed:
            continue

        for pp_sample in processed:
            step_count += 1
            sample_buf.append(round(pp_sample, 6))

            afferents = encoder.step(pp_sample)

            if not encoder.is_calibrated:
                dn_buf.append(0)
                if step_count % broadcast_every == 0 and sample_buf:
                    await _broadcast(_json_dumps({
                        "t": step_count, "samples": list(sample_buf),
                        "dn_flags": list(dn_buf), "spikes": [],
                        "control": 0, "confidence": 0,
                        "channel": channel_idx,
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
                if channel_idx == 0:  # only expose primary channel for live tuning
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
                # Capture membrane state only at broadcast time (not every step)
                l1_membrane_snapshot = pipeline_obj.template.mem.detach().numpy()
                broadcast_msg = {
                    "t": step_count,
                    "channel": channel_idx,
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
                    "l1_membrane": np.round(l1_membrane_snapshot, 2).tolist(),
                }
                if pipeline_refs.get("network_visible", True):
                    broadcast_msg["l1_membrane"] = [round(v, 2) for v in l1_membrane_snapshot]
                if pipeline_obj and pipeline_obj.dec_layer is not None:
                    broadcast_msg["dec_spikes"] = sorted(dec_spike_set)
                    broadcast_msg["dec_hex"] = f"0x{pipeline_obj.dec_layer.hex_output:04X}"
                await _broadcast(_json_dumps(broadcast_msg))
                sample_buf.clear()
                dn_buf.clear()
                l1_spike_set.clear()
                dec_spike_set.clear()

    elapsed_total = time.perf_counter() - t0
    print(
        f"\n   🏁 Finished: {step_count} processed samples "
        f"in {elapsed_total:.1f}s"
    )
    if _has_gt:
        # Count remaining unmatched GT spikes as FN
        while _gt_ptr < len(_all_gt):
            if _gt_ptr not in _gt_matched:
                _fn += 1
            _gt_ptr += 1
        _p = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _r = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        _fh = (1.25 * _p * _r) / (0.25 * _p + _r) if (_p + _r) > 0 else 0.0
        _lat = (np.mean(_latencies) / _native_fs * 1000) if _latencies else 0.0
        print(f"   📊 Final accuracy — P:{_p:.3f}  R:{_r:.3f}  F½:{_fh:.3f}")
        print(f"      TP:{_tp}  FP:{_fp}  FN:{_fn}  Latency:{_lat:.2f}ms")


# ── Multi-channel pipeline step (chunk-based) ────────────────────────────────
async def _process_stream_multi(
    cfg: Config,
    sample_source,
    *,
    pace_realtime: bool = False,
    pace_fs: float | None = None,
    gt_spike_trains: dict | None = None,
):
    """
    Multi-channel streaming pipeline using ChannelBank + batched GPU layers.

    ``sample_source`` yields ``(frame_idx, samples)`` where *samples* is
    an ndarray ``[C]`` (one value per channel).
    """
    C = cfg.n_channels
    broadcast_stride = cfg.multichannel_broadcast_stride()
    chunk_size = cfg.preprocess.decimation_factor * 20 if cfg.preprocess.enable_decimation else 80

    bank, effective_cfg = build_multichannel(cfg)
    pipeline_refs["selected_ch"] = 0
    pipeline_refs["viz_detail"] = True
    pipeline_refs["network_visible"] = True

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = (cfg.control_target_host, cfg.udp_control_port)

    step_count = 0
    sample_buf: list[list[float]] = [[] for _ in range(C)]
    dn_buf: list[list[int]] = [[] for _ in range(C)]
    l1_spike_sets: list[set[int]] = [set() for _ in range(C)]
    dec_spike_sets: list[set[int]] = [set() for _ in range(C)]
    last_controls = np.zeros(C)
    last_confidences = np.zeros(C)
    last_noise_gates = np.ones(C)
    last_inhibitions = np.zeros(C, dtype=bool)
    convergence_snapshot: list[float] = []
    probe_convergence_snapshot: list[list[float]] = []

    # ── Live accuracy tracking (multichannel synthetic GT) ─────────────
    _mc_has_gt = gt_spike_trains is not None
    _mc_native_fs = pace_fs or float(cfg.sampling_rate_hz)
    _mc_delta_samp = int(2.0 * 1e-3 * _mc_native_fs)
    _mc_det_debounce = int(1.0 * 1e-3 * _mc_native_fs)
    _mc_all_gt: np.ndarray = np.empty(0, dtype=np.int64)
    if _mc_has_gt:
        _mc_all_gt = np.sort(np.concatenate(list(gt_spike_trains.values())))
    # Mutable state dict — mutated inside _process_one closure without nonlocal
    _mc_st = {
        'tp': 0, 'fp': 0, 'fn': 0,
        'gt_ptr': 0, 'last_det': -99999, 'frame': 0,
        'latencies': [], 'matched': set(),
    }

    # Chunk accumulator for batched preprocessing
    raw_chunk = np.zeros((chunk_size, C), dtype=np.float64)
    chunk_idx = 0

    dec_info = (
        f"  decimation {cfg.sampling_rate_hz}→{bank.preprocessors[0].effective_fs} Hz"
        if bank.preprocessors[0].do_decimate
        else ""
    )
    print(
        f"   ⏳ Multi-channel ({C}ch) calibrating encoder …{dec_info}"
        f"  device={bank.device}  chunk={chunk_size}"
    )

    t0 = time.perf_counter()

    async for frame_idx, raw_samples in sample_source:
        raw_arr = np.asarray(raw_samples, dtype=np.float64)
        if raw_arr.ndim == 0:
            raw_arr = raw_arr.reshape(1)
        if len(raw_arr) < C:
            padded = np.zeros(C, dtype=np.float64)
            padded[:len(raw_arr)] = raw_arr
            raw_arr = padded

        # Accumulate raw samples into chunk for batched preprocessing
        raw_chunk[chunk_idx] = raw_arr
        chunk_idx += 1
        if chunk_idx < chunk_size:
            if pace_realtime and pace_fs and frame_idx % 500 == 0:
                expected = frame_idx / pace_fs
                elapsed = time.perf_counter() - t0
                if expected > elapsed:
                    await asyncio.sleep(expected - elapsed)
            continue
        chunk_idx = 0

        # Preprocess entire chunk (one sosfilt call per channel)
        decimated_block = bank.step_preprocess_chunk(raw_chunk)
        if len(decimated_block) == 0:
            continue

        # ── Helper: process one full-step result dict ───────────────────────────
        # Defined inside the chunk loop so it closes over all local state.
        async def _process_one(result: dict, sc_i: int) -> None:
            nonlocal convergence_snapshot, probe_convergence_snapshot
            dn_spikes   = result["dn_spikes"]
            l1_spikes   = result["l1_spikes"]
            dec_spikes  = result["dec_spikes"]
            controls    = result["controls"]
            confidences = result["confidences"]
            hex_outputs = result["hex_outputs"]

            l1_cpu = (l1_spikes.cpu().numpy()
                      if isinstance(l1_spikes, torch.Tensor) else l1_spikes)
            dec_cpu = (dec_spikes.cpu().numpy()
                       if isinstance(dec_spikes, torch.Tensor) and dec_spikes is not None
                       else None)

            for ch in range(C):
                dn_buf[ch].append(int(dn_spikes[ch]))
                last_controls[ch]    = controls[ch]
                last_confidences[ch] = confidences[ch]
                last_noise_gates[ch] = result["noise_gates"][ch]
                last_inhibitions[ch] = result["inhibitions"][ch]
                l1_spike_sets[ch].update(np.flatnonzero(l1_cpu[ch]).tolist())
                if dec_cpu is not None:
                    dec_spike_sets[ch].update(np.flatnonzero(dec_cpu[ch]).tolist())

            # ── Live accuracy: match any-channel L1 fire to merged GT ──
            _mc_st['frame'] += 1
            if _mc_has_gt:
                _fi = _mc_st['frame']
                # Expire old GT spikes as FN
                while (_mc_st['gt_ptr'] < len(_mc_all_gt) and
                       _mc_all_gt[_mc_st['gt_ptr']] < _fi - _mc_delta_samp):
                    if _mc_st['gt_ptr'] not in _mc_st['matched']:
                        _mc_st['fn'] += 1
                    _mc_st['gt_ptr'] += 1
                # Any channel fired L1?
                any_ch_fired = bool(np.any(l1_cpu))
                if any_ch_fired and _fi - _mc_st['last_det'] > _mc_det_debounce:
                    _mc_st['last_det'] = _fi
                    best_dist = float('inf')
                    best_gi = -1
                    for gi in range(_mc_st['gt_ptr'], len(_mc_all_gt)):
                        gt_t = int(_mc_all_gt[gi])
                        if gt_t > _fi + _mc_delta_samp:
                            break
                        if gi in _mc_st['matched']:
                            continue
                        dist = abs(_fi - gt_t)
                        if dist <= _mc_delta_samp and dist < best_dist:
                            best_dist = dist
                            best_gi = gi
                    if best_gi >= 0:
                        _mc_st['tp'] += 1
                        _mc_st['matched'].add(best_gi)
                        _mc_st['latencies'].append(best_dist)
                    else:
                        _mc_st['fp'] += 1

            if "convergence_spikes" in result and result["convergence_spikes"] is not None:
                convergence_snapshot = [
                    float(x) for x in result["convergence_spikes"].cpu().tolist()
                ]
            pcs = result.get("probe_convergence_spikes")
            if pcs:
                probe_convergence_snapshot = [
                    [float(x) for x in t.cpu().tolist()] for t in pcs
                ]
            else:
                probe_convergence_snapshot = []

            sc = pipeline_refs.get("selected_ch", 0)
            if bank.dec_layer is not None:
                hw = int(hex_outputs[sc])
                if hw > 0:
                    ctrl_sock.sendto(struct.pack("!H", hw), ctrl_addr)
            elif abs(controls[sc]) > 0.05:
                ctrl_sock.sendto(
                    struct.pack("!ff", float(controls[sc]), float(confidences[sc])),
                    ctrl_addr,
                )

            if sc_i % broadcast_stride == 0 and sample_buf[sc]:
                _n_l1 = cfg.l1.n_neurons
                ch_summary = []
                for ch in range(C):
                    ch_summary.append({
                        "ch": ch,
                        "dn": bool(dn_spikes[ch]),
                        "n_spikes": len(l1_spike_sets[ch]),
                        "control": round(float(last_controls[ch]), 4),
                        "dec_hex": f"0x{int(hex_outputs[ch]):04X}",
                        "spk_hist": _spike_hist(l1_spike_sets[ch], _n_l1),
                        "samples": sample_buf[ch][-30:],
                    })
                ps = max(1, int(cfg.probe_size))
                n_probes_mc = (C + ps - 1) // ps
                cc = cfg.convergence
                viz_detail = pipeline_refs.get("viz_detail", True)
                dec_active_sel = (
                    bank.dec_layer is not None and int(hex_outputs[sc]) != 0
                )
                broadcast_msg = {
                    "t": sc_i,
                    "selected_ch": sc,
                    "n_channels": C,
                    "effective_fs": bank.preprocessors[0].effective_fs,
                    "probe_size": ps,
                    "n_probes": n_probes_mc,
                    "n_local_neurons": cc.n_local_neurons,
                    "n_global_neurons": cc.n_global_neurons,
                    "viz_detail": viz_detail,
                    "samples": list(sample_buf[sc]),
                    "dn_flags": list(dn_buf[sc]),
                    "spikes": sorted(l1_spike_sets[sc]),
                    "control": round(float(last_controls[sc]), 4),
                    "confidence": round(float(last_confidences[sc]), 4),
                    "dn_th": round(bank.batched_attention.threshold, 2) if bank.batched_attention is not None else 0,
                    "noise_gate": round(float(last_noise_gates[sc]), 4),
                    "inhibition_active": bool(last_inhibitions[sc]),
                    "channels": ch_summary,
                    "dn_open_selected": bool(dn_spikes[sc]),
                    "dec_gate_active": dec_active_sel,
                    "dec_unit_threshold": round(bank.dec_layer.unit_threshold, 2) if bank.dec_layer is not None else 16.5,
                    "dec_dn_window_ms": round(bank.dec_layer._dn_window_samples / bank.preprocessors[0].effective_fs * 1000, 2) if bank.dec_layer is not None else 5.0,
                    "n_l1": cfg.l1.n_neurons,
                }
                if viz_detail and pipeline_refs.get("network_visible", True) and bank.template:
                    _mem_row = bank.template.mem[sc].detach().cpu()
                    broadcast_msg["l1_membrane"] = [
                        round(float(v), 2) for v in _mem_row.tolist()
                    ]
                if bank.dec_layer is not None:
                    broadcast_msg["dec_spikes"] = (
                        sorted(dec_spike_sets[sc]) if viz_detail else []
                    )
                    broadcast_msg["dec_hex"] = f"0x{int(hex_outputs[sc]):04X}"
                if viz_detail and convergence_snapshot:
                    broadcast_msg["convergence"] = [
                        round(v, 3) for v in convergence_snapshot
                    ]
                    broadcast_msg["global_convergence"] = [
                        round(v, 3) for v in convergence_snapshot
                    ]
                if viz_detail and probe_convergence_snapshot:
                    broadcast_msg["probe_convergence"] = [
                        [round(v, 3) for v in row] for row in probe_convergence_snapshot
                    ]
                if _mc_has_gt and (_mc_st['tp'] + _mc_st['fp'] + _mc_st['fn']) > 0:
                    _p = _mc_st['tp'] / (_mc_st['tp'] + _mc_st['fp']) if (_mc_st['tp'] + _mc_st['fp']) > 0 else 0.0
                    _r = _mc_st['tp'] / (_mc_st['tp'] + _mc_st['fn']) if (_mc_st['tp'] + _mc_st['fn']) > 0 else 0.0
                    _fh = (1.25 * _p * _r) / (0.25 * _p + _r) if (_p + _r) > 0 else 0.0
                    _lat = (np.mean(_mc_st['latencies']) / _mc_native_fs * 1000) if _mc_st['latencies'] else 0.0
                    broadcast_msg["accuracy"] = {
                        "precision": round(_p, 4),
                        "recall":    round(_r, 4),
                        "f_half":    round(_fh, 4),
                        "tp": _mc_st['tp'], "fp": _mc_st['fp'], "fn": _mc_st['fn'],
                        "latency_ms": round(_lat, 2),
                        "n_gt": len(_mc_all_gt),
                        "gt_progress": round(_mc_st['gt_ptr'] / len(_mc_all_gt), 4) if len(_mc_all_gt) > 0 else 0,
                    }
                await _broadcast(json.dumps(broadcast_msg))
                for ch in range(C):
                    sample_buf[ch].clear()
                    dn_buf[ch].clear()
                    l1_spike_sets[ch].clear()
                    dec_spike_sets[ch].clear()

        # ── Calibration pass: per-row encode until bank is complete ─────────
        dec_n = len(decimated_block)
        block_start = dec_n   # row index where block API takes over; dec_n = "not yet"

        for row_i in range(dec_n):
            dec_row = decimated_block[row_i]
            step_count += 1
            for ch in range(C):
                sample_buf[ch].append(round(float(dec_row[ch]), 6))

            afferents = bank.step_encode_row(dec_row)

            if afferents is None:
                for ch in range(C):
                    dn_buf[ch].append(0)
                sc = pipeline_refs.get("selected_ch", 0)
                _ps = max(1, int(cfg.probe_size))
                if step_count % broadcast_stride == 0 and sample_buf[sc]:
                    await _broadcast(json.dumps({
                        "t": step_count,
                        "n_channels": C,
                        "probe_size": _ps,
                        "n_probes": (C + _ps - 1) // _ps,
                        "effective_fs": bank.preprocessors[0].effective_fs,
                        "samples": list(sample_buf[sc]),
                        "dn_flags": list(dn_buf[sc]),
                        "spikes": [],
                        "control": 0, "confidence": 0,
                        "selected_ch": sc,
                        "dn_open_selected": False,
                        "dec_gate_active": False,
                    }))
                    for ch2 in range(C):
                        sample_buf[ch2].clear()
                        dn_buf[ch2].clear()
                if step_count % 2000 == 0:
                    calib_counts = [enc._sample_count for enc in bank.encoders]
                    print(
                        f"   ⏳ Calibrating… "
                        f"min {min(calib_counts)}/{effective_cfg.encoder.noise_init_samples}"
                    )
                continue

            if not bank._completed:
                bank.complete()
                pipeline_refs["bank"] = bank
                pipeline_refs["effective_fs"] = bank.preprocessors[0].effective_fs
                for ch in range(C):
                    enc = bank.encoders[ch]
                    print(
                        f"   ✅ Ch {ch}: {enc.n_centers} centres × "
                        f"{enc.twindow} delays = {enc.n_afferents} afferents"
                    )
                components = ["DN", "L1"]
                if bank.inhibitors[0]:
                    components.append("Inhibitor")
                if bank.noise_gates[0]:
                    components.append("NoiseGate")
                if bank.dec_layer:
                    components.append("DEC(16)")
                if bank.global_convergence:
                    n_loc = cfg.convergence.n_local_neurons
                    n_glob = bank.global_convergence.n
                    n_pr = len(bank.local_convergence)
                    components.append(f"Conv({n_pr}×local{n_loc}→global{n_glob})")
                eff_fs = bank.preprocessors[0].effective_fs
                bstride = cfg.multichannel_broadcast_stride()
                print(
                    f"   ✅ Pipeline ready [{' → '.join(components)}] × {C}ch — "
                    f"device={bank.device}"
                )
                print(
                    f"   📡 UI WebSocket: every {bstride} sample(s) "
                    f"(~{eff_fs / max(bstride, 1):.1f} frames/s cap, "
                    f"broadcast_max_hz_mc={cfg.broadcast_max_hz_mc})"
                )

            # First calibrated row: encoder already stepped — use single-step path
            result = bank.step_full(afferents, dec_row.tolist())
            await _process_one(result, step_count)
            block_start = row_i + 1   # hand off remaining rows to block API
            break

        # ── Block API: remaining calibrated rows in one Numba call ──────────
        if bank._completed and block_start < dec_n:
            remaining = decimated_block[block_start:]
            N_rem = len(remaining)
            sc_base = step_count   # step count at start of remaining block

            # Accumulate sample_buf for all remaining rows up-front
            for ri in range(N_rem):
                step_count += 1
                dec_row_r = remaining[ri]
                for ch in range(C):
                    sample_buf[ch].append(round(float(dec_row_r[ch]), 6))

            # One Numba call for attention + template; per-row for DEC/decoders
            results_block = bank.step_full_block(remaining)
            for ri, result_r in enumerate(results_block):
                await _process_one(result_r, sc_base + ri + 1)

        if step_count % 20 == 0:
            await asyncio.sleep(0)

    elapsed_total = time.perf_counter() - t0
    print(
        f"\n   🏁 Finished: {step_count} processed samples × {C}ch "
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
    """Yield samples from an LSL stream (single or multi-channel)."""
    from mne_lsl.stream import StreamLSL

    lsl_cfg = cfg.lsl
    C = cfg.n_channels
    print(f"   🔍 Searching for LSL stream '{lsl_cfg.stream_name}' …")

    stream = StreamLSL(bufsize=lsl_cfg.bufsize_sec, name=lsl_cfg.stream_name)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: stream.connect(acquisition_delay=0.001, timeout=60)
    )

    sfreq = stream.info["sfreq"]
    ch_names = list(stream.ch_names)

    # Resolve channel indices
    if C > 1:
        if lsl_cfg.pick_channels:
            ch_indices = [ch_names.index(n) for n in lsl_cfg.pick_channels if n in ch_names]
            if len(ch_indices) < C:
                ch_indices = list(range(min(C, len(ch_names))))
        else:
            ch_indices = list(range(min(C, len(ch_names))))
        picked = [ch_names[i] for i in ch_indices]
        print(
            f"   ✅ Connected: '{lsl_cfg.stream_name}'  |"
            f"  {sfreq:.0f} Hz  |  {len(ch_indices)} channels: {picked}"
        )
    else:
        if lsl_cfg.pick_channel and lsl_cfg.pick_channel in ch_names:
            ch_indices = [ch_names.index(lsl_cfg.pick_channel)]
        else:
            ch_indices = [0]
        print(
            f"   ✅ Connected: '{lsl_cfg.stream_name}'  |"
            f"  {sfreq:.0f} Hz  |  ch: {ch_names[ch_indices[0]]}"
        )

    multi = len(ch_indices) > 1
    frame_idx = 0
    while True:
        n_new = stream.n_new_samples
        if n_new == 0:
            await asyncio.sleep(lsl_cfg.poll_interval_s)
            continue

        data, _ts = stream.get_data(winsize=n_new / sfreq)
        for i in range(data.shape[1]):
            if multi:
                yield frame_idx, data[ch_indices, i].astype(np.float32)
            else:
                yield frame_idx, float(data[ch_indices[0], i])
            frame_idx += 1
        await asyncio.sleep(0)


async def _synthetic_source(cfg: Config):
    """Yield samples from a SpikeInterface synthetic recording."""
    import spikeinterface.full as si

    syn = cfg.synthetic
    C = cfg.n_channels
    print(f"   🧪 Generating synthetic recording …")
    print(f"      Duration : {syn.duration_s} s  |  Fs : {syn.fs} Hz  |  Channels : {C}")
    print(f"      Units    : {syn.num_units}  |  Noise : {syn.noise_level}")

    recording, sorting = si.generate_ground_truth_recording(
        durations=[syn.duration_s],
        sampling_frequency=float(syn.fs),
        num_channels=syn.num_channels,
        num_units=syn.num_units,
        seed=syn.seed,
    )

    for uid in sorting.unit_ids:
        train = sorting.get_unit_spike_train(unit_id=uid)
        rate = len(train) / syn.duration_s
        print(f"      Unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")

    all_traces = recording.get_traces(segment_index=0)  # [T, C]
    if C == 1:
        traces = all_traces[:, 0]
    else:
        traces = all_traces[:, :C]
    n_total = traces.shape[0]
    t0 = time.perf_counter()

    for frame_idx in range(n_total):
        if C == 1:
            yield frame_idx, float(traces[frame_idx])
        else:
            yield frame_idx, traces[frame_idx]

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
async def _array_source(traces: np.ndarray, fs: float, name: str = "array",
                        emit_done: bool = True):
    """Yield samples from a pre-loaded numpy array at real-time pace."""
    n = len(traces)
    print(f"   ▶  Streaming {name}: {n} samples at {fs:.0f} Hz ({n/fs:.1f}s)")
    t0 = time.perf_counter()
    for i in range(n):
        if is_multi:
            yield i, traces[i]  # ndarray [C]
        else:
            yield i, float(traces[i])
        if i % 500 == 0:
            expected = i / fs
            elapsed = time.perf_counter() - t0
            if expected > elapsed:
                await asyncio.sleep(expected - elapsed)
    print(f"   ⏹  Playback complete: {name}")
    if emit_done:
        await _broadcast(_json_dumps(
            {"mode_change": {"mode": "idle", "state": "finished"}}
        ))
    while True:
        await asyncio.sleep(1)


# ── Runtime mode launcher ─────────────────────────────────────────────────────
async def _launch_mode(mode: str, **kwargs) -> dict:
    """Cancel current streams and start a new source. Returns status dict."""
    global _stream_tasks, _current_mode, _num_channels, pipeline_refs

    # Cancel all existing stream tasks
    for _task in _stream_tasks:
        if not _task.done():
            _task.cancel()
            try:
                await _task
            except (asyncio.CancelledError, Exception):
                pass
    _stream_tasks.clear()

    pipeline_refs.clear()
    _current_mode = mode
    await _broadcast(_json_dumps(
        {"mode_change": {"mode": mode, "state": "starting"}}
    ))

    cfg = _base_cfg

    try:
        if mode == "synthetic":
            try:
                import spikeinterface.full as si
            except ImportError:
                _current_mode = "error"
                return {
                    "status": "error",
                    "message": "SpikeInterface not installed. "
                    "Run: uv pip install -e '.[eval]'",
                }
            from dataclasses import replace as _repl
            import spikeinterface.full as si

            syn = cfg.synthetic
            num_channels = int(kwargs.get("num_channels", syn.num_channels))
            syn = _repl(
                syn,
                duration_s=float(kwargs.get("duration_s", syn.duration_s)),
                num_units=int(kwargs.get("num_units", syn.num_units)),
                noise_level=float(kwargs.get("noise_level", syn.noise_level)),
                num_channels=num_channels,
            )
            cfg = _repl(cfg, synthetic=syn, sampling_rate_hz=syn.fs)
            _num_channels = num_channels

            if num_channels > 1:
                # Generate all channels at once, run N parallel pipelines
                recording, _ = si.generate_ground_truth_recording(
                    durations=[syn.duration_s],
                    sampling_frequency=float(syn.fs),
                    num_channels=num_channels,
                    num_units=syn.num_units,
                    seed=syn.seed,
                )
                traces = recording.get_traces(segment_index=0)  # (n_samples, n_ch)
                for ch_idx in range(num_channels):
                    source = _array_source(
                        traces[:, ch_idx], float(syn.fs), f"ch{ch_idx}",
                        emit_done=(ch_idx == 0),
                    )
                    _stream_tasks.append(asyncio.create_task(
                        _process_stream(
                            cfg, source, channel_idx=ch_idx,
                            pace_realtime=syn.realtime, pace_fs=float(syn.fs),
                        )
                    ))
            else:
                source = _synthetic_source(cfg)
                _stream_tasks.append(asyncio.create_task(
                    _process_stream(
                        cfg, source, channel_idx=0,
                        pace_realtime=syn.realtime, pace_fs=float(syn.fs),
                    )
                ))
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
                C = cfg.n_channels
                if C > 1:
                    available = len(raw.ch_names)
                    n_pick = min(C, available)
                    raw.pick(raw.ch_names[:n_pick])
                else:
                    ch = p.stem
                    if ch in raw.ch_names:
                        raw.pick([ch])
                    elif len(raw.ch_names) > 1:
                        raw.pick([raw.ch_names[0]])
                sfreq = int(raw.info["sfreq"])
                data = raw.get_data()  # [n_picked, T]
                if C > 1:
                    traces = data.T  # [T, C]
                else:
                    traces = data[0]  # [T]
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
            _stream_tasks.append(asyncio.create_task(
                _process_stream(
                    cfg, source, channel_idx=0, pace_realtime=True, pace_fs=float(sfreq)
                )
            ))
            return {
                "status": "ok",
                "mode": "file",
                "file": p.name,
                "sfreq": sfreq,
                "duration_s": round(len(traces) / sfreq, 1),
            }

        elif mode == "lsl":
            _stream_tasks.append(asyncio.create_task(sim_loop_lsl(cfg)))
            return {"status": "ok", "mode": "lsl"}

        elif mode == "electrode":
            queue = kwargs.get("queue", asyncio.Queue(maxsize=4096))
            _stream_tasks.append(asyncio.create_task(
                sim_loop_electrode(cfg, queue)
            ))
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

    n_l1 = cfg.l1.n_neurons
    print(f"\n⚡ SNN Agent  (pipeline server)")
    print(f"   L1 neurons   →  {n_l1}")
    print(f"   Channels     →  {cfg.n_channels}")
    print(f"   Device       →  {dev}")
    print(f"   Strategy     →  {cfg.decoder.strategy}")
    print(f"   WebSocket    →  ws://localhost:{cfg.ws_port}")
    print(f"   Mode         →  {mode}")
    print(f"   Dashboard    →  run ./start.sh for the browser UI  (port 8000)\n")

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

        await asyncio.Future()  # run forever


_BEST_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "best_config.json"


def _load_optimized_config(path: Path | None = None) -> "Config | None":
    """Load best_config.json (or *path*) and return a Config built from its parameters.

    Returns ``None`` if the file doesn't exist or can't be parsed.
    """
    target = path if path is not None else _BEST_CONFIG_PATH
    if not target.exists():
        return None
    try:
        with open(target) as f:
            data = json.load(f)
        params = data.get("parameters", {})
        if not params:
            return None
        # Strip non-config keys that the optimizer stores inside "parameters"
        params = {k: v for k, v in params.items()
                  if k not in ("f_half", "accuracy", "precision", "recall")}
        return Config.from_flat(params)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠  Could not load {target}: {exc}")
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
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a config JSON file (overrides --no-optimized)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel synthetic channels (default: 1)",
    )
    args = parser.parse_args()

    # ── Build config: file / optimized params → CLI overrides ────────────────────
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            try:
                with open(cfg_path) as _f:
                    _data = json.load(_f)
                _params = {k: v for k, v in _data.get("parameters", {}).items()
                           if k not in ("f_half", "accuracy", "precision", "recall")}
                cfg = Config.from_flat(_params) if _params else DEFAULT_CONFIG
                print(f"✓  Loaded config from {cfg_path}")
            except Exception as exc:
                print(f"⚠  Could not load {cfg_path}: {exc}")
                cfg = DEFAULT_CONFIG
        else:
            print(f"⚠  Config file not found: {cfg_path}")
            cfg = DEFAULT_CONFIG
    elif not args.no_optimized:
        optimized = _load_optimized_config()
        if optimized is not None:
            cfg = optimized
            print(f"✓  Loaded optimized hyperparameters from {_BEST_CONFIG_PATH.name}")
        else:
            cfg = DEFAULT_CONFIG
            print("ℹ  No optimized config found — using built-in defaults")
    else:
        cfg = DEFAULT_CONFIG

    if args.mode:
        cfg = cfg.with_overrides(mode=args.mode)

    if args.channels is not None:
        from dataclasses import replace as _repl
        _syn = _repl(cfg.synthetic, num_channels=args.channels)
        cfg = _repl(cfg, synthetic=_syn)

    try:
        asyncio.run(_async_main(cfg))
    except KeyboardInterrupt:
        print("\nAgent stopped.")


if __name__ == "__main__":
    main()
