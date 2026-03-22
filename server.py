"""
server.py — asyncio event loop for the SNN agent.

Supports three modes (set in config.py):

  "electrode" — ANNet-derived mode.  UDP raw electrode samples in,
                UDP control signal out, temporal receptive field encoding,
                attention neuron gating, L1 template STDP.
  "lsl"       — Lab Streaming Layer mode.  Receives data from an LSL stream
                (e.g. Neuralynx .ncs via lsl_player.py).  Same pipeline as
                electrode mode but input comes from LSL instead of UDP.
  "synthetic" — SpikeInterface ground-truth mode.  Generates a synthetic
                recording with known spike times for benchmarking.

Services:
  1. UDP  port <udp_electrode_port> — electrode mode input
  2. WS   port <ws_port>           — streams spikes to browser
  3. HTTP port <http_port>         — serves index.html
  4. UDP  port <udp_control_port>  — control signal output

Run:
  python server.py
"""

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

from config import CFG
from snn import TemplateLayer
from encoder import Preprocessor, SpikeEncoder, AttentionNeuron
from decoder import ControlDecoder


# ── Shared mutable state ──────────────────────────────────────────────────────
ws_clients = set()             # active WebSocket connections
pipeline   = {}                # live refs to pipeline objects (for WS control)

BROADCAST_EVERY = CFG.get("broadcast_every", 5)

# Electrode mode frame format: magic(uint16) + sample(float32)
UDP_MAGIC       = 0xABCD
ELEC_FMT        = "!Hf"
ELEC_FRAME_SIZE = struct.calcsize(ELEC_FMT)


# ── Pre-flight port cleanup ───────────────────────────────────────────────────
def _free_port(port: int) -> None:
    """Best-effort kill of any process holding *port* (macOS / Linux)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        for pid_str in out.splitlines():
            pid = int(pid_str)
            if pid != os.getpid():
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)  # let the OS release the socket
    except (subprocess.CalledProcessError, OSError, ValueError):
        pass  # nothing on that port — fine


# ── HTTP static file server ───────────────────────────────────────────────────
class _StaticHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files from the project directory. Silences access log noise."""
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(Path(__file__).parent), **kw)

    def log_message(self, *_):
        pass  # suppress per-request log lines


class _ReusableHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True


def _run_http(port: int) -> None:
    _ReusableHTTPServer(("", port), _StaticHandler).serve_forever()


# ── UDP input receiver ────────────────────────────────────────────────────────
class UDPReceiver(asyncio.DatagramProtocol):
    """Receives binary-mode frames OR electrode-mode samples (same structure)."""
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
            self.queue.put_nowait(fields[1])  # value (int) or sample (float)
        except asyncio.QueueFull:
            pass  # drop under backpressure


# ── WebSocket handler (bidirectional) ─────────────────────────────────────────
async def ws_handler(websocket) -> None:
    ws_clients.add(websocket)
    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)
                if "dn_threshold" in data:
                    _dn = pipeline.get("dn")
                    if _dn is not None:
                        _dn.threshold = float(data["dn_threshold"])
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


# ── Simulation loop (electrode mode — ANNet-derived) ──────────────────────────
async def sim_loop_electrode(queue: asyncio.Queue) -> None:
    """
    Streaming pipeline:
      sample → SpikeEncoder → AttentionNeuron → TemplateLayer → ControlDecoder
    All layers run in-process, sample-by-sample.  No inter-stage files.
    """
    cfg = CFG

    # Build pipeline components
    preproc = Preprocessor(cfg)
    # Override sampling rate for downstream components if decimating
    enc_cfg = dict(cfg)
    enc_cfg["sampling_rate_hz"] = preproc.effective_fs
    encoder = SpikeEncoder(enc_cfg)
    # AttentionNeuron and TemplateLayer are created AFTER encoder calibration
    dn:      AttentionNeuron | None = None
    l1:      TemplateLayer   | None = None
    decoder: ControlDecoder  | None = None

    # UDP socket for outgoing control signal
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = (cfg["control_target_host"], cfg["udp_control_port"])

    step_count = 0
    last_control = 0.0
    last_confidence = 0.0
    sample_buf   = []       # batched raw samples for browser waveform
    dn_buf       = []       # per-sample DN fire flags (0/1)
    l1_spike_set = set()    # union of L1 neuron IDs across batch

    dec_info = (f"  decimation {cfg['sampling_rate_hz']}→{preproc.effective_fs} Hz"
                if preproc.do_decimate else "")
    print(f"   ⏳ Calibrating encoder (collecting noise statistics)…{dec_info}")

    while True:
        # Drain queue — always use freshest sample
        sample = None
        while not queue.empty():
            sample = queue.get_nowait()

        if sample is None:
            await asyncio.sleep(0.0001)  # 100 µs idle poll
            continue

        # ── Preprocess (bandpass + decimation) ────────────────────────────
        processed = preproc.step(float(sample))
        if not processed:
            continue  # decimated away

        for pp_sample in processed:
            step_count += 1
            sample_buf.append(round(pp_sample, 6))

            # ── Encode ────────────────────────────────────────────────────
            afferents = encoder.step(pp_sample)

            if not encoder.is_calibrated:
                dn_buf.append(0)
                if step_count % BROADCAST_EVERY == 0 and sample_buf:
                    await _broadcast(json.dumps({
                        "t": step_count, "samples": list(sample_buf),
                        "dn_flags": list(dn_buf), "spikes": [],
                        "control": 0, "confidence": 0,
                    }))
                    sample_buf.clear(); dn_buf.clear()
                if step_count % 2000 == 0:
                    print(f"   ⏳ Calibrating… {step_count}/{enc_cfg['enc_noise_init_samples']}")
                continue

            # First step after calibration: build downstream components
            if dn is None:
                n_aff = encoder.n_afferents
                dn      = AttentionNeuron(enc_cfg, n_aff)
                pipeline["dn"] = dn
                l1 = TemplateLayer(cfg, n_aff)
                decoder = ControlDecoder(enc_cfg, cfg["l1_n_neurons"])
                print(f"   ✅ Encoder calibrated: {encoder.n_centers} centers × "
                      f"{encoder.twindow} delays = {n_aff} afferents")
                print(f"   ✅ Pipeline ready — processing at {preproc.effective_fs} Hz")

            # ── Attention neuron ──────────────────────────────────────────
            dn_spike = dn.step(afferents)
            dn_buf.append(int(dn_spike))

            # ── Template layer (L1) ──────────────────────────────────────
            l1_spikes = l1.step(afferents, dn_spike)
            for idx in np.flatnonzero(l1_spikes):
                l1_spike_set.add(int(idx))

            # ── Control decoder ──────────────────────────────────────────
            result = decoder.step(l1_spikes, dn_spike)
            if result is not None:
                ctrl_val, confidence = result
                last_control    = ctrl_val
                last_confidence = confidence
                ctrl_sock.sendto(
                    struct.pack("!ff", ctrl_val, confidence), ctrl_addr)

            # ── Broadcast to browser (batched) ───────────────────────────
            if step_count % BROADCAST_EVERY == 0 and sample_buf:
                await _broadcast(json.dumps({
                    "t":          step_count,
                    "samples":    list(sample_buf),
                    "dn_flags":   list(dn_buf),
                    "spikes":     sorted(l1_spike_set),
                    "control":    round(last_control, 4),
                    "confidence": round(last_confidence, 4),
                    "dn_th":      round(dn.threshold, 2) if dn else 0,
                }))
                sample_buf.clear(); dn_buf.clear(); l1_spike_set.clear()

        # Yield to event loop periodically (every 100 samples ≈ 1.25 ms)
        if step_count % 100 == 0:
            await asyncio.sleep(0)


# ── Simulation loop (LSL mode — NCS playback via mne-lsl) ─────────────────────
async def sim_loop_lsl() -> None:
    """
    Receives samples from an LSL stream (e.g. Neuralynx .ncs replayed by
    lsl_player.py) and processes them through the pipeline:
      sample → Preprocessor → SpikeEncoder → AttentionNeuron → TemplateLayer → Decoder
    """
    from mne_lsl.stream import StreamLSL     # lazy import — only needed in LSL mode

    cfg = CFG
    stream_name  = cfg.get("lsl_stream_name", "NCS-Replay")
    pick_channel = cfg.get("lsl_pick_channel", None)
    bufsize      = cfg.get("lsl_bufsize_sec", 5.0)
    poll_sleep   = cfg.get("lsl_poll_interval_s", 0.0005)

    print(f"   🔍 Searching for LSL stream '{stream_name}' …")

    # ── Connect to LSL stream (blocks until found or timeout) ─────────
    stream = StreamLSL(bufsize=bufsize, name=stream_name)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: stream.connect(acquisition_delay=0.001, timeout=60),
    )

    sfreq   = stream.info["sfreq"]
    ch_names = list(stream.ch_names)

    # ── Channel selection ─────────────────────────────────────────────
    if pick_channel and pick_channel in ch_names:
        ch_idx = ch_names.index(pick_channel)
    else:
        ch_idx = 0

    # Override sampling rate so preprocessor uses the right time base
    cfg["sampling_rate_hz"] = int(sfreq)

    print(f"   ✅ Connected: '{stream_name}'  |"
          f"  {sfreq:.0f} Hz  |  ch: {ch_names[ch_idx]}")

    # ── Build pipeline ────────────────────────────────────────────────
    preproc = Preprocessor(cfg)
    enc_cfg = dict(cfg)
    enc_cfg["sampling_rate_hz"] = preproc.effective_fs
    encoder = SpikeEncoder(enc_cfg)

    dn:      AttentionNeuron | None = None
    l1:      TemplateLayer   | None = None
    decoder: ControlDecoder  | None = None

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = (cfg["control_target_host"], cfg["udp_control_port"])

    step_count      = 0
    last_control    = 0.0
    last_confidence = 0.0
    sample_buf      = []
    dn_buf          = []
    l1_spike_set    = set()

    dec_info = (f"  decimation {int(sfreq)}→{preproc.effective_fs} Hz"
                if preproc.do_decimate else "")
    print(f"   ⏳ Calibrating encoder (collecting noise statistics)…{dec_info}")

    while True:
        n_new = stream.n_new_samples
        if n_new == 0:
            await asyncio.sleep(poll_sleep)
            continue

        data, _ts = stream.get_data(winsize=n_new / sfreq)

        for i in range(data.shape[1]):
            raw_sample = float(data[ch_idx, i])

            # ── Preprocess (bandpass + decimation) ────────────────────
            processed = preproc.step(raw_sample)
            if not processed:
                continue

            for pp_sample in processed:
                step_count += 1
                sample_buf.append(round(pp_sample, 6))

                # ── Encode ────────────────────────────────────────────
                afferents = encoder.step(pp_sample)

                if not encoder.is_calibrated:
                    dn_buf.append(0)
                    if step_count % BROADCAST_EVERY == 0 and sample_buf:
                        await _broadcast(json.dumps({
                            "t": step_count, "samples": list(sample_buf),
                            "dn_flags": list(dn_buf), "spikes": [],
                            "control": 0, "confidence": 0,
                        }))
                        sample_buf.clear(); dn_buf.clear()
                    if step_count % 2000 == 0:
                        print(f"   ⏳ Calibrating… "
                              f"{step_count}/{enc_cfg['enc_noise_init_samples']}")
                    continue

                # First step after calibration → build downstream
                if dn is None:
                    n_aff = encoder.n_afferents
                    dn      = AttentionNeuron(enc_cfg, n_aff)
                    pipeline["dn"] = dn
                    l1 = TemplateLayer(cfg, n_aff)
                    decoder = ControlDecoder(enc_cfg, cfg["l1_n_neurons"])
                    print(f"   ✅ Encoder calibrated: {encoder.n_centers} centers × "
                          f"{encoder.twindow} delays = {n_aff} afferents")
                    print(f"   ✅ Pipeline ready — processing at "
                          f"{preproc.effective_fs} Hz")

                # ── Attention neuron ──────────────────────────────────
                dn_spike = dn.step(afferents)
                dn_buf.append(int(dn_spike))

                # ── Template layer (L1) ──────────────────────────────
                l1_spikes = l1.step(afferents, dn_spike)
                for idx in np.flatnonzero(l1_spikes):
                    l1_spike_set.add(int(idx))

                # ── Control decoder ──────────────────────────────────
                result = decoder.step(l1_spikes, dn_spike)
                if result is not None:
                    ctrl_val, confidence = result
                    last_control    = ctrl_val
                    last_confidence = confidence
                    ctrl_sock.sendto(
                        struct.pack("!ff", ctrl_val, confidence), ctrl_addr)

                # ── Broadcast to browser (batched) ───────────────────
                if step_count % BROADCAST_EVERY == 0 and sample_buf:
                    await _broadcast(json.dumps({
                        "t":          step_count,
                        "samples":    list(sample_buf),
                        "dn_flags":   list(dn_buf),
                        "spikes":     sorted(l1_spike_set),
                        "control":    round(last_control, 4),
                        "confidence": round(last_confidence, 4),
                        "dn_th":      round(dn.threshold, 2) if dn else 0,
                    }))
                    sample_buf.clear(); dn_buf.clear(); l1_spike_set.clear()

        # Yield to event loop after processing each chunk
        await asyncio.sleep(0)


# ── Simulation loop (synthetic mode — SpikeInterface ground truth) ─────────────
async def sim_loop_synthetic() -> None:
    """
    Generates a synthetic recording via spikeinterface with known ground-truth
    spike trains, then processes it through the pipeline.  Useful for
    development and benchmarking without hardware or NCS files.
    """
    import spikeinterface.full as si          # lazy — heavy dep

    cfg = CFG

    duration_s  = cfg.get("synth_duration_s", 20.0)
    fs          = cfg.get("synth_fs", 30_000)
    num_units   = cfg.get("synth_num_units", 2)
    noise_level = cfg.get("synth_noise_level", 8.0)
    seed        = cfg.get("synth_seed", 42)
    pace_rt     = cfg.get("synth_realtime", True)

    print(f"   🧪 Generating synthetic recording …")
    print(f"      Duration : {duration_s} s  |  Fs : {fs} Hz")
    print(f"      Units    : {num_units}  |  Noise : {noise_level}")

    recording, sorting = si.generate_ground_truth_recording(
        durations=[duration_s],
        sampling_frequency=float(fs),
        num_channels=1,
        num_units=num_units,
        seed=seed,
    )

    for uid in sorting.unit_ids:
        train = sorting.get_unit_spike_train(unit_id=uid)
        rate  = len(train) / duration_s
        print(f"      Unit {uid}: {len(train)} spikes ({rate:.1f} Hz)")

    # Override sampling rate for preprocessor
    cfg["sampling_rate_hz"] = int(fs)

    # ── Build pipeline ────────────────────────────────────────────────
    preproc = Preprocessor(cfg)
    enc_cfg = dict(cfg)
    enc_cfg["sampling_rate_hz"] = preproc.effective_fs
    encoder = SpikeEncoder(enc_cfg)

    dn:      AttentionNeuron | None = None
    l1:      TemplateLayer   | None = None
    decoder: ControlDecoder  | None = None

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = (cfg["control_target_host"], cfg["udp_control_port"])

    step_count      = 0
    last_control    = 0.0
    last_confidence = 0.0
    sample_buf      = []
    dn_buf          = []
    l1_spike_set    = set()

    dec_info = (f"  decimation {fs}→{preproc.effective_fs} Hz"
                if preproc.do_decimate else "")
    print(f"   ⏳ Calibrating encoder …{dec_info}")

    traces = recording.get_traces(segment_index=0)[:, 0]  # (n_samples,)
    n_total = len(traces)
    t0 = time.perf_counter()

    for frame_idx in range(n_total):
        raw_sample = float(traces[frame_idx])

        processed = preproc.step(raw_sample)
        if not processed:
            continue

        for pp_sample in processed:
            step_count += 1
            sample_buf.append(round(pp_sample, 6))

            afferents = encoder.step(pp_sample)

            if not encoder.is_calibrated:
                dn_buf.append(0)
                if step_count % BROADCAST_EVERY == 0 and sample_buf:
                    await _broadcast(json.dumps({
                        "t": step_count, "samples": list(sample_buf),
                        "dn_flags": list(dn_buf), "spikes": [],
                        "control": 0, "confidence": 0,
                    }))
                    sample_buf.clear(); dn_buf.clear()
                if step_count % 2000 == 0:
                    print(f"   ⏳ Calibrating… "
                          f"{step_count}/{enc_cfg['enc_noise_init_samples']}")
                # Real-time pacing (based on raw recording clock)
                if pace_rt and frame_idx % 500 == 0:
                    expected = frame_idx / fs
                    elapsed  = time.perf_counter() - t0
                    if expected > elapsed:
                        await asyncio.sleep(expected - elapsed)
                    else:
                        await asyncio.sleep(0)
                continue

            if dn is None:
                n_aff = encoder.n_afferents
                dn      = AttentionNeuron(enc_cfg, n_aff)
                pipeline["dn"] = dn
                l1 = TemplateLayer(cfg, n_aff)
                decoder = ControlDecoder(enc_cfg, cfg["l1_n_neurons"])
                print(f"   ✅ Encoder calibrated: {encoder.n_centers} centres × "
                      f"{encoder.twindow} delays = {n_aff} afferents")
                print(f"   ✅ Pipeline ready — processing at "
                      f"{preproc.effective_fs} Hz")

            dn_spike = dn.step(afferents)
            dn_buf.append(int(dn_spike))

            l1_spikes = l1.step(afferents, dn_spike)
            for idx in np.flatnonzero(l1_spikes):
                l1_spike_set.add(int(idx))

            result = decoder.step(l1_spikes, dn_spike)
            if result is not None:
                ctrl_val, confidence = result
                last_control    = ctrl_val
                last_confidence = confidence
                ctrl_sock.sendto(
                    struct.pack("!ff", ctrl_val, confidence), ctrl_addr)

            if step_count % BROADCAST_EVERY == 0 and sample_buf:
                await _broadcast(json.dumps({
                    "t":          step_count,
                    "samples":    list(sample_buf),
                    "dn_flags":   list(dn_buf),
                    "spikes":     sorted(l1_spike_set),
                    "control":    round(last_control, 4),
                    "confidence": round(last_confidence, 4),
                    "dn_th":      round(dn.threshold, 2) if dn else 0,
                }))
                sample_buf.clear(); dn_buf.clear(); l1_spike_set.clear()

        # Pace to recording real-time
        if pace_rt and frame_idx % 500 == 0:
            expected = frame_idx / fs
            elapsed  = time.perf_counter() - t0
            if expected > elapsed:
                await asyncio.sleep(expected - elapsed)
            else:
                await asyncio.sleep(0)
        elif frame_idx % 500 == 0:
            await asyncio.sleep(0)  # yield even in fast mode

    elapsed_total = time.perf_counter() - t0
    print(f"\n   🏁 Synthetic recording finished: {step_count} processed samples "
          f"in {elapsed_total:.1f}s")
    print(f"      Ground-truth sorting saved in-memory; "
          f"use ground_truth_generator.py for scoring.")
    # Keep WS alive so the browser can scroll back
    while True:
        await asyncio.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    cfg   = CFG
    mode  = cfg.get("mode", "electrode")
    queue = asyncio.Queue(maxsize=4096)

    # 0. Free stale ports before binding
    _free_port(cfg["ws_port"])
    _free_port(cfg["http_port"])

    # 1. HTTP static server — background daemon thread
    threading.Thread(
        target=_run_http, args=(cfg["http_port"],), daemon=True
    ).start()

    # 2. WebSocket server
    async def run_ws() -> None:
        async with websockets.serve(
            ws_handler, "0.0.0.0", cfg["ws_port"],
            reuse_address=True,
            reuse_port=True,
        ):
            await asyncio.Future()

    loop = asyncio.get_running_loop()

    if mode == "synthetic":
        # ── Synthetic (SpikeInterface) mode ───────────────────────────────
        n_l1 = cfg["l1_n_neurons"]
        dur  = cfg.get("synth_duration_s", 20.0)
        print(f"\n⚡ SNN Agent  [synthetic mode]")
        print(f"   L1 neurons   →  {n_l1}")
        print(f"   Strategy     →  {cfg['ctrl_strategy']}")
        print(f"   Duration     →  {dur} s")
        print(f"   Browser      →  http://localhost:{cfg['http_port']}")
        print(f"   WebSocket    →  ws://localhost:{cfg['ws_port']}\n")

        await asyncio.gather(sim_loop_synthetic(), run_ws())

    elif mode == "lsl":
        # ── LSL mode ──────────────────────────────────────────────────────
        lsl_name = cfg.get("lsl_stream_name", "NCS-Replay")
        n_l1 = cfg["l1_n_neurons"]
        print(f"\n⚡ SNN Agent  [LSL mode]")
        print(f"   L1 neurons   →  {n_l1}")
        print(f"   Strategy     →  {cfg['ctrl_strategy']}")
        print(f"   LSL stream   →  '{lsl_name}'")
        print(f"   Browser      →  http://localhost:{cfg['http_port']}")
        print(f"   Control out  →  UDP port {cfg['udp_control_port']}  "
              f"(pack: !ff  control, confidence)")
        print(f"   WebSocket    →  ws://localhost:{cfg['ws_port']}\n")

        await asyncio.gather(sim_loop_lsl(), run_ws())

    else:
        # ── Electrode mode ────────────────────────────────────────────────
        await loop.create_datagram_endpoint(
            lambda: UDPReceiver(queue, ELEC_FMT, ELEC_FRAME_SIZE),
            local_addr=("0.0.0.0", cfg["udp_electrode_port"]),
        )

        n_l1 = cfg["l1_n_neurons"]
        print(f"\n⚡ SNN Agent  [electrode mode]")
        print(f"   L1 neurons   →  {n_l1}")
        print(f"   Strategy     →  {cfg['ctrl_strategy']}")
        print(f"   Browser      →  http://localhost:{cfg['http_port']}")
        print(f"   Electrode in →  UDP port {cfg['udp_electrode_port']}  "
              f"(pack: !Hf  magic=0xABCD, sample_f32)")
        print(f"   Control out  →  UDP port {cfg['udp_control_port']}  "
              f"(pack: !ff  control, confidence)")
        print(f"   WebSocket    →  ws://localhost:{cfg['ws_port']}\n")

        await asyncio.gather(sim_loop_electrode(queue), run_ws())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent stopped.")
