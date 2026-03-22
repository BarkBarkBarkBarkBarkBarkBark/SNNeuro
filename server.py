"""
server.py — asyncio event loop for the SNN agent.

Supports three modes (set in config.py):

  "binary"    — Original mode.  UDP binary integer input, R-STDP learning.
  "electrode" — ANNet-derived mode.  UDP raw electrode samples in,
                UDP control signal out, temporal receptive field encoding,
                attention neuron gating, L1 template STDP.
  "lsl"       — Lab Streaming Layer mode.  Receives data from an LSL stream
                (e.g. Neuralynx .ncs via lsl_player.py).  Same pipeline as
                electrode mode but input comes from LSL instead of UDP.

Services (both modes):
  1. UDP  port <udp_port>           — binary mode input  OR
     UDP  port <udp_electrode_port> — electrode mode input
  2. WS   port <ws_port>           — streams spikes to browser; accepts reward
  3. HTTP port <http_port>         — serves index.html
  4. UDP  port <udp_control_port>  — electrode mode control signal output

Run:
  python server.py
"""

import asyncio
import json
import socket
import struct
import threading
import http.server
from pathlib import Path

import numpy as np
import websockets

from config import CFG
from snn import Network, TemplateLayer
from encoder import SpikeEncoder, AttentionNeuron
from decoder import ControlDecoder


# ── Shared mutable state ──────────────────────────────────────────────────────
net        = Network(CFG)      # binary mode network (always instantiated)
ws_clients = set()             # active WebSocket connections
reward     = [0.0]             # current reward; updated by browser over WebSocket
pipeline   = {}                # live refs to pipeline objects (for WS control)

UDP_MAGIC       = 0xABCD
UDP_FMT         = "!HI"
UDP_FRAME_SIZE  = struct.calcsize(UDP_FMT)
BROADCAST_EVERY = CFG.get("broadcast_every", 5)

# Electrode mode frame format: magic(uint16) + sample(float32)
ELEC_FMT        = "!Hf"
ELEC_FRAME_SIZE = struct.calcsize(ELEC_FMT)


# ── HTTP static file server ───────────────────────────────────────────────────
class _StaticHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files from the project directory. Silences access log noise."""
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(Path(__file__).parent), **kw)

    def log_message(self, *_):
        pass  # suppress per-request log lines


def _run_http(port: int) -> None:
    http.server.HTTPServer(("", port), _StaticHandler).serve_forever()


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
                if "reward" in data:
                    reward[0] = max(-1.0, min(1.0, float(data["reward"])))
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


# ── Simulation loop (binary mode — original) ──────────────────────────────────
async def sim_loop_binary(queue: asyncio.Queue) -> None:
    cfg        = CFG
    dt         = cfg["dt"]
    bw_in      = cfg["input_bit_width"]
    current_in = 0

    while True:
        while not queue.empty():
            current_in = queue.get_nowait()

        in_spikes            = Network.encode(current_in, bw_in)
        h_spikes, out_spikes = net.step(in_spikes)
        net.stdp_update(in_spikes, reward=reward[0])

        if net.t % BROADCAST_EVERY == 0:
            await _broadcast(json.dumps({
                "t":      net.t,
                "spikes": np.where(h_spikes)[0].tolist(),
                "out":    Network.decode(out_spikes),
            }))

        await asyncio.sleep(dt)


# ── Simulation loop (electrode mode — ANNet-derived) ──────────────────────────
async def sim_loop_electrode(queue: asyncio.Queue) -> None:
    """
    Streaming pipeline:
      sample → SpikeEncoder → AttentionNeuron → TemplateLayer → ControlDecoder
    All layers run in-process, sample-by-sample.  No inter-stage files.
    """
    cfg = CFG

    # Build pipeline components
    encoder = SpikeEncoder(cfg)
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

    print("   ⏳ Calibrating encoder (collecting noise statistics)…")

    while True:
        # Drain queue — always use freshest sample
        sample = None
        while not queue.empty():
            sample = queue.get_nowait()

        if sample is None:
            await asyncio.sleep(0.0001)  # 100 µs idle poll
            continue

        step_count += 1

        # ── Encode ────────────────────────────────────────────────────────
        afferents = encoder.step(float(sample))

        if not encoder.is_calibrated:
            # Still collecting noise statistics
            if step_count % 2000 == 0:
                print(f"   ⏳ Calibrating… {step_count}/{cfg['enc_noise_init_samples']}")
            continue

        # First step after calibration: build downstream components
        if dn is None:
            n_aff = encoder.n_afferents
            dn      = AttentionNeuron(cfg, n_aff)
            pipeline["dn"] = dn
            if cfg.get("backend") == "torch":
                from snn_torch import TorchTemplateLayer
                l1 = TorchTemplateLayer(cfg, n_aff)
            else:
                l1 = TemplateLayer(cfg, n_aff)
            decoder = ControlDecoder(cfg, cfg["l1_n_neurons"])
            print(f"   ✅ Encoder calibrated: {encoder.n_centers} centers × "
                  f"{encoder.twindow} delays = {n_aff} afferents")
            print(f"   ✅ Pipeline ready — processing samples")

        # ── Attention neuron ──────────────────────────────────────────────
        dn_spike = dn.step(afferents)

        # ── Template layer (L1) ──────────────────────────────────────────
        l1_spikes = l1.step(afferents, dn_spike)

        # ── Control decoder ──────────────────────────────────────────────
        result = decoder.step(l1_spikes, dn_spike)
        if result is not None:
            ctrl_val, confidence = result
            last_control    = ctrl_val
            last_confidence = confidence
            # Send control signal via UDP
            ctrl_sock.sendto(
                struct.pack("!ff", ctrl_val, confidence), ctrl_addr)

        # ── Broadcast to browser ─────────────────────────────────────────
        if step_count % BROADCAST_EVERY == 0:
            firing = np.where(l1_spikes)[0].tolist() if l1 is not None else []
            await _broadcast(json.dumps({
                "t":          step_count,
                "spikes":     firing,
                "dn":         int(dn_spike),
                "control":    round(last_control, 4),
                "confidence": round(last_confidence, 4),
                "sample":     round(float(sample), 6),
                "dn_th":      round(dn.threshold, 2) if dn else 0,
            }))

        # Yield to event loop periodically (every 100 samples ≈ 1.25 ms)
        if step_count % 100 == 0:
            await asyncio.sleep(0)


# ── Simulation loop (LSL mode — NCS playback via mne-lsl) ─────────────────────
async def sim_loop_lsl() -> None:
    """
    Receives samples from an LSL stream (e.g. Neuralynx .ncs replayed by
    lsl_player.py) and processes them through the same pipeline as electrode
    mode:  sample → SpikeEncoder → AttentionNeuron → TemplateLayer → Decoder
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
    # run connect in executor so we don't freeze the event loop
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

    # Override sampling rate so encoder / decoder use the right time base
    cfg["sampling_rate_hz"] = int(sfreq)

    print(f"   ✅ Connected: '{stream_name}'  |"
          f"  {sfreq:.0f} Hz  |  ch: {ch_names[ch_idx]}")

    # ── Build pipeline ────────────────────────────────────────────────
    encoder = SpikeEncoder(cfg)
    dn:      AttentionNeuron | None = None
    l1:      TemplateLayer   | None = None
    decoder: ControlDecoder  | None = None

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_addr = (cfg["control_target_host"], cfg["udp_control_port"])

    step_count      = 0
    last_control    = 0.0
    last_confidence = 0.0

    print("   ⏳ Calibrating encoder (collecting noise statistics)…")

    while True:
        n_new = stream.n_new_samples
        if n_new == 0:
            await asyncio.sleep(poll_sleep)
            continue

        # Retrieve only the freshly-arrived samples
        data, _ts = stream.get_data(winsize=n_new / sfreq)
        # data shape: (n_channels, n_samples)

        for i in range(data.shape[1]):
            sample = float(data[ch_idx, i])
            step_count += 1

            # ── Encode ────────────────────────────────────────────────
            afferents = encoder.step(sample)

            if not encoder.is_calibrated:
                if step_count % 2000 == 0:
                    print(f"   ⏳ Calibrating… "
                          f"{step_count}/{cfg['enc_noise_init_samples']}")
                continue

            # First step after calibration → build downstream
            if dn is None:
                n_aff = encoder.n_afferents
                dn      = AttentionNeuron(cfg, n_aff)
                pipeline["dn"] = dn
                if cfg.get("backend") == "torch":
                    from snn_torch import TorchTemplateLayer
                    l1 = TorchTemplateLayer(cfg, n_aff)
                else:
                    l1 = TemplateLayer(cfg, n_aff)
                decoder = ControlDecoder(cfg, cfg["l1_n_neurons"])
                print(f"   ✅ Encoder calibrated: {encoder.n_centers} centers × "
                      f"{encoder.twindow} delays = {n_aff} afferents")
                print(f"   ✅ Pipeline ready — processing samples")

            # ── Attention neuron ──────────────────────────────────────
            dn_spike = dn.step(afferents)

            # ── Template layer (L1) ──────────────────────────────────
            l1_spikes = l1.step(afferents, dn_spike)

            # ── Control decoder ──────────────────────────────────────
            result = decoder.step(l1_spikes, dn_spike)
            if result is not None:
                ctrl_val, confidence = result
                last_control    = ctrl_val
                last_confidence = confidence
                ctrl_sock.sendto(
                    struct.pack("!ff", ctrl_val, confidence), ctrl_addr)

            # ── Broadcast to browser ─────────────────────────────────
            if step_count % BROADCAST_EVERY == 0:
                firing = (
                    np.where(l1_spikes)[0].tolist() if l1 is not None else []
                )
                await _broadcast(json.dumps({
                    "t":          step_count,
                    "spikes":     firing,
                    "dn":         int(dn_spike),
                    "control":    round(last_control, 4),
                    "confidence": round(last_confidence, 4),
                    "sample":     round(sample, 6),
                    "dn_th":      round(dn.threshold, 2) if dn else 0,
                }))

        # Yield to event loop after processing each chunk
        await asyncio.sleep(0)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    cfg   = CFG
    mode  = cfg.get("mode", "binary")
    queue = asyncio.Queue(maxsize=4096)

    # 1. HTTP static server — background daemon thread
    threading.Thread(
        target=_run_http, args=(cfg["http_port"],), daemon=True
    ).start()

    # 2. WebSocket server
    async def run_ws() -> None:
        async with websockets.serve(ws_handler, "0.0.0.0", cfg["ws_port"]):
            await asyncio.Future()

    loop = asyncio.get_running_loop()

    if mode == "lsl":
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

    elif mode == "electrode":
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

    else:
        # ── Binary mode (original) ───────────────────────────────────────
        await loop.create_datagram_endpoint(
            lambda: UDPReceiver(queue, UDP_FMT, UDP_FRAME_SIZE),
            local_addr=("0.0.0.0", cfg["udp_port"]),
        )

        n_in  = cfg["input_bit_width"]
        n_h   = cfg["n_hidden"]
        n_out = cfg["output_bit_width"]
        print(f"\n⚡ SNN Agent  [{n_in} → {n_h} → {n_out} neurons]")
        print(f"   Browser    →  http://localhost:{cfg['http_port']}")
        print(f"   UDP input  →  port {cfg['udp_port']}   (run: python test_sender.py)")
        print(f"   WebSocket  →  ws://localhost:{cfg['ws_port']}")
        print(f"   Reward     →  use ✅/❌ buttons in browser, or send {{\"reward\": ±1}} via WS\n")

        await asyncio.gather(sim_loop_binary(queue), run_ws())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent stopped.")
