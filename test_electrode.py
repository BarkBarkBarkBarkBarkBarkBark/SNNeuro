"""
test_electrode.py — Send synthetic electrode signal to a running SNN agent.

Generates a simulated 80 kHz extracellular recording with:
  - Gaussian background noise
  - Occasional spike waveforms (2–3 distinct shapes) embedded in the noise

The agent must already be running in electrode mode:
  python server.py

Usage:
  python test_electrode.py                  # default: 2 spike templates, SNR=5
  python test_electrode.py --snr 3          # lower signal-to-noise ratio
  python test_electrode.py --templates 4    # more distinct spike shapes
  python test_electrode.py --duration 30    # seconds of signal to send

Frame format:  struct.pack('!Hf', 0xABCD, sample_float32)

Also listens on UDP port 9002 for control signal output from the SNN
and prints it to the console.
"""

import socket
import struct
import time
import sys
import threading
import numpy as np

HOST  = "127.0.0.1"
PORT  = 9001          # must match udp_electrode_port in config.py
MAGIC = 0xABCD
FMT   = "!Hf"
SR    = 80_000        # sampling rate (Hz)

CTRL_PORT = 9002      # control signal output from the SNN

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ── Spike waveform templates ─────────────────────────────────────────────────
def make_templates(n_templates: int = 2, n_samples: int = 40) -> list[np.ndarray]:
    """
    Generate distinct synthetic extracellular spike waveforms.
    Each is a short (0.5 ms) biphasic or triphasic shape.
    """
    rng = np.random.default_rng(seed=123)
    t = np.linspace(0, 1, n_samples)
    templates = []
    for i in range(n_templates):
        # Base shape: damped sinusoid with varying frequency and phase
        freq = 2.0 + i * 0.7
        phase = i * 0.5
        amp = 0.7 + rng.random() * 0.6
        waveform = amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-3.0 * t)
        # Add a small secondary bump
        waveform += 0.3 * amp * np.sin(2 * np.pi * (freq * 1.5) * t) * np.exp(-5.0 * t)
        templates.append(waveform.astype(np.float32))
    return templates


def generate_signal(duration_s: float, snr: float, templates: list[np.ndarray],
                    firing_rate_hz: float = 30.0) -> np.ndarray:
    """
    Generate a synthetic electrode signal.
      - Background: Gaussian noise with σ=1
      - Spikes: templates embedded at random times, scaled by snr
    """
    n_samples = int(duration_s * SR)
    rng = np.random.default_rng(seed=42)

    signal = rng.standard_normal(n_samples).astype(np.float32)

    # Embed spikes
    n_spikes = int(duration_s * firing_rate_hz)
    spike_times = np.sort(rng.integers(100, n_samples - 100, size=n_spikes))
    spike_labels = rng.integers(0, len(templates), size=n_spikes)

    for t_spike, label in zip(spike_times, spike_labels):
        wf = templates[label]
        end = min(t_spike + len(wf), n_samples)
        n = end - t_spike
        signal[t_spike:end] += snr * wf[:n]

    return signal


# ── Control signal listener ──────────────────────────────────────────────────
def listen_control():
    """Background thread: print control signals received from the SNN."""
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.bind(("0.0.0.0", CTRL_PORT))
    ctrl_sock.settimeout(1.0)
    n_ctrl = 0
    while True:
        try:
            data, _ = ctrl_sock.recvfrom(64)
            if len(data) >= 8:
                ctrl_val, confidence = struct.unpack("!ff", data[:8])
                n_ctrl += 1
                if n_ctrl % 10 == 1:  # print every 10th to avoid spam
                    print(f"   ← CONTROL  val={ctrl_val:+.4f}  "
                          f"conf={confidence:.4f}  (#{n_ctrl})")
        except socket.timeout:
            pass
        except Exception:
            break


# ── Main sender loop ─────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    snr = 5.0
    n_templates = 2
    duration = 10.0

    # Parse args
    if "--snr" in args:
        idx = args.index("--snr")
        snr = float(args[idx + 1])
    if "--templates" in args:
        idx = args.index("--templates")
        n_templates = int(args[idx + 1])
    if "--duration" in args:
        idx = args.index("--duration")
        duration = float(args[idx + 1])

    templates = make_templates(n_templates)
    signal = generate_signal(duration, snr, templates)

    print(f"\n🔬 Electrode Test Sender")
    print(f"   Signal      →  {duration}s @ {SR} Hz  ({len(signal)} samples)")
    print(f"   Templates   →  {n_templates} waveform shapes, SNR={snr}")
    print(f"   Sending to  →  {HOST}:{PORT}")
    print(f"   Listening   →  control output on port {CTRL_PORT}")
    print(f"   Press Ctrl+C to stop\n")

    # Start control listener in background
    threading.Thread(target=listen_control, daemon=True).start()

    # Send samples at real-time rate (batch to avoid per-sample overhead)
    batch_size = 80   # 80 samples = 1 ms at 80 kHz
    batch_delay = batch_size / SR

    i = 0
    t_start = time.perf_counter()

    try:
        while i < len(signal):
            batch_end = min(i + batch_size, len(signal))
            for j in range(i, batch_end):
                sock.sendto(struct.pack(FMT, MAGIC, float(signal[j])), (HOST, PORT))
            i = batch_end

            # Pace to real-time
            elapsed = time.perf_counter() - t_start
            expected = i / SR
            if expected > elapsed:
                time.sleep(expected - elapsed)

            # Progress
            if i % (SR * 2) == 0:  # every 2 seconds
                pct = 100 * i / len(signal)
                print(f"   → {pct:.0f}%  ({i}/{len(signal)} samples)")

    except KeyboardInterrupt:
        pass

    elapsed = time.perf_counter() - t_start
    print(f"\n   Done. Sent {i} samples in {elapsed:.2f}s "
          f"(effective rate: {i/elapsed:.0f} Hz)")


if __name__ == "__main__":
    main()
