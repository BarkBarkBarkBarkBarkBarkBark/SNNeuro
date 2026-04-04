"""
snn_agent.io.test_electrode — Synthetic electrode signal sender for testing.

Generates 80 kHz simulated extracellular recording with Gaussian noise
and embedded spike waveforms, sends via UDP to a running SNN agent.
Also listens for control signal output.

Usage::

    snn-test-electrode               # default: 2 templates, SNR=5
    snn-test-electrode --snr 3
    snn-test-electrode --duration 30
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time

import numpy as np

__all__ = ["main"]

HOST = "127.0.0.1"
PORT = 9001
MAGIC = 0xABCD
FMT = "!Hf"
SR = 80_000
CTRL_PORT = 9002


def make_templates(n_templates: int = 2, n_samples: int = 40) -> list[np.ndarray]:
    rng = np.random.default_rng(seed=123)
    t = np.linspace(0, 1, n_samples)
    templates = []
    for i in range(n_templates):
        freq = 2.0 + i * 0.7
        phase = i * 0.5
        amp = 0.7 + rng.random() * 0.6
        wf = amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-3.0 * t)
        wf += 0.3 * amp * np.sin(2 * np.pi * (freq * 1.5) * t) * np.exp(-5.0 * t)
        templates.append(wf.astype(np.float32))
    return templates


def generate_signal(
    duration_s: float, snr: float, templates: list[np.ndarray], firing_rate_hz: float = 30.0
) -> np.ndarray:
    n_samples = int(duration_s * SR)
    rng = np.random.default_rng(seed=42)
    signal = rng.standard_normal(n_samples).astype(np.float32)

    n_spikes = int(duration_s * firing_rate_hz)
    spike_times = np.sort(rng.integers(100, n_samples - 100, size=n_spikes))
    spike_labels = rng.integers(0, len(templates), size=n_spikes)

    for t_spike, label in zip(spike_times, spike_labels):
        wf = templates[label]
        end = min(t_spike + len(wf), n_samples)
        n = end - t_spike
        signal[t_spike:end] += snr * wf[:n]

    return signal


def _listen_control():
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
                if n_ctrl % 10 == 1:
                    print(
                        f"   ← CONTROL  val={ctrl_val:+.4f}  "
                        f"conf={confidence:.4f}  (#{n_ctrl})"
                    )
        except socket.timeout:
            pass
        except Exception:
            break


def main() -> None:
    """CLI entry point (``snn-test-electrode``)."""
    parser = argparse.ArgumentParser(description="Send synthetic electrode signal")
    parser.add_argument("--snr", type=float, default=5.0)
    parser.add_argument("--templates", type=int, default=2)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    templates = make_templates(args.templates)
    signal = generate_signal(args.duration, args.snr, templates)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("\n🔬 Electrode Test Sender")
    print(f"   Signal      →  {args.duration}s @ {SR} Hz  ({len(signal)} samples)")
    print(f"   Templates   →  {args.templates} waveform shapes, SNR={args.snr}")
    print(f"   Sending to  →  {HOST}:{PORT}")
    print(f"   Listening   →  control output on port {CTRL_PORT}")
    print("   Press Ctrl+C to stop\n")

    threading.Thread(target=_listen_control, daemon=True).start()

    batch_size = 80
    i = 0
    t_start = time.perf_counter()

    try:
        while i < len(signal):
            batch_end = min(i + batch_size, len(signal))
            for j in range(i, batch_end):
                sock.sendto(struct.pack(FMT, MAGIC, float(signal[j])), (HOST, PORT))
            i = batch_end

            elapsed = time.perf_counter() - t_start
            expected = i / SR
            if expected > elapsed:
                time.sleep(expected - elapsed)

            if i % (SR * 2) == 0:
                pct = 100 * i / len(signal)
                print(f"   → {pct:.0f}%  ({i}/{len(signal)} samples)")

    except KeyboardInterrupt:
        pass

    elapsed = time.perf_counter() - t_start
    print(f"\n   Done. Sent {i} samples in {elapsed:.2f}s ({i / elapsed:.0f} Hz)")


if __name__ == "__main__":
    main()
