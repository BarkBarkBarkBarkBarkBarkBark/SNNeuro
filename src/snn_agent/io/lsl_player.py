"""
snn_agent.io.lsl_player — Stream a Neuralynx .ncs recording over LSL.

Pushes samples at the original sampling rate so downstream consumers
see the same timing as a live acquisition system.

Requires: ``pip install snn-agent[lsl]``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mne
from mne_lsl.player import PlayerLSL

__all__ = ["main"]


def main() -> None:
    """CLI entry point (``snn-lsl``)."""
    parser = argparse.ArgumentParser(
        description="Stream Neuralynx .ncs files over LSL at real-time rate.",
    )
    parser.add_argument("path", help="Path to a .ncs file or directory.")
    parser.add_argument("--channel", "-c", default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--name", default="NCS-Replay")
    parser.add_argument("--n-repeat", type=float, default=float("inf"))
    args = parser.parse_args()

    path = Path(args.path).resolve()

    if path.is_file() and path.suffix.lower() == ".ncs":
        input_dir = path.parent
        default_channel = path.stem
    elif path.is_dir():
        ncs_files = sorted(path.glob("*.ncs")) + sorted(path.glob("*.NCS"))
        if not ncs_files:
            sys.exit(f"No .ncs files found in {path}")
        input_dir = path
        default_channel = None
    else:
        sys.exit(f"Not a .ncs file or directory: {path}")

    print(f"\n📂 Loading Neuralynx data from: {input_dir}")
    raw = mne.io.read_raw_neuralynx(input_dir, preload=True)

    print(f"   Channels : {raw.ch_names}")
    print(f"   Sfreq    : {raw.info['sfreq']:.0f} Hz")
    print(f"   Duration : {raw.times[-1]:.2f} s  ({raw.n_times} samples)")

    if len(raw.ch_names) > 1:
        pick = args.channel or default_channel
        if pick:
            if pick not in raw.ch_names:
                close = [
                    c
                    for c in raw.ch_names
                    if c.lower() in pick.lower() or pick.lower() in c.lower()
                ]
                if close:
                    print(f"   ⚠️  '{pick}' not found, using: {close[0]}")
                    pick = close[0]
                else:
                    sys.exit(f"Channel '{pick}' not in {raw.ch_names}.")
            raw.pick([pick])
            print(f"   Picked   : {pick}")
        else:
            print("   ℹ️  Multiple channels — streaming all.")
    else:
        print(f"   Picked   : {raw.ch_names[0]}")

    n_rep = np.inf if np.isinf(args.n_repeat) else int(args.n_repeat)
    player = PlayerLSL(raw, chunk_size=args.chunk_size, n_repeat=n_rep, name=args.name)
    player.start()

    sfreq = raw.info["sfreq"]
    chunk_latency_ms = args.chunk_size / sfreq * 1000

    print("\n▶️  LSL stream started")
    print(f"   Name         : {args.name}")
    print(f"   Sfreq        : {sfreq:.0f} Hz")
    print(f"   Chunk size   : {args.chunk_size}  ({chunk_latency_ms:.2f} ms)")
    print(f"   Repeat       : {'∞' if np.isinf(n_rep) else int(n_rep)}")
    print("\n   Waiting for consumers…  (Ctrl+C to stop)\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        print("\n⏹  Player stopped.")


if __name__ == "__main__":
    main()
